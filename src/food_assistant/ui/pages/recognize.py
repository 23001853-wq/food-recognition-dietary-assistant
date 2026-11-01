import hashlib
import json
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import streamlit as st

from food_assistant.config import Settings
from food_assistant.domain.errors import AppError
from food_assistant.services.image_service import ImageService
from food_assistant.ui.client import APIClient, disclaimer, food_label
from food_assistant.ui.components.image_cropper import render_cropper
from food_assistant.ui.components.nutrition_editor import render_estimate
from food_assistant.ui.components.prediction_card import render_predictions
from food_assistant.ui.labels import PORTION_LABELS, meal_label, portion_label


def render():
    st.title("Nhận diện món ăn")
    st.caption("Tải ảnh → Xác nhận món → Chọn khẩu phần → Lưu bữa ăn")
    api = APIClient()
    models = api.request("GET", "/api/v1/models")
    foods = api.request("GET", "/api/v1/foods")
    if models is None or foods is None:
        return
    selectors = [
        m.get("selector") or m["name"]
        for m in models
        if m.get("is_mock") or (m.get("trained") and m.get("enabled", True))
    ]
    if not selectors:
        st.info(
            "MODEL_NOT_READY — Chưa có model đã huấn luyện. "
            "Chỉ bật Mock AI trong môi trường phát triển."
        )
        return
    model = st.selectbox("Mô hình", selectors)
    if model == "mock":
        st.warning(
            "MOCK AI — Kết quả giả lập xác định, không phải nhận diện thực hoặc kết quả thí nghiệm."
        )
    st.markdown('<div class="food-section-label">01 · ẢNH MÓN ĂN</div>', unsafe_allow_html=True)
    source = st.radio(
        "Chọn nguồn ảnh", ["Tải ảnh", "Chụp ảnh"], horizontal=True, key="image_source"
    )
    if st.session_state.get("previous_image_source") != source:
        for key in (
            "recognition_result",
            "recognition_signature",
            "meal_request_id",
            "recognition_saved",
            "confirmed_food",
            "confirm_food_checkbox",
            "nutrition_context",
        ):
            st.session_state.pop(key, None)
        st.session_state["previous_image_source"] = source
    if source == "Chụp ảnh":
        st.caption(
            "Cho phép trình duyệt dùng camera, đặt món trong khung hình rồi chụp. "
            "Camera cần localhost hoặc kết nối HTTPS; bạn có thể chuyển sang Tải ảnh."
        )
        upload = st.camera_input("Chụp món ăn", key="food_camera")
    else:
        upload = st.file_uploader("Ảnh JPEG/PNG", type=["jpg", "jpeg", "png"], key="food_upload")
        st.caption("Ảnh rõ, đủ sáng và có một món chính giúp bạn xác nhận kết quả dễ hơn.")
    if upload is None:
        for key in [
            "recognition_result",
            "recognition_signature",
            "meal_request_id",
            "recognition_saved",
            "confirmed_food",
            "confirm_food_checkbox",
            "nutrition_context",
        ]:
            st.session_state.pop(key, None)
        return
    data = upload.getvalue()
    try:
        settings = Settings()
        image = ImageService(
            settings.max_upload_mb * 1024 * 1024, settings.max_image_pixels
        ).validate_and_decode(data, upload.type)
    except AppError as exc:
        st.error(exc.message)
        st.session_state.pop("recognition_result", None)
        return
    preview, guidance = st.columns([1.4, 1])
    with preview:
        st.image(image, caption="Ảnh đầu vào", width="stretch")
        box = render_cropper(image)
    with guidance:
        with st.container(border=True):
            st.markdown("#### Sẵn sàng phân tích")
            st.write("1. Kiểm tra ảnh hoặc chọn vùng chứa món chính.")
            st.write("2. Phân tích và xác nhận món phù hợp.")
            st.write("3. Chọn khẩu phần rồi lưu vào nhật ký.")
            st.caption("Ảnh chụp và ảnh tải lên đều dùng cùng quy trình kiểm tra và nhận diện.")
    signature = hashlib.sha256(data + json.dumps([box, model, source]).encode()).hexdigest()
    if st.session_state.get("recognition_signature") != signature:
        st.session_state.pop("recognition_result", None)
        st.session_state["recognition_signature"] = signature
        st.session_state["meal_request_id"] = str(uuid4())
        st.session_state.pop("recognition_saved", None)
        st.session_state.pop("confirmed_food", None)
        st.session_state.pop("confirm_food_checkbox", None)
    if st.button("Phân tích", type="primary"):
        with st.spinner("Đang phân tích ảnh..."):
            result = api.request(
                "POST",
                "/api/v1/predictions",
                files={"file": (upload.name, data, upload.type)},
                data={
                    "model_name": model,
                    "top_k": 3,
                    **({"crop_box": json.dumps(box)} if box else {}),
                },
            )
        if result:
            st.session_state["recognition_result"] = result
            st.session_state["meal_request_id"] = str(uuid4())
            st.session_state.pop("recognition_saved", None)
            st.session_state.pop("confirm_food_checkbox", None)
    result = st.session_state.get("recognition_result")
    if not result:
        return
    st.divider()
    st.markdown(
        '<div class="food-section-label">02 · XÁC NHẬN & DINH DƯỠNG</div>', unsafe_allow_html=True
    )
    render_predictions(result)
    by_id = {f["id"]: f for f in foods}
    ids = list(by_id) + ["other"]
    predicted = next(
        (f["id"] for f in foods if f["dataset_key"] == result["predictions"][0]["class_key"]),
        "other",
    )
    selected = st.selectbox(
        "Xác nhận hoặc sửa món",
        ids,
        index=ids.index(predicted),
        format_func=lambda i: "Món khác / chưa hỗ trợ" if i == "other" else food_label(by_id[i]),
        key="confirmed_food",
    )
    custom = st.text_input("Tên món khác") if selected == "other" else None
    portion = st.selectbox("Khẩu phần", list(PORTION_LABELS), index=1, format_func=portion_label)
    context = (result["prediction_id"], selected, portion)
    if st.session_state.get("nutrition_context") != context:
        for key in (
            "nutrition_use_grams",
            "nutrition_grams",
            "nutrition_manual",
            "nutrition_kcal",
            "confirm_food_checkbox",
        ):
            st.session_state.pop(key, None)
        st.session_state["nutrition_context"] = context
    base = (
        api.request("GET", f"/api/v1/nutrition/{selected}", params={"portion": portion})
        if selected != "other"
        else None
    )
    custom_grams = portion == "custom" or st.checkbox(
        "Nhập gram thực tế", key="nutrition_use_grams"
    )
    suggested = (base or {}).get("grams")
    grams = (
        st.number_input(
            "Gram",
            min_value=0.01,
            max_value=10000.0,
            value=float(suggested) if suggested is not None else None,
            key="nutrition_grams",
            placeholder="Nhập khối lượng thực tế",
        )
        if custom_grams
        else None
    )
    manual = st.checkbox("Tự nhập kcal (lưu nguồn: người dùng nhập)", key="nutrition_manual")
    kcal = (
        st.number_input(
            "Kcal tự nhập",
            min_value=0.0,
            max_value=100000.0,
            value=None,
            key="nutrition_kcal",
            placeholder="Nhập kcal đã biết",
        )
        if manual
        else None
    )
    if selected != "other":
        estimate = api.request(
            "GET",
            f"/api/v1/nutrition/{selected}",
            params={
                "portion": portion,
                **({"grams": grams} if grams is not None else {}),
                **({"manual_kcal": kcal} if kcal is not None else {}),
            },
        )
        if estimate:
            render_estimate(estimate)
        else:
            st.info("Chưa có dữ liệu dinh dưỡng hoặc khối lượng tham khảo cho khẩu phần này.")
    elif kcal is not None:
        render_estimate({"estimated_kcal": kcal, "source_type": "user_input"})
    else:
        st.info(
            "Món chưa có dữ liệu tham khảo. Có thể lưu với kcal chưa xác định hoặc tự nhập kcal."
        )
    consumed = st.date_input("Ngày ăn", datetime.now(ZoneInfo(Settings().app_timezone)).date())
    meal_type = st.selectbox(
        "Bữa ăn", ["breakfast", "lunch", "dinner", "snack"], format_func=meal_label
    )
    confirmed = st.checkbox("Tôi đã kiểm tra món và khẩu phần", key="confirm_food_checkbox")
    if st.button(
        "Xác nhận và lưu",
        disabled=not confirmed
        or bool(st.session_state.get("recognition_saved"))
        or (manual and kcal is None)
        or (custom_grams and grams is None),
    ):
        payload = {
            "consumed_on": str(consumed),
            "meal_type": meal_type,
            "client_request_id": st.session_state["meal_request_id"],
            "items": [
                {
                    "food_class_id": None if selected == "other" else selected,
                    "custom_food_name": custom,
                    "prediction_id": result["prediction_id"],
                    "portion_size": portion,
                    "grams": grams,
                    "manual_kcal": kcal,
                }
            ],
        }
        saved = api.request("POST", "/api/v1/meals", json=payload)
        if saved:
            st.session_state["recognition_saved"] = saved["id"]
            st.success("Đã lưu vào PostgreSQL. Bạn có thể xem trong Nhật ký.")
    disclaimer()
