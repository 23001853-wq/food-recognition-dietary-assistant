from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import streamlit as st

from food_assistant.config import Settings
from food_assistant.ui.client import APIClient, disclaimer, food_label
from food_assistant.ui.labels import meal_label, nutrient_value, portion_label


def render():
    st.title("Nhật ký bữa ăn")
    st.caption("Mỗi bữa ăn là một ghi nhận nhỏ. Xem lại khẩu phần và dinh dưỡng bạn đã xác nhận.")
    api = APIClient()
    today = datetime.now(ZoneInfo(Settings().app_timezone)).date()
    first, second, third = st.columns(3)
    start = first.date_input("Từ ngày", today - timedelta(days=6))
    end = second.date_input("Đến ngày", today)
    meal_filter = third.selectbox(
        "Lọc loại bữa", ["all", "breakfast", "lunch", "dinner", "snack"], format_func=meal_label
    )
    meals = api.request(
        "GET",
        "/api/v1/meals",
        params={
            "start": str(start),
            "end": str(end),
            **({"meal_type": meal_filter} if meal_filter != "all" else {}),
        },
    )
    foods = api.request("GET", "/api/v1/foods") or []
    catalog = {f["id"]: food_label(f) for f in foods}
    with st.expander("Thêm món thủ công"):
        with st.form("manual_meal"):
            name = st.text_input("Tên món")
            kind = st.selectbox(
                "Loại bữa", ["breakfast", "lunch", "dinner", "snack"], format_func=meal_label
            )
            consumed = st.date_input("Ngày ghi nhận", today)
            known = st.checkbox("Có kcal do tôi nhập")
            kcal = st.number_input("Kcal", min_value=0.0, max_value=100000.0)
            if st.form_submit_button("Thêm vào nhật ký"):
                result = api.request(
                    "POST",
                    "/api/v1/meals",
                    json={
                        "consumed_on": str(consumed),
                        "meal_type": kind,
                        "client_request_id": str(uuid4()),
                        "items": [
                            {"custom_food_name": name, "manual_kcal": kcal if known else None}
                        ],
                    },
                )
                if result:
                    st.rerun()
    if meals is None:
        return
    if not meals:
        st.info("Chưa có bữa ăn trong khoảng ngày đã chọn.")
    for meal in meals:
        with st.expander(
            f"📖 {meal['consumed_on']} · {meal_label(meal['meal_type'])} · {len(meal['items'])} món"
        ):
            rows = [
                {
                    **item,
                    "food_name": catalog.get(
                        item["confirmed_food_class_id"], item["custom_food_name"]
                    ),
                }
                for item in meal["items"]
            ]
            st.dataframe(
                [
                    {
                        "Món": row["food_name"],
                        "Khẩu phần": portion_label(row["portion_size"]),
                        "Gram": row["estimated_grams"]
                        if row["estimated_grams"] is not None
                        else "Chưa xác định",
                        "Kcal ước tính": row["estimated_kcal"]
                        if row["estimated_kcal"] is not None
                        else "Chưa có dữ liệu",
                        "Protein (g)": nutrient_value(row.get("protein_g")),
                        "Carbohydrate (g)": nutrient_value(row.get("carbohydrate_g")),
                        "Chất béo (Fat, g)": nutrient_value(row.get("fat_g")),
                        "Nguồn": "Người dùng nhập (user_input)"
                        if row["source_type"] == "user_input"
                        else row["nutrition_source_snapshot"] or "Chưa có dữ liệu",
                    }
                    for row in rows
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Dinh dưỡng được lưu tại thời điểm ghi nhận bữa ăn. "
                "Chỉ số thiếu không được coi là 0."
            )
            notes = st.text_input("Ghi chú bữa", meal["notes"] or "", key=f"notes_{meal['id']}")
            consumed = st.date_input(
                "Ngày bữa ăn",
                datetime.fromisoformat(meal["consumed_on"]).date(),
                key=f"date_{meal['id']}",
            )
            kinds = ["breakfast", "lunch", "dinner", "snack"]
            kind = st.selectbox(
                "Loại bữa ăn",
                kinds,
                index=kinds.index(meal["meal_type"]),
                format_func=meal_label,
                key=f"kind_{meal['id']}",
            )
            if st.button("Lưu ghi chú", key=f"save_{meal['id']}"):
                if (
                    api.request(
                        "PATCH",
                        f"/api/v1/meals/{meal['id']}",
                        json={"notes": notes, "consumed_on": str(consumed), "meal_type": kind},
                    )
                    is not None
                ):
                    st.success("Đã cập nhật")
            st.caption("Sửa món và khẩu phần bên dưới. Khi lưu, dinh dưỡng sẽ được tính lại.")
            st.divider()
            drafts = [
                {
                    "food_class_id": i["confirmed_food_class_id"],
                    "custom_food_name": i["custom_food_name"],
                    "prediction_id": i["prediction_id"],
                    "portion_size": i["portion_size"],
                    "grams": i["estimated_grams"],
                    "manual_kcal": i["estimated_kcal"]
                    if i["calculation_method"] == "manual"
                    else None,
                    "notes": i["notes"],
                }
                for i in meal["items"]
            ]
            edited = []
            for index, draft in enumerate(drafts):
                prefix = f"edit_{meal['id']}_{index}"
                options = list(catalog) + ["other"]
                selected = st.selectbox(
                    "Món ăn",
                    options,
                    index=options.index(draft["food_class_id"] or "other"),
                    format_func=lambda k: catalog.get(k, "Món khác"),
                    key=prefix + "food",
                )
                name = (
                    st.text_input(
                        "Tên món khác", draft["custom_food_name"] or "", key=prefix + "name"
                    )
                    if selected == "other"
                    else None
                )
                portion = st.selectbox(
                    "Khẩu phần",
                    ["small", "medium", "large", "custom"],
                    index=["small", "medium", "large", "custom"].index(draft["portion_size"]),
                    key=prefix + "portion",
                    format_func=portion_label,
                )
                has_grams = st.checkbox(
                    "Có khối lượng gram", draft["grams"] is not None, key=prefix + "hasgrams"
                )
                grams = (
                    st.number_input(
                        "Gram",
                        min_value=0.01,
                        max_value=10000.0,
                        value=float(draft["grams"] or 100),
                        key=prefix + "grams",
                    )
                    if has_grams
                    else None
                )
                manual = st.checkbox(
                    "Tự nhập năng lượng", draft["manual_kcal"] is not None, key=prefix + "manual"
                )
                kcal = (
                    st.number_input(
                        "Kcal tự nhập",
                        min_value=0.0,
                        max_value=100000.0,
                        value=float(draft["manual_kcal"] or 0),
                        key=prefix + "kcal",
                    )
                    if manual
                    else None
                )
                remove = st.checkbox("Bỏ món này", key=prefix + "remove")
                if not remove:
                    edited.append(
                        {
                            **draft,
                            "food_class_id": None if selected == "other" else selected,
                            "custom_food_name": name,
                            "portion_size": portion,
                            "grams": grams,
                            "manual_kcal": kcal,
                        }
                    )
            recalc = st.checkbox(
                "Xác nhận tính lại dinh dưỡng cho các món sửa", key=f"recalc_{meal['id']}"
            )
            if st.button("Lưu các món", key=f"items_{meal['id']}", disabled=not recalc):
                if not edited:
                    if api.request("DELETE", f"/api/v1/meals/{meal['id']}"):
                        st.rerun()
                else:
                    if api.request("PATCH", f"/api/v1/meals/{meal['id']}", json={"items": edited}):
                        st.rerun()
            confirmed = st.checkbox("Xác nhận xóa bữa ăn", key=f"delete_confirm_{meal['id']}")
            if st.button("Xóa bữa ăn", disabled=not confirmed, key=f"delete_{meal['id']}"):
                if api.request("DELETE", f"/api/v1/meals/{meal['id']}") is not None:
                    st.rerun()
    disclaimer()
