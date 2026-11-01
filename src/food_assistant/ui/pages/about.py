import streamlit as st

from food_assistant.ui.client import APIClient, disclaimer


def render():
    st.title("Thông tin mô hình")
    st.caption("Tìm hiểu mô hình nhận diện và cách bữa ăn được ghi nhận trong hệ thống.")
    with st.container(border=True):
        st.markdown("#### Từ ảnh đến nhật ký của bạn")
        steps = st.columns(4)
        for column, title, detail in zip(
            steps,
            ["📷 Ảnh món ăn", "🧠 Nhận diện AI", "🥗 Dinh dưỡng", "📖 Nhật ký"],
            [
                "Tải ảnh hoặc chọn vùng món chính",
                "Xem dự đoán và xác nhận món",
                "Chọn khẩu phần, tra nguồn tham khảo",
                "Lưu bữa ăn và theo dõi thống kê",
            ],
        ):
            column.markdown(f"**{title}**")
            column.caption(detail)
    models = APIClient().request("GET", "/api/v1/models")
    if models:
        for model in models:
            names = {
                "efficientnet_b0": "EfficientNet-B0",
                "resnet18": "ResNet18",
                "hog_svm": "HOG + SVM",
                "mock": "Mô phỏng (Mock)",
            }
            name = model.get("model_name", model.get("name", "Mô hình"))
            with st.container(border=True):
                st.subheader(names.get(name, name))
                if model.get("production"):
                    st.success("Đang được chọn cho nhận diện production")
                if model.get("is_mock"):
                    st.warning("Mock chỉ dùng phát triển, không phải mô hình đã huấn luyện.")
                if model.get("smoke_only"):
                    st.warning("Smoke training trên tập nhỏ; chưa đủ điều kiện production.")
                left, right = st.columns(2)
                left.caption("Phiên bản")
                left.write(model.get("run_id", model.get("version", "Chưa có dữ liệu")))
                right.caption("Bộ dữ liệu")
                right.write(model.get("dataset_version", "Chưa có dữ liệu"))
                with st.expander("Chi tiết kỹ thuật và kết quả đã lưu"):
                    st.json(model)
    else:
        st.info("Chưa có model release.")
    st.write(
        "Một ảnh/crop dự đoán một món chính. Món hỗn hợp cần crop thủ công. "
        "Score chưa được hiệu chuẩn."
    )
    st.markdown(
        "Dataset: [Food-101, ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)."
    )
    disclaimer()
