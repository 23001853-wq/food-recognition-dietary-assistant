import streamlit as st

from food_assistant.ui.client import APIClient, disclaimer
from food_assistant.ui.theme import render_hero


def render():
    render_hero()
    health = APIClient().request("GET", "/health")
    if health:
        st.success("Backend và PostgreSQL đang kết nối")
        if health["mock_enabled"]:
            st.warning("Chế độ phát triển: có Mock AI. Dự đoán mock chỉ dùng kiểm thử hệ thống.")
    st.subheader("Bữa ăn của bạn trong 7 ngày")
    data = APIClient().request("GET", "/api/v1/dashboard/summary")
    if data:
        a, b, c = st.columns(3)
        a.metric("Bữa ăn (7 ngày)", data["meal_count"])
        b.metric("Món đã ghi", data["item_count"])
        c.metric(
            "Kcal ước tính",
            "Chưa có dữ liệu"
            if data["item_count"] and not data["known_count"]
            else data["estimated_kcal"],
        )
        st.caption(f"Tổng các món có dữ liệu; {data['missing_count']} món chưa có kcal.")
    st.subheader("Một bữa ăn, ba bước đơn giản")
    for column, title, description in zip(
        st.columns(3),
        ["📷 01 · Nhận diện", "🥑 02 · Xem dinh dưỡng", "📖 03 · Lưu nhật ký"],
        [
            "Tải ảnh món ăn, xem các dự đoán và xác nhận món phù hợp.",
            "Chọn khẩu phần hoặc nhập gram để xem kcal và các chất dinh dưỡng tham khảo.",
            "Lưu bữa ăn đã xác nhận, xem lại lịch sử và theo dõi thống kê mỗi ngày.",
        ],
    ):
        column.markdown(
            f'<div class="food-feature"><strong>{title}</strong><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
    st.caption("Chọn Nhận diện món ăn trong thanh điều hướng để bắt đầu.")
    disclaimer()
