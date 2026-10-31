"""Dashboard scaffold page."""

import streamlit as st


def render() -> None:
    st.title('Dashboard')
    st.info('Chưa triển khai tổng hợp nhật ký. Biểu đồ sẽ dùng dữ liệu thực từ PostgreSQL.')
    st.caption("Năng lượng chỉ là ước tính theo khẩu phần/khối lượng tham khảo, "
               "không đo trực tiếp từ ảnh.")
