"""Food Recognition & Dietary Assistant scaffold page."""

import streamlit as st


def render() -> None:
    st.title('Food Recognition & Dietary Assistant')
    st.info('Bộ khung dự án theo PLAN.md. Chức năng nhận diện và nhật ký sẽ được triển khai theo từng phase.')
    st.caption("Năng lượng chỉ là ước tính theo khẩu phần/khối lượng tham khảo, "
               "không đo trực tiếp từ ảnh.")
