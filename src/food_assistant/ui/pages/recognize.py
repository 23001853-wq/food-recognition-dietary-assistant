"""Recognize Food scaffold page."""

import streamlit as st


def render() -> None:
    st.title('Recognize Food')
    st.info('MODEL_NOT_READY — Chưa triển khai inference và chưa có model artifact đã huấn luyện.')
    st.caption("Năng lượng chỉ là ước tính theo khẩu phần/khối lượng tham khảo, "
               "không đo trực tiếp từ ảnh.")
