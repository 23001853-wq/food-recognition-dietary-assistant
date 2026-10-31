"""Model Info / About scaffold page."""

import streamlit as st


def render() -> None:
    st.title('Model Info / About')
    st.info('Dự kiến so sánh HOG + SVM, ResNet18 và EfficientNet-B0. Chưa có kết quả huấn luyện hoặc đánh giá.')
    st.caption("Năng lượng chỉ là ước tính theo khẩu phần/khối lượng tham khảo, "
               "không đo trực tiếp từ ảnh.")
