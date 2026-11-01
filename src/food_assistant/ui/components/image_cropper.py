import streamlit as st

from food_assistant.domain.errors import AppError
from food_assistant.services.image_service import CropService


def render_cropper(image):
    if not st.checkbox("Crop thủ công một món"):
        return None
    x = st.slider("Khoảng ngang", 0.0, 1.0, (0.0, 1.0), 0.01)
    y = st.slider("Khoảng dọc", 0.0, 1.0, (0.0, 1.0), 0.01)
    box = [x[0], y[0], x[1], y[1]]
    try:
        st.image(CropService.crop(image, box), caption="Xem trước crop")
    except AppError as exc:
        st.error(exc.message)
    return box
