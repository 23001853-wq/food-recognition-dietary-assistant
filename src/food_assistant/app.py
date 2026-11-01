"""Streamlit entry point for the project scaffold."""

import streamlit as st

from food_assistant.ui.pages import about, dashboard, diary, home, recognize
from food_assistant.ui.theme import apply_theme


def main() -> None:
    st.set_page_config(page_title="Nhật ký món ăn · Project 18", page_icon="🍽️", layout="wide")
    apply_theme()
    with st.sidebar:
        st.markdown("### 🥗 Nhật ký món ăn")
        st.caption("Nhận diện • Dinh dưỡng • Nhật ký")
    pages = [
        st.Page(home.render, title="Trang chủ", icon="🏡", url_path="home", default=True),
        st.Page(recognize.render, title="Nhận diện món ăn", icon="📷", url_path="recognize"),
        st.Page(diary.render, title="Nhật ký bữa ăn", icon="📖", url_path="diary"),
        st.Page(dashboard.render, title="Thống kê dinh dưỡng", icon="📊", url_path="dashboard"),
        st.Page(about.render, title="Thông tin mô hình", icon="🧠", url_path="about"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
