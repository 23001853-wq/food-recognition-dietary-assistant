"""Streamlit entry point for the project scaffold."""

import streamlit as st

from food_assistant.ui.pages import about, dashboard, diary, home, recognize


def main() -> None:
    st.set_page_config(page_title="Food Recognition & Dietary Assistant", page_icon="🍽️")
    pages = [
        st.Page(home.render, title="Home", url_path="home", default=True),
        st.Page(recognize.render, title="Recognize Food", url_path="recognize"),
        st.Page(diary.render, title="Meal Diary", url_path="diary"),
        st.Page(dashboard.render, title="Dashboard", url_path="dashboard"),
        st.Page(about.render, title="Model Info / About", url_path="about"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
