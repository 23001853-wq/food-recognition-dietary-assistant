"""Streamlit event flow with injected upload bytes and the PostgreSQL TestClient API."""

from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration


def test_frontend_confirm_save_and_image_change(client, image_bytes, monkeypatch):
    upload = {"data": image_bytes}
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *args, **kwargs: SimpleNamespace(
            name="fixture.png", type="image/png", getvalue=lambda: upload["data"]
        ),
    )

    def request(method, url, **kwargs):
        parts = urlsplit(url)
        kwargs.pop("timeout", None)
        return client.request(method, parts.path, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_string(
        "from food_assistant.ui.pages.recognize import render\nrender()", default_timeout=15
    ).run()
    assert not app.exception
    next(box for box in app.selectbox if box.label == "Mô hình").set_value("mock").run()
    next(button for button in app.button if button.label == "Phân tích").click().run()
    assert not app.exception
    assert app.session_state["recognition_result"]["is_mock"]
    foods = client.get("/api/v1/foods").json()
    pizza = next(food for food in foods if food["dataset_key"] == "pizza")
    food_selector = next(box for box in app.selectbox if box.label == "Xác nhận hoặc sửa món")
    food_selector.set_value(pizza["id"]).run()
    assert next(m for m in app.metric if m.label == "Kcal ước tính").value == "536.00"
    app.checkbox(key="nutrition_use_grams").check().run()
    app.number_input(key="nutrition_grams").set_value(50.0).run()
    assert next(m for m in app.metric if m.label == "Kcal ước tính").value == "134.00"
    cream = next(food for food in foods if food["dataset_key"] == "ice_cream")
    app.selectbox(key="confirmed_food").set_value(cream["id"]).run()
    assert not app.checkbox(key="nutrition_use_grams").value
    assert next(m for m in app.metric if m.label == "Kcal ước tính").value == "207.00"
    app.selectbox(key="confirmed_food").set_value(pizza["id"]).run()
    next(box for box in app.selectbox if box.label == "Khẩu phần").set_value("large").run()
    assert next(m for m in app.metric if m.label == "Kcal ước tính").value == "804.00"
    next(box for box in app.selectbox if box.label == "Khẩu phần").set_value("medium").run()
    app.checkbox(key="confirm_food_checkbox").check().run()
    save = next(button for button in app.button if button.label == "Xác nhận và lưu")
    assert not save.disabled
    save.click().run()
    assert not app.exception
    meal_id = app.session_state["recognition_saved"]
    meal = client.get("/api/v1/meals/" + meal_id).json()
    assert meal["items"][0]["confirmed_food_class_id"] == pizza["id"]
    assert float(meal["items"][0]["estimated_kcal"]) == 536
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "blue").save(buffer, "PNG")
    upload["data"] = buffer.getvalue()
    app.run()
    assert not app.exception
    assert "recognition_result" not in app.session_state.filtered_state
    assert "recognition_saved" not in app.session_state.filtered_state
