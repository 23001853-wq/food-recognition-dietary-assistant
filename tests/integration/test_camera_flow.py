"""Camera bytes use the upload contract; source changes invalidate prior confirmations."""

from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration


def test_camera_capture_save_and_source_switch(client, image_bytes, monkeypatch):
    camera = {"capture": None}
    jpeg = BytesIO()
    Image.new("RGB", (64, 64), "green").save(jpeg, "JPEG")
    captured = SimpleNamespace(name="camera.jpg", type="image/jpeg", getvalue=jpeg.getvalue)
    monkeypatch.setattr(st, "camera_input", lambda *a, **k: camera["capture"])
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda *a, **k: SimpleNamespace(
            name="uploaded.png",
            type="image/png",
            getvalue=lambda: image_bytes,
        ),
    )
    calls = []

    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        calls.append((method, urlsplit(url).path))
        return client.request(method, urlsplit(url).path, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_string(
        "from food_assistant.ui.pages.recognize import render\nrender()",
        default_timeout=15,
    ).run()
    next(box for box in app.selectbox if box.label == "Mô hình").set_value("mock").run()
    app.radio(key="image_source").set_value("Chụp ảnh").run()
    assert not app.exception
    assert not any(button.label == "Phân tích" for button in app.button)
    camera["capture"] = SimpleNamespace(name="bad.jpg", type="image/jpeg", getvalue=lambda: b"bad")
    app.run()
    assert app.error and not app.exception
    assert not any(method == "POST" for method, path in calls)
    camera["capture"] = captured
    app.run()
    next(button for button in app.button if button.label == "Phân tích").click().run()
    assert not app.exception
    assert app.session_state["recognition_result"]["is_mock"] is True
    pizza = next(f for f in client.get("/api/v1/foods").json() if f["dataset_key"] == "pizza")
    app.selectbox(key="confirmed_food").set_value(pizza["id"]).run()
    app.checkbox(key="confirm_food_checkbox").check().run()
    next(button for button in app.button if button.label == "Xác nhận và lưu").click().run()
    assert not app.exception
    meal_id = app.session_state["recognition_saved"]
    item = client.get("/api/v1/meals/" + meal_id).json()["items"][0]
    assert float(item["estimated_kcal"]) == 536
    assert item["protein_g"] is not None
    app.radio(key="image_source").set_value("Tải ảnh").run()
    assert not app.exception
    assert "recognition_result" not in app.session_state.filtered_state
    assert "recognition_saved" not in app.session_state.filtered_state
    assert calls.count(("POST", "/api/v1/predictions")) == 1
    camera["capture"] = None
    app.radio(key="image_source").set_value("Chụp ảnh").run()
    assert not app.exception
    assert not any(button.label == "Xác nhận và lưu" for button in app.button)
