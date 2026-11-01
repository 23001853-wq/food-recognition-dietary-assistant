"""Stored macro values and localized labels survive the UI/HTTP boundary."""

from datetime import datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from food_assistant.config import PROJECT_ROOT, Settings

pytestmark = pytest.mark.integration


def test_diary_displays_stored_macros_and_vietnamese_enums(client, monkeypatch):
    foods = client.get("/api/v1/foods").json()
    curry = next(food for food in foods if food["dataset_key"] == "chicken_curry")
    today = str(datetime.now(ZoneInfo(Settings().app_timezone)).date())
    created = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": today,
            "meal_type": "lunch",
            "items": [
                {"food_class_id": curry["id"], "portion_size": "medium"},
                {"custom_food_name": "Món chưa có nguồn"},
            ],
        },
    )
    assert created.status_code == 201
    meal = created.json()
    stored = client.get("/api/v1/meals/" + meal["id"]).json()
    item = next(i for i in stored["items"] if i["confirmed_food_class_id"] == curry["id"])
    assert all(item[field] is not None for field in ("protein_g", "carbohydrate_g", "fat_g"))

    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        return client.request(method, urlsplit(url).path, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_string(
        "from food_assistant.ui.pages.diary import render\nrender()", default_timeout=15
    ).run()
    assert not app.exception
    table = app.dataframe[0].value
    row = table.loc[table["Món"] == curry["display_name_vi"]].iloc[0]
    assert row["Khẩu phần"] == "Vừa"
    for column, field in (
        ("Protein (g)", "protein_g"),
        ("Carbohydrate (g)", "carbohydrate_g"),
        ("Chất béo (Fat, g)", "fat_g"),
    ):
        assert row[column] == item[field]
        unknown = table.loc[table["Món"] == "Món chưa có nguồn"].iloc[0]
        assert unknown[column] == "Chưa có dữ liệu"
    assert any("Bữa trưa" in expander.label for expander in app.expander)
    meal_selector = next(box for box in app.selectbox if box.label == "Loại bữa ăn")
    assert meal_selector.options == ["Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"]
    assert next(box for box in app.selectbox if box.label == "Khẩu phần").options == [
        "Nhỏ",
        "Vừa",
        "Lớn",
        "Tùy chỉnh",
    ]
    # Localized labels must not be sent as enum values to the backend.
    meal_selector.set_value("dinner").run()
    app.button(key=f"save_{meal['id']}").click().run()
    assert not app.exception
    updated = client.get("/api/v1/meals/" + meal["id"]).json()
    assert updated["meal_type"] == "dinner"
    assert updated["items"] == stored["items"]

    dashboard = AppTest.from_string(
        "from food_assistant.ui.pages.dashboard import render\nrender()", default_timeout=15
    ).run()
    assert not dashboard.exception
    about = AppTest.from_string(
        "from food_assistant.ui.pages.about import render\nrender()", default_timeout=15
    ).run()
    assert not about.exception
    main = AppTest.from_file(
        str(PROJECT_ROOT / "src/food_assistant/app.py"), default_timeout=15
    ).run()
    assert not main.exception
    assert any("food-hero" in block.value for block in main.markdown)
