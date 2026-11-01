from decimal import Decimal

import pytest
from sqlalchemy import func, select

from food_assistant.config import PROJECT_ROOT
from food_assistant.persistence.orm_models import FoodClass, FoodNutritionMapping, NutritionFood
from food_assistant.persistence.seed import seed_nutrition

pytestmark = pytest.mark.integration


def catalog(client):
    return {food["dataset_key"]: food["id"] for food in client.get("/api/v1/foods").json()}


@pytest.mark.parametrize(
    "food,kcal,grams,record",
    [
        ("hamburger", "394.50", "150.00", "170694"),
        ("ice_cream", "207.00", "100.00", "167575"),
        ("pizza", "536.00", "200.00", "170317"),
    ],
)
def test_automatic_estimate_and_diary_snapshot(client, food, kcal, grams, record):
    food_id = catalog(client)[food]
    url = f"/api/v1/nutrition/{food_id}"
    estimate = client.get(url, params={"portion": "medium"}).json()
    assert Decimal(estimate["estimated_kcal"]) == Decimal(kcal)
    assert estimate["source_record_id"] == record
    assert estimate["mass_source"] == "design_assumption"
    custom = client.get(url, params={"portion": "custom", "grams": 50}).json()
    assert Decimal(custom["estimated_kcal"]) == Decimal(estimate["kcal_per_100g"]) / 2
    small = client.get(url, params={"portion": "small"}).json()
    assert Decimal(small["estimated_kcal"]) < Decimal(kcal)
    created = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "lunch",
            "items": [{"food_class_id": food_id, "portion_size": "medium"}],
        },
    )
    assert created.status_code == 201, created.text
    meal = created.json()
    item = meal["items"][0]
    assert Decimal(item["estimated_grams"]) == Decimal(grams)
    assert Decimal(item["estimated_kcal"]) == Decimal(kcal)
    assert item["source_type"] == "reference"
    assert item["nutrition_snapshot"]["record_id"] == record
    assert item["nutrition_snapshot"]["mass_source"] == "design_assumption"
    with client.app.state.session_factory.begin() as session:
        summary = seed_nutrition(session)
        assert summary["profiles_added"] == summary["mappings_added"] == 0
    assert client.get("/api/v1/meals/" + meal["id"]).json() == meal


def test_manual_and_unknown_are_distinct_and_not_zero(client):
    food_id = catalog(client)["hamburger"]
    preview = client.get(f"/api/v1/nutrition/{food_id}?portion=medium&manual_kcal=125").json()
    assert preview["source_type"] == "user_input" and preview["protein_g"] is None
    created = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "snack",
            "items": [
                {"food_class_id": food_id, "portion_size": "medium", "manual_kcal": "125"},
                {"custom_food_name": "Unknown recipe"},
            ],
        },
    )
    assert created.status_code == 201, created.text
    items = created.json()["items"]
    manual = next(i for i in items if i["source_type"] == "user_input")
    missing = next(i for i in items if i["source_type"] == "unavailable")
    assert manual["nutrition_source_snapshot"] == "user_input"
    assert manual["nutrition_snapshot"]["source_type"] == "user_input"
    assert manual["kcal_per_100g_snapshot"] is None
    assert missing["estimated_kcal"] is None
    summary = client.get("/api/v1/dashboard/summary?start=2026-10-31&end=2026-10-31").json()
    assert Decimal(summary["estimated_kcal"]) == 125
    assert summary["known_count"] == summary["missing_count"] == summary["user_input_count"] == 1
    assert summary["daily"][0]["known_count"] == 1


def test_unmapped_food_does_not_crash_and_null_is_preserved(client):
    with client.app.state.session_factory.begin() as session:
        food_id = session.scalar(select(FoodClass.id).where(FoodClass.dataset_key == "ice_cream"))
        mapping = session.scalar(
            select(FoodNutritionMapping).where(
                FoodNutritionMapping.food_class_id == food_id,
                FoodNutritionMapping.is_default.is_(True),
            )
        )
        mapping.is_default = False
    estimate = client.get(f"/api/v1/nutrition/{food_id}?portion=medium").json()
    assert estimate["status"] == "unavailable" and estimate["estimated_kcal"] is None
    assert client.get(f"/api/v1/nutrition/{food_id}?grams=-1").status_code == 422
    assert client.get(f"/api/v1/nutrition/{food_id}?portion=invalid").status_code == 422
    assert client.get(f"/api/v1/nutrition/{food_id}?manual_kcal=NaN").status_code == 422


def test_unknown_only_dashboard_and_home(client, monkeypatch):
    from urllib.parse import urlsplit

    import httpx
    from streamlit.testing.v1 import AppTest

    response = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "dinner",
            "items": [{"custom_food_name": "No reference recipe"}],
        },
    )
    assert response.status_code == 201
    summary = client.get("/api/v1/dashboard/summary?start=2026-10-31&end=2026-10-31").json()
    assert summary["known_count"] == 0 and summary["missing_count"] == 1
    assert summary["daily"][0]["known_count"] == 0

    def request(method, url, **kwargs):
        kwargs.pop("timeout", None)
        if urlsplit(url).path == "/api/v1/dashboard/summary":
            kwargs["params"] = {"start": "2026-10-31", "end": "2026-10-31"}
        return client.request(method, urlsplit(url).path, **kwargs)

    monkeypatch.setattr(httpx, "request", request)
    app = AppTest.from_string(
        "from food_assistant.ui.pages.home import render\nrender()", default_timeout=15
    ).run()
    assert not app.exception
    assert (
        next(metric for metric in app.metric if metric.label == "Kcal ước tính").value
        == "Chưa có dữ liệu"
    )


def test_seed_idempotence_and_historical_snapshot(client):
    food_id = catalog(client)["pizza"]
    created = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "dinner",
            "items": [{"food_class_id": food_id, "portion_size": "medium"}],
        },
    ).json()
    with client.app.state.session_factory.begin() as session:
        before = session.scalar(select(func.count()).select_from(NutritionFood))
        seed_nutrition(session)
        assert before == session.scalar(select(func.count()).select_from(NutritionFood))
        profile = session.scalar(
            select(NutritionFood).where(NutritionFood.source_record_id == "170317")
        )
        profile.kcal_per_100g = Decimal("1")
    assert client.get("/api/v1/meals/" + created["id"]).json() == created
    with client.app.state.session_factory.begin() as session:
        with pytest.raises(ValueError, match="Existing source profile differs"):
            seed_nutrition(session)


def test_production_cpu_single_image_smoke(client, image_bytes, monkeypatch):
    """One synthetic upload checks real serving compatibility, never test-set accuracy."""
    pointer = PROJECT_ROOT / "artifacts/production/current.json"
    if not pointer.exists():
        pytest.skip("Production pointer not available; no artifact downloads")
    monkeypatch.setattr(client.app.state.settings, "inference_device", "cpu")
    response = client.post(
        "/api/v1/predictions",
        files={
            "file": ("synthetic-serving-smoke.png", image_bytes, "image/png"),
        },
        data={"model_name": "production", "top_k": 3},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["is_mock"] is False
    assert len(result["predictions"]) == 3
    foods = catalog(client)
    assert all(p["class_key"] in foods for p in result["predictions"])
    food_id = foods[result["predictions"][0]["class_key"]]
    estimate = client.get(f"/api/v1/nutrition/{food_id}?portion=medium").json()
    assert estimate["status"] == "available"
    saved = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "lunch",
            "items": [
                {
                    "food_class_id": food_id,
                    "prediction_id": result["prediction_id"],
                    "portion_size": "medium",
                }
            ],
        },
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["items"][0]["estimated_kcal"] == estimate["estimated_kcal"]


def test_read_only_verification_script(client, monkeypatch):
    import runpy
    from contextlib import nullcontext

    import httpx

    module = runpy.run_path(str(PROJECT_ROOT / "scripts/check_nutrition.py"))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: nullcontext(client))
    results = module["check"]("http://testserver", ["hamburger", "ice_cream", "pizza"])
    assert len(results) == 12


def test_new_reference_version_preserves_existing_default(client, tmp_path):
    from food_assistant.nutrition.catalog import read_csv, write_csv
    from food_assistant.services.nutrition_service import NutritionService

    rows = read_csv(PROJECT_ROOT / "data/nutrition/source_catalog.csv")
    profile = next(row for row in rows if row["source_record_id"] == "170317")
    # Explicit test-only revision to exercise version identity, never application data.
    revised = {**profile, "source_version": profile["source_version"] + " TEST revision"}
    mappings = read_csv(PROJECT_ROOT / "data/nutrition/curated_mapping.csv")
    mapping = next(row for row in mappings if row["dataset_key"] == "pizza")
    write_csv(tmp_path / "source_catalog.csv", rows + [revised])
    write_csv(
        tmp_path / "curated_mapping.csv",
        mappings
        + [
            {
                **mapping,
                "source_version": revised["source_version"],
            }
        ],
    )
    with client.app.state.session_factory.begin() as session:
        summary = seed_nutrition(session, tmp_path)
        assert summary == {"profiles_added": 1, "mappings_added": 1, "defaults_preserved": 1}
        food_id = session.scalar(select(FoodClass.id).where(FoodClass.dataset_key == "pizza"))
        assert NutritionService(session).lookup(food_id).source_version == profile["source_version"]
