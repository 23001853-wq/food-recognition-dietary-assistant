from decimal import Decimal

import pytest

pytestmark = pytest.mark.integration


def test_reference_and_missing_totals(client):
    foods = client.get("/api/v1/foods").json()
    pizza = next(f for f in foods if f["dataset_key"] == "pizza")
    estimate = client.get("/api/v1/nutrition/" + pizza["id"] + "?portion=medium").json()
    assert float(estimate["estimated_kcal"]) == 536
    result = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "lunch",
            "items": [
                {"food_class_id": pizza["id"], "portion_size": "medium"},
                {"custom_food_name": "unknown"},
            ],
        },
    )
    assert result.status_code == 201
    snapshot = result.json()["items"][0]["nutrition_snapshot"]
    assert snapshot["record_id"] == "170317"
    summary = client.get("/api/v1/dashboard/summary?start=2026-10-31&end=2026-10-31").json()
    assert summary["missing_count"] == 1
    assert float(summary["estimated_kcal"]) == 536
    for field in ("estimated_kcal", "protein_g", "carbohydrate_g", "fat_g"):
        expected = result.json()["items"][0][field]
        assert Decimal(str(summary["nutrients"][field]["total"])) == Decimal(expected)
        assert summary["nutrients"][field]["known_count"] == 1
        assert summary["nutrients"][field]["missing_count"] == 1
        assert Decimal(str(summary["daily"][0]["nutrients"][field]["total"])) == Decimal(expected)
    assert (
        client.get("/api/v1/dashboard/daily?start=2026-10-31&end=2026-10-31").json()
        == summary["daily"]
    )


def test_dashboard_missing_macros_stay_null(client):
    result = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "dinner",
            "items": [
                {"custom_food_name": "Manual energy only", "manual_kcal": "100"},
                {"custom_food_name": "Unknown recipe"},
            ],
        },
    )
    assert result.status_code == 201
    summary = client.get("/api/v1/dashboard/summary?start=2026-10-31&end=2026-10-31").json()
    for field in ("protein_g", "carbohydrate_g", "fat_g"):
        assert summary["nutrients"][field] == {"total": None, "known_count": 0, "missing_count": 2}
    assert float(summary["nutrients"]["estimated_kcal"]["total"]) == 100
    empty = client.get("/api/v1/dashboard/summary?start=2026-11-01&end=2026-11-01").json()
    assert all(value["total"] is None for value in empty["nutrients"].values())
