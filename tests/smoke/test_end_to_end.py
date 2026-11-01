from uuid import uuid4

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def test_mock_upload_save_update_dashboard_delete(client, image_bytes):
    assert client.get("/health").status_code == 200
    foods = client.get("/api/v1/foods").json()
    response = client.post(
        "/api/v1/predictions",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"model_name": "mock"},
    )
    assert response.status_code == 200, response.text
    prediction = response.json()
    assert prediction["is_mock"] is True
    assert client.get("/api/v1/predictions/" + prediction["prediction_id"]).json() == prediction
    payload = {
        "consumed_on": "2026-10-31",
        "meal_type": "lunch",
        "client_request_id": str(uuid4()),
        "items": [
            {
                "food_class_id": foods[0]["id"],
                "prediction_id": prediction["prediction_id"],
                "portion_size": "custom",
                "grams": "100",
            }
        ],
    }
    response = client.post("/api/v1/meals", json=payload)
    assert response.status_code == 201, response.text
    meal = response.json()
    assert meal["items"][0]["estimated_kcal"] is not None
    assert client.post("/api/v1/meals", json=payload).json()["id"] == meal["id"]
    summary = client.get("/api/v1/dashboard/summary?start=2026-10-31&end=2026-10-31").json()
    assert summary["meal_count"] == 1 and summary["missing_count"] == 0
    patch = {"items": [{"custom_food_name": "manual food", "manual_kcal": "250", "grams": "100"}]}
    assert client.patch("/api/v1/meals/" + meal["id"], json=patch).status_code == 200
    summary = client.get("/api/v1/dashboard/summary?start=2026-10-31&end=2026-10-31").json()
    assert float(summary["estimated_kcal"]) == 250 and summary["missing_count"] == 0
    assert client.delete("/api/v1/meals/" + meal["id"]).status_code == 204
    assert client.get("/api/v1/meals/" + meal["id"]).status_code == 404


def test_bad_upload_does_not_predict(client):
    response = client.post(
        "/api/v1/predictions",
        files={"file": ("fake.png", b"not png", "image/png")},
        data={"model_name": "mock"},
    )
    assert response.status_code == 422


def test_rolls_back_partial_meal(client):
    payload = {
        "consumed_on": "2026-10-31",
        "meal_type": "dinner",
        "items": [{"custom_food_name": "valid"}, {"food_class_id": str(uuid4())}],
    }
    assert client.post("/api/v1/meals", json=payload).status_code == 422
    assert client.get("/api/v1/meals?start=2026-10-31&end=2026-10-31").json() == []


def test_idempotency_conflict_and_validation(client):
    payload = {
        "consumed_on": "2026-10-31",
        "meal_type": "snack",
        "client_request_id": str(uuid4()),
        "items": [{"custom_food_name": "test"}],
    }
    assert client.post("/api/v1/meals", json=payload).status_code == 201
    payload["meal_type"] = "lunch"
    assert client.post("/api/v1/meals", json=payload).status_code == 409
    payload["items"][0]["grams"] = -1
    assert client.post("/api/v1/meals", json=payload).status_code == 422
    assert client.get("/api/v1/meals?start=2026-11-01&end=2026-10-31").status_code == 422
