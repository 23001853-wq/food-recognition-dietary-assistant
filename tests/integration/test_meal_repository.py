import pytest

pytestmark = pytest.mark.integration


def test_null_patch_and_empty_items_rejected(client):
    result = client.post(
        "/api/v1/meals",
        json={
            "consumed_on": "2026-10-31",
            "meal_type": "lunch",
            "items": [{"custom_food_name": "test"}],
        },
    )
    meal_id = result.json()["id"]
    for patch in [{"consumed_on": None}, {"meal_type": None}, {"items": []}, {"items": None}]:
        assert client.patch("/api/v1/meals/" + meal_id, json=patch).status_code == 422
