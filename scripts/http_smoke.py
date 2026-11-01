"""Live HTTP end-to-end smoke. Removes only the meal created by this command."""

import argparse
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mock")
    parser.add_argument(
        "--image", type=Path, help="Real local JPEG/PNG; required for a real predictor"
    )
    args = parser.parse_args()
    if args.model != "mock" and args.image is None:
        parser.error(
            "Real inference smoke requires --image with a real validation/local food photo"
        )
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        image = BytesIO()
        if args.image:
            image.write(args.image.read_bytes())
            with Image.open(args.image) as source:
                content_type = Image.MIME[source.format]
            filename = args.image.name
        else:
            Image.new("RGB", (128, 128), "orange").save(image, "PNG")
            content_type, filename = "image/png", "smoke.png"
        response = client.post(
            "/api/v1/predictions",
            files={"file": (filename, image.getvalue(), content_type)},
            data={"model_name": args.model, "top_k": 3},
        )
        response.raise_for_status()
        prediction = response.json()
        assert prediction["is_mock"] == (args.model == "mock")
        foods = client.get("/api/v1/foods").json()
        pizza = next(food for food in foods if food["dataset_key"] == "pizza")
        today = str(datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date())
        response = client.post(
            "/api/v1/meals",
            json={
                "consumed_on": today,
                "meal_type": "snack",
                "client_request_id": str(uuid4()),
                "notes": "Automated smoke verification; deleted by the same command",
                "items": [
                    {
                        "food_class_id": pizza["id"],
                        "portion_size": "medium",
                        "prediction_id": prediction["prediction_id"],
                    }
                ],
            },
        )
        response.raise_for_status()
        meal = response.json()
        meal_id = meal["id"]
        try:
            assert float(meal["items"][0]["estimated_kcal"]) == 536
            assert meal["items"][0]["nutrition_snapshot"]["record_id"] == "170317"
            client.get(f"/api/v1/meals/{meal_id}").raise_for_status()
            summary = client.get("/api/v1/dashboard/summary", params={"start": today, "end": today})
            summary.raise_for_status()
            assert summary.json()["meal_count"] >= 1
            client.patch(
                f"/api/v1/meals/{meal_id}", json={"notes": "Smoke update verified"}
            ).raise_for_status()
            print(
                "PASS: HTTP upload -> predictor -> sourced nutrition -> "
                "PostgreSQL meal -> dashboard -> update"
            )
            print(
                f"is_mock={prediction['is_mock']}; model={prediction['model']['name']}; "
                "not a food accuracy evaluation"
            )
        finally:
            client.delete(f"/api/v1/meals/{meal_id}").raise_for_status()
            assert client.get(f"/api/v1/meals/{meal_id}").status_code == 404
        print("PASS: own smoke meal deleted; existing meals untouched")


if __name__ == "__main__":
    main()
