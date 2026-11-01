"""Read-only HTTP verification of sourced food/portion calculations; no AI evaluation."""

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal

import httpx

from food_assistant.config import PROJECT_ROOT, Settings
from food_assistant.nutrition.catalog import identity, read_csv


def check(base_url, foods):
    root = PROJECT_ROOT / "data/nutrition"
    references = {identity(row): row for row in read_csv(root / "source_catalog.csv")}
    mappings = {
        row["dataset_key"]: row
        for row in read_csv(root / "curated_mapping.csv")
        if row["is_default"] == "true"
    }
    results = []
    with httpx.Client(base_url=base_url, timeout=15) as client:
        response = client.get("/api/v1/foods")
        response.raise_for_status()
        ids = {row["dataset_key"]: row["id"] for row in response.json()}
        for key in foods:
            profile = references[identity(mappings[key])]
            energy = Decimal(profile["kcal_per_100g"])
            for portion in ("small", "medium", "large", "custom"):
                grams = (
                    Decimal("123.45")
                    if portion == "custom"
                    else Decimal(profile[portion + "_grams"])
                )
                params = {
                    "portion": portion,
                    **({"grams": str(grams)} if portion == "custom" else {}),
                }
                response = client.get(f"/api/v1/nutrition/{ids[key]}", params=params)
                response.raise_for_status()
                value = response.json()
                expected = (energy * grams / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if (
                    value.get("status") != "available"
                    or value.get("source_record_id") != profile["source_record_id"]
                    or value.get("source_version") != profile["source_version"]
                    or Decimal(str(value.get("grams"))) != grams
                    or Decimal(str(value.get("estimated_kcal"))) != expected
                ):
                    raise ValueError(
                        f"{key}/{portion}: API differs from curated catalog. "
                        "Apply seed_nutrition.py and restart the updated backend; "
                        "an existing alternative default mapping is preserved, not replaced."
                    )
                results.append(
                    {
                        "food": key,
                        "portion": portion,
                        "grams": str(grams),
                        "kcal": str(expected),
                        "fdc_id": value["source_record_id"],
                    }
                )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=Settings().api_base_url)
    parser.add_argument("--foods", nargs="+", default=["hamburger", "ice_cream", "pizza"])
    args = parser.parse_args()
    print(json.dumps(check(args.base_url, args.foods), indent=2))
