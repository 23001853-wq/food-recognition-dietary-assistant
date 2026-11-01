"""Idempotent catalog seed. Never fabricates nutrition references or model metrics."""

import json
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select, text

from food_assistant.config import PROJECT_ROOT
from food_assistant.nutrition.catalog import (
    NUTRIENTS,
    PORTIONS,
    numeric,
    read_csv,
    validate_reference,
)
from food_assistant.persistence.orm_models import (
    FoodClass,
    FoodNutritionMapping,
    NutritionFood,
    User,
)

DEMO_USER_ID = uuid.UUID("f00da550-0000-4000-8000-000000000001")

VI_NAMES = {
    "apple_pie": "Bánh táo",
    "caesar_salad": "Salad Caesar",
    "cheesecake": "Bánh phô mai",
    "chicken_curry": "Cà ri gà",
    "french_fries": "Khoai tây chiên",
    "hamburger": "Hamburger",
    "ice_cream": "Kem",
    "omelette": "Trứng tráng",
    "pizza": "Pizza",
    "sushi": "Sushi",
}


def seed(session):
    if session.get(User, DEMO_USER_ID) is None:
        session.add(User(id=DEMO_USER_ID, display_name="Local demo"))
    config = json.loads(
        (PROJECT_ROOT / "data/nutrition/label_catalog.json").read_text(encoding="utf-8")
    )
    for label in config["labels"]:
        key = label["class_key"]
        if session.scalar(select(FoodClass).where(FoodClass.dataset_key == key)) is None:
            session.add(
                FoodClass(
                    dataset_key=key,
                    display_name_en=label["display_name_en"],
                    display_name_vi=label.get("display_name_vi"),
                    label_version=config["dataset_version"],
                )
            )
    session.flush()
    return seed_nutrition(session)


def seed_nutrition(session, catalog_dir=None):
    """Append reviewed versioned profiles; never replace defaults or historical meals."""
    catalog_dir = catalog_dir or PROJECT_ROOT / "data/nutrition"
    rows = read_csv(catalog_dir / "source_catalog.csv")
    mappings = read_csv(catalog_dir / "curated_mapping.csv")
    for row in rows:
        validate_reference(row)
        timestamp = datetime.fromisoformat(row["retrieved_at"])
        if timestamp.tzinfo is None:
            raise ValueError("Source retrieval timestamp must have a timezone")
    if session.bind.dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(180018002)"))
    summary = {"profiles_added": 0, "mappings_added": 0, "defaults_preserved": 0}
    for row in rows:
        if row["reviewed"] != "true" or not row["provenance_url"].startswith("https://"):
            continue
        profile = session.scalar(
            select(NutritionFood).where(
                NutritionFood.source_name == row["source_name"],
                NutritionFood.source_record_id == row["source_record_id"],
                NutritionFood.source_version == row["source_version"],
            )
        )
        if profile is None:
            values = {
                key: Decimal(row[key]) if row.get(key) else None
                for key in (
                    "kcal_per_100g",
                    "protein_per_100g",
                    "carbohydrate_per_100g",
                    "fat_per_100g",
                    "small_grams",
                    "medium_grams",
                    "large_grams",
                )
            }
            session.add(
                NutritionFood(
                    **values,
                    name=row["name"],
                    source_name=row["source_name"],
                    source_record_id=row["source_record_id"],
                    source_version=row["source_version"],
                    provenance_url=row["provenance_url"],
                    mapping_note=row["mapping_note"],
                    retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
                    reviewed=True,
                )
            )
            summary["profiles_added"] += 1
        else:
            for field in (*NUTRIENTS, *PORTIONS):
                if getattr(profile, field) != numeric(row.get(field), positive=field in PORTIONS):
                    raise ValueError(
                        f"Existing source profile differs: {row['source_record_id']}/{field}; "
                        "use a new version"
                    )
    session.flush()
    for row in mappings:
        food = session.scalar(select(FoodClass).where(FoodClass.dataset_key == row["dataset_key"]))
        profile = session.scalar(
            select(NutritionFood).where(
                NutritionFood.source_name == row["source_name"],
                NutritionFood.source_record_id == row["source_record_id"],
                NutritionFood.source_version == row["source_version"],
                NutritionFood.reviewed.is_(True),
            )
        )
        if food is None or profile is None:
            raise ValueError(f"Unknown food or versioned reference: {row['dataset_key']}")
        if food and profile:
            exists = session.scalar(
                select(FoodNutritionMapping).where(
                    FoodNutritionMapping.food_class_id == food.id,
                    FoodNutritionMapping.nutrition_food_id == profile.id,
                )
            )
            if exists is None:
                current = session.scalar(
                    select(FoodNutritionMapping).where(
                        FoodNutritionMapping.food_class_id == food.id,
                        FoodNutritionMapping.is_default.is_(True),
                    )
                )
                wanted = row["is_default"] == "true"
                if wanted and current:
                    summary["defaults_preserved"] += 1
                session.add(
                    FoodNutritionMapping(
                        food_class_id=food.id,
                        nutrition_food_id=profile.id,
                        is_default=wanted and current is None,
                        confidence_note=row["confidence_note"],
                    )
                )
                summary["mappings_added"] += 1
                session.flush()
    return summary
