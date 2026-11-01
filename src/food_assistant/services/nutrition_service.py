"""Reviewed references only. Snapshot nutrient values and preserve unknowns."""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from sqlalchemy import select

from food_assistant.domain.schemas import NutritionEstimate
from food_assistant.nutrition.catalog import profile_notes
from food_assistant.persistence.orm_models import FoodNutritionMapping, NutritionFood
from food_assistant.services.portion_service import PortionService, validated_amount


def scaled(value, grams):
    if value is None or grams is None:
        return None
    try:
        amount, mass = Decimal(str(value)), Decimal(str(grams))
        if not amount.is_finite() or not mass.is_finite() or amount < 0 or mass <= 0:
            return None
        return (amount * mass / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


class NutritionService:
    def __init__(self, session):
        self.session = session

    def lookup(self, food_id):
        return self.session.scalar(
            select(NutritionFood)
            .join(FoodNutritionMapping, FoodNutritionMapping.nutrition_food_id == NutritionFood.id)
            .where(
                FoodNutritionMapping.food_class_id == food_id,
                FoodNutritionMapping.is_default.is_(True),
                NutritionFood.reviewed.is_(True),
            )
        )

    def estimate(self, profile, portion, grams=None, manual_kcal=None):
        eligible = (
            profile is not None
            and profile.reviewed
            and (profile.provenance_url or "").startswith("https://")
        )
        selected = PortionService.resolve(profile if eligible else None, portion, grams)
        grams = selected.grams
        common = {
            "grams": grams,
            "portion_grams": selected.profiles,
            "mass_source": selected.source,
            "portion_note": selected.note,
        }
        if manual_kcal is not None:
            return NutritionEstimate(
                **common,
                estimated_kcal=validated_amount(manual_kcal, energy=True),
                source="user_input",
                source_name="Người dùng nhập",
                source_type="user_input",
                status="manual",
                snapshot={
                    "source_type": "user_input",
                    "mass_source": selected.source,
                    "portion_size": str(portion),
                    "portion_note": selected.note,
                },
                calculation_method="manual",
            )
        if not eligible:
            return NutritionEstimate(**common, unavailable_reason="no_reviewed_reference")
        notes = profile_notes(profile)
        snapshot = {
            "name": profile.name,
            "source_name": profile.source_name,
            "record_id": profile.source_record_id,
            "source_version": profile.source_version,
            "provenance_url": profile.provenance_url,
            "mapping_note": profile.mapping_note,
            "source_type": "reference",
            "portion_size": str(portion),
            "mass_source": selected.source,
            "portion_note": selected.note,
            "match_note": notes.get("match_note"),
            "retrieved_at": str(getattr(profile, "retrieved_at", None)),
            "kcal_per_100g": str(profile.kcal_per_100g)
            if profile.kcal_per_100g is not None
            else None,
            "protein_per_100g": str(profile.protein_per_100g)
            if profile.protein_per_100g is not None
            else None,
            "carbohydrate_per_100g": str(profile.carbohydrate_per_100g)
            if profile.carbohydrate_per_100g is not None
            else None,
            "fat_per_100g": str(profile.fat_per_100g) if profile.fat_per_100g is not None else None,
        }
        kcal = scaled(profile.kcal_per_100g, grams)
        return NutritionEstimate(
            **common,
            estimated_kcal=kcal,
            protein_g=scaled(profile.protein_per_100g, grams),
            carbohydrate_g=scaled(profile.carbohydrate_per_100g, grams),
            fat_g=scaled(profile.fat_per_100g, grams),
            source=profile.provenance_url,
            kcal_per_100g=profile.kcal_per_100g,
            snapshot=snapshot,
            calculation_method="reference_formula" if kcal is not None else "unknown",
            status="available" if kcal is not None else "unavailable",
            source_type="reference",
            unavailable_reason=None
            if kcal is not None
            else "missing_grams"
            if grams is None
            else "missing_energy",
            reference_name=profile.name,
            source_name=profile.source_name,
            source_record_id=profile.source_record_id,
            source_version=profile.source_version,
            match_note=notes.get("match_note"),
        )
