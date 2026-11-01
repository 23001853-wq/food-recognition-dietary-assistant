from decimal import Decimal
from types import SimpleNamespace

import pytest

from food_assistant.domain.schemas import MealItemDraft
from food_assistant.services.nutrition_service import NutritionService


def test_formula_and_macros():
    # Deliberately synthetic reference fixture, never seeded into the application.
    profile = SimpleNamespace(
        reviewed=True,
        provenance_url="https://example.org/test-only",
        name="test fixture",
        source_name="synthetic",
        source_record_id="fixture",
        source_version="v1",
        mapping_note="unit test only",
        kcal_per_100g=Decimal("250"),
        protein_per_100g=Decimal("10"),
        carbohydrate_per_100g=None,
        fat_per_100g=Decimal("2"),
        medium_grams=Decimal("180"),
    )
    result = NutritionService(None).estimate(profile, "medium")
    assert result.estimated_kcal == Decimal("450.00")
    assert result.protein_g == Decimal("18.00")
    assert result.carbohydrate_g is None
    profile.kcal_per_100g = 1
    assert result.snapshot["kcal_per_100g"] == "250"


def test_unknown_is_not_zero():
    assert NutritionService(None).estimate(None, "small").estimated_kcal is None
    assert NutritionService(None).estimate(None, "custom", Decimal(100)).estimated_kcal is None


def test_manual_zero_is_valid():
    result = NutritionService(None).estimate(None, "custom", manual_kcal=Decimal(0))
    assert result.estimated_kcal == 0 and result.calculation_method == "manual"


@pytest.mark.parametrize(
    "field,value", [("grams", -1), ("grams", 0), ("manual_kcal", -1), ("grams", "NaN")]
)
def test_invalid_inputs(field, value):
    with pytest.raises(ValueError):
        MealItemDraft(custom_food_name="test", **{field: value})
