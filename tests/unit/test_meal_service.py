import pytest

from food_assistant.domain.schemas import MealDraft


def test_reject_empty_and_missing_food():
    for items in [[], [{}], [{"custom_food_name": "   "}]]:
        with pytest.raises(ValueError):
            MealDraft(consumed_on="2026-10-31", meal_type="lunch", items=items)
