from collections import Counter
from decimal import Decimal

from food_assistant.persistence.repositories import MealRepository

NUTRIENTS = ("estimated_kcal", "protein_g", "carbohydrate_g", "fat_g")


def nutrient_totals(items):
    """Aggregate persisted snapshots only; a missing nutrient is never invented as zero."""
    result = {}
    for field in NUTRIENTS:
        known = [getattr(item, field) for item in items if getattr(item, field) is not None]
        result[field] = {
            "total": sum(known, Decimal(0)) if known else None,
            "known_count": len(known),
            "missing_count": len(items) - len(known),
        }
    return result


class DashboardService:
    def __init__(self, session, user_id):
        self.repository = MealRepository(session, user_id)

    def summary(self, start, end):
        meals = self.repository.list(start, end)
        items = [item for meal in meals for item in meal.items]
        days, types, foods, day_items = {}, {}, Counter(), {}
        for meal in meals:
            day = days.setdefault(
                str(meal.consumed_on),
                {
                    "date": str(meal.consumed_on),
                    "estimated_kcal": Decimal(0),
                    "missing_count": 0,
                    "known_count": 0,
                },
            )
            for item in meal.items:
                day_items.setdefault(str(meal.consumed_on), []).append(item)
                if item.estimated_kcal is None:
                    day["missing_count"] += 1
                else:
                    day["known_count"] += 1
                    day["estimated_kcal"] += item.estimated_kcal
                    types[meal.meal_type] = (
                        types.get(meal.meal_type, Decimal(0)) + item.estimated_kcal
                    )
                foods[str(item.confirmed_food_class_id or item.custom_food_name)] += 1
        for date, day in days.items():
            day["nutrients"] = nutrient_totals(day_items.get(date, []))
        return {
            "meal_count": len(meals),
            "item_count": len(items),
            "estimated_kcal": sum(
                (i.estimated_kcal for i in items if i.estimated_kcal is not None), Decimal(0)
            ),
            "missing_count": sum(i.estimated_kcal is None for i in items),
            "known_count": sum(i.estimated_kcal is not None for i in items),
            "user_input_count": sum(i.calculation_method == "manual" for i in items),
            "reference_count": sum(i.calculation_method == "reference_formula" for i in items),
            "daily": sorted(days.values(), key=lambda x: x["date"]),
            "by_meal_type": types,
            "frequent_foods": dict(foods.most_common(10)),
            "nutrients": nutrient_totals(items),
        }
