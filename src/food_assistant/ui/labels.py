"""Vietnamese display labels; API and database enum values stay stable."""

MEAL_LABELS = {
    "all": "Tất cả bữa ăn",
    "breakfast": "Bữa sáng",
    "lunch": "Bữa trưa",
    "dinner": "Bữa tối",
    "snack": "Bữa phụ",
}
PORTION_LABELS = {
    "small": "Nhỏ",
    "medium": "Vừa",
    "large": "Lớn",
    "custom": "Tùy chỉnh",
}


def meal_label(value):
    return MEAL_LABELS.get(value, value)


def portion_label(value):
    return PORTION_LABELS.get(value, value)


def nutrient_value(value):
    """Display persisted values, including a real zero, without inventing missing macros."""
    return "Chưa có dữ liệu" if value is None else str(value)
