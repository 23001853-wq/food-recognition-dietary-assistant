"""Resolve per-food stored mass profiles or explicit user grams; no vision mass inference."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from food_assistant.domain.errors import AppError
from food_assistant.nutrition.catalog import profile_notes

PORTIONS = ("small", "medium", "large", "custom")
PORTION_LABELS = {
    "small": "Nhỏ (S)",
    "medium": "Vừa (M)",
    "large": "Lớn (L)",
    "custom": "Tự nhập gram",
}


def validated_amount(value, *, energy=False):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise AppError(
            "INVALID_PORTION" if not energy else "INVALID_KCAL", "Giá trị nhập không hợp lệ", 422
        ) from exc
    maximum = 100000 if energy else 10000
    if (
        not number.is_finite()
        or number < 0
        or (not energy and number == 0)
        or number > maximum
        or number.as_tuple().exponent < -2
    ):
        raise AppError(
            "INVALID_KCAL" if energy else "INVALID_PORTION",
            "Giá trị phải hợp lệ, tối đa hai chữ số thập phân",
            422,
        )
    return number


@dataclass(frozen=True)
class PortionSelection:
    grams: Decimal | None
    source: str
    profiles: dict[str, Decimal | None]
    note: str | None


class PortionService:
    @staticmethod
    def resolve(profile, portion, grams=None):
        if portion not in PORTIONS:
            raise AppError("INVALID_PORTION", "Khẩu phần không hợp lệ", 422)
        profiles = {
            key: getattr(profile, f"{key}_grams", None) if profile else None
            for key in PORTIONS
            if key != "custom"
        }
        notes = profile_notes(profile) if profile else {}
        if grams is not None:
            return PortionSelection(
                validated_amount(grams),
                "user_input",
                profiles,
                "Khối lượng do người dùng nhập, không đo từ ảnh.",
            )
        mass = profiles.get(portion)
        if mass is not None:
            mass = validated_amount(mass)
        return PortionSelection(
            mass,
            notes.get("portion_basis", "unspecified") if mass else "unavailable",
            profiles,
            notes.get("portion_note"),
        )
