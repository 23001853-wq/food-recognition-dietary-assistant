"""Shared domain vocabulary from PLAN.md."""

from enum import StrEnum


class MealType(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class PortionSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    CUSTOM = "custom"


class RecognitionStatus(StrEnum):
    EMPTY = "EMPTY"
    IMAGE_READY = "IMAGE_READY"
    ANALYZING = "ANALYZING"
    RESULT_READY = "RESULT_READY"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    SAVED = "SAVED"
    ERROR = "ERROR"


class CalculationMethod(StrEnum):
    REFERENCE_FORMULA = "reference_formula"
    MANUAL = "manual"
    UNKNOWN = "unknown"
