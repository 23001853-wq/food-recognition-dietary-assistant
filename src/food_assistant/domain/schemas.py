"""Versioned wire contracts shared by HTTP endpoints and inference."""

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from food_assistant.domain.enums import MealType, PortionSize

PositiveDecimal = Annotated[Decimal, Field(gt=0, le=10000, max_digits=10, decimal_places=2)]
EnergyDecimal = Annotated[Decimal, Field(ge=0, le=100000, max_digits=10, decimal_places=2)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ModelIdentity(Contract):
    name: str
    version: str
    dataset_version: str


class Prediction(Contract):
    class_id: int = Field(ge=0)
    class_key: str
    display_name_vi: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class PredictionResult(Contract):
    schema_version: Literal["1.0"] = "1.0"
    prediction_id: UUID = Field(default_factory=uuid4)
    model: ModelIdentity
    is_mock: bool
    predictions: list[Prediction] = Field(min_length=1)
    inference_time_ms: float = Field(ge=0, allow_inf_nan=False)


class MealItemDraft(Contract):
    food_class_id: UUID | None = None
    custom_food_name: str | None = Field(default=None, min_length=1, max_length=200)
    prediction_id: UUID | None = None
    portion_size: PortionSize = PortionSize.CUSTOM
    grams: PositiveDecimal | None = None
    manual_kcal: EnergyDecimal | None = None
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_food(self):
        if not self.food_class_id and not (self.custom_food_name or "").strip():
            raise ValueError("Select a food or supply a custom food name")
        return self


class MealDraft(Contract):
    consumed_on: date
    meal_type: MealType
    notes: str | None = Field(default=None, max_length=2000)
    items: list[MealItemDraft] = Field(min_length=1, max_length=50)
    client_request_id: UUID = Field(default_factory=uuid4)


class MealPatch(Contract):
    consumed_on: date | None = None
    meal_type: MealType | None = None
    notes: str | None = Field(default=None, max_length=2000)
    items: list[MealItemDraft] | None = Field(default=None, min_length=1, max_length=50)

    @model_validator(mode="after")
    def no_null_required_fields(self):
        for key in ("consumed_on", "meal_type", "items"):
            if key in self.model_fields_set and getattr(self, key) is None:
                raise ValueError(f"{key} cannot be null")
        return self


class FoodOut(Contract):
    id: UUID
    dataset_key: str
    display_name_en: str
    display_name_vi: str | None
    label_version: str


class ItemOut(Contract):
    id: UUID
    confirmed_food_class_id: UUID | None
    custom_food_name: str | None
    prediction_id: UUID | None
    portion_size: str
    estimated_grams: Decimal | None
    estimated_kcal: Decimal | None
    protein_g: Decimal | None
    carbohydrate_g: Decimal | None
    fat_g: Decimal | None
    kcal_per_100g_snapshot: Decimal | None
    nutrition_source_snapshot: str | None
    nutrition_snapshot: dict | None
    calculation_method: str
    is_user_corrected: bool
    notes: str | None

    @computed_field
    @property
    def source_type(self) -> str:
        if self.calculation_method == "manual":
            return "user_input"
        return "reference" if self.calculation_method == "reference_formula" else "unavailable"


class MealOut(Contract):
    id: UUID
    consumed_on: date
    meal_type: str
    notes: str | None
    items: list[ItemOut]
    created_at: datetime


class NutritionEstimate(Contract):
    grams: Decimal | None = None
    estimated_kcal: Decimal | None = None
    protein_g: Decimal | None = None
    carbohydrate_g: Decimal | None = None
    fat_g: Decimal | None = None
    source: str | None = None
    kcal_per_100g: Decimal | None = None
    snapshot: dict | None = None
    calculation_method: Literal["reference_formula", "manual", "unknown"] = "unknown"
    status: Literal["available", "unavailable", "manual"] = "unavailable"
    source_type: Literal["reference", "user_input", "unavailable"] = "unavailable"
    unavailable_reason: str | None = None
    reference_name: str | None = None
    source_name: str | None = None
    source_record_id: str | None = None
    source_version: str | None = None
    portion_grams: dict[str, Decimal | None] = Field(default_factory=dict)
    mass_source: str = "unavailable"
    portion_note: str | None = None
    match_note: str | None = None
