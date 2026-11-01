"""Business schema. Nutrition/profile names preserve the original PLAN vocabulary."""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from food_assistant.persistence.database import Base


def now():
    return datetime.now(UTC)


class Identity:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class User(Identity, Base):
    __tablename__ = "users"
    display_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(255), unique=True)


class FoodClass(Identity, Base):
    __tablename__ = "food_classes"
    dataset_key: Mapped[str] = mapped_column(String(150), unique=True)
    display_name_en: Mapped[str] = mapped_column(String(200))
    display_name_vi: Mapped[str | None] = mapped_column(String(200))
    label_version: Mapped[str] = mapped_column(String(100), default="poc-v1")
    is_active: Mapped[bool] = mapped_column(default=True)


class NutritionFood(Identity, Base):
    __tablename__ = "nutrition_foods"
    name: Mapped[str] = mapped_column(String(200))
    kcal_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    protein_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    carbohydrate_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    fat_per_100g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    source_name: Mapped[str] = mapped_column(String(200))
    source_record_id: Mapped[str | None] = mapped_column(String(120))
    provenance_url: Mapped[str | None] = mapped_column(Text)
    source_version: Mapped[str] = mapped_column(String(100), default="unversioned")
    unit: Mapped[str] = mapped_column(String(20), default="per_100g")
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    small_grams: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    medium_grams: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    large_grams: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    mapping_note: Mapped[str | None] = mapped_column(Text)
    reviewed: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = tuple(
        CheckConstraint(f"{field} IS NULL OR {field} {'>' if 'grams' in field else '>='} 0")
        for field in (
            "kcal_per_100g",
            "protein_per_100g",
            "carbohydrate_per_100g",
            "fat_per_100g",
            "small_grams",
            "medium_grams",
            "large_grams",
        )
    ) + (
        CheckConstraint("unit = 'per_100g'"),
        UniqueConstraint("source_name", "source_record_id", "source_version"),
    )


class FoodNutritionMapping(Identity, Base):
    __tablename__ = "food_nutrition_mappings"
    food_class_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("food_classes.id"))
    nutrition_food_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nutrition_foods.id"))
    is_default: Mapped[bool] = mapped_column(default=False)
    confidence_note: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("food_class_id", "nutrition_food_id"),
        Index(
            "uq_default_nutrition",
            "food_class_id",
            unique=True,
            postgresql_where=text("is_default = true"),
        ),
    )


class ModelVersion(Identity, Base):
    __tablename__ = "model_versions"
    model_name: Mapped[str] = mapped_column(String(80))
    run_id: Mapped[str] = mapped_column(String(100), unique=True)
    dataset_version: Mapped[str] = mapped_column(String(100))
    num_classes: Mapped[int] = mapped_column(Integer)
    artifact_dir: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[dict | None] = mapped_column(JSONB)
    is_production: Mapped[bool] = mapped_column(default=False)
    __table_args__ = (
        CheckConstraint("num_classes > 0"),
        Index(
            "uq_production_model",
            "is_production",
            unique=True,
            postgresql_where=text("is_production = true"),
        ),
    )


class PredictionHistory(Identity, Base):
    __tablename__ = "prediction_history"
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    image_path: Mapped[str | None] = mapped_column(Text)
    image_sha256: Mapped[str | None] = mapped_column(String(64))
    crop_box_json: Mapped[list | None] = mapped_column(JSONB)
    response_json: Mapped[dict] = mapped_column(JSONB)
    is_mock: Mapped[bool] = mapped_column(Boolean)
    __table_args__ = (Index("ix_predictions_user_created", "user_id", "created_at"),)


class MealLog(Identity, Base):
    __tablename__ = "meal_logs"
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    client_request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    request_hash: Mapped[str] = mapped_column(String(64))
    consumed_on: Mapped[date] = mapped_column(Date)
    meal_type: Mapped[str] = mapped_column(String(20))
    notes: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    items: Mapped[list["MealItem"]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by="MealItem.created_at"
    )
    __table_args__ = (
        UniqueConstraint("user_id", "client_request_id"),
        CheckConstraint("meal_type IN ('breakfast','lunch','dinner','snack')"),
        Index("ix_meals_user_date", "user_id", "consumed_on"),
    )


class MealItem(Identity, Base):
    __tablename__ = "meal_items"
    meal_log_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("meal_logs.id", ondelete="CASCADE"), index=True
    )
    confirmed_food_class_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("food_classes.id"))
    custom_food_name: Mapped[str | None] = mapped_column(String(200))
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("prediction_history.id"))
    portion_size: Mapped[str] = mapped_column(String(20))
    estimated_grams: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    kcal_per_100g_snapshot: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    estimated_kcal: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    protein_g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    carbohydrate_g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    fat_g: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    nutrition_source_snapshot: Mapped[str | None] = mapped_column(Text)
    nutrition_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    calculation_method: Mapped[str] = mapped_column(String(30))
    is_user_corrected: Mapped[bool] = mapped_column(default=False)
    notes: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (
        CheckConstraint("confirmed_food_class_id IS NOT NULL OR custom_food_name IS NOT NULL"),
        CheckConstraint("portion_size IN ('small','medium','large','custom')"),
        CheckConstraint("calculation_method IN ('reference_formula','manual','unknown')"),
        CheckConstraint("estimated_grams IS NULL OR estimated_grams > 0"),
    ) + tuple(
        CheckConstraint(f"{key} IS NULL OR {key} >= 0")
        for key in (
            "estimated_kcal",
            "kcal_per_100g_snapshot",
            "protein_g",
            "carbohydrate_g",
            "fat_g",
        )
    )
