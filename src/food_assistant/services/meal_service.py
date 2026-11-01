"""Validated transactional meals with retry-safe creation and immutable snapshots."""

import hashlib

from sqlalchemy import select, text

from food_assistant.domain.errors import AppError
from food_assistant.persistence.orm_models import FoodClass, MealItem, MealLog, PredictionHistory
from food_assistant.persistence.repositories import MealRepository
from food_assistant.services.nutrition_service import NutritionService


class MealService:
    def __init__(self, session, user_id):
        self.session, self.user_id = session, user_id
        self.repository = MealRepository(session, user_id)

    def build_items(self, drafts):
        items = []
        nutrition = NutritionService(self.session)
        for draft in drafts:
            food = self.session.get(FoodClass, draft.food_class_id) if draft.food_class_id else None
            if draft.food_class_id and (food is None or not food.is_active):
                raise AppError("UNSUPPORTED_FOOD", "Món không tồn tại hoặc đã ngừng hỗ trợ", 422)
            prediction = (
                self.session.get(PredictionHistory, draft.prediction_id)
                if draft.prediction_id
                else None
            )
            if draft.prediction_id and (prediction is None or prediction.user_id != self.user_id):
                raise AppError("NOT_FOUND", "Không tìm thấy dự đoán", 404)
            profile = nutrition.lookup(food.id) if food else None
            estimate = nutrition.estimate(
                profile, draft.portion_size, draft.grams, draft.manual_kcal
            )
            corrected = bool(
                prediction
                and (
                    food is None
                    or food.dataset_key != prediction.response_json["predictions"][0]["class_key"]
                )
            )
            items.append(
                MealItem(
                    confirmed_food_class_id=draft.food_class_id,
                    custom_food_name=draft.custom_food_name,
                    prediction_id=draft.prediction_id,
                    portion_size=draft.portion_size,
                    estimated_grams=estimate.grams,
                    estimated_kcal=estimate.estimated_kcal,
                    protein_g=estimate.protein_g,
                    carbohydrate_g=estimate.carbohydrate_g,
                    fat_g=estimate.fat_g,
                    kcal_per_100g_snapshot=estimate.kcal_per_100g,
                    nutrition_source_snapshot=estimate.source,
                    nutrition_snapshot=estimate.snapshot,
                    calculation_method=estimate.calculation_method,
                    is_user_corrected=corrected,
                    notes=draft.notes,
                )
            )
        return items

    def create(self, draft):
        fingerprint = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
        # Transaction-scoped advisory lock serializes retries across backend processes.
        lock = int.from_bytes(
            hashlib.sha256(str(draft.client_request_id).encode()).digest()[:8], "big", signed=True
        )
        self.session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})
        existing = self.session.scalar(
            select(MealLog).where(
                MealLog.user_id == self.user_id,
                MealLog.client_request_id == draft.client_request_id,
            )
        )
        if existing:
            if existing.request_hash != fingerprint:
                raise AppError("IDEMPOTENCY_CONFLICT", "Mã yêu cầu đã dùng với nội dung khác", 409)
            return existing
        meal = MealLog(
            user_id=self.user_id,
            client_request_id=draft.client_request_id,
            request_hash=fingerprint,
            consumed_on=draft.consumed_on,
            meal_type=draft.meal_type,
            notes=draft.notes,
            items=self.build_items(draft.items),
        )
        self.session.add(meal)
        self.session.flush()
        return meal

    def update(self, meal_id, patch):
        meal = self.repository.get(meal_id)
        for key, value in patch.model_dump(exclude_unset=True, exclude={"items"}).items():
            setattr(meal, key, value)
        if patch.items is not None:
            meal.items = self.build_items(patch.items)
        self.session.flush()
        return meal

    def delete(self, meal_id):
        self.session.delete(self.repository.get(meal_id))
        self.session.flush()
