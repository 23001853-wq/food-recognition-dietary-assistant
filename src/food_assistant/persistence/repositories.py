"""Queries scoped to the local demo user. Services own mutation transactions."""

from sqlalchemy import select

from food_assistant.domain.errors import AppError
from food_assistant.persistence.orm_models import MealLog


class MealRepository:
    def __init__(self, session, user_id):
        self.session, self.user_id = session, user_id

    def get(self, meal_id):
        meal = self.session.scalar(
            select(MealLog).where(MealLog.id == meal_id, MealLog.user_id == self.user_id)
        )
        if meal is None:
            raise AppError("NOT_FOUND", "Không tìm thấy bữa ăn", 404)
        return meal

    def list(self, start, end, meal_type=None):
        query = select(MealLog).where(
            MealLog.user_id == self.user_id, MealLog.consumed_on.between(start, end)
        )
        if meal_type:
            query = query.where(MealLog.meal_type == meal_type)
        return list(
            self.session.scalars(query.order_by(MealLog.consumed_on.desc(), MealLog.created_at))
        )
