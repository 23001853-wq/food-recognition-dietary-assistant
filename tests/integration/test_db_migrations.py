import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from food_assistant.persistence.orm_models import FoodClass, User
from food_assistant.persistence.seed import seed

pytestmark = pytest.mark.integration


def test_migration_schema(pg_engine):
    inspector = inspect(pg_engine)
    assert {
        "users",
        "food_classes",
        "nutrition_foods",
        "food_nutrition_mappings",
        "meal_logs",
        "meal_items",
        "prediction_history",
        "model_versions",
    } <= set(inspector.get_table_names())
    assert any(
        i["name"] == "uq_default_nutrition" and i["unique"]
        for i in inspector.get_indexes("food_nutrition_mappings")
    )


def test_idempotent_seed(pg_engine):
    with pg_engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection) as session:
            seed(session)
            first = session.scalar(select(func.count()).select_from(FoodClass))
            seed(session)
            assert session.scalar(select(func.count()).select_from(FoodClass)) == first
            assert session.scalar(select(func.count()).select_from(User)) == 1
        transaction.rollback()
