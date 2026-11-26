import json
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import jsonschema
import pytest
from pydantic import ValidationError

from food_assistant.domain.schemas import MealOut, PredictionResult


def test_published_contract_is_current():
    schema = json.loads(Path("contracts/prediction.schema.json").read_text())
    example = json.loads(Path("contracts/prediction.mock.example.json").read_text())
    assert schema == PredictionResult.model_json_schema()
    jsonschema.validate(example, schema)
    assert example["is_mock"] is True


def test_meal_output_timestamp_is_normalized_to_utc():
    """Serialize equivalent database timestamps with a stable UTC offset."""
    local_zone = timezone(timedelta(hours=7))
    local_timestamp = datetime(2026, 11, 25, 16, 10, tzinfo=local_zone)
    meal = MealOut(
        id=uuid4(),
        consumed_on=date(2026, 11, 25),
        meal_type="lunch",
        notes=None,
        items=[],
        created_at=local_timestamp,
    )

    assert meal.created_at == datetime(2026, 11, 25, 9, 10, tzinfo=UTC)
    assert meal.model_dump(mode="json")["created_at"] == "2026-11-25T09:10:00Z"


def test_meal_output_rejects_naive_timestamp():
    """Reject a meal timestamp whose instant cannot be determined safely."""
    with pytest.raises(ValidationError, match="created_at must include a timezone"):
        MealOut(
            id=uuid4(),
            consumed_on=date(2026, 11, 25),
            meal_type="lunch",
            notes=None,
            items=[],
            created_at=datetime(2026, 11, 25, 9, 10),
        )
