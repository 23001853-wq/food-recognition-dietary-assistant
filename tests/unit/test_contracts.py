import json
from pathlib import Path

import jsonschema

from food_assistant.domain.schemas import PredictionResult


def test_published_contract_is_current():
    schema = json.loads(Path("contracts/prediction.schema.json").read_text())
    example = json.loads(Path("contracts/prediction.mock.example.json").read_text())
    assert schema == PredictionResult.model_json_schema()
    jsonschema.validate(example, schema)
    assert example["is_mock"] is True
