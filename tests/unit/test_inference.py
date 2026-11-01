import pytest
from PIL import Image

from food_assistant.config import Settings
from food_assistant.domain.errors import AppError
from food_assistant.domain.schemas import PredictionResult
from food_assistant.services.inference_service import InferenceService


def test_mock_deterministic_and_schema():
    service = InferenceService(Settings(_env_file=None, allow_mock_ai=True))
    a = service.predict(Image.new("RGB", (16, 16)), "mock")
    b = service.predict(Image.new("RGB", (16, 16)), "mock")
    assert a.is_mock and a.predictions == b.predictions
    assert PredictionResult.model_validate_json(a.model_dump_json()) == a
    assert len(a.predictions) == 3


def test_mock_disabled():
    with pytest.raises(AppError) as exc:
        InferenceService(Settings(_env_file=None)).predict(Image.new("RGB", (16, 16)), "mock")
    assert exc.value.code == "MOCK_DISABLED"


def test_production_forbids_mock():
    with pytest.raises(ValueError):
        Settings(_env_file=None, app_env="production", allow_mock_ai=True)


def test_missing_model(tmp_path):
    service = InferenceService(
        Settings(_env_file=None, artifact_root=tmp_path, model_pointer=tmp_path / "missing")
    )
    with pytest.raises(AppError) as exc:
        service.predict(Image.new("RGB", (16, 16)))
    assert exc.value.code == "MODEL_NOT_READY"
