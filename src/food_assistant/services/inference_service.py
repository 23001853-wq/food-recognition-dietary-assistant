"""Common inference interface. Mock is explicit and never silently replaces real AI."""

import hashlib
import time

from food_assistant.domain.errors import AppError
from food_assistant.domain.schemas import ModelIdentity, Prediction, PredictionResult
from food_assistant.persistence.seed import VI_NAMES


class InferenceService:
    def __init__(self, settings):
        self.settings = settings

    def predict(self, image, model_name="production", top_k=3):
        if not 1 <= top_k <= 10:
            raise AppError("INVALID_TOP_K", "top_k phải từ 1 đến 10", 422)
        start = time.perf_counter()
        if model_name == "mock":
            if not self.settings.allow_mock_ai or self.settings.app_env == "production":
                raise AppError(
                    "MOCK_DISABLED", "Mock AI chưa được bật hoặc bị cấm trong production", 403
                )
            keys = list(VI_NAMES)
            offset = hashlib.sha256(image.tobytes()).digest()[0] % len(keys)
            indices = [(offset + i) % len(keys) for i in range(min(top_k, len(keys)))]
            predictions = [
                Prediction(
                    class_id=i,
                    class_key=keys[i],
                    display_name_vi=VI_NAMES[keys[i]],
                    score=0.6 / (rank + 1),
                )
                for rank, i in enumerate(indices)
            ]
            identity = ModelIdentity(
                name="mock", version="deterministic-v1", dataset_version="mock-poc-v1"
            )
        else:
            from food_assistant.services.model_registry import ModelRegistry

            release = ModelRegistry(self.settings).load(model_name)
            predictions = release.predict(image, top_k)
            identity = ModelIdentity(
                name=release.config["model_name"],
                version=release.config["run_id"],
                dataset_version=release.config["dataset_version"],
            )
        return PredictionResult(
            model=identity,
            is_mock=model_name == "mock",
            predictions=predictions,
            inference_time_ms=(time.perf_counter() - start) * 1000,
        )
