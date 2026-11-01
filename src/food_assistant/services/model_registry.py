"""Trusted local releases with checksums, stable labels and consistent preprocessing."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from food_assistant.domain.errors import AppError
from food_assistant.domain.schemas import Prediction


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Release:
    def __init__(self, directory, device="cpu"):
        import joblib
        import torch

        from food_assistant.ml.model_factory import create_model
        from food_assistant.ml.train_utils import device_for
        from food_assistant.ml.transforms import build_transform

        self.directory = Path(directory)
        checksums = read_json(self.directory / "checksum.sha256")
        required = {
            "model_config.json",
            "labels.json",
            "class_to_idx.json",
            "idx_to_class.json",
            "preprocessing.json",
        }
        if not required <= checksums.keys():
            raise ValueError("Missing required artifact checksums")
        for name, digest in checksums.items():
            path = (self.directory / name).resolve()
            if (
                not path.is_relative_to(self.directory.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Artifact integrity check failed")
        self.config = read_json(self.directory / "model_config.json")
        self.labels = read_json(self.directory / "labels.json")
        labels = self.labels["labels"]
        mapping = read_json(self.directory / "class_to_idx.json")
        inverse = read_json(self.directory / "idx_to_class.json")
        if (
            len(labels) != self.config["num_classes"]
            or [x["class_id"] for x in labels] != list(range(len(labels)))
            or mapping != {x["class_key"]: x["class_id"] for x in labels}
            or inverse != {str(x["class_id"]): x["class_key"] for x in labels}
            or self.labels["dataset_version"] != self.config["dataset_version"]
        ):
            raise ValueError("Artifact class map mismatch")
        if not self.config.get("trained"):
            raise ValueError("Model has not completed training")
        self.preproc = read_json(self.directory / "preprocessing.json")
        self.device = device_for("cpu" if self.config["model_name"] == "hog_svm" else device)
        if self.config["model_name"] == "hog_svm":
            if "model.joblib" not in checksums:
                raise ValueError("Missing model checksum")
            self.model = joblib.load(self.directory / "model.joblib")
            if list(self.model.classes_) != list(range(len(labels))):
                raise ValueError("SVM class order mismatch")
        else:
            if "model.pth" not in checksums:
                raise ValueError("Missing model checksum")
            self.model = create_model(self.config["model_name"], len(labels))
            self.model.load_state_dict(
                torch.load(self.directory / "model.pth", map_location="cpu", weights_only=True)
            )
            self.model.to(self.device).eval()
            self.transform = build_transform(self.preproc)

    def predict(self, image, top_k=3):
        import numpy as np
        import torch

        from food_assistant.ml.hog_features import extract_hog

        if self.config["model_name"] == "hog_svm":
            decision = np.asarray(self.model.decision_function([extract_hog(image, self.preproc)]))[
                0
            ]
            if np.ndim(decision) == 0:
                decision = np.array([-decision, decision])
            # Normalized decision scores for ranking, not calibrated probabilities.
            scores = np.exp(decision - max(decision))
            scores /= scores.sum()
        else:
            with torch.inference_mode():
                scores = (
                    self.model(self.transform(image).unsqueeze(0).to(self.device))
                    .softmax(1)[0]
                    .cpu()
                    .numpy()
                )
        indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        labels = self.labels["labels"]
        return [
            Prediction(
                class_id=int(i),
                class_key=labels[i]["class_key"],
                display_name_vi=labels[i]["display_name_vi"],
                score=float(scores[i]),
            )
            for i in indices
        ]


@lru_cache(maxsize=4)
def cached_release(directory, checksum_hash, device):
    return Release(directory, device)


class ModelRegistry:
    def __init__(self, settings):
        self.settings = settings

    def catalog(self):
        result = []
        if self.settings.allow_mock_ai:
            result.append(
                {"name": "mock", "version": "deterministic-v1", "is_mock": True, "trained": False}
            )
        for path in self.settings.artifact_root.glob("*/*/model_config.json"):
            data = read_json(path)
            public = {
                key: data[key]
                for key in (
                    "model_name",
                    "run_id",
                    "dataset_version",
                    "num_classes",
                    "smoke_only",
                    "pretrained",
                    "trained",
                    "training_samples",
                    "validation_samples",
                )
                if key in data
            }
            for split in ("val", "test"):
                metrics = path.parent / f"metrics_{split}.json"
                if metrics.exists():
                    public[f"metrics_{split}"] = read_json(metrics)
            result.append(
                {
                    **public,
                    "is_mock": False,
                    "enabled": (path.parent / "checksum.sha256").exists()
                    and (not data.get("smoke_only") or self.settings.allow_smoke_models),
                    "selector": f"{data['model_name']}/{data['run_id']}",
                }
            )
        if self.settings.model_pointer.exists():
            try:
                pointer = read_json(self.settings.model_pointer)
                selected = next(
                    row
                    for row in result
                    if row.get("selector") == pointer["artifact"]
                    and row.get("trained")
                    and row.get("enabled")
                )
                result.insert(0, {**selected, "selector": "production", "production": True})
            except (ValueError, KeyError, OSError, StopIteration):
                pass
        return result

    def load(self, selector="production"):
        try:
            if selector == "production":
                pointer = read_json(self.settings.model_pointer)
                selector = pointer["artifact"]
            root = self.settings.artifact_root.resolve()
            directory = (root / selector).resolve()
            if not directory.is_relative_to(root) or len(Path(selector).parts) != 2:
                raise ValueError("Invalid model selector")
            checksum = hashlib.sha256((directory / "checksum.sha256").read_bytes()).hexdigest()
            release = cached_release(str(directory), checksum, self.settings.inference_device)
            if release.config.get("smoke_only") and (
                self.settings.app_env == "production" or not self.settings.allow_smoke_models
            ):
                raise ValueError("Smoke-only model is disabled; explicitly opt in for development")
            return release
        except FileNotFoundError as exc:
            raise AppError(
                "MODEL_NOT_READY", "Chưa có model artifact được huấn luyện", 503
            ) from exc
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            raise AppError(
                "ARTIFACT_MISMATCH", "Model artifact không hợp lệ hoặc không tương thích", 503
            ) from exc
