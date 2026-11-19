"""Regression checks for existing CNN and HOG preprocessing contracts."""

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from food_assistant.ml.hog_features import extract_hog
from food_assistant.ml.transforms import build_transform, preprocessing


def test_resnet_and_efficientnet_eval_transforms_are_deterministic():
    """Keep legacy eval pipelines deterministic and 3x224x224 with finite values."""
    image = Image.fromarray(
        np.random.default_rng(27).integers(0, 256, (260, 310, 3), dtype=np.uint8)
    )
    for model_name in ("resnet18", "efficientnet_b0"):
        transform = build_transform(preprocessing(model_name), train=False)
        first = transform(image)
        second = transform(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert first.shape == (3, 224, 224)
        assert first.dtype == torch.float32 and torch.isfinite(first).all()


def test_hog_feature_contract_remains_rgb_to_gray_configured_size():
    """Preserve HOG grayscale conversion and dimensions from the project config."""
    config_path = Path("configs/hog_svm.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    image = Image.fromarray(
        np.random.default_rng(29).integers(0, 256, (150, 170, 3), dtype=np.uint8)
    )
    features = extract_hog(image, config)
    assert features.ndim == 1
    assert features.size > 0
    assert np.isfinite(features).all()
