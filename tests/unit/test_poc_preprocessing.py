import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

from food_assistant.ml.datasets import ManifestDataset
from food_assistant.ml.transforms import build_transform, preprocessing


@pytest.mark.parametrize(
    "name,weights",
    [
        ("resnet18", ResNet18_Weights.IMAGENET1K_V1),
        ("efficientnet_b0", EfficientNet_B0_Weights.IMAGENET1K_V1),
    ],
)
def test_eval_transform_matches_official_weights_without_download(name, weights):
    photo = Image.fromarray(
        np.random.default_rng(42).integers(0, 255, (180, 320, 3), dtype=np.uint8)
    )
    actual = build_transform(preprocessing(name))(photo)
    expected = weights.transforms()(photo)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    train_names = [
        type(step).__name__ for step in build_transform(preprocessing(name), True).transforms
    ]
    eval_names = [type(step).__name__ for step in build_transform(preprocessing(name)).transforms]
    assert "RandomResizedCrop" in train_names and "RandomHorizontalFlip" in train_names
    assert not any("Random" in name for name in eval_names)


def test_dataset_applies_exif_before_rgb(tmp_path):
    photo = Image.new("RGB", (32, 16))
    exif = photo.getexif()
    exif[274] = 6
    photo.save(tmp_path / "photo.jpg", exif=exif)
    dataset = object.__new__(ManifestDataset)
    dataset.root = tmp_path
    dataset.rows = [{"relative_path": "photo.jpg", "class_id": "0"}]
    dataset.transform = None
    pixels, label = dataset[0]
    assert pixels.shape == (32, 16, 3) and label == 0
