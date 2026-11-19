"""End-to-end image tensor smoke tests over a tiny synthetic manifest."""

import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from food_assistant.ml.datasets import ManifestDataset, prepare_food101
from food_assistant.ml.transforms import build_transform, preprocessing


def test_manifest_dataset_and_dataloader_preserve_eval_tensor_contract(tmp_path):
    """Load a synthetic prepared split through PIL, production transforms and DataLoader."""
    raw_root = tmp_path / "raw"
    (raw_root / "meta").mkdir(parents=True)
    (raw_root / "images" / "apple_pie").mkdir(parents=True)
    (raw_root / "images" / "baby_back_ribs").mkdir(parents=True)
    class_names = ["apple_pie", "baby_back_ribs"]
    (raw_root / "meta" / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    train_ids = []
    test_ids = []
    for class_index, class_name in enumerate(class_names):
        for image_index in range(5):
            image_id = f"{class_name}/{image_index:02d}"
            image_path = raw_root / "images" / class_name / f"{image_index:02d}.jpg"
            pixels = np.full((280, 280, 3), 30 + class_index * 90 + image_index, dtype=np.uint8)
            Image.fromarray(pixels).save(image_path)
            (test_ids if image_index == 4 else train_ids).append(image_id)
    (raw_root / "meta" / "train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (raw_root / "meta" / "test.txt").write_text("\n".join(test_ids), encoding="utf-8")
    output = tmp_path / "manifest"
    prepare_food101(
        raw_root, output, class_names, version="synthetic-v1", seed=42, val_fraction=0.25
    )

    dataset = ManifestDataset(output, "train", transform=build_transform(preprocessing("resnet18")))
    batch_images, batch_labels = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch_images.shape == (2, 3, 224, 224)
    assert batch_images.dtype == torch.float32
    assert torch.isfinite(batch_images).all()
    assert batch_labels.tolist() in ([0, 1], [1, 0], [0, 0], [1, 1])

    with (output / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual_test_ids = {row["image_id"] for row in rows if row["split"] == "test"}
    assert actual_test_ids == set(test_ids)
