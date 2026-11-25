"""Verify the complete official Food-101 dataset without modifying source images."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from PIL import Image

from food_assistant.ml.food101_verifier import _write_reports, audit_food101

FALLBACK_RESIZE = 256
FALLBACK_CROP = 224
NORMALIZE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class FallbackValidationTransform:
    """Provide deterministic model-shaped tensors when torchvision is unavailable."""

    def __call__(self, image: Image.Image):
        """Resize, center-crop, and ImageNet-normalize an RGB PIL image."""
        import torch

        width, height = image.size
        scale = FALLBACK_RESIZE / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
        )
        left = (resized.width - FALLBACK_CROP) // 2
        top = (resized.height - FALLBACK_CROP) // 2
        cropped = resized.crop((left, top, left + FALLBACK_CROP, top + FALLBACK_CROP))
        pixels = np.asarray(cropped, dtype=np.float32) / 255.0
        normalized = (pixels - NORMALIZE_MEAN) / NORMALIZE_STD
        return torch.from_numpy(normalized.transpose(2, 0, 1).copy())


def verify_loader(dataset_root: Path, manifest_dir: Path, batch_size: int) -> dict:
    """Read a batch from both official 101-class metadata and the frozen POC manifest."""
    import torch
    from torch.utils.data import DataLoader

    from food_assistant.ml.datasets import ManifestDataset, OfficialFood101Dataset

    result = {
        "status": "PASS",
        "batch_size_requested": batch_size,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    pipeline_transform = None
    try:
        from food_assistant.ml.transforms import build_transform, preprocessing

        pipeline_transform = build_transform(preprocessing("resnet18"), train=True)
        result["preprocessing"] = {"status": "AVAILABLE", "model": "resnet18", "mode": "training"}
    except Exception as exc:
        pipeline_transform = FallbackValidationTransform()
        result["preprocessing"] = {
            "status": "BLOCKED",
            "error": f"{type(exc).__name__}: {exc}",
            "fallback": (
                "deterministic Resize(256), CenterCrop(224), ImageNet normalization; "
                "does not test training augmentation"
            ),
        }

    official = OfficialFood101Dataset(dataset_root, "train", transform=pipeline_transform)
    official_batch = next(
        iter(DataLoader(official, batch_size=batch_size, shuffle=False, num_workers=0))
    )
    official_images, official_labels = official_batch
    expected_shape = (batch_size, 3, 224, 224)
    official_ok = (
        len(official.classes) == 101
        and len(official) == 75_750
        and official_images.shape[0] == batch_size
        and official_labels.dtype == torch.int64
        and int(official_labels.min()) >= 0
        and int(official_labels.max()) < len(official.classes)
    )
    official_ok = official_ok and tuple(official_images.shape) == expected_shape
    official_ok = official_ok and official_images.dtype == torch.float32
    result["official_101_class"] = {
        "status": "PASS" if official_ok else "FAIL",
        "classes": len(official.classes),
        "train_samples": len(official),
        "class_id_range": [int(official_labels.min()), int(official_labels.max())],
        "class_id_dtype": str(official_labels.dtype),
        "class_keys_in_batch": [official.classes[int(index)] for index in official_labels],
        "image_shape": list(official_images.shape),
        "image_dtype": str(official_images.dtype),
        "preprocessing_checked": result["preprocessing"]["status"] == "AVAILABLE",
    }

    try:
        poc = ManifestDataset(
            manifest_dir,
            "train",
            transform=pipeline_transform,
            limit_per_class=1,
            root_override=dataset_root,
        )
        poc_batch = next(iter(DataLoader(poc, batch_size=batch_size, shuffle=False, num_workers=0)))
        poc_images, poc_labels = poc_batch
        poc_ok = (
            len(poc.labels["labels"]) == 10
            and poc_images.shape[0] == batch_size
            and tuple(poc_images.shape[1:]) == (3, 224, 224)
            and poc_images.dtype == torch.float32
            and poc_labels.dtype == torch.int64
            and int(poc_labels.min()) >= 0
            and int(poc_labels.max()) < len(poc.labels["labels"])
        )
        result["poc_10_class"] = {
            "status": "PASS" if poc_ok else "FAIL",
            "classes": len(poc.labels["labels"]),
            "batch_shape": list(poc_images.shape),
            "batch_dtype": str(poc_images.dtype),
            "class_id_dtype": str(poc_labels.dtype),
            "class_keys_in_batch": [
                poc.labels["labels"][int(index)]["class_key"] for index in poc_labels
            ],
            "class_id_range": [int(poc_labels.min()), int(poc_labels.max())],
            "manifest_dir": str(manifest_dir.resolve()),
            "manifest_modified": False,
        }
    except Exception as exc:
        result["poc_10_class"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    result["status"] = (
        "PASS"
        if result["official_101_class"]["status"] == "PASS"
        and result.get("poc_10_class", {}).get("status") == "PASS"
        and result["preprocessing"]["status"] == "AVAILABLE"
        else "PARTIAL"
        if result["official_101_class"]["status"] == "PASS"
        else "FAIL"
    )
    return result


def main() -> None:
    """Run a quick or resumable full audit and optionally read DataLoader batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/food-101"))
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports/food101"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--dataloader", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.progress_every < 1 or args.batch_size < 1:
        parser.error("--progress-every and --batch-size must be positive")

    report = audit_food101(args.root, args.report_dir, args.mode, args.progress_every)
    if args.dataloader:
        try:
            report["data_loader"] = verify_loader(args.root, args.manifest_dir, args.batch_size)
        except Exception as exc:
            report["data_loader"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        if report["overall"] == "PASS" and report["data_loader"]["status"] != "PASS":
            report["overall"] = "INCOMPLETE"
        _write_reports(args.report_dir, report)
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "report_json": str(args.report_dir / "food101_report.json"),
                "report_markdown": str(args.report_dir / "food101_report.md"),
            },
            indent=2,
        )
    )
    if report["overall"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
