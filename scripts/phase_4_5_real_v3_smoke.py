"""Run a bounded, read-only processing and DataLoader smoke test on real v3 images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from food_assistant.ml.datasets import FIELDS, ManifestDataset, read_json
from food_assistant.ml.image_ops import (
    adjust_brightness,
    adjust_contrast,
    canny_edges,
    center_crop,
    chw_to_hwc,
    closing,
    dilation,
    erosion,
    extract_regions,
    gamma_correct,
    gaussian_blur,
    hwc_to_chw,
    median_filter,
    opening,
    resize_image,
    rgb_to_bgr,
    rgb_to_grayscale,
    rgb_to_hsv,
    rotate_image,
    to_float01,
    to_rgb_array,
)
from food_assistant.ml.near_duplicate import SimilarityPolicy, compare_pair
from food_assistant.ml.transforms import build_transform, preprocessing

SEED = 4505
TRAIN_COUNT = 20
VAL_COUNT = 10
CLASS_COUNT = 20


class _CaptureTransform:
    """Capture EXIF-corrected RGB inputs in memory, then invoke a production transform."""

    def __init__(self, transform):
        """Initialize a transform wrapper and its ordered capture list."""
        self.transform = transform
        self.images = []

    def __call__(self, image):
        """Capture the image presented by ManifestDataset and return transformed pixels."""
        self.images.append(
            {"pixels": np.asarray(image.convert("RGB")).copy(), "size": list(image.size)}
        )
        return self.transform(image)


def _select_rows(manifest_path):
    """Select one seeded train image for classes 0-19 and one val image for classes 0-9."""
    candidates = {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("v3 manifest header differs from ManifestDataset schema")
        for row in reader:
            class_id = int(row["class_id"])
            if class_id < CLASS_COUNT and row["split"] in {"train", "val"}:
                candidates.setdefault((row["split"], class_id), []).append(row)
    chooser = random.Random(SEED)
    selected = []
    for split, class_ids in (("train", range(TRAIN_COUNT)), ("val", range(VAL_COUNT))):
        for class_id in class_ids:
            options = sorted(candidates.get((split, class_id), []), key=lambda row: row["image_id"])
            if not options:
                raise ValueError(f"No v3 {split} candidate for class ID {class_id}")
            selected.append(chooser.choice(options))
    if len({row["image_id"] for row in selected}) != len(selected):
        raise ValueError("Sample contains repeated image IDs")
    return selected


def _mse(image, reference):
    """Calculate mean squared error between equal-shaped numeric image arrays."""
    return float(np.mean((image.astype(np.float32) - reference.astype(np.float32)) ** 2))


def _validate_all_sample_pixels(captures):
    """Check color, layout, range and dimensions for captured real sample images."""
    results = {}
    for split, entries in captures.items():
        checked = 0
        sizes = []
        for entry in entries:
            rgb = to_rgb_array(entry["pixels"])
            gray = rgb_to_grayscale(rgb)
            hsv = rgb_to_hsv(rgb)
            normalized_range = to_float01(rgb)
            layout = chw_to_hwc(hwc_to_chw(rgb))
            if gray.shape != rgb.shape[:2] or hsv.shape != rgb.shape:
                raise ValueError(f"{split}: color-space shape mismatch in real sample")
            if (
                normalized_range.dtype != np.float32
                or normalized_range.min() < 0
                or normalized_range.max() > 1
            ):
                raise ValueError(f"{split}: float pixel range invalid in real sample")
            if not np.array_equal(layout, rgb):
                raise ValueError(f"{split}: HWC/CHW round trip failed in real sample")
            checked += 1
            sizes.append(list(rgb.shape[:2]))
        results[split] = {"images_checked_in_memory": checked, "hw_sizes": sizes}
    return results


def _write_sample_manifest(output, rows, metadata, labels):
    """Write a sample-only copy of the consumer schema outside immutable split folders."""
    directory = output / "sample_manifest"
    directory.mkdir()
    with (directory / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    sample_metadata = {
        "schema_version": metadata["schema_version"],
        "manifest_schema_version": metadata["manifest_schema_version"],
        "dataset_version": metadata["dataset_version"],
        "root": ".",
        "sample_only": True,
        "source_manifest": "data/splits/v3/dataset_manifest.csv",
        "source_v3_fingerprint": metadata["v3_fingerprint"],
        "source_fingerprint": metadata["source_fingerprint"],
        "inventory_sha256": metadata["inventory_sha256"],
        "sampling_seed": SEED,
        "sampling_policy": (
            "seeded choice: one train image per class ID 0-19 and one val image per class ID 0-9"
        ),
        "image_count": len(rows),
        "class_count": CLASS_COUNT,
        "split_counts": dict(Counter(row["split"] for row in rows)),
    }
    (directory / "dataset_metadata.json").write_text(
        json.dumps(sample_metadata, indent=2), encoding="utf-8"
    )
    return directory


def _make_loaders(directory, raw_root):
    """Decode selected images once via ResNet DataLoader and test both profiles in memory."""
    captures = {}
    metrics = {}
    sample_rows = {}
    resnet_metrics = {}
    for split, is_train in (("train", True), ("val", False)):
        capture = _CaptureTransform(build_transform(preprocessing("resnet18"), train=is_train))
        dataset = ManifestDataset(directory, split, transform=capture, root_override=raw_root)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batches = []
        observed_labels = []
        for image_batch, label_batch in loader:
            batches.append(image_batch)
            observed_labels.extend(int(value) for value in label_batch.tolist())
        combined = torch.cat(batches)
        expected_labels = [int(row["class_id"]) for row in dataset.rows]
        if observed_labels != expected_labels:
            raise ValueError(f"ResNet18/{split}: labels do not match sample manifest")
        if combined.dtype != torch.float32 or not torch.isfinite(combined).all():
            raise ValueError(f"ResNet18/{split}: invalid model input tensor")
        result = {
            "samples": len(dataset),
            "batch_shapes": [list(batch.shape) for batch in batches],
            "combined_shape": list(combined.shape),
            "dtype": str(combined.dtype),
            "finite": bool(torch.isfinite(combined).all()),
            "label_dtype": "int64",
            "labels_match_manifest": True,
            "random_augmentation": is_train,
            "real_manifest_dataloader": True,
            "decoded_rgb_sizes": [image["size"] for image in capture.images],
        }
        captures[split] = capture.images
        sample_rows[split] = dataset.rows
        if not is_train:
            repeated = torch.stack(
                [
                    build_transform(preprocessing("resnet18"))(Image.fromarray(image["pixels"]))
                    for image in capture.images
                ]
            )
            result["eval_repeat_max_abs_difference"] = float(
                torch.max(torch.abs(repeated - combined))
            )
            if not torch.equal(repeated, combined):
                raise ValueError("ResNet18/val: eval preprocessing was not deterministic")
        resnet_metrics[split] = result
    metrics["resnet18"] = resnet_metrics
    efficientnet_metrics = {}
    for split, is_train in (("train", True), ("val", False)):
        transform = build_transform(preprocessing("efficientnet_b0"), train=is_train)
        tensors = [transform(Image.fromarray(image["pixels"])) for image in captures[split]]
        combined = torch.stack(tensors)
        batches = [combined[start : start + 4] for start in range(0, len(tensors), 4)]
        result = {
            "samples": len(tensors),
            "batch_shapes": [list(batch.shape) for batch in batches],
            "combined_shape": list(combined.shape),
            "dtype": str(combined.dtype),
            "finite": bool(torch.isfinite(combined).all()),
            "labels_from_manifest": [int(row["class_id"]) for row in sample_rows[split]],
            "label_dtype": "int64 (manifest IDs)",
            "random_augmentation": is_train,
            "real_image_source": "same once-decoded sample retained in memory",
            "real_manifest_dataloader": False,
        }
        if not result["finite"] or combined.dtype != torch.float32:
            raise ValueError(f"EfficientNet-B0/{split}: invalid model input tensor")
        if not is_train:
            repeated = torch.stack(
                [transform(Image.fromarray(image["pixels"])) for image in captures[split]]
            )
            result["eval_repeat_max_abs_difference"] = float(
                torch.max(torch.abs(repeated - combined))
            )
            if not torch.equal(repeated, combined):
                raise ValueError("EfficientNet-B0/val: eval preprocessing was not deterministic")
        efficientnet_metrics[split] = result
    metrics["efficientnet_b0"] = efficientnet_metrics
    return metrics, captures


def _image_ops(pixels, row, output):
    """Exercise color, pixel, geometry, filtering, morphology and edge operations on one sample."""
    rgb = to_rgb_array(pixels)
    bgr = rgb_to_bgr(rgb)
    bgr_back = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = rgb_to_grayscale(rgb)
    hsv = rgb_to_hsv(rgb)
    hsv_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    float_rgb = to_float01(rgb)
    chw = hwc_to_chw(rgb)
    height, width = rgb.shape[:2]
    resized = resize_image(rgb, (max(32, min(height, 256)), max(32, min(width, 256))), "area")
    cropped = center_crop(rgb, (max(1, int(height * 0.85)), max(1, int(width * 0.85))))
    flipped = np.ascontiguousarray(rgb[:, ::-1])
    rotated = rotate_image(rgb, 5, enabled=True)
    brightness = adjust_brightness(rgb, 1.1)
    contrast = adjust_contrast(rgb, 1.1)
    gamma = gamma_correct(rgb, 0.9)
    noise = np.random.default_rng(SEED).normal(0, 12, rgb.shape)
    noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    gaussian = gaussian_blur(noisy, 5, sigma=1.0)
    median = median_filter(noisy, 3)
    mask = gray > np.percentile(gray, 65)
    morphology = {
        "erosion_pixels": int(erosion(mask, 3).sum()),
        "dilation_pixels": int(dilation(mask, 3).sum()),
        "opening_pixels": int(opening(mask, 3).sum()),
        "closing_pixels": int(closing(mask, 3).sum()),
    }
    edges = canny_edges(rgb, 50, 120)
    regions = extract_regions(mask, max(4, int(height * width * 0.001)))
    panels = [
        (rgb, "RGB"),
        (gray, "Grayscale"),
        (hsv, "HSV"),
        (brightness, "Brightness 1.1"),
        (contrast, "Contrast 1.1"),
        (gamma, "Gamma 0.9"),
        (resized, "Resize"),
        (cropped, "Center crop"),
        (flipped, "Horizontal flip"),
        (rotated, "Rotation 5° opt-in"),
        (gaussian, f"Gaussian blur MSE {_mse(gaussian, rgb):.1f}"),
        (median, f"Median MSE {_mse(median, rgb):.1f}"),
    ]
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output / "real_v3_processing.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (mask, edges, cv2.Sobel(gray, cv2.CV_32F, 1, 0)),
        ("Intensity mask (not GT)", "Canny", "Sobel"),
    ):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output / "real_v3_edges_regions.png", dpi=140)
    plt.close(figure)
    return {
        "image_id": row["image_id"],
        "split": row["split"],
        "class": row["class_key"],
        "source_shape_hwc": list(rgb.shape),
        "source_dtype": str(rgb.dtype),
        "gray_shape": list(gray.shape),
        "hsv_shape": list(hsv.shape),
        "float_dtype": str(float_rgb.dtype),
        "float_range": [float(float_rgb.min()), float(float_rgb.max())],
        "bgr_roundtrip_equal": bool(np.array_equal(rgb, bgr_back)),
        "hsv_roundtrip_max_abs_error": int(
            np.abs(hsv_back.astype(np.int16) - rgb.astype(np.int16)).max()
        ),
        "chw_shape": list(chw.shape),
        "layout_roundtrip_equal": bool(np.array_equal(chw_to_hwc(chw), rgb)),
        "resize_shape": list(resized.shape),
        "crop_shape": list(cropped.shape),
        "flip_shape": list(flipped.shape),
        "rotation_shape": list(rotated.shape),
        "brightness_range": [int(brightness.min()), int(brightness.max())],
        "contrast_range": [int(contrast.min()), int(contrast.max())],
        "gamma_range": [int(gamma.min()), int(gamma.max())],
        "noise_mse_vs_source": _mse(noisy, rgb),
        "gaussian_mse_vs_source": _mse(gaussian, rgb),
        "median_mse_vs_source": _mse(median, rgb),
        "morphology": morphology,
        "edge_pixel_count": int(np.count_nonzero(edges)),
        "region_count": len(regions),
        "region_examples": regions[:20],
        "mask_caveat": (
            "Intensity-percentile threshold demonstration; not ground-truth food segmentation."
        ),
    }


def _near_duplicate(pixels, row, output):
    """Compare one real sample with in-memory derivatives; never compare distinct v3 rows."""
    original = to_rgb_array(pixels)
    height, width = original.shape[:2]
    resized = cv2.resize(original, (max(32, width * 3 // 4), max(32, height * 3 // 4)))
    buffer = io.BytesIO()
    Image.fromarray(original).save(buffer, format="JPEG", quality=82)
    jpeg_data = buffer.getvalue()
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"))
    bright = adjust_brightness(original, 1.08)
    crop = original[
        max(0, height // 20) : max(1, height - height // 20),
        max(0, width // 20) : max(1, width - width // 20),
    ]
    flipped = np.ascontiguousarray(original[:, ::-1])
    variants = {
        "resize": resized,
        "jpeg_recompression": recompressed,
        "brightness": bright,
        "small_crop": crop,
        "horizontal_flip": flipped,
    }
    policy = SimilarityPolicy(hash_size=8, max_hamming_distance=12, min_ssim=0.72)
    results = {}
    for name, variant in variants.items():
        variant_buffer = io.BytesIO()
        Image.fromarray(variant).save(variant_buffer, format="JPEG", quality=82)
        results[name] = {
            "source_manifest_sha256": row["checksum"],
            "derived_variant_sha256": hashlib.sha256(variant_buffer.getvalue()).hexdigest(),
            **compare_pair(original, variant, policy),
        }
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    display = [("source", original), *variants.items()]
    for axis, (name, image) in zip(axes.flat, display):
        evidence = results.get(name)
        caption = (
            "source"
            if evidence is None
            else f"H={evidence['hamming_distance']} / SSIM={evidence['ssim']:.3f}"
        )
        axis.imshow(image)
        axis.set_title(f"{name}: {caption}")
        axis.axis("off")
    figure.savefig(output / "real_v3_near_duplicate.png", dpi=140)
    plt.close(figure)
    return {
        "source_image_id": row["image_id"],
        "source_split": row["split"],
        "policy": {
            "hash_size": policy.hash_size,
            "max_hamming_distance": policy.max_hamming_distance,
            "min_ssim": policy.min_ssim,
        },
        "source_serialized_sha256": hashlib.sha256(jpeg_data).hexdigest(),
        "variants": results,
        "scope": (
            "one selected real v3 image and in-memory derivatives only; not a near-duplicate audit"
        ),
    }


def main():
    """Run the sample-bounded v3 smoke test and write evidence to a new report directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/elt/phase_4_5_real_v3_smoke")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    repo = Path.cwd().resolve()
    config = yaml.safe_load((repo / "configs/food101_elt_full_v3.yaml").read_text(encoding="utf-8"))
    split_root = repo / config["paths"]["manifest_root"]
    raw_root = repo / config["paths"]["raw_root"]
    metadata = read_json(split_root / "dataset_metadata.json")
    labels = read_json(split_root / "labels.json")
    quality = read_json(split_root / "quality_report.json")
    if (
        metadata.get("schema_version") != "3.0"
        or quality.get("quality_gates", {}).get("overall") != "PASS"
    ):
        raise ValueError("v3 metadata/quality gate does not match the accepted baseline")
    rows = _select_rows(split_root / "dataset_manifest.csv")
    (args.output / "sampling_metadata.json").write_text(
        json.dumps(
            {
                "source_manifest": "data/splits/v3/dataset_manifest.csv",
                "source_v3_fingerprint": metadata["v3_fingerprint"],
                "seed": SEED,
                "selection_policy": (
                    "seeded choice from sorted IDs; one train image for class IDs 0-19 "
                    "and one val image for 0-9"
                ),
                "unique_image_count": len(rows),
                "split_counts": dict(Counter(row["split"] for row in rows)),
                "class_coverage": sorted({row["class_key"] for row in rows}),
                "samples": [
                    {
                        "sample_id": f"{row['split']}-{row['class_id']}",
                        "image_id": row["image_id"],
                        "split": row["split"],
                        "class_name": row["class_key"],
                        "class_id": int(row["class_id"]),
                        "relative_path": row["relative_path"],
                        "sampling_seed": SEED,
                    }
                    for row in rows
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for row in rows:
        path = (raw_root / row["relative_path"]).resolve()
        if not path.is_relative_to(raw_root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"Selected v3 sample is missing or unsafe: {row['image_id']}")
    sample_dir = _write_sample_manifest(args.output, rows, metadata, labels)
    torch.manual_seed(SEED)
    loader_metrics, captured = _make_loaders(sample_dir, raw_root)
    sample_pixel_validation = _validate_all_sample_pixels(captured)
    representative = next(row for row in rows if row["split"] == "train")
    train_rows = [row for row in rows if row["split"] == "train"]
    representative_index = next(
        index
        for index, row in enumerate(train_rows)
        if row["image_id"] == representative["image_id"]
    )
    operations = _image_ops(
        captured["train"][representative_index]["pixels"], representative, args.output
    )
    duplicate = _near_duplicate(
        captured["train"][representative_index]["pixels"], representative, args.output
    )
    metrics = {
        "image_operations": operations,
        "near_duplicate_demo": duplicate,
        "dataloader": loader_metrics,
        "all_sample_pixel_validation": sample_pixel_validation,
    }
    (args.output / "smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": len(rows),
                "train": TRAIN_COUNT,
                "validation": VAL_COUNT,
                "class_coverage": len({row["class_key"] for row in rows}),
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
