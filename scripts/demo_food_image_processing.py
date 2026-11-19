"""Render reproducible synthetic image-processing and near-copy research demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from food_assistant.ml.image_ops import (
    adjust_brightness,
    canny_edges,
    closing,
    gaussian_blur,
    median_filter,
    opening,
    resize_image,
    rgb_to_grayscale,
    rgb_to_hsv,
    threshold_mask,
)
from food_assistant.ml.near_duplicate import SimilarityPolicy, compare_pair
from food_assistant.ml.transforms import build_transform, preprocessing


def _synthetic_food(seed: int, size: int = 192) -> np.ndarray:
    """Create a bounded food-like synthetic scene for reproducible demonstrations."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 45 + 0.45 * x + 18 * np.sin(y / 11)
    base[..., 1] = 70 + 0.35 * y + 12 * np.cos(x / 13)
    base[..., 2] = 35 + 0.15 * (x + y)
    center = (size // 2, size // 2)
    radius = (x - center[0]) ** 2 + (y - center[1]) ** 2
    plate = radius < (size * 0.39) ** 2
    base[plate] = np.stack(
        (184 + 10 * np.sin(x[plate] / 17), 170 + 8 * np.cos(y[plate] / 15), 143 + x[plate] * 0),
        axis=-1,
    )
    food = radius < (size * 0.26) ** 2
    texture = rng.normal(0, 14, (food.sum(), 3))
    base[food] = np.clip(np.array([177, 91, 38]) + texture, 0, 255)
    for center_x, center_y in ((75, 85), (120, 105), (95, 130)):
        nugget = (x - center_x) ** 2 + (y - center_y) ** 2 < 15**2
        base[nugget] = np.array([211, 155, 73]) + rng.normal(0, 8, (nugget.sum(), 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def _write_processing_panel(image: np.ndarray, output: Path, seed: int) -> dict[str, float]:
    """Render representation, pixel, geometry, filtering, morphology and edge examples."""
    rng = np.random.default_rng(seed)
    gaussian_noisy = np.clip(
        image.astype(np.float32) + rng.normal(0, 18, image.shape), 0, 255
    ).astype(np.uint8)
    salt_pepper_noisy = image.copy()
    points = rng.integers(0, image.shape[0], (900, 2))
    salt_pepper_noisy[points[:, 0], points[:, 1]] = rng.choice((0, 255), (len(points), 3)).astype(
        np.uint8
    )
    gaussian = gaussian_blur(gaussian_noisy, 5, sigma=1.2)
    median = median_filter(salt_pepper_noisy, 3)
    reference_float = image.astype(np.float32)
    mse_metrics = {
        "gaussian_noise_mse": float(
            np.mean((gaussian_noisy.astype(np.float32) - reference_float) ** 2)
        ),
        "gaussian_filtered_mse": float(
            np.mean((gaussian.astype(np.float32) - reference_float) ** 2)
        ),
        "salt_pepper_mse": float(
            np.mean((salt_pepper_noisy.astype(np.float32) - reference_float) ** 2)
        ),
        "median_filtered_mse": float(np.mean((median.astype(np.float32) - reference_float) ** 2)),
    }
    mask = threshold_mask(image, 130)
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    panels = [
        (image, "RGB HWC uint8"),
        (adjust_brightness(image, 1.2), "Brightness x1.2"),
        (resize_image(image, (128, 128)), "Resize 128×128"),
        (image[:, ::-1], "Horizontal flip"),
        (gaussian_noisy, f"Gaussian noise, MSE={mse_metrics['gaussian_noise_mse']:.1f}"),
        (gaussian, f"Gaussian, MSE={mse_metrics['gaussian_filtered_mse']:.1f}"),
        (salt_pepper_noisy, f"Salt/pepper, MSE={mse_metrics['salt_pepper_mse']:.1f}"),
        (median, f"Median, MSE={mse_metrics['median_filtered_mse']:.1f}"),
        (canny_edges(image, 45, 110), "Canny edges"),
    ]
    for axis, (pixels, title) in zip(axes.flat, panels):
        axis.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("RGB source")
    axes[1].imshow(rgb_to_grayscale(image), cmap="gray")
    axes[1].set_title("Grayscale")
    axes[2].imshow(rgb_to_hsv(image))
    axes[2].set_title("OpenCV HSV channels")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output.with_name("food_image_color_spaces.png"), dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    pixel_variants = [
        (image, "Original"),
        (adjust_brightness(image, 1.2), "Brightness ×1.2"),
        (
            np.clip(image.astype(np.float32) * 1.4 - 40, 0, 255).astype(np.uint8),
            "Contrast demo ×1.4",
        ),
        (np.rint(255 * (image.astype(np.float32) / 255) ** 0.7).astype(np.uint8), "Gamma 0.7"),
    ]
    for axis, (pixels, title) in zip(axes, pixel_variants):
        axis.imshow(pixels)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output.with_name("food_image_pixel_ops.png"), dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    geometry_variants = [
        (image, "Source"),
        (resize_image(image, (128, 128)), "Resize 128×128"),
        (image[24:-24, 24:-24], "Center crop 144×144"),
    ]
    for axis, (pixels, title) in zip(axes, geometry_variants):
        axis.imshow(pixels)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output.with_name("food_image_geometry.png"), dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Threshold mask")
    axes[1].imshow(opening(mask, 5), cmap="gray")
    axes[1].set_title("Morphological opening")
    axes[2].imshow(closing(mask, 5), cmap="gray")
    axes[2].set_title("Morphological closing")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output.with_name("food_image_morphology.png"), dpi=150)
    plt.close(figure)
    return mse_metrics


def _near_duplicate_metrics(
    image: np.ndarray, policy: SimilarityPolicy, output: Path
) -> dict[str, object]:
    """Measure a small synthetic labeled pair set and calculate confusion statistics."""
    resized = np.asarray(Image.fromarray(image).resize((140, 140), Image.Resampling.LANCZOS))
    crop = image[5:-5, 5:-5]
    flipped = np.ascontiguousarray(image[:, ::-1])
    brightened = np.clip(image.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
    encoded = Image.fromarray(image)
    import io

    buffer = io.BytesIO()
    encoded.save(buffer, format="JPEG", quality=82)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"))
    palette_similar = image[..., ::-1].copy()
    random_image = np.random.default_rng(993).integers(0, 256, image.shape, dtype=np.uint8)
    fixtures = [
        ("exact copy", image, True),
        ("resize", resized, True),
        ("JPEG recompression", recompressed, True),
        ("brightness change", brightened, True),
        ("slight crop", crop, True),
        ("horizontal flip", flipped, True),
        ("same layout, different channel palette", palette_similar, False),
        ("unrelated texture", random_image, False),
    ]
    records = []
    expected = []
    predicted = []
    for fixture_name, candidate, is_copy in fixtures:
        evidence = compare_pair(image, candidate, policy)
        expected.append(int(is_copy))
        predicted.append(int(evidence["suspicious"]))
        source_digest = hashlib.sha256(image.tobytes()).hexdigest()
        candidate_digest = hashlib.sha256(candidate.tobytes()).hexdigest()
        records.append(
            {
                "fixture": fixture_name,
                "expected_copy": is_copy,
                "sha256_same": source_digest == candidate_digest,
                **evidence,
            }
        )
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    display_pairs = [
        ("reference", image),
        ("resized", resized),
        ("JPEG", recompressed),
        ("brightened", brightened),
        ("crop", crop),
        ("flip", flipped),
        ("palette changed", palette_similar),
        ("unrelated", random_image),
    ]
    for axis, (fixture_name, candidate) in zip(axes.flat, display_pairs):
        axis.imshow(candidate)
        axis.set_title(fixture_name)
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.savefig(output / "near_duplicate_fixtures.png", dpi=150)
    plt.close(figure)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        expected, predicted, labels=[0, 1]
    ).ravel()
    return {
        "policy": {
            "hash_size": policy.hash_size,
            "max_hamming_distance": policy.max_hamming_distance,
            "min_ssim": policy.min_ssim,
        },
        "pairs": records,
        "confusion": {
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "true_negative": int(true_negative),
            "false_negative": int(false_negative),
            "precision": float(precision_score(expected, predicted, zero_division=0)),
            "recall": float(recall_score(expected, predicted, zero_division=0)),
        },
        "scope": "synthetic fixture only; not a full-v3 scan or validated food-label policy",
    }


def _write_train_augmentation_panel(image: np.ndarray, output: Path, seed: int) -> None:
    """Show a sampled train-only crop/flip without changing production transform settings."""
    torch.manual_seed(seed)
    transform = build_transform(preprocessing("resnet18"), train=True)
    tensor = transform(Image.fromarray(image))
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
    augmented = np.clip(tensor.numpy() * std + mean, 0, 1).transpose(1, 2, 0)
    figure, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Synthetic source")
    axes[1].imshow(augmented)
    axes[1].set_title("Existing train transform sample")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output / "train_augmentation.png", dpi=150)
    plt.close(figure)


def main() -> int:
    """Write synthetic demonstrations and quantitative metrics to a new output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/elt/phase_4_5_demos")
    )
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing demo output: {arguments.output}")
    arguments.output.mkdir(parents=True)
    image = _synthetic_food(arguments.seed)
    pixel_metrics = _write_processing_panel(
        image, arguments.output / "food_image_processing.png", arguments.seed + 1
    )
    duplicate_metrics = _near_duplicate_metrics(
        image, SimilarityPolicy(8, 12, 0.72), arguments.output
    )
    _write_train_augmentation_panel(image, arguments.output, arguments.seed + 2)
    metrics = {
        "seed": arguments.seed,
        "pixel_metrics": pixel_metrics,
        "near_duplicate_metrics": duplicate_metrics,
        "provenance": "synthetic data generated locally; no Food-101 files read",
    }
    (arguments.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(arguments.output), **metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
