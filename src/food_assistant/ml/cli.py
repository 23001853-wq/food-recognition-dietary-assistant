"""Dataset verification, preparation and frozen-candidate evaluation commands."""

import argparse
import csv
import hashlib
import platform
import time
from pathlib import Path

import yaml
from PIL import Image

from food_assistant.data_pipeline.atomic import resource_lock
from food_assistant.ml.datasets import (
    ManifestDataset,
    audit_manifest,
    prepare_food101,
    read_json,
    write_json,
)
from food_assistant.services.model_registry import Release


def create_splits():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/food-101"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes_poc.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(args.classes.read_text(encoding="utf-8"))
    print(prepare_food101(args.root, args.output, config["classes"], config["version"], args.seed))


def verify_dataset():
    """Audit an immutable manifest and write its report outside the manifest directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/reports/food101/manifest_quality_report.json")
    )
    args = parser.parse_args()
    metadata = read_json(args.manifest_dir / "dataset_metadata.json")
    with (args.manifest_dir / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stats = audit_manifest(rows, metadata["root"])
    for row in rows:
        with Image.open(Path(metadata["root"]) / row["relative_path"]) as image:
            image.verify()
    with resource_lock(args.report, timeout_seconds=0):
        write_json(args.report, stats)
    print(stats)


def evaluate_models():
    import torch

    from food_assistant.ml.metrics import export_metrics
    from food_assistant.ml.train_utils import seal_artifact

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    args = parser.parse_args()
    is_full_v3_test = (
        args.split == "test"
        and not args.smoke
        and read_json(args.artifact / "model_config.json").get("dataset_version") == "v3"
    )
    final_test_attempt = None
    if is_full_v3_test:
        if not args.freeze_record or not args.protocol or not args.evidence_output:
            parser.error(
                "Full-v3 final test requires --freeze-record, --protocol and "
                "separate --evidence-output"
            )
        from food_assistant.ml.full_v3_release import claim_final_test_attempt

        if args.evidence_output.exists():
            parser.error("Final test evidence target exists; refusing repeated evaluation")
        final_test_attempt = claim_final_test_attempt(
            args.freeze_record,
            args.evidence_output,
            args.artifact,
            args.protocol,
        )
    torch.set_num_threads(2)
    release = Release(args.artifact, args.device)
    frozen = None
    if args.protocol:
        if release.config.get("dataset_version") == "v3":
            from food_assistant.ml.full_v3_release import verify_protocol

            verify_protocol(args.protocol)
        else:
            from food_assistant.ml.protocol import verify_frozen_candidate

            frozen = verify_frozen_candidate(args.protocol, args.artifact, args.device)
    elif not args.smoke and release.config.get("dataset_version") == "poc-v1":
        parser.error("Freeze all three POC candidates first and provide --protocol")
    data = ManifestDataset(args.manifest_dir, args.split, limit_per_class=2 if args.smoke else None)
    fingerprint = hashlib.sha256(
        (args.manifest_dir / "dataset_manifest.csv").read_bytes()
    ).hexdigest()
    if fingerprint != release.config["manifest_sha256"] or data.labels != release.labels:
        raise ValueError("Evaluation manifest or labels differ from training")
    if frozen and (
        frozen["manifest_sha256"] != fingerprint
        or len(data) != frozen["test_samples"]
        or args.split != "test"
        or args.smoke
    ):
        raise ValueError("Evaluation split/sample count differs from frozen protocol")
    output = (
        args.evidence_output
        if is_full_v3_test
        else args.artifact / ("evaluation_smoke" if args.smoke else f"evaluation_{args.split}")
    )
    if (output / "metrics.json").exists():
        raise ValueError("Evaluation already exists; refusing repeated test evaluation")
    if is_full_v3_test:
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / "evaluation_attempt.json",
            {"status": "STARTED", "freeze_record": str(args.freeze_record.resolve())},
        )
    truth, predictions, timings, details = [], [], [], []
    top5_correct = 0
    # Warm up using validation image; test images never tune any hyperparameters.
    warmup = ManifestDataset(args.manifest_dir, "val", limit_per_class=1)
    with Image.open(warmup.root / warmup.rows[0]["relative_path"]) as image:
        for _ in range(3):
            release.predict(image.convert("RGB"), 3)
    for row in data.rows:
        with Image.open(data.root / row["relative_path"]) as image:
            start = time.perf_counter()
            result = release.predict(image.convert("RGB"), 5)
            timings.append((time.perf_counter() - start) * 1000)
        truth.append(int(row["class_id"]))
        predictions.append(result[0].class_id)
        top5_correct += int(any(item.class_id == truth[-1] for item in result))
        details.append(
            {
                "image_path": row["relative_path"],
                "true_class": row["class_key"],
                "predicted_class": result[0].class_key,
                "score": result[0].score,
                "error_group": "UNREVIEWED"
                if result[0].class_id != int(row["class_id"])
                else "CORRECT",
            }
        )
    metrics = export_metrics(output, truth, predictions, data.labels["labels"], timings)
    metrics["top5_accuracy"] = top5_correct / len(truth)
    metrics.update(
        model_name=release.config["model_name"],
        run_id=release.config["run_id"],
        dataset_version=release.config["dataset_version"],
        manifest_sha256=fingerprint,
        split=args.split,
        smoke_only=args.smoke or release.config["smoke_only"],
        artifact_size_mb=sum(p.stat().st_size for p in args.artifact.glob("model.*")) / 2**20,
        device=str(release.device),
        batch_size=1,
        warmup_iterations=3,
        latency_scope="image preprocessing plus inference; excludes file decode",
        num_classes=len(data.labels["labels"]),
        labels_sha256=hashlib.sha256(
            (args.manifest_dir / "labels.json").read_bytes()
        ).hexdigest(),
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        if args.protocol
        else None,
        hardware={
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", details)
    write_json(
        output / "evaluation_metadata.json",
        {
            **metrics,
            "labels": data.labels,
            "preprocessing": release.preproc,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )
    if not args.smoke and not is_full_v3_test:
        write_json(args.artifact / f"metrics_{args.split}.json", metrics)
        seal_artifact(args.artifact)
    if is_full_v3_test:
        metrics["frozen_candidate_id"] = read_json(final_test_attempt)["candidate_run_id"]
        metrics["freeze_record_sha256"] = hashlib.sha256(
            args.freeze_record.read_bytes()
        ).hexdigest()
        write_json(output / "final_test_evidence.json", metrics)
    write_json(output / "evaluation_complete.json", {"complete": True, "sample_count": len(data)})
    if is_full_v3_test:
        from food_assistant.ml.full_v3_release import complete_final_test_attempt

        complete_final_test_attempt(final_test_attempt, output, args.freeze_record)
    print(metrics)


def analyze_failures():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in read_json(args.predictions) if r["error_group"] != "CORRECT"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_class", "predicted_class", "score", "error_group"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Exported {len(rows)} failures for manual taxonomy review; no cause inferred automatically"
    )


def select_production():
    from food_assistant.ml.release_selection import rank_candidates

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("artifacts/reports/validation_selection.csv"),
    )
    args = parser.parse_args()
    if len(args.artifacts) < 2:
        parser.error("At least two independently trained methods are required")
    candidates = []
    for directory in args.artifacts:
        config = read_json(directory / "model_config.json")
        val = read_json(directory / "metrics_val.json")
        labels = read_json(directory / "labels.json")["labels"]
        checksums_path = directory / "checksum.sha256"
        if not checksums_path.is_file():
            raise ValueError(f"Candidate has no sealed checksum inventory: {directory}")
        inventory = read_json(checksums_path)
        model_file = directory / (
            "model.joblib" if config.get("model_name") == "hog_svm" else "model.pth"
        )
        essential_files = (
            model_file,
            directory / "model_config.json",
            directory / "labels.json",
            directory / "preprocessing.json",
            directory / "metrics_val.json",
        )
        if any(not path.is_file() or path.name not in inventory for path in essential_files):
            raise ValueError("Candidate model checkpoint is missing from its seal")
        for path in essential_files:
            with path.open("rb") as handle:
                actual_digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual_digest != inventory[path.name]:
                raise ValueError(f"Candidate evidence checksum mismatch: {path.name}")
            if path == model_file:
                checksum = actual_digest
        artifact_size = model_file.stat().st_size
        candidates.append(
            {
                "config": config,
                "val": val,
                "labels": labels,
                "artifact": directory,
                "artifact_checksum": checksum,
                "artifact_size_bytes": artifact_size,
            }
        )
    ranked = rank_candidates(candidates)
    chosen = ranked[0]
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_name",
        "run_id",
        "dataset_version",
        "num_classes",
        "sample_count",
        "validation_macro_f1",
        "top1_accuracy",
        "validation_top1_accuracy",
        "validation_p95_ms",
        "artifact_size_bytes",
        "artifact_checksum",
    ]
    with args.comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            row = {
                **candidate["config"],
                "sample_count": candidate["val"]["sample_count"],
                "top1_accuracy": candidate["val"]["top1_accuracy"],
                "validation_macro_f1": candidate["val"]["macro_f1"],
                "validation_top1_accuracy": candidate["val"]["top1_accuracy"],
                "validation_p95_ms": candidate["val"]["p95_ms"],
                "artifact_size_bytes": candidate["artifact_size_bytes"],
                "artifact_checksum": candidate["artifact_checksum"],
            }
            writer.writerow({key: row.get(key) for key in columns})
    print(
        f"Validation selection only: {chosen['config']['model_name']}/"
        f"{chosen['config']['run_id']}; no test data read and no production pointer changed."
    )
