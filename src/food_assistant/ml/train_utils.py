"""Offline training and artifact exports. Smoke runs are explicitly ineligible for production."""

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from food_assistant.config import PROJECT_ROOT
from food_assistant.ml.datasets import ManifestDataset, write_json
from food_assistant.ml.hog_feature_store import build_feature_store, open_feature_store
from food_assistant.ml.hog_features import extract_hog
from food_assistant.ml.metrics import compute_metrics, top_k_accuracy_from_scores
from food_assistant.ml.model_factory import create_model
from food_assistant.ml.transforms import build_transform, preprocessing
from food_assistant.utils.process_lock import training_lock


def device_for(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; explicitly choose CPU")
    return torch.device(name)


def accumulation_window(index, batches, accumulation):
    """Use the actual last window size so a partial accumulation is not underweighted."""
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    start = (index // accumulation) * accumulation
    size = min(accumulation, batches - start)
    return size, (index + 1) % accumulation == 0 or index + 1 == batches


def estimate_hog_peak_bytes(train_rows, validation_rows, feature_count=8100):
    """Estimate a conservative HOG fit peak including scaler and estimator copies.

    This is a planning gate, not a measured peak: it budgets one float32 and two
    float64 train representations, one float32 and one float64 validation
    representation, plus 1 GiB for Python, OS and estimator overhead.
    """
    if min(train_rows, validation_rows, feature_count) < 1:
        raise ValueError("HOG memory estimate dimensions must be positive")
    train_bytes = train_rows * feature_count * (4 + 8 + 8)
    validation_bytes = validation_rows * feature_count * (4 + 8)
    return train_bytes + validation_bytes + 1024**3


def hog_memory_preflight(evidence):
    """Compare the conservative HOG estimator estimate with current free RAM."""
    import psutil

    required = estimate_hog_peak_bytes(
        evidence["split_counts"]["train"], evidence["split_counts"]["val"]
    )
    available = psutil.virtual_memory().available
    return {
        "required_bytes": required,
        "available_bytes": available,
        "ready": available >= required,
    }


def seal_artifact(output):
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(output).iterdir())
        if p.is_file() and p.name != "checksum.sha256"
    }
    write_json(Path(output) / "checksum.sha256", checksums)


def export_contract(output, name, run_id, dataset, config, smoke, preproc, pretrained):
    output.mkdir(parents=True, exist_ok=False)
    labels = dataset.labels
    write_json(output / "labels.json", labels)
    mapping = {row["class_key"]: row["class_id"] for row in labels["labels"]}
    write_json(output / "class_to_idx.json", mapping)
    write_json(output / "idx_to_class.json", {str(v): k for k, v in mapping.items()})
    manifest = Path(config["manifest_dir"]) / "dataset_manifest.csv"
    labels_digest = hashlib.sha256(
        (Path(config["manifest_dir"]) / "labels.json").read_bytes()
    ).hexdigest()
    write_json(
        output / "model_config.json",
        {
            "model_name": name,
            "run_id": run_id,
            "num_classes": len(mapping),
            "dataset_version": labels["dataset_version"],
            "smoke_only": smoke or dataset.metadata.get("is_synthetic", False),
            "pretrained": pretrained,
            "trained": False,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "labels_sha256": labels_digest,
            "source_fingerprint": dataset.metadata.get("source_fingerprint"),
            "inventory_sha256": dataset.metadata.get("inventory_sha256"),
            "v3_fingerprint": dataset.metadata.get("v3_fingerprint"),
            "dataset_root": dataset.metadata["root"],
            "source": dataset.metadata.get("source"),
            "torch_version": torch.__version__,
            "device": "cpu" if name == "hog_svm" else str(device_for(config.get("device", "auto"))),
        },
    )
    write_json(output / "preprocessing.json", preproc)
    (output / "train_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    (output / "source_commit.txt").write_text(
        commit.stdout.strip() or "unavailable", encoding="utf-8"
    )
    changed = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    sources = [
        *PROJECT_ROOT.joinpath("src").rglob("*.py"),
        *PROJECT_ROOT.joinpath("configs").rglob("*.yaml"),
        PROJECT_ROOT / "pyproject.toml",
    ]
    write_json(
        output / "source_state.json",
        {
            "working_tree_dirty": bool(changed.stdout.strip()),
            "file_sha256": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sources
            },
        },
    )


def run_training(name):
    parser = argparse.ArgumentParser(description=f"Train {name} offline")
    parser.add_argument("--config", type=Path, default=Path(f"configs/{name}.yaml"))
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/manifest and model shape without locks, artifacts or training",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Architecture smoke only; not transfer learning",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.resume and name == "hog_svm":
        parser.error("HOG/SVM has no mid-fit resume; use a fresh run after the active fit finishes")
    if args.dry_run and args.resume:
        parser.error(
            "--dry-run validates a fresh run; validate resume with checkpoint contract tests"
        )
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,69}", args.run_id):
        parser.error("run-id must be a safe identifier of at most 70 characters")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.manifest_dir:
        config["manifest_dir"] = str(args.manifest_dir)
    else:
        config.setdefault("manifest_dir", "data/splits/v1")
    if args.device:
        config["device"] = args.device
    _validate_training_binding(name, config)
    if args.no_pretrained and not args.smoke:
        parser.error("--no-pretrained is allowed only with --smoke")
    pretrained = config.get("pretrained", not args.no_pretrained and name != "hog_svm")
    if not isinstance(pretrained, bool):
        parser.error("config pretrained must be a boolean")
    if name == "hog_svm" and pretrained:
        parser.error("HOG-SVM does not support pretrained initialization")
    if args.dry_run:
        evidence = _validate_training_dry_run(name, config)
        print(
            "DRY_RUN PASS: "
            f"model={name}, dataset={evidence['dataset_version']}, "
            f"classes={evidence['class_count']}, manifest_sha256={evidence['manifest_sha256']}, "
            f"output_shape={evidence['output_shape']}"
        )
        if name == "hog_svm":
            memory = hog_memory_preflight(evidence)
            state = "READY" if memory["ready"] else "BLOCKED"
            print(
                f"HOG_MEMORY_{state}: estimated_peak_gib="
                f"{memory['required_bytes'] / 1024**3:.2f}, "
                f"available_gib={memory['available_bytes'] / 1024**3:.2f}"
            )
        return
    if name == "hog_svm" and not args.smoke:
        evidence = _validate_training_dry_run(name, config)
        memory = hog_memory_preflight(evidence)
        if not memory["ready"]:
            raise RuntimeError(
                "HOG fit memory gate blocked before image decoding: estimated conservative "
                f"peak={memory['required_bytes'] / 1024**3:.2f} GiB, "
                f"currently available={memory['available_bytes'] / 1024**3:.2f} GiB"
            )
    with training_lock(name, args.output_root):
        started = time.perf_counter()
        torch.hub.set_dir(str(PROJECT_ROOT / "artifacts/pretrained"))
        for key in ("head_epochs", "fine_tune_epochs", "batch_size"):
            value = getattr(args, key)
            if value is not None:
                if value < 1:
                    parser.error(f"{key} must be positive")
                config[key] = value
        if name != "hog_svm":
            device_for(config.get("device", "auto"))
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,69}", run_id):
            parser.error("run-id must be a safe identifier of at most 70 characters")
        output = args.output_root / name / run_id
        if output.exists():
            parser.error(f"Run folder already exists: {output}. Keep it and choose a new run-id.")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(2)
        limit = 2 if args.smoke else None
        preproc = config if name == "hog_svm" else preprocessing(name)
        train = ManifestDataset(
            args.manifest_dir,
            "train",
            None if name == "hog_svm" else build_transform(preproc, True),
            limit,
        )
        val = ManifestDataset(
            args.manifest_dir, "val", None if name == "hog_svm" else build_transform(preproc), limit
        )
        if config.get("expected_num_classes") is not None:
            if len(train.labels["labels"]) != config["expected_num_classes"]:
                raise ValueError("Unexpected class count for this experiment")
        if config.get("expected_dataset_version") is not None:
            if train.labels["dataset_version"] != config["expected_dataset_version"]:
                raise ValueError("Unexpected dataset version for this experiment")
        if args.resume:
            previous = read_resume_contract(args.resume, name, train, preproc, config)
        else:
            previous = None
        export_contract(
            output,
            name,
            run_id,
            train,
            config,
            args.smoke,
            preproc,
            previous["pretrained"] if previous else pretrained,
        )
        history = []
        if name == "hog_svm":
            cache_root = args.output_root.resolve().parent / "feature_cache/hog_svm" / run_id
            train_store = build_feature_store(
                train, config, cache_root / "train", manifest_dir=args.manifest_dir, split="train"
            )
            val_store = build_feature_store(
                val, config, cache_root / "val", manifest_dir=args.manifest_dir, split="val"
            )
            train_x, train_y = open_feature_store(train_store)
            val_x, val_y = open_feature_store(val_store)
            print(
                "HOG features are disk-backed float32 memmaps; LinearSVC still requires an "
                "in-memory fit and may allocate estimator copies.",
                flush=True,
            )
            best_score, best_model = -1, None
            for c in config["c_candidates"]:
                model = make_pipeline(
                    StandardScaler(),
                    LinearSVC(
                        C=c,
                        random_state=config["seed"],
                        dual=config.get("dual", False),
                        max_iter=config.get("max_iter", 2000),
                        tol=config.get("tol", 0.001),
                    ),
                )
                print(f"Fitting LinearSVC C={c}, dual={config.get('dual', False)}", flush=True)
                fit_started = time.perf_counter()
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_x, train_y)
                for warning in fit_warnings:
                    print(f"SVM warning: {warning.message}", flush=True)
                metric = compute_metrics(val_y, model.predict(val_x), len(train.labels["labels"]))
                history.append(
                    {
                        "epoch": len(history),
                        "trial_C": c,
                        "train_loss": "",
                        "val_macro_f1": metric["macro_f1"],
                        "fit_seconds": time.perf_counter() - fit_started,
                        "iterations": int(model[-1].n_iter_),
                        "converged": not any(
                            issubclass(w.category, ConvergenceWarning) for w in fit_warnings
                        ),
                    }
                )
                print(f"C={c}: validation macro-F1={metric['macro_f1']:.4f}", flush=True)
                if metric["macro_f1"] > best_score:
                    best_score, best_model = metric["macro_f1"], model
                    joblib.dump(best_model, output / "best_candidate.joblib")
                write_json(output / "svm_trials.json", history)
            joblib.dump(best_model, output / "model.joblib")
            best_metric = compute_metrics(
                val_y, best_model.predict(val_x), len(train.labels["labels"])
            )
            best_metric["top5_accuracy"] = top_k_accuracy_from_scores(
                val_y,
                best_model.decision_function(val_x),
                min(5, len(train.labels["labels"])),
            )
        else:
            device = device_for(config.get("device", "auto"))
            use_amp = device.type == "cuda" and config.get("amp", False)
            accumulation = config.get("gradient_accumulation_steps", 1)
            if not isinstance(accumulation, int) or accumulation < 1:
                raise ValueError("gradient_accumulation_steps must be a positive integer")
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = create_model(
                name,
                len(train.labels["labels"]),
                pretrained=pretrained and args.resume is None,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["head_lr"], weight_decay=config["weight_decay"]
            )
            loader = DataLoader(
                train,
                batch_size=2 if args.smoke else config["batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            validation = DataLoader(
                val,
                batch_size=2 if args.smoke else config["batch_size"],
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            start_epoch, best_score, stale = 0, -1, 0
            best_metric, best_state = None, None
            if args.resume:
                resume = torch.load(args.resume, map_location=device, weights_only=True)
                if (
                    resume["manifest_sha256"]
                    != json.loads((output / "model_config.json").read_text())["manifest_sha256"]
                ):
                    raise ValueError("Resume manifest mismatch")
                model.load_state_dict(resume["model"])
                optimizer.load_state_dict(resume["optimizer"])
                start_epoch, best_score = resume["epoch"] + 1, resume["best_score"]
                best_metric, best_state = resume["best_metric"], resume["best_model"]
                if resume.get("scaler"):
                    scaler.load_state_dict(resume["scaler"])
                history = resume.get("history", [])
                if "torch_rng" in resume:
                    torch.set_rng_state(resume["torch_rng"].cpu())
                    random.setstate(resume["python_rng"])
                    state = resume["numpy_rng"]
                    np.random.set_state(
                        (
                            state[0],
                            np.array(state[1], dtype=np.uint32),
                            state[2],
                            state[3],
                            state[4],
                        )
                    )
                    if device.type == "cuda" and resume.get("cuda_rng"):
                        torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng"]])
                stale = resume.get("stale", 0)
                save_checkpoint(best_state, output / "model.pth")
            epochs = 1 if args.smoke else config["head_epochs"] + config["fine_tune_epochs"]
            if start_epoch >= epochs:
                raise ValueError("Resume checkpoint has already completed configured epochs")
            for epoch in range(start_epoch, epochs):
                head_only = epoch < config["head_epochs"] and not args.smoke
                for key, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        not head_only or key.startswith(("fc.", "classifier."))
                    )
                model.train()
                if head_only:
                    # Freeze backbone batch normalization as well as its parameters.
                    for module in model.modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            module.eval()
                for group in optimizer.param_groups:
                    group["lr"] = config["head_lr"] if head_only else config["fine_tune_lr"]
                losses = []
                optimizer.zero_grad(set_to_none=True)
                for batch_index, (images, targets) in enumerate(loader):
                    images = images.to(device, non_blocking=device.type == "cuda")
                    targets = targets.to(device, non_blocking=device.type == "cuda")
                    window, step = accumulation_window(batch_index, len(loader), accumulation)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        loss = torch.nn.functional.cross_entropy(model(images), targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "Non-finite training loss; checkpoint not marked trained"
                        )
                    scaler.scale(loss / window).backward()
                    if step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    losses.append(loss.item())
                    if len(losses) % 50 == 0:
                        print(
                            f"epoch={epoch} batch={len(losses)}/{len(loader)} "
                            f"loss={loss.item():.4f}",
                            flush=True,
                        )
                truth, predicted, top5_correct = [], [], 0
                model.eval()
                with torch.inference_mode():
                    for images, targets in validation:
                        logits = model(images.to(device))
                        predicted.extend(logits.argmax(1).cpu().tolist())
                        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
                        top5_correct += int((top5 == targets.to(device)[:, None]).any(1).sum())
                        truth.extend(targets.tolist())
                metric = compute_metrics(truth, predicted, len(train.labels["labels"]))
                metric["top5_accuracy"] = top5_correct / len(truth)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "val_macro_f1": metric["macro_f1"],
                    }
                )
                print(
                    f"epoch={epoch} loss={np.mean(losses):.4f} "
                    f"val_macro_f1={metric['macro_f1']:.4f}",
                    flush=True,
                )
                if metric["macro_f1"] > best_score:
                    best_score, stale, best_metric = metric["macro_f1"], 0, metric
                    save_checkpoint(model.state_dict(), output / "model.pth")
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    stale += 1
                save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "best_metric": {key: float(value) for key, value in best_metric.items()},
                        "best_model": best_state,
                        "scaler": scaler.state_dict(),
                        "history": history,
                        "stale": stale,
                        "torch_rng": torch.get_rng_state(),
                        "python_rng": random.getstate(),
                        "numpy_rng": numpy_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "manifest_sha256": json.loads((output / "model_config.json").read_text())[
                            "manifest_sha256"
                        ],
                    },
                    output / "resume.pth",
                )
                if stale >= config.get("patience", 5):
                    break
        with (output / "training_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "trial_C",
                    "train_loss",
                    "val_macro_f1",
                    "fit_seconds",
                    "iterations",
                    "converged",
                ],
            )
            writer.writeheader()
            writer.writerows(history)
        metadata = json.loads((output / "model_config.json").read_text())
        metadata.update(trained=True, training_samples=len(train), validation_samples=len(val))
        if metadata.get("dataset_version") == "v3":
            protocol_path = PROJECT_ROOT / "docs/ml/full_v3_101class_training_protocol_v1.md"
            protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            best_metric.update(
                split="val",
                dataset_version="v3",
                num_classes=metadata["num_classes"],
                manifest_sha256=metadata["manifest_sha256"],
                labels_sha256=metadata["labels_sha256"],
                protocol_sha256=protocol_digest,
                smoke_only=bool(metadata["smoke_only"]),
            )
        write_json(output / "metrics_val.json", best_metric)
        metadata.update(
            training_seconds=time.perf_counter() - started,
            amp_enabled=name != "hog_svm" and use_amp,
            cuda_version=torch.version.cuda,
        )
        if name != "hog_svm" and device.type == "cuda":
            metadata.update(
                gpu_name=torch.cuda.get_device_name(device),
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20,
            )
        write_json(output / "model_config.json", metadata)
        seal_artifact(output)
        print(f"TRAINED {'SMOKE ONLY' if args.smoke else 'CANDIDATE'}: {output}")


def numpy_rng_state():
    state = np.random.get_state()
    return (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4]))


def _validate_training_binding(name, config):
    """Bind a training config to its manifest, labels, dataset version and source evidence."""
    manifest_dir = Path(config["manifest_dir"]).resolve()
    manifest_path = manifest_dir / "dataset_manifest.csv"
    labels_path = manifest_dir / "labels.json"
    metadata_path = manifest_dir / "dataset_metadata.json"
    if not all(path.is_file() for path in (manifest_path, labels_path, metadata_path)):
        raise ValueError("Training manifest directory is incomplete")
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = labels_data.get("labels", [])
    version = labels_data.get("dataset_version")
    if config.get("model_name") != name:
        raise ValueError("Training config model_name does not match the requested trainer")
    if not labels or [row.get("class_id") for row in labels] != list(range(len(labels))):
        raise ValueError("Training labels must use contiguous, stable class IDs")
    if version != metadata.get("dataset_version"):
        raise ValueError("Manifest metadata and label dataset versions differ")
    if config.get("expected_num_classes", len(labels)) != len(labels):
        raise ValueError("Training config class count differs from the manifest labels")
    if config.get("expected_dataset_version", version) != version:
        raise ValueError("Training config dataset version differs from the manifest")
    with manifest_path.open("rb") as handle:
        manifest_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    checks = {
        "expected_manifest_sha256": manifest_digest,
        "expected_labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "expected_source_fingerprint": metadata.get("source_fingerprint"),
        "expected_inventory_sha256": metadata.get("inventory_sha256"),
        "expected_v3_fingerprint": metadata.get("v3_fingerprint"),
    }
    for key, actual in checks.items():
        if key in config and config[key] != actual:
            raise ValueError(f"Training config fingerprint mismatch: {key}")
    return {"manifest_dir": manifest_dir, "metadata": metadata, "labels": labels_data}


def _validate_training_dry_run(name, config):
    """Validate manifest contracts and model output shape without touching training state."""
    binding = _validate_training_binding(name, config)
    manifest_dir = binding["manifest_dir"]
    metadata_path = manifest_dir / "dataset_metadata.json"
    labels_path = manifest_dir / "labels.json"
    manifest_path = manifest_dir / "dataset_manifest.csv"
    if not all(path.is_file() for path in (metadata_path, labels_path, manifest_path)):
        raise ValueError("Training manifest directory is incomplete")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = labels_data.get("labels", [])
    if not labels or [row.get("class_id") for row in labels] != list(range(len(labels))):
        raise ValueError("Training labels must use contiguous, stable class IDs")
    dataset_version = labels_data.get("dataset_version")
    if dataset_version != metadata.get("dataset_version"):
        raise ValueError("Manifest metadata and label dataset versions differ")
    if config.get("model_name") != name:
        raise ValueError("Training config model_name does not match the requested trainer")
    if config.get("expected_num_classes", len(labels)) != len(labels):
        raise ValueError("Training config class count differs from the manifest labels")
    if config.get("expected_dataset_version", dataset_version) != dataset_version:
        raise ValueError("Training config dataset version differs from the manifest")
    image_root = Path(metadata["root"])
    if not image_root.is_absolute():
        image_root = manifest_dir / image_root
    if not image_root.resolve().is_dir():
        raise ValueError("Manifest image root is unavailable")

    mapping = {row["class_id"]: row["class_key"] for row in labels}
    split_counts = Counter()
    split_classes = {split: set() for split in ("train", "val", "test")}
    image_ids = set()
    content_hashes = set()
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "image_id",
            "relative_path",
            "class_id",
            "class_key",
            "split",
            "checksum",
            "dataset_version",
        }
        if not required_fields <= set(reader.fieldnames or []):
            raise ValueError("Training manifest schema is incomplete")
        for row in reader:
            split = row["split"]
            class_id = int(row["class_id"])
            relative_path = Path(row["relative_path"])
            digest = row["checksum"]
            if split not in split_classes or mapping.get(class_id) != row["class_key"]:
                raise ValueError("Manifest contains an invalid split or class mapping")
            if row["dataset_version"] != dataset_version:
                raise ValueError("Manifest row dataset version differs from labels")
            if (
                not row["image_id"]
                or row["image_id"] in image_ids
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or "\\" in row["relative_path"]
            ):
                raise ValueError("Manifest contains duplicate IDs or unsafe image paths")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise ValueError("Manifest contains an invalid SHA-256 checksum")
            if digest in content_hashes:
                raise ValueError("Manifest contains an exact duplicate image checksum")
            image_ids.add(row["image_id"])
            content_hashes.add(digest)
            split_counts[split] += 1
            split_classes[split].add(row["class_key"])

    expected_counts = metadata.get("split_counts")
    if expected_counts and dict(split_counts) != expected_counts:
        raise ValueError("Manifest split counts differ from published metadata")
    expected_classes = set(mapping.values())
    if any(split_classes[split] != expected_classes for split in split_classes):
        raise ValueError("Every published split must contain the complete class mapping")
    with manifest_path.open("rb") as handle:
        manifest_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    labels_digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    expected_digests = {
        "expected_manifest_sha256": manifest_digest,
        "expected_labels_sha256": labels_digest,
        "expected_source_fingerprint": metadata.get("source_fingerprint"),
        "expected_inventory_sha256": metadata.get("inventory_sha256"),
        "expected_v3_fingerprint": metadata.get("v3_fingerprint"),
    }
    for key, actual in expected_digests.items():
        if key in config and config[key] != actual:
            raise ValueError(f"Training config fingerprint mismatch: {key}")

    if name == "hog_svm":
        sample = np.zeros((*config["image_size"][::-1], 3), dtype=np.uint8)
        output_shape = extract_hog(sample, config).shape
    else:
        import gc

        model = create_model(name, len(labels), pretrained=False).cpu().eval()
        crop = preprocessing(name)["crop"]
        with torch.inference_mode():
            output = model(torch.zeros((1, 3, crop, crop), dtype=torch.float32))
        output_shape = tuple(output.shape)
        del output, model
        gc.collect()
        if output_shape != (1, len(labels)):
            raise ValueError("Model output does not match the manifest class count")
    return {
        "dataset_version": dataset_version,
        "class_count": len(labels),
        "split_counts": dict(split_counts),
        "manifest_sha256": manifest_digest,
        "labels_sha256": labels_digest,
        "output_shape": output_shape,
    }


def save_checkpoint(value, target):
    pending = target.with_name(target.name + ".pending")
    torch.save(value, pending)
    pending.replace(target)


def read_resume_contract(checkpoint, name, dataset, preproc, config):
    root = checkpoint.parent
    previous = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    prior_preproc = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    prior_config = yaml.safe_load((root / "train_config.yaml").read_text(encoding="utf-8"))
    if previous["model_name"] != name or labels != dataset.labels or prior_preproc != preproc:
        raise ValueError("Resume architecture, labels or preprocessing mismatch")
    if previous.get("smoke_only"):
        raise ValueError("Smoke checkpoints cannot be resumed into a formal experiment")
    resume_mutable = {"fine_tune_epochs"}
    for key in set(prior_config) | set(config):
        if key not in resume_mutable and prior_config.get(key) != config.get(key):
            raise ValueError(f"Resume training configuration mismatch: {key}")
    if config.get("fine_tune_epochs", 0) < prior_config.get("fine_tune_epochs", 0):
        raise ValueError("Resume fine_tune_epochs may only stay fixed or increase")
    return previous
