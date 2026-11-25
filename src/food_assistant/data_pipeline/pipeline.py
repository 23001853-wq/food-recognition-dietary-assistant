"""Resumable, offline-first orchestration for the Food-101 ELT stages."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from food_assistant.data_pipeline.atomic import (
    atomic_write_json,
    atomic_write_text,
    checkpoint_digest,
    resource_lock,
)
from food_assistant.data_pipeline.config import PipelineConfig, load_pipeline_config

STAGES = ("inspect", "extract", "load", "quality", "transform", "validate", "report")
RUN_SCHEMA_VERSION = 1


class PipelineBlocked(RuntimeError):
    """Represent an ELT prerequisite that cannot safely be satisfied."""


def _utc_now() -> str:
    """Return a timezone-aware ISO timestamp for state records."""
    return datetime.now(UTC).isoformat()


def _file_digest(path: Path) -> str:
    """Hash one configuration or evidence file without logging its contents."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_order(stage: str) -> tuple[str, ...]:
    """Return the DAG prefix required to reach a requested stage."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of: {', '.join(STAGES)}")
    return STAGES[: STAGES.index(stage) + 1]


def _default_report_dir(workspace: Path, run_id: str) -> Path:
    """Resolve the immutable per-run report directory under the repository artifacts."""
    return workspace / "artifacts" / "reports" / "elt" / run_id


def _existing_state(path: Path, config_hash: str, run_id: str) -> dict[str, Any] | None:
    """Read a resume checkpoint only when its envelope and run identity are valid."""
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume state is corrupt: {exc}") from exc
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != RUN_SCHEMA_VERSION
        or state.get("run_id") != run_id
        or state.get("config_hash") != config_hash
        or not isinstance(state.get("stages"), dict)
    ):
        raise ValueError("Resume state schema, run id, or config fingerprint mismatch")
    return state


def _discover_resume_id(workspace: Path, config_hash: str) -> str:
    """Find the newest valid run for this config without creating or changing files."""
    report_root = workspace / "artifacts" / "reports" / "elt"
    candidates: list[tuple[str, str]] = []
    if not report_root.is_dir():
        raise FileNotFoundError("No prior ELT run exists to resume.")
    for state_path in report_root.glob("*/run_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(state, dict)
            and state.get("schema_version") == RUN_SCHEMA_VERSION
            and state.get("run_id") == state_path.parent.name
            and state.get("config_hash") == config_hash
            and isinstance(state.get("stages"), dict)
        ):
            candidates.append((str(state.get("updated_at", "")), state_path.parent.name))
    if not candidates:
        raise FileNotFoundError("No valid prior ELT state matches the current config fingerprint.")
    candidates.sort()
    return candidates[-1][1]


def _render_markdown(result: dict[str, Any]) -> str:
    """Render machine state into a concise human-readable run report."""
    lines = [
        "# Food-101 ELT orchestration",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Overall: **{result['status']}**",
        f"- Config fingerprint: `{result['config_hash']}`",
        "",
        "| Stage | Status | Reason | Duration (s) |",
        "|---|---|---|---:|",
    ]
    for name, entry in result["stages"].items():
        reason = str(entry.get("reason") or entry.get("skip_reason") or "").replace("|", "\\|")
        lines.append(
            f"| {name} | {entry.get('status')} | {reason} | "
            f"{entry.get('duration_seconds', 0):.3f} |"
        )
    lines.extend(["", "No training or model inference is performed by this pipeline.", ""])
    return "\n".join(lines)


def _raw_exists(config: PipelineConfig) -> bool:
    """Check for the expected Food-101 root without traversing its image inventory."""
    return (config.raw_root / "images").is_dir() and (config.raw_root / "meta").is_dir()


def _approved_source(config: PipelineConfig) -> dict[str, Any] | None:
    """Load and validate the repository's explicit academic approval without logging values."""
    try:
        from scripts.download_food101 import _load_approval

        return _load_approval(config.approval_file)
    except (OSError, ValueError, PermissionError):
        return None


def _plan(config: PipelineConfig, stages: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Build a read-only stage plan without creating directories, locks, or state."""
    raw_present = _raw_exists(config)
    planned: dict[str, dict[str, str]] = {}
    for name in stages:
        if name == "inspect":
            planned[name] = {
                "status": "RUN",
                "reason": "Inspect configured paths and existing raw presence.",
            }
        elif name in {"extract", "load"} and raw_present:
            planned[name] = {
                "status": "RUN",
                "reason": "Inspect raw checkpoint/fingerprints before deciding to skip this stage.",
            }
        elif name in {"extract", "load"} and config.download_policy == "disabled":
            planned[name] = {
                "status": "BLOCKED",
                "reason": "Raw missing and download disabled; offline mode will not connect.",
            }
        elif name in {"extract", "load"}:
            planned[name] = {
                "status": "BLOCKED",
                "reason": "No explicit source approval/evidence was supplied to the runner.",
            }
        elif name == "quality":
            planned[name] = {
                "status": "RUN" if raw_present else "BLOCKED",
                "reason": "Run the existing Food-101 integrity verifier."
                if raw_present
                else "Raw tree is absent.",
            }
        elif name == "transform":
            transform_supported = (
                config.dataset_scope == "poc" and config.split_version == "poc-v2"
            ) or (config.dataset_scope == "full" and config.split_version in {"v2", "v3"})
            planned[name] = {
                "status": "RUN" if raw_present and transform_supported else "BLOCKED",
                "reason": "Run the configured immutable transform only after quality PASS."
                if raw_present and transform_supported
                else "No compatible Transform builder exists for this scope/version."
                if raw_present
                else "Raw tree is absent.",
            }
        elif name == "validate":
            transform_supported = (
                config.dataset_scope == "poc" and config.split_version == "poc-v2"
            ) or (config.dataset_scope == "full" and config.split_version in {"v2", "v3"})
            can_reach_validation = raw_present and transform_supported
            planned[name] = {
                "status": "RUN" if can_reach_validation else "BLOCKED",
                "reason": "Validate the manifest after prerequisite stages pass."
                if can_reach_validation
                else "A prerequisite stage is blocked; Validate will not run.",
            }
        else:
            planned[name] = {"status": "RUN", "reason": "Write this run's report after execution."}
    return planned


def _dispatch(
    name: str,
    config_path: Path,
    config: PipelineConfig,
    report_dir: Path,
    prior_stages: dict[str, Any],
) -> dict[str, Any]:
    """Execute one production stage through the repository's existing data-pipeline APIs."""
    if name == "inspect":
        if not _raw_exists(config):
            return {
                "status": "PASS",
                "reason": "Raw tree absent; acquisition policy will decide next step.",
            }
        from scripts.download_food101 import verify_existing_raw

        try:
            workspace = config_path.parents[1]
            baseline_path = (
                workspace / "data/reports/elt/v2/phase_2_5_verification_retry/food101_report.json"
            )
            evidence = verify_existing_raw(config.raw_root, baseline_path)
        except (OSError, ValueError, RuntimeError) as exc:
            return {
                "status": "BLOCKED",
                "reason": f"Existing raw does not match trusted Phase 2.5 evidence: {exc}",
                "raw_verified": False,
            }
        return {
            **evidence,
            "status": "PASS",
            "raw_verified": True,
            "reason": "Existing raw matches Phase 2.5 signatures and inventory evidence.",
        }
    if name == "extract":
        if _raw_exists(config):
            if prior_stages.get("inspect", {}).get("raw_verified") is not True:
                return {
                    "status": "BLOCKED",
                    "reason": "Raw exists but has no verified source fingerprint; refusing reuse.",
                }
            return {
                "status": "SKIPPED",
                "skip_reason": "Phase 2.5 source/inventory evidence matches current raw.",
            }
        if config.download_policy == "disabled":
            return {"status": "BLOCKED", "reason": "Raw missing and download policy is disabled."}
        approval = _approved_source(config)
        if approval is None:
            return {"status": "BLOCKED", "reason": "Valid academic source approval is unavailable."}
        from food_assistant.data_pipeline.extract import RetryPolicy, download_sequential

        result = download_sequential(
            approval.get("archive_url", ""),
            config.archive,
            approval=approval,
            official_sha256=approval.get("official_sha256"),
            retry_policy=RetryPolicy(
                max_attempts=config.retries + 1,
                base_delay_seconds=config.backoff_seconds,
            ),
            lock_timeout_seconds=config.lock_timeout_seconds,
        )
        return {**result, "status": result.get("state", "PASS")}
    if name == "load":
        if _raw_exists(config):
            if prior_stages.get("inspect", {}).get("raw_verified") is not True:
                return {
                    "status": "BLOCKED",
                    "reason": "Raw exists but has no verified source fingerprint; refusing reuse.",
                }
            return {
                "status": "SKIPPED",
                "skip_reason": "Verified raw is already published; extraction is unnecessary.",
            }
        if config.download_policy != "approved_only":
            return {
                "status": "BLOCKED",
                "reason": "Load requires approved_only policy and a source-bound archive.",
            }
        if config.raw_root.name != "food-101":
            return {
                "status": "BLOCKED",
                "reason": (
                    "The safe extractor only publishes the official food-101 directory name."
                ),
            }
        approval = _approved_source(config)
        if approval is None:
            return {"status": "BLOCKED", "reason": "Valid academic source approval is unavailable."}
        from food_assistant.data_pipeline.extract import RetryPolicy, download_sequential

        archive_evidence = download_sequential(
            approval.get("archive_url", ""),
            config.archive,
            approval=approval,
            official_sha256=approval.get("official_sha256"),
            retry_policy=RetryPolicy(
                max_attempts=config.retries + 1,
                base_delay_seconds=config.backoff_seconds,
            ),
            lock_timeout_seconds=config.lock_timeout_seconds,
        )
        from scripts.download_food101 import safe_extract

        target_root = safe_extract(
            config.archive,
            config.raw_root.parent,
            {
                "dataset": "food101",
                "source_url": approval.get("source_url"),
                "academic_approval": approval.get("academic_approval"),
                "academic_approval_date": approval.get("academic_approval_date"),
                "citation": approval.get("citation"),
                "use": approval.get("use"),
                "rights_status": approval.get("rights_status"),
                "download_sha256": archive_evidence["local_sha256"],
                "local_sha256": archive_evidence["local_sha256"],
                "official_sha256": approval.get("official_sha256"),
                "official_checksum_verified": archive_evidence.get(
                    "official_checksum_verified", False
                ),
            },
        )
        if target_root.resolve() != config.raw_root.resolve():
            raise PipelineBlocked("Safe Load target does not match configured raw_root.")
        return {
            "status": "PASS",
            "raw_root": str(target_root),
            "archive_fingerprint": archive_evidence.get("local_sha256"),
        }
    if name == "quality":
        from food_assistant.ml.food101_verifier import audit_food101

        audit = audit_food101(
            config.raw_root,
            report_dir / "quality",
            mode="full",
            reject_cross_split_duplicates=False,
        )
        raw_pass = audit.get("overall") == "PASS"
        return {
            "status": "PASS" if raw_pass else audit.get("overall", "INCOMPLETE"),
            "raw_integrity_status": "PASS" if raw_pass else audit.get("overall", "INCOMPLETE"),
            "official_split_leakage": audit.get("train_test_duplicate_hashes", 0),
            "audit_report": str(report_dir / "quality" / "food101_report.json"),
            "details": audit,
        }
    if name == "transform":
        if config.dataset_scope == "poc" and config.split_version == "poc-v2":
            from food_assistant.data_pipeline.transform import build_poc_transform

            result = build_poc_transform(
                config_path,
                baseline_report_path=report_dir / "quality" / "food101_report.json",
            )
        elif config.split_version == "v2":
            from food_assistant.data_pipeline.splits import build_clean_split

            result = build_clean_split(
                config_path,
                baseline_report_path=report_dir / "quality" / "food101_report.json",
            )
        elif config.dataset_scope == "full" and config.split_version == "v3":
            from food_assistant.data_pipeline.split_v3 import build_training_split_v3

            workspace = config_path.parents[1]
            result = build_training_split_v3(
                config_path,
                v2_config_path=workspace / "configs" / "food101_elt_full_v2.yaml",
                baseline_report_path=report_dir / "quality" / "food101_report.json",
            )
        else:
            return {
                "status": "BLOCKED",
                "reason": (
                    "No Transform contract for "
                    f"scope={config.dataset_scope}, version={config.split_version}."
                ),
            }
        stage_status = result.get("state", result.get("status", result.get("overall", "PASS")))
        return {**result, "status": stage_status}
    if name == "validate":
        if config.dataset_scope == "full" and config.split_version == "v3":
            from food_assistant.data_pipeline.validate_v3 import validate_full_v3

            result = validate_full_v3(config_path)
            return {**result, "status": result.get("status", "INCOMPLETE")}
        if not config.manifest_root.is_dir():
            return {"status": "INCOMPLETE", "reason": "Manifest output is missing."}
        from food_assistant.ml.datasets import ManifestDataset

        splits = {}
        for split_name in ("train", "val", "test"):
            try:
                dataset = ManifestDataset(
                    config.manifest_root,
                    split_name,
                    root_override=config.raw_root,
                )
            except (FileNotFoundError, ValueError) as exc:
                if split_name == "val" and config.split_version == "v2":
                    continue
                return {"status": "FAIL", "reason": f"{split_name}: {exc}"}
            splits[split_name] = len(dataset)
        return {"status": "PASS", "split_counts": splits}
    raise ValueError(f"Unsupported stage: {name}")


def run_pipeline(
    config_path: Path,
    *,
    stage: str = "all",
    dry_run: bool = False,
    resume: bool = False,
    report_dir: Path | None = None,
    run_id: str | None = None,
    workspace: Path | None = None,
    stage_handlers: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run or preflight the requested ELT DAG prefix with atomic resumable state."""
    root = Path(workspace or Path.cwd()).resolve()
    resolved_config = Path(config_path)
    resolved_config = (
        (root / resolved_config).resolve()
        if not resolved_config.is_absolute()
        else resolved_config.resolve()
    )
    config = load_pipeline_config(resolved_config, root)
    requested_stages = _stage_order(stage if stage != "all" else "report")
    stages = requested_stages if requested_stages[-1] == "report" else (*requested_stages, "report")
    config_hash = _file_digest(resolved_config)
    if resume and run_id is None:
        selected_run_id = (
            Path(report_dir).name
            if report_dir is not None
            else _discover_resume_id(root, config_hash)
        )
    else:
        selected_run_id = run_id or (
            f"elt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        )
    if Path(selected_run_id).name != selected_run_id or selected_run_id in {".", ".."}:
        raise ValueError("run_id must be a safe single path component")
    output_dir = (
        Path(report_dir).resolve() if report_dir else _default_report_dir(root, selected_run_id)
    )
    if not output_dir.is_relative_to(root / "artifacts" / "reports" / "elt"):
        raise ValueError("report directory must remain under artifacts/reports/elt")
    if dry_run:
        return {
            "run_id": selected_run_id,
            "status": "DRY_RUN",
            "config_hash": config_hash,
            "report_dir": str(output_dir),
            "plan": _plan(config, stages),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "run_state.json"
    handlers = stage_handlers or {}
    with resource_lock(output_dir / "run_state.json", config.lock_timeout_seconds):
        state = _existing_state(state_path, config_hash, selected_run_id) if resume else None
        if state is None:
            if state_path.exists():
                raise FileExistsError("Run state exists; pass --resume to validate and continue it")
            state = {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": selected_run_id,
                "config_hash": config_hash,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "stages": {},
            }
        for name in stages:
            if name == "report":
                continue
            previous = state["stages"].get(name, {})
            if resume and name in {"extract", "load"} and previous.get("status") == "SKIPPED":
                if previous.get("config_hash") != config_hash:
                    raise ValueError(
                        f"Cannot resume stale stage {name}: config fingerprint mismatch"
                    )
                continue
            attempt = int(previous.get("attempts", 0)) + 1
            entry: dict[str, Any] = {
                "stage_id": name,
                "status": "RUNNING",
                "started_at": _utc_now(),
                "attempts": attempt,
                "config_hash": config_hash,
                "reason": None,
                "error_type": None,
                "artifact_fingerprints": {},
            }
            state["stages"][name] = entry
            state["updated_at"] = _utc_now()
            atomic_write_json(state_path, state)
            started = time.monotonic()
            try:
                result = (
                    handlers[name](
                        config_path=resolved_config, config=config, report_dir=output_dir
                    )
                    if name in handlers
                    else _dispatch(name, resolved_config, config, output_dir, state["stages"])
                )
                status = result.get("status", "PASS")
                if status not in {"PASS", "SKIPPED", "FAIL", "INCOMPLETE", "BLOCKED"}:
                    result = {
                        **result,
                        "reason": f"Unsupported stage status {status!r}; failing closed.",
                    }
                    status = "INCOMPLETE"
                entry.update(result)
                entry["status"] = status
                entry["reason"] = result.get("reason") or result.get("skip_reason")
                evidence = result.get("details", result)
                if not entry.get("artifact_fingerprints"):
                    entry["artifact_fingerprints"] = {
                        key: evidence[key]
                        for key in ("source_fingerprint", "inventory_sha256", "split_fingerprint")
                        if isinstance(evidence, dict) and evidence.get(key)
                    }
                if isinstance(evidence, dict) and evidence.get("source_fingerprint"):
                    state["input_fingerprint"] = evidence["source_fingerprint"]
                elif isinstance(evidence, dict) and evidence.get("inventory_sha256"):
                    state["input_fingerprint"] = evidence["inventory_sha256"]
                if status not in {"PASS", "SKIPPED"}:
                    for downstream in stages[stages.index(name) + 1 :]:
                        if downstream == "report":
                            continue
                        state["stages"][downstream] = {
                            "stage_id": downstream,
                            "status": "BLOCKED",
                            "reason": f"Blocked by {name}={status}.",
                            "skip_reason": None,
                            "attempts": 0,
                            "config_hash": config_hash,
                            "artifact_fingerprints": {},
                        }
                    break
            except Exception as exc:
                entry.update(
                    {"status": "FAIL", "error_type": type(exc).__name__, "reason": str(exc)}
                )
                for downstream in stages[stages.index(name) + 1 :]:
                    if downstream != "report":
                        state["stages"][downstream] = {
                            "stage_id": downstream,
                            "status": "BLOCKED",
                            "reason": f"Blocked by {name} failure.",
                            "attempts": 0,
                            "config_hash": config_hash,
                            "artifact_fingerprints": {},
                        }
                break
            finally:
                entry["duration_seconds"] = time.monotonic() - started
                entry["finished_at"] = _utc_now()
                state["updated_at"] = _utc_now()
                atomic_write_json(state_path, state)
        state["stages"]["report"] = {
            "stage_id": "report",
            "status": "PASS",
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "duration_seconds": 0.0,
            "attempts": 1,
            "config_hash": config_hash,
            "reason": "JSON and Markdown run reports published atomically.",
            "artifact_fingerprints": {},
        }
        statuses = [value.get("status") for value in state["stages"].values()]
        state["status"] = (
            "PASS"
            if all(value in {"PASS", "SKIPPED"} for value in statuses)
            else next(
                (value for value in ("FAIL", "INCOMPLETE", "BLOCKED") if value in statuses),
                "INCOMPLETE",
            )
        )
        state["updated_at"] = _utc_now()
        state["report_fingerprint"] = checkpoint_digest(state["stages"])
        atomic_write_json(state_path, state)
        atomic_write_json(output_dir / "run_report.json", state)
        atomic_write_text(output_dir / "run_report.md", _render_markdown(state))
        return state
