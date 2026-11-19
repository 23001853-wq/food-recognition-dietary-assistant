"""Preflight or atomically publish the full Food-101 training split v3."""

import argparse
import json
from pathlib import Path

from food_assistant.data_pipeline.atomic import atomic_write_json, resource_lock
from food_assistant.data_pipeline.config import load_pipeline_config
from food_assistant.data_pipeline.split_v3 import (
    V2_CONFIG_PATH,
    build_training_split_v3,
)
from food_assistant.data_pipeline.splits import REPORT_PATH, RawBaselineMismatch


def main() -> None:
    """Run read-only preflight by default or publish when --publish is supplied."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/food101_elt_full_v3.yaml"))
    parser.add_argument("--v2-config", type=Path, default=V2_CONFIG_PATH)
    parser.add_argument("--baseline-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    config = load_pipeline_config(args.config, Path.cwd())
    try:
        result = build_training_split_v3(
            args.config,
            args.v2_config,
            args.baseline_report,
            publish=args.publish,
        )
    except Exception as exc:
        target = config.report_root / "failure_report.json"
        status = "BLOCKED" if isinstance(exc, (RawBaselineMismatch, FileExistsError)) else "FAIL"
        with resource_lock(target, timeout_seconds=config.lock_timeout_seconds):
            atomic_write_json(
                target,
                {
                    "schema_version": "3.0",
                    "status": status,
                    "complete": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
