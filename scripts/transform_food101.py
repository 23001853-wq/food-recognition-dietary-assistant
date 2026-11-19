"""Build or preflight the immutable POC Food-101 Transform manifest."""

import argparse
import json
from pathlib import Path

from food_assistant.data_pipeline.transform import build_poc_transform


def main() -> int:
    """Run the configured Transform quality gate and print its structured result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/food101_elt_poc_v2.yaml"),
        help="Versioned POC Transform configuration.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Verify and plan without creating output, lock, staging, or reports.",
    )
    arguments = parser.parse_args()
    result = build_poc_transform(arguments.config, publish=not arguments.preflight)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
