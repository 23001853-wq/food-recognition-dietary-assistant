"""Check a measured POC release; --apply explicitly switches local config away from mock."""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from food_assistant.config import PROJECT_ROOT, Settings
from food_assistant.ml.protocol import digest
from food_assistant.services.model_registry import ModelRegistry


def update_environment(path, values):
    path = Path(path)
    original = path.read_text(encoding="utf-8")
    backup = path.with_name(".env.before-real-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
    backup.write_text(original, encoding="utf-8")
    seen, lines = set(), []
    for line in original.splitlines():
        match = re.match(r"^([A-Z_]+)=", line)
        if match and match[1] in values:
            key = match[1]
            lines.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in values.items() if key not in seen)
    pending = path.with_name(".env.pending-real")
    pending.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pending.replace(path)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    settings = Settings().model_copy(
        update={"allow_smoke_models": False, "inference_device": "cpu"}
    )
    release = ModelRegistry(settings).load("production")
    metrics = json.loads((release.directory / "metrics_test.json").read_text(encoding="utf-8"))
    if (
        release.config["num_classes"] != 10
        or release.config["dataset_version"] != "poc-v1"
        or metrics.get("split") != "test"
        or metrics.get("smoke_only")
        or metrics.get("sample_count") != 2500
        or not metrics.get("protocol_sha256")
        or metrics["manifest_sha256"] != release.config["manifest_sha256"]
        or not (release.directory / "evaluation_test/evaluation_complete.json").exists()
    ):
        raise SystemExit("Release does not have a complete frozen POC test evaluation")
    if digest(PROJECT_ROOT / "data/splits/v1/dataset_manifest.csv") != metrics["manifest_sha256"]:
        raise SystemExit("Release does not match the current POC manifest")
    print(
        json.dumps(
            {
                "release": release.directory.name,
                "model": release.config["model_name"],
                "test_metrics": metrics,
                "apply": args.apply,
            },
            indent=2,
        )
    )
    if args.apply:
        backup = update_environment(
            PROJECT_ROOT / ".env",
            {
                "ALLOW_MOCK_AI": "false",
                "ALLOW_SMOKE_MODELS": "false",
                "DEFAULT_MODEL": "production",
                "INFERENCE_DEVICE": "cpu",
            },
        )
        print(f"Local configuration updated; secret backup: {backup.name}")
        print(
            "Restart backend/frontend yourself, then run "
            "smoke_test.py --model production --image PATH"
        )
    else:
        print("CHECK ONLY: config unchanged. Use --apply after reviewing measured results.")


if __name__ == "__main__":
    main()
