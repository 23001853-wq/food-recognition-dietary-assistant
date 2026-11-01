import argparse
from pathlib import Path

from food_assistant.ml.protocol import freeze_poc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Freeze exactly three trained POC candidates before test"
    )
    parser.add_argument("--artifacts", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze_poc(args.artifacts, args.manifest_dir, args.output)
    print(f"Frozen protocol written: {args.output}. No test images evaluated.")
