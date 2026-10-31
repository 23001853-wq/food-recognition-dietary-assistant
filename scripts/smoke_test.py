"""Validate the scaffold using Python 3.11 standard-library tools only."""

import ast
import csv
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        "PLAN.md", "README.md", "pyproject.toml", ".env.example", "docker-compose.yml",
        "alembic.ini", "migrations/env.py", "migrations/script.py.mako",
        "configs/data.yaml", "configs/classes_poc.yaml", "configs/hog_svm.yaml",
        "configs/resnet18.yaml", "configs/efficientnet_b0.yaml", "configs/inference.yaml",
        "src/food_assistant/app.py", "src/food_assistant/config.py",
        "src/food_assistant/persistence/database.py", "docs/tasks.md",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            raise SystemExit(f"Missing required file: {name}")
    sources = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
               *ROOT.joinpath("tests").rglob("*.py"), *ROOT.joinpath("migrations").rglob("*.py")]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in [ROOT / "pyproject.toml", ROOT / ".streamlit/config.toml"]:
        tomllib.loads(path.read_text(encoding="utf-8"))
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    names = re.findall(r"\| D\d+ \| `([^`]+\.mmd)`", plan)
    if len(names) != 12:
        raise SystemExit("Expected 12 planned diagram names")
    for name in names:
        if not (ROOT / "diagrams" / name).is_file():
            raise SystemExit(f"Missing diagram source: {name}")
    for name in ["data/nutrition/source_catalog.csv", "data/nutrition/curated_mapping.csv",
                 "artifacts/reports/experiment_comparison.csv"]:
        with (ROOT / name).open(encoding="utf-8", newline="") as handle:
            if len(list(csv.reader(handle))) != 1:
                raise SystemExit(f"Scaffold should contain header only: {name}")
    if (ROOT / "artifacts/production/current.json").exists():
        raise SystemExit("Scaffold should not contain a fabricated production pointer")
    print(f"PASS: structure, {len(sources)} Python files, TOML, 12 diagram files, empty CSVs.")
    print("Business tests, model inference, PostgreSQL and Mermaid rendering were not checked.")


if __name__ == "__main__":
    main()
