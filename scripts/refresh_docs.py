"""Refresh derivative docs from current contracts; preserve historical PLAN separately."""
# ruff: noqa: E501

from pathlib import Path

from food_assistant.persistence import orm_models  # noqa: F401
from food_assistant.persistence.database import Base


def main():
    texts = {
        "docs/deployment.md": "# Deployment\n\nSee [current deployment guide](../DEPLOYMENT_GUIDE.md). FastAPI is now part of MVP, following the authorized v2 architecture. Observed runtime results are recorded in IMPLEMENTATION_STATUS.md and TEST_REPORT.md.",
        "docs/experiment_protocol.md": "# Experiment protocol\n\nSee [dataset guide](../DATASET_GUIDE.md), [training guide](../MODEL_TRAINING_GUIDE.md), and configs/experiment_budget.yaml. The short CPU budget is frozen before test evaluation. Original Food-101 test remains independent; class order and manifest digest must match across methods. Synthetic smoke metrics are not food results. The original detailed research proposal remains in PLAN_v1_historical.md.",
        "docs/evaluation_report.md": "# Evaluation report\n\nSee [actual model evaluation status](../MODEL_EVALUATION_REPORT.md). Only script-produced measurements may be reported. Never substitute ImageNet accuracy or synthetic smoke scores.",
        "docs/nutrition_assumptions.md": "# Nutrition assumptions\n\nCurrent sourced profile is USDA SR Legacy FDC 170317, cooked regular-crust cheese pizza. The archive SHA256 and exact source are in data/nutrition/provenance.json. Nutrients scale per 100g; S/M/L 100/200/300g is an openly declared demonstration convention. Other recipes vary. Unknown values remain NULL; manual kcal is labeled manual. Meal items snapshot all reference values and provenance. Energy is estimated from selected/entered mass, never measured from image pixels. This is not personalized medical advice.",
        "docs/limitations.md": "# Limits\n\nSingle-user local demo; no public authentication/authorization. One crop yields one primary class, with manual correction. Scores are uncalibrated. No automatic ingredient/mass/detection inference. Only one sourced representative nutrition profile is currently curated; other foods may remain unavailable. CUDA hardware exists but installed PyTorch is CPU-only. Training and dataset status change during execution: consult IMPLEMENTATION_STATUS.md, not historical design targets. No local Vietnamese domain-shift photos have been collected. Short POC experiments are not full-101 or converged model benchmarks.",
        "docs/tasks.md": "# Implementation tasks\n\nThe active phase-by-phase checklist, executed commands, blockers and remaining dependencies are maintained in [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md). This is a solo developer project; no team handoff is generated. Full original research roadmap is preserved in PLAN_v1_historical.md.",
        "docs/diagram_requirements.md": "# Diagram requirements\n\nArchitecture v2: Browser → Streamlit → FastAPI → services → PostgreSQL / trusted model artifacts. Sources in diagrams/ reflect this boundary; the ERD is generated from current ORM by scripts/update_diagrams.py. Diagrams are explanatory designs, not evidence of trained-model performance. Future clients/auth/detection remain explicitly Future. No image/weight dataset is stored in PostgreSQL. Prediction metadata and meal snapshots preserve provenance. Mermaid sources have not been rendered by a Mermaid runtime in this session; no SVG/PNG output is claimed.",
        "tests/README.md": "# Tests\n\nSee [executed test report](../TEST_REPORT.md). Unit, PostgreSQL integration, HTTP smoke, and CPU ML smoke are implemented. Run pytest from the repository root. Integration tests require TEST_DATABASE_URL with a separate database ending in _test; no application DB reset is allowed. ML smoke tests use temporary synthetic fixtures and random CNN initialization, never external data or pretrained downloads. Install the ml and dev extras to run the complete suite.",
        "migrations/README.md": "# Migrations\n\nBusiness schema revision cba28587a601 is implemented and has run against PostgreSQL. Use alembic upgrade head and alembic check. No automatic downgrade or database reset. Tests migrate only the configured test database. See DATABASE_GUIDE.md for constraints, naming equivalence and snapshot policy.",
    }
    lines = [
        "# Current data dictionary",
        "",
        "Generated from SQLAlchemy ORM; see DATABASE_GUIDE.md for design rules.",
        "",
    ]
    for table in Base.metadata.sorted_tables:
        lines += [
            f"## {table.name}",
            "",
            "| Column | Type | Nullable | Keys |",
            "|---|---|---|---|",
        ]
        for column in table.columns:
            keys = (["PK"] if column.primary_key else []) + [
                "FK → " + str(fk.target_fullname) for fk in column.foreign_keys
            ]
            lines.append(
                f"| {column.name} | {column.type} | {column.nullable} | {', '.join(keys)} |"
            )
        lines += [""]
    texts["docs/data_dictionary.md"] = "\n".join(lines)
    files = sorted(Path("diagrams").glob("*.mmd"))
    texts["diagrams/README.md"] = (
        "# Architecture v2 diagrams\n\nD02/D03/D05/D06/D07/D10/D12 updated for FastAPI and current ORM. Other sources retain applicable original research/user-flow designs. D09 is the intended recognition state model, while the UI implements its behavior through session-state flags. Sources have not been rendered/validated by Mermaid runtime; no SVG/PNG is claimed.\n\n**FUTURE ARCHITECTURE — NOT IMPLEMENTED IN MVP** applies to D12 future clients/auth/detection, not to the now-implemented FastAPI backend. D11 dates remain illustrative.\n\n"
        + "\n".join(f"- [{p.name}]({p.name})" for p in files)
    )
    for name, text in texts.items():
        Path(name).write_text(text.rstrip() + "\n", encoding="utf-8")
    print("Refreshed derivative docs for v2; historical PLAN retained")


if __name__ == "__main__":
    main()
