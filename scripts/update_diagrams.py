"""Refresh architecture sources for the authorized FastAPI architecture, without fake renders."""

from pathlib import Path

from food_assistant.persistence import orm_models  # noqa: F401
from food_assistant.persistence.database import Base


def main():
    diagrams = {
        "02-container-mvp.mmd": """flowchart TB
    User["Browser"] --> UI["Streamlit frontend"]
    UI -->|HTTP / JSON / multipart| API["FastAPI backend"]
    API --> S["Application services"]
    S --> R["SQLAlchemy repositories / transactions"]
    R --> PG[("PostgreSQL app data")]
    S --> I["InferenceService: explicit mock or real release"]
    I --> FS[("Trusted local artifacts")]
    OFF["Offline Food-101 training and evaluation"] --> FS
    S --> N["Reviewed nutrition / nullable estimates"]
    N --> R
    S --> U[("Optional upload storage")]
""",
        "03-component-mvp.mmd": """flowchart LR
    UI["Streamlit pages"] --> C["HTTP API client"]
    C --> API["FastAPI validation and response contracts"]
    API --> IMG["ImageService / CropService"]
    IMG --> INF["InferenceService"]
    INF --> REG["ModelRegistry / trusted release"]
    API --> NUT["NutritionService"]
    API --> MEAL["MealService: idempotency + snapshots"]
    API --> DASH["DashboardService"]
    MEAL --> NUT
    NUT --> DB[("SQLAlchemy / PostgreSQL")]
    MEAL --> DB
    DASH --> DB
""",
        "05-inference-sequence.mmd": """sequenceDiagram
    actor U as User
    participant UI as Streamlit
    participant API as FastAPI
    participant IS as ImageService / CropService
    participant IN as InferenceService
    participant MR as ModelRegistry
    participant DB as PostgreSQL
    U->>UI: Upload image and choose model/crop
    UI->>API: POST predictions multipart
    API->>IS: Validate MIME, bytes, pixels, decode, EXIF and crop
    alt Invalid image/crop
        IS-->>API: Validation error
        API-->>UI: Structured 4xx error
    else Valid image
        API->>IN: predict(image, model_name, top_k)
        alt Explicit mock and development enabled
            IN-->>API: Deterministic response, is_mock=true
        else Real model requested
            IN->>MR: Validate checksums, labels, version, preprocessing
            alt Missing or inconsistent artifact
                MR-->>API: MODEL_NOT_READY / ARTIFACT_MISMATCH
            else Trained artifact
                MR-->>IN: Loaded release
                IN-->>API: Top-k scores, is_mock=false
            end
        end
        API->>DB: Store prediction metadata (raw image opt-in)
        DB-->>API: Commit
        API-->>UI: Versioned PredictionResult
        UI-->>U: Require label/portion confirmation
    end
""",
        "06-meal-save-sequence.mmd": """sequenceDiagram
    actor U as User
    participant UI as Streamlit
    participant API as FastAPI
    participant MS as MealService
    participant NS as NutritionService
    participant DB as PostgreSQL
    U->>UI: Confirm corrected food, portion and date
    UI->>API: POST meals with client_request_id
    API->>MS: Validated draft within transaction
    MS->>DB: Advisory lock / check retry fingerprint
    alt Same key, different request
        MS-->>API: 409 conflict
    else New meal
        MS->>NS: Lookup reviewed reference and resolve mass
        NS-->>MS: Snapshot nutrients/source or unknown/manual
        MS->>DB: INSERT meal and all items
        alt Any failure
            DB-->>API: Rollback
            API-->>UI: Save failed
        else Valid transaction
            DB-->>API: Commit
            API-->>UI: Saved IDs (201)
        end
    end
""",
        "10-deployment.mmd": """flowchart TB
    B["Browser: localhost:8501"] --> ST["Streamlit frontend container or Windows venv"]
    ST -->|HTTP :8000| API["FastAPI backend container or Windows venv"]
    API -->|SQLAlchemy / psycopg| PG[("PostgreSQL container or local service")]
    PG --> VOL[("Persistent DB volume")]
    API --> FS[("Trusted artifact mount / optional uploads")]
    TR["Separate offline training: CPU/CUDA or Colab"] --> FS
    NOTE["Loopback host bindings; single-user demo; no public auth"]
    API -.-> NOTE
""",
        "12-future-architecture.mmd": """flowchart LR
    NOTE["FUTURE ARCHITECTURE - NOT IMPLEMENTED IN MVP"]
    UI["Future React/mobile clients"] --> API["Existing FastAPI services"]
    API --> AUTH["Future authentication and authorization"]
    API --> OBJ[("Future object storage")]
    API --> OBS["Future multi-user monitoring"]
    API --> ML["Future detection / segmentation only with suitable training data"]
    NOTE -.-> UI
""",
    }
    for name, source in diagrams.items():
        Path("diagrams", name).write_text(
            "%% Architecture v2; see PLAN.md and implementation status.\n" + source,
            encoding="utf-8",
        )
    relations = """erDiagram
    users ||--o{ meal_logs : owns
    users ||--o{ prediction_history : requests
    model_versions o|--o{ prediction_history : audits
    meal_logs ||--|{ meal_items : contains
    prediction_history o|--o{ meal_items : referenced_by
    food_classes o|--o{ meal_items : confirmed_as
    food_classes ||--o{ food_nutrition_mappings : mapped_by
    nutrition_foods ||--o{ food_nutrition_mappings : matched_to
"""
    for table in Base.metadata.sorted_tables:
        relations += f"    {table.name} {{\n"
        for column in table.columns:
            kind = column.type.__visit_name__.lower()
            suffix = " PK" if column.primary_key else " FK" if column.foreign_keys else ""
            relations += f"        {kind} {column.name}{suffix}\n"
        relations += "    }\n"
    Path("diagrams/07-erd-postgresql.mmd").write_text(relations, encoding="utf-8")
    print("Updated FastAPI architecture sources and ERD from current ORM; no rendering claimed")


if __name__ == "__main__":
    main()
