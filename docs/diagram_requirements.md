# Diagram Requirements

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

## 23. BỘ SƠ ĐỒ MERMAID ĐỂ DIAGRAMMING AGENT DỰNG TRỰC TIẾP

### 23.0. Quy chuẩn xuất sơ đồ

**Thư mục đích:** `diagrams/`. Đối với mỗi `DD-title` dưới đây, xuất `DD-title.mmd` và, nếu render được, `DD-title.svg` hoặc `DD-title.png` với cùng nội dung. Có thể dùng draw.io/Figma/PlantUML để trình bày lại đẹp hơn nhưng **không thay đổi ý nghĩa kiến trúc**.

**Style:** sạch, tươi sáng, nhất quán; `UI` xanh nhạt, `AI` tím nhạt, `Database` xanh lá/xanh teal, `Storage` vàng nhạt, `External` xám, `Error/Warning` đỏ nhạt; không mã hóa ý nghĩa chỉ bằng màu. Thể hiện nhãn tiếng Việt và tên technical English khi cần.

**Nguyên tắc tính trung thực:** Sơ đồ Mermaid dưới đây là **đặc tả mục tiêu** của hệ thống dự kiến, chưa phải bằng chứng hệ thống đã xây dựng. Diagramming agent không thêm tỷ lệ accuracy mẫu, danh tính user, hoặc kcal giả định lên sơ đồ sản phẩm. Mọi cạnh biểu diễn một hướng dữ liệu hoặc thao tác hợp lệ. Nếu không chắc, ưu tiên sơ đồ mục tiêu đã nêu và ghi chú là `planned`.

**Danh mục 12 diagram bắt buộc cho bộ tài liệu:**

| ID | File | Diagram type | Trả lời câu hỏi |
|---|---|---|---|
| D01 | `01-system-context.mmd` | System context | Ai dùng hệ thống, dữ liệu tới từ đâu? |
| D02 | `02-container-mvp.mmd` | C4-like container flowchart | Thành phần runtime MVP nào tồn tại? |
| D03 | `03-component-mvp.mmd` | Component/service | Các service và ranh giới trách nhiệm? |
| D04 | `04-data-training.mmd` | Flowchart | Train/evaluate/select model ra sao? |
| D05 | `05-inference-sequence.mmd` | Sequence | Upload/crop/predict ra top-k như thế nào? |
| D06 | `06-meal-save-sequence.mmd` | Sequence | Confirm/estimate/save/rollback thế nào? |
| D07 | `07-erd-postgresql.mmd` | ER diagram | Các bảng, khóa ngoại, quan hệ? |
| D08 | `08-user-journey.mmd` | Flowchart | Người dùng đi từ ảnh đến dashboard? |
| D09 | `09-state-recognize.mmd` | State diagram | Các trạng thái và nhánh lỗi của màn hình? |
| D10 | `10-deployment.mmd` | Deployment flowchart | Chạy local có những node nào? |
| D11 | `11-roadmap.mmd` | Gantt | Phụ thuộc và lịch làm theo phase? |
| D12 | `12-future-architecture.mmd` | Future architecture | Kiến trúc sau MVP khác thế nào? |

### D01 — System Context

```mermaid
flowchart LR
    U["Người dùng cuối"]
    R["Developer / Researcher"]
    C["Nutrition Curator"]
    SYS["Food Recognition & Dietary Assistant<br/>MVP local/demo"]
    FOOD["Food-101<br/>dataset ảnh và nhãn"]
    NUT["Nguồn dinh dưỡng<br/>bảng tham khảo có xuất xứ"]

    U -->|"Upload, crop, xác nhận, lưu và xem bữa ăn"| SYS
    SYS -->|"Dự đoán món, kcal ước tính, nhật ký"| U
    R -->|"Train, evaluate, chọn model release"| SYS
    FOOD -->|"Ảnh/nhãn dùng cho offline training"| R
    NUT -->|"Dữ liệu tham chiếu kcal/100g"| C
    C -->|"Review nutrition mapping/portion"| SYS
```

**Không có mũi tên Food-101 → inference request trực tiếp:** training là quy trình offline; runtime sử dụng trained artifact.

### D02 — Container Architecture MVP

```mermaid
flowchart TB
    USER["User / Browser"]
    subgraph APP["Python Application - Modular Monolith"]
        UI["Streamlit UI<br/>Home / Recognize / Diary / Dashboard / About"]
        SVC["Application Services<br/>Image / Inference / Nutrition / Meal / Dashboard"]
        ORM["Repositories + SQLAlchemy<br/>Transaction / Unit of Work"]
        PRED["PyTorch Runtime<br/>EfficientNet-B0 or selected model"]
    end
    DB[("PostgreSQL<br/>App records")]
    FS[("Filesystem<br/>Model artifacts + optional user uploads")]
    OFF["Offline pipeline<br/>Food-101 → train → evaluate"]

    USER <-->|"Streamlit interaction"| UI
    UI -->|"Internal Python calls"| SVC
    SVC -->|"CRUD and queries"| ORM
    ORM <-->|"SQL"| DB
    SVC -->|"Classify crop"| PRED
    PRED -->|"Read checkpoint + label map"| FS
    OFF -->|"Publish approved model release"| FS
    SVC -->|"Temporary file read/write if enabled"| FS
```

**Ghi chú:** Không có FastAPI, React, remote LLM, Kafka hoặc YOLO trong sơ đồ này.

### D03 — Component Diagram MVP

```mermaid
flowchart LR
    UI["Streamlit Pages"]
    IMG["ImageService"]
    CROP["CropService"]
    INF["InferenceService"]
    MODEL["ModelRegistry"]
    NUT["NutritionService"]
    MEAL["MealService"]
    DASH["DashboardService"]
    REPO["Repository Layer"]
    DB[("PostgreSQL")]
    ART[("Production Artifact")]

    UI --> IMG
    UI --> CROP
    UI --> INF
    UI --> NUT
    UI --> MEAL
    UI --> DASH
    IMG --> CROP
    INF --> MODEL
    MODEL --> ART
    NUT --> REPO
    MEAL --> NUT
    MEAL --> REPO
    DASH --> REPO
    REPO --> DB
```

**Rule:** `InferenceService` trả top-k chứ không ghi thẳng meal log. Chỉ `MealService` xử lý save sau user confirmation.

### D04 — Training and Model Selection Pipeline

```mermaid
flowchart TD
    A["Food-101 original<br/>train / test"] --> B["Validate images, labels, counts"]
    B --> C["Select POC classes or 101 full"]
    C --> D["Stratified split original train<br/>train_internal / val_internal"]
    D --> L["Locked test from original test"]
    D --> E["Shared train/val manifest"]

    E --> F1["HOG extraction<br/>train Linear SVM"]
    E --> F2["ResNet18 ImageNet weights<br/>replace head + fine-tune"]
    E --> F3["EfficientNet-B0 ImageNet weights<br/>replace head + fine-tune"]

    F1 --> V1["Tune with validation"]
    F2 --> V2["Checkpoint by validation"]
    F3 --> V3["Checkpoint by validation"]

    V1 --> G["Freeze candidate settings"]
    V2 --> G
    V3 --> G
    L --> H["Final test comparison"]
    G --> H
    H --> I["Top-1, macro-F1, confusion matrix<br/>latency, artifact size"]
    I --> J{"Select production model<br/>on measured evidence"}
    J --> K["Publish model + config + label map"]
    K --> M["artifacts/production/current.json"]

    D -.->|"Test never used for tuning"| H
```

**Bắt buộc:** Giữ `test` độc lập đến bước H; POC subset và full 101 là **hai thí nghiệm khác nhau**.

### D05 — Inference Sequence (không lưu bữa ăn)

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit Recognize
    participant IS as ImageService
    participant CS as CropService
    participant IN as InferenceService
    participant MR as ModelRegistry
    participant FS as Model Artifact FS

    U->>UI: Upload JPG/PNG
    UI->>IS: validate_and_decode(bytes)
    alt Invalid/too large image
        IS-->>UI: INVALID_IMAGE / IMAGE_TOO_LARGE
        UI-->>U: Hiển thị lỗi, yêu cầu ảnh khác
    else Valid image
        IS-->>UI: PIL RGB image
        UI-->>U: Preview and crop controls
        opt User selects crop
            U->>UI: Choose crop box
            UI->>CS: crop(image, normalized_box)
            CS-->>UI: Cropped RGB
        end
        U->>UI: Click Analyze
        UI->>IN: predict(RGB image/crop, top_k=3)
        IN->>MR: load_production_model()
        MR->>FS: Read model, preprocess, label mapping
        FS-->>MR: Trusted release files
        MR-->>IN: Model + transforms + labels
        alt Missing/inconsistent model
            IN-->>UI: MODEL_NOT_READY / ARTIFACT_MISMATCH
            UI-->>U: Báo lỗi, không tự tạo kết quả
        else Model ready
            Note over IN: eval + inference_mode<br/>resize/normalize match artifact
            IN-->>UI: Top-3 scores, model run_id, uncertain flag
            UI-->>U: Show predictions; require confirmation
        end
    end
```

**Note:** `ModelRegistry` có thể cache, không mở checkpoint từ disk trên mọi request. Sơ đồ mô tả dependency logic, không bắt buộc disk read lại mỗi lần.

### D06 — Nutrition Estimation and Meal Save Sequence

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit Recognize
    participant NS as NutritionService
    participant MS as MealService
    participant RP as Repositories
    participant DB as PostgreSQL

    U->>UI: Chọn/sửa món đã dự đoán
    U->>UI: Chọn S/M/L hoặc grams
    UI->>NS: lookup_and_estimate(confirmed_food, portion)
    NS->>RP: Find reviewed default nutrition profile
    RP->>DB: SELECT nutrition profile and mapping
    DB-->>RP: Profile or none
    RP-->>NS: Profile or none
    alt No reviewed source / missing grams
        NS-->>UI: kcal unavailable, request manual grams
        UI-->>U: Thông báo giới hạn, cho chỉnh tay
    else Valid reference + grams
        Note over NS: estimated_kcal = kcal_per_100g × grams / 100
        NS-->>UI: Estimated kcal + provenance + assumptions
        UI-->>U: Hiển thị kcal ước tính và nguồn
    end
    U->>UI: Confirm and Save
    UI->>MS: create_meal_with_items(validated_draft)
    MS->>RP: Begin transaction
    RP->>DB: INSERT meal_log and meal_items
    alt DB error
        DB-->>RP: Error
        RP->>DB: ROLLBACK
        RP-->>MS: SAVE_FAILED
        MS-->>UI: Error
        UI-->>U: Chưa lưu, có thể thử lại
    else Commit success
        RP->>DB: COMMIT
        DB-->>RP: Committed
        RP-->>MS: Saved IDs
        MS-->>UI: Success + IDs
        UI-->>U: Đã lưu, có thể xem Diary
    end
```

**Invariant:** Khi kcal không có nguồn thì vẫn lưu được `estimated_kcal=NULL`; `manual` kcal phải có dấu vết người dùng nhập.

### D07 — PostgreSQL ER Diagram

```mermaid
erDiagram
    users ||--o{ meal_logs : owns
    users ||--o{ prediction_history : requests
    model_versions ||--o{ prediction_history : generates
    meal_logs ||--|{ meal_items : contains
    prediction_history o|--o| meal_items : referenced_by
    food_classes o|--o{ meal_items : confirmed_as
    food_classes ||--o{ food_nutrition_mappings : mapped_by
    nutrition_foods ||--o{ food_nutrition_mappings : matched_to

    users {
        uuid id PK
        string display_name
        string email UK
        timestamptz created_at
    }
    food_classes {
        uuid id PK
        string dataset_key UK
        string display_name_en
        string display_name_vi
        boolean is_active
        timestamptz created_at
    }
    nutrition_foods {
        uuid id PK
        string name
        decimal kcal_per_100g
        string source_name
        string source_record_id
        string provenance_url
        timestamptz retrieved_at
        decimal small_grams
        decimal medium_grams
        decimal large_grams
        boolean reviewed
        string mapping_note
    }
    food_nutrition_mappings {
        uuid id PK
        uuid food_class_id FK
        uuid nutrition_food_id FK
        boolean is_default
        string confidence_note
    }
    model_versions {
        uuid id PK
        string model_name
        string run_id UK
        int num_classes
        string artifact_dir
        json metrics_json
        boolean is_production
    }
    prediction_history {
        uuid id PK
        uuid user_id FK
        uuid model_version_id FK
        string image_path
        string image_sha256
        json crop_box_json
        json topk_json
        decimal top1_score
        boolean uncertain
        timestamptz created_at
    }
    meal_logs {
        uuid id PK
        uuid user_id FK
        date consumed_on
        string meal_type
        string notes
        timestamptz created_at
        timestamptz updated_at
    }
    meal_items {
        uuid id PK
        uuid meal_log_id FK
        uuid confirmed_food_class_id FK
        uuid prediction_id FK
        string custom_food_name
        string portion_size
        decimal estimated_grams
        decimal kcal_per_100g_snapshot
        decimal estimated_kcal
        string nutrition_source_snapshot
        string calculation_method
        boolean is_user_corrected
        timestamptz created_at
        timestamptz updated_at
    }
```

**ERD note:** `model_versions` quản lý release; `class_to_idx` *thuộc model artifact* chứ không phải bảng nghiệp vụ. `prediction_history` và `confirmed_food_class_id` đều có thể NULL theo nghiệp vụ. Mermaid `||--|{ meal_items` biểu thị bản ghi bữa ăn có thể có từ một món sau khi commit; ứng dụng không lưu meal log rỗng cuối transaction. Diagramming agent phải review optionality khi triển khai ORM.

### D08 — User Journey / Activity Flow

```mermaid
flowchart TD
    START(["Open Recognize"]) --> UP["Upload image"]
    UP --> VAL{"Valid image?"}
    VAL -- No --> ERR["Show error and retry"]
    ERR --> UP
    VAL -- Yes --> PREV["Preview / optional manual crop"]
    PREV --> ANA["Analyze crop with trained model"]
    ANA --> READY{"Model available?"}
    READY -- No --> MODELERR["Show model setup error"]
    READY -- Yes --> TOP["Display top-3 + raw scores"]
    TOP --> CONF["User confirms/corrects food"]
    CONF --> PORT["Choose portion S/M/L or custom gram"]
    PORT --> NUT{"Reviewed nutrition mapping and grams?"}
    NUT -- Yes --> KC["Calculate estimated kcal + source"]
    NUT -- No --> UNK["Kcal unavailable / manual entry"]
    KC --> META["Choose date + meal type"]
    UNK --> META
    META --> SAVE["Confirm and save"]
    SAVE --> DBS{"DB committed?"}
    DBS -- No --> DBERR["Show save failed, retry"]
    DBERR --> SAVE
    DBS -- Yes --> DIARY["Meal Diary"]
    DIARY --> DASH["Dashboard"]
    DIARY --> MORE{"Another dish in same photo?"}
    MORE -- Yes --> PREV
    MORE -- No --> END(["Finish"])
    DASH --> END
```

### D09 — Recognize Screen State Machine

```mermaid
stateDiagram-v2
    [*] --> EMPTY
    EMPTY --> IMAGE_READY: upload valid image
    EMPTY --> ERROR: invalid image
    IMAGE_READY --> ANALYZING: click Analyze
    IMAGE_READY --> EMPTY: clear image
    ANALYZING --> RESULT_READY: model returns top-k
    ANALYZING --> ERROR: model/inference error
    RESULT_READY --> NEEDS_CONFIRMATION: low score or unknown
    RESULT_READY --> RESULT_READY: user edits label/portion
    NEEDS_CONFIRMATION --> RESULT_READY: user confirms corrected food
    RESULT_READY --> SAVED: DB commit success
    RESULT_READY --> ERROR: save error
    RESULT_READY --> IMAGE_READY: change image or crop
    SAVED --> IMAGE_READY: analyze another crop
    SAVED --> EMPTY: clear/reset
    ERROR --> IMAGE_READY: retry with valid image
    ERROR --> EMPTY: reset
```

**Quy tắc:** `RESULT_READY → IMAGE_READY` khi thay crop để bỏ kết quả model trước; `SAVED` chỉ sau PostgreSQL commit.

### D10 — Local Deployment Diagram

```mermaid
flowchart TB
    B["Web Browser<br/>http://localhost:8501"]
    subgraph HOST["Developer Laptop / Local Machine"]
        ST["Python venv<br/>Streamlit app & services"]
        RT["PyTorch inference<br/>CPU or CUDA device"]
        FILES[("Local Storage<br/>artifacts + optional uploads")]
        subgraph DBENV["PostgreSQL Environment"]
            PG[("PostgreSQL<br/>port 5432 - local/internal")]
            VOL[("Persistent DB volume/data")]
        end
        TRAIN["Offline training scripts<br/>CPU/GPU or external Colab"]
    end
    B <-->|"User interactions"| ST
    ST --> RT
    RT -->|"Load release"| FILES
    ST <-->|"SQLAlchemy / psycopg"| PG
    PG --> VOL
    TRAIN -->|"Release trained model to file storage"| FILES

    NOTES["No public access, no auth in single-user MVP"]
    ST -.-> NOTES
```

**Nếu training trên Colab**, diễn giải `TRAIN` là máy/cloud khác; chỉ copy release artifacts cần thiết sang host demo và xác minh checksum. PostgreSQL có thể là service local hoặc Compose container, không vẽ một DB cloud mặc định.

### D11 — Roadmap Gantt

```mermaid
gantt
    title Project 18 - 10 week implementation plan (proposed)
    dateFormat YYYY-MM-DD
    axisFormat %d/%m
    section Foundation
    Setup and architecture          :a1, 2026-11-02, 7d
    Dataset and EDA                :a2, after a1, 7d
    section ML Experiments
    HOG and SVM baseline           :a3, after a2, 7d
    ResNet18 and EfficientNet      :a4, after a3, 14d
    section App and Database
    PostgreSQL and nutrition       :a5, after a3, 7d
    Streamlit MVP                  :a6, after a4, 7d
    section Validation
    Robustness and error analysis  :a7, after a6, 7d
    Report diagrams and delivery   :a8, after a7, 14d
```

**Note:** Mốc 2026-11-02 chỉ là điểm bắt đầu minh họa sau ngày viết tài liệu, **không phải deadline đã được thầy xác nhận**. Agent có thể vẽ P4 Database song song CNN training; sơ đồ Gantt trên mô tả dependency tối thiểu và mang tính tham khảo.

### D12 — Future Architecture (KHÔNG PHẢI MVP)

```mermaid
flowchart LR
    USER["Web / Mobile Users"]
    FE["React or Mobile Client<br/>Future"]
    API["FastAPI REST<br/>Future"]
    AUTH["Auth & Authorization<br/>Future"]
    SV["Application Services"]
    AI["Model Inference Runtime"]
    DB[("PostgreSQL")]
    OBJ[("Object Storage<br/>Future")]
    OBS["Monitoring / Audit<br/>Future"]
    USER --> FE
    FE -->|"HTTPS / JSON / multipart"| API
    API --> AUTH
    API --> SV
    SV --> AI
    SV --> DB
    SV --> OBJ
    API --> OBS
    AI --> OBS
```

**Cảnh báo in đậm trong output:** `FUTURE ARCHITECTURE — NOT IMPLEMENTED IN MVP`. Không vẽ YOLO/segmentation trong sơ đồ tương lai như tính năng đã sẵn có nếu chưa có dữ liệu detection/segmentation.

---

## 24. MAPPING SƠ ĐỒ ↔ MODULE ↔ TEST

| Diagram | Modules | Tests/checks |
|---|---|---|
| D01 context | All, data sources | Coverage yêu cầu đề |
| D02 container | UI, services, persistence, ML | Smoke end-to-end |
| D03 component | Python services/repo | Unit/Integration contract |
| D04 training | `scripts/train_*`, `ml/*` | Split leakage, metrics validity |
| D05 inference sequence | `recognize`, `image_service`, `inference_service` | UT-01..08, E2E-01 |
| D06 meal save sequence | `nutrition_service`, `meal_service`, ORM | UT-09..14, IT-03..10 |
| D07 ERD | `orm_models.py`, migrations | IT-01, IT-02, referential tests |
| D08 user journey | UI and business services | E2E all scenarios |
| D09 UI state | Streamlit `st.session_state` | image-change, retry, save cases |
| D10 deployment | `.env`, Compose, app startup | Fresh clone smoke test |
| D11 roadmap | Phase/task list | Phases exit criteria |
| D12 future architecture | No MVP modules | Explicitly labeled Future |

## 25. CHECKLIST RÀ SOÁT SƠ ĐỒ CỦA AGENT

- [ ] Đủ 12 diagram names như mục 23.0.
- [ ] Mermaid source syntactically valid hoặc nêu chính xác lỗi chưa render.
- [ ] Trên container diagram, app là modular monolith với Streamlit + Python services.
- [ ] `Food-101` chỉ nằm trong training/data pipeline; runtime đọc artifact.
- [ ] `EfficientNet-B0` là candidate, **không** tự tuyên bố là winner.
- [ ] Có cả ResNet18 và HOG+SVM trong comparison.
- [ ] DB là PostgreSQL, ORM SQLAlchemy, migration Alembic.
- [ ] Nutrition data được map có nguồn; không AI estimate gram chính xác từ 1 ảnh.
- [ ] Có bước user correction trước khi lưu.
- [ ] Một ảnh/crop → một class chính; multi-dish bằng manual crop.
- [ ] Mọi meal save chạy transaction và rollback khi lỗi.
- [ ] ERD chính xác FK, nullable và default mapping.
- [ ] Ký hiệu nguồn external/dataset/model artifact khác database.
- [ ] Future architecture được tách biệt, gắn nhãn Future/Not in MVP.
- [ ] Không có số accuracy/kcal/runs giả trong sơ đồ kết quả.
- [ ] Mỗi sơ đồ có tiêu đề, legend nếu cần, tên file và note quan trọng.

---
