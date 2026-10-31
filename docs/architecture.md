# Architecture

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

## 11. KIẾN TRÚC HỆ THỐNG

### 11.1. Kiến trúc MVP chuẩn — Modular Monolith

```text
┌───────────────────────────────────────────────────────────────┐
│ Browser                                                       │
│  Streamlit UI: Recognize / Diary / Dashboard / About          │
└───────────────────────────┬───────────────────────────────────┘
                            │ internal Python calls
                            v
┌───────────────────────────────────────────────────────────────┐
│ Application Service Layer (Python)                            │
│  ImageService | InferenceService | NutritionService            │
│  MealService  | DashboardService | ModelRegistry              │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
                v                               v
┌──────────────────────────────┐    ┌───────────────────────────┐
│ AI Runtime                   │    │ PostgreSQL                │
│ production artifact + PyTorch│    │ meal, classes, nutrition  │
└──────────────────────────────┘    └───────────────────────────┘
                │
                v
┌──────────────────────────────┐
│ Local file storage           │
│ image uploads / artifacts    │
└──────────────────────────────┘

Separate offline pipeline:
Food-101 → split/preprocess → train 3 methods → evaluate → model artifacts
                                               |
                                               v
                                  artifacts/production/current.json
```

**Đặc điểm:** Streamlit chỉ điều phối presentation, gọi trực tiếp service Python; không có FastAPI/HTTP nội bộ trong MVP. SQLAlchemy nằm sau repository layer. InferenceService chỉ đọc model artifact và class mapping, không truy cập train dataset khi xử lý yêu cầu.

### 11.2. Module dependencies

```text
ui/pages  -> services         -> repositories -> SQLAlchemy/PostgreSQL
           -> inference       -> model_loader -> model artifacts
           -> image_processing-> Pillow/OpenCV
           -> nutrition       -> repositories
           -> dashboard       -> repositories
training/scripts -> datasets/transforms/models -> checkpoints & reports
```

- UI không gọi `torch.load`, `session.execute`, hoặc truy vấn SQL trực tiếp.
- `training/` không import Streamlit.
- Inference module không được sửa meal records.
- Unit test service chạy với fake repositories; integration test chạy PostgreSQL test DB.
- Có một session DB **trên mỗi transaction**, không giữ shared mutable Session trong cache toàn cục.

### 11.3. Tương lai (không thuộc MVP)

`React/Mobile → FastAPI (REST) → Application Services → PyTorch inference + PostgreSQL + Object Storage`. Cần authentication, authorization, API versioning, rate limiting, file retention và deployment đa instance. **Không** dùng sơ đồ này làm hiện trạng v1.0.

### 11.4. Runtime deployment MVP

- Windows 11 / Linux host; Python virtual environment chạy Streamlit.
- PostgreSQL chạy local service hoặc Docker container (Docker Compose được khuyên dùng).
- `models/` và `artifacts/` là thư mục local có quyền đọc; uploads nằm ở thư mục riêng không nằm trong web root.
- `.env` chứa `DATABASE_URL`, storage config, timezone, model pointer; không commit credentials.
- Cấu hình port Streamlit mặc định 8501, PostgreSQL mặc định 5432 (có thể đổi); DB không phơi công khai qua Internet ở demo.
- Single-user local/demo; nếu deploy public cần thêm authentication và privacy controls trước.

---
