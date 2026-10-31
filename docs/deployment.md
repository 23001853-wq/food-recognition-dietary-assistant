# Deployment

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

## 16. ENVIRONMENT VÀ HƯỚNG DẪN TRIỂN KHAI KỸ THUẬT

### 16.1. Local development

1. Cài Python 3.11, Git, PostgreSQL (hoặc Docker Desktop nếu dùng Compose), VS Code.
2. Tạo virtual environment.
3. Cài dependency từ `pyproject.toml`/lockfile sau khi agent tạo.
4. Copy `.env.example` thành `.env`, đặt `DATABASE_URL`.
5. Start PostgreSQL; kiểm tra readiness.
6. `alembic upgrade head` để tạo schema.
7. Chạy seed thực phẩm và demo user (chỉ nguồn đáng tin).
8. Tải dataset Food-101 vào `data/raw/`.
9. Tạo manifest bằng `create_splits.py`; chạy kiểm tra leakage.
10. Train baseline và CNN (trên local GPU hoặc Colab).
11. Evaluate và đặt production model pointer.
12. Chạy `streamlit run src/food_assistant/app.py` **sau khi đảm bảo package import path hoạt động** (nên cài project ở editable mode).
13. Upload ảnh và demo đủ luồng.

### 16.2. Ví dụ `.env.example` (không để password thật)

```dotenv
APP_ENV=development
DATABASE_URL=postgresql+psycopg://food_user:CHANGE_ME@localhost:5432/food_assistant
APP_TIMEZONE=Asia/Ho_Chi_Minh
MODEL_POINTER=artifacts/production/current.json
UPLOAD_DIR=storage/uploads
TEMP_DIR=storage/temp
MAX_UPLOAD_MB=10
MAX_IMAGE_PIXELS=20000000
SAVE_UPLOADED_IMAGES=false
SAVE_PREDICTION_HISTORY=false
INFERENCE_DEVICE=auto
```

**Lưu ý:** `DATABASE_URL` trên chỉ là mẫu. Phải đặt secret thật riêng, không commit. `INFERENCE_DEVICE=auto` chọn CUDA nếu khả dụng, nếu không dùng CPU; code phải bắt lỗi GPU thiếu RAM và có fallback/notification hợp lý, không âm thầm đổi model.

### 16.3. Docker Compose local

- 01 PostgreSQL service `postgres`; optional Adminer/pgAdmin chỉ ở môi trường dev (không bắt buộc).
- Mount named volume cho DB, healthcheck, không dùng mật khẩu mẫu thực tế.
- App Streamlit có thể chạy host venv; nếu containerize Streamlit, `DATABASE_URL` phải dùng hostname service `postgres` thay vì localhost.
- Có quy trình migration và backup trước deploy, không phụ thuộc DB GUI.

### 16.4. Training on Google Colab (khi không có NVIDIA GPU)

- Mount Drive hoặc tải dataset theo nguồn đã duyệt; không xáo trộn split.
- Đặt seed, class config, source commit, package versions.
- Tạo checkpoints và metric trong run folder.
- Download/sync **artifact đã tạo** sang máy chạy Streamlit; kiểm tra SHA256.
- Đo latency inference **trên máy demo**, tách khỏi tốc độ train của Colab.

---
