# Data Dictionary

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

## 12. DATABASE POSTGRESQL — ENTITY, COLUMN, RULE

### 12.1. Nguyên tắc

- Dùng PostgreSQL làm DB duy nhất cho ứng dụng nghiệp vụ.
- Tạo schema qua SQLAlchemy ORM + Alembic migrations.
- Mọi ID nghiệp vụ là UUID; có thể dùng `uuid.uuid4()` từ Python cho tính tương thích.
- Timestamps dùng `TIMESTAMP WITH TIME ZONE`; phân biệt `created_at`, `updated_at`.
- `NUMERIC(10,2)` cho gram/kcal, hạn chế lỗi tiền số float trong thống kê đơn giản.
- DB lưu đường dẫn ảnh tương đối, metadata, không lưu binary toàn bộ Food-101 vào DB.
- **Single-user MVP:** tạo `demo_user` cố định qua seed; `user_id` trong `meal_logs` để chuẩn bị multi-user nhưng chưa có tính năng đăng nhập. Không tuyên bố dữ liệu đã cách ly an toàn giữa nhiều user.
- `prediction_history` là lịch sử inference (nếu người dùng đồng ý lưu); `meal_items` là bản ghi đã xác nhận. Không nhầm prediction chưa xác nhận với món đã ăn.

### 12.2. Bảng `users`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | User ID |
| display_name | VARCHAR(120) | NOT NULL | Demo user name |
| email | VARCHAR(255) | NULL, UNIQUE | Không bắt buộc MVP |
| created_at | TIMESTAMPTZ | NOT NULL | Created UTC |

### 12.3. Bảng `food_classes`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Food class UUID |
| dataset_key | VARCHAR(150) | UNIQUE, NOT NULL | `pizza`, `sushi`, ... |
| display_name_en | VARCHAR(200) | NOT NULL | Food class English |
| display_name_vi | VARCHAR(200) | NULL | Tên hiển thị Việt |
| is_active | BOOLEAN | DEFAULT true | Cho phép chọn trong UI |
| created_at | TIMESTAMPTZ | NOT NULL | Created |

**Không** đặt `class_idx` trên `food_classes` như chân lý độc lập; class index thuộc **model release**, có thể đổi khi train lại. Lấy từ `model_class_labels` hoặc artifact mapping.

### 12.4. Bảng `nutrition_foods`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Profile UUID |
| name | VARCHAR(200) | NOT NULL | Tên mục dinh dưỡng |
| kcal_per_100g | NUMERIC(10,2) | NULL, CHECK >= 0 | Kcal tham khảo |
| source_name | VARCHAR(200) | NOT NULL | Ví dụ USDA FDC |
| source_record_id | VARCHAR(120) | NULL | External FDC ID nếu có |
| provenance_url | TEXT | NULL | Link nguồn |
| retrieved_at | TIMESTAMPTZ | NULL | Ngày truy xuất |
| small_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định cho S |
| medium_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định cho M |
| large_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định cho L |
| mapping_note | TEXT | NULL | Giả định/công thức |
| reviewed | BOOLEAN | DEFAULT false | Đã review mapping/nguồn |
| created_at | TIMESTAMPTZ | NOT NULL | Created |
| updated_at | TIMESTAMPTZ | NOT NULL | Last updated |

### 12.5. Bảng `food_nutrition_mappings`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Mapping UUID |
| food_class_id | UUID | FK → food_classes | Class của model |
| nutrition_food_id | UUID | FK → nutrition_foods | Profile lựa chọn |
| is_default | BOOLEAN | NOT NULL DEFAULT false | Profile mặc định |
| confidence_note | TEXT | NULL | Ghi chú mapping |
| created_at | TIMESTAMPTZ | NOT NULL | Created |

Unique `(food_class_id, nutrition_food_id)`; tạo **partial unique index** đảm bảo tối đa một row `is_default=true` mỗi `food_class_id`. Chỉ sử dụng auto lookup khi `nutrition_foods.reviewed=true` và `kcal_per_100g IS NOT NULL`.

### 12.6. Bảng `model_versions`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Release UUID |
| model_name | VARCHAR(80) | NOT NULL | `resnet18`, `efficientnet_b0`, `hog_svm` |
| run_id | VARCHAR(100) | UNIQUE, NOT NULL | Training run |
| num_classes | INT | CHECK > 0 | Số class |
| artifact_dir | TEXT | NOT NULL | Artifact local path |
| metrics_json | JSONB | NULL | Các metrics đã công bố |
| is_production | BOOLEAN | NOT NULL DEFAULT false | Model chạy demo |
| created_at | TIMESTAMPTZ | NOT NULL | Created |

Chỉ tối đa một production model trong DB (`partial unique index` trên constant `is_production=true`), đồng bộ với production pointer file theo thủ tục release. **MVP ưu tiên pointer file là nguồn model runtime**, DB chỉ để audit.

### 12.7. Bảng `prediction_history`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Prediction event |
| user_id | UUID | FK → users | Demo user |
| model_version_id | UUID | FK → model_versions | Provenance |
| image_path | TEXT | NULL | Nếu user cho phép lưu ảnh |
| image_sha256 | VARCHAR(64) | NULL | Optional fingerprint |
| crop_box_json | JSONB | NULL | Normalized crop coordinates |
| topk_json | JSONB | NOT NULL | `[{dataset_key, score}, ...]` |
| top1_score | NUMERIC(7,6) | NULL | Raw softmax score |
| uncertain | BOOLEAN | NOT NULL DEFAULT false | Theo threshold calibrated |
| created_at | TIMESTAMPTZ | NOT NULL | Inference time |

`prediction_history` không bắt buộc ghi với mỗi infer; có thể thiết lập `SAVE_PREDICTION_HISTORY=false` mặc định để hạn chế lưu ảnh/data. Nếu lưu prediction, yêu cầu về privacy áp dụng cho cả metadata.

### 12.8. Bảng `meal_logs`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Meal log UUID |
| user_id | UUID | FK → users | Owner |
| consumed_on | DATE | NOT NULL | Ngày ăn (địa phương) |
| meal_type | VARCHAR(20) | CHECK enum values | breakfast/lunch/dinner/snack |
| notes | TEXT | NULL | Ghi chú bữa |
| created_at | TIMESTAMPTZ | NOT NULL | Created |
| updated_at | TIMESTAMPTZ | NOT NULL | Updated |

Một meal log có nhiều `meal_items`. Có thể tạo nhiều logs cùng `consumed_on + meal_type` nếu thực tế ăn nhiều bữa phụ; không unique cặp ngày/bữa.

### 12.9. Bảng `meal_items`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Meal item UUID |
| meal_log_id | UUID | FK → meal_logs (ON DELETE CASCADE) | Bữa chứa item |
| confirmed_food_class_id | UUID | FK → food_classes, NULL | Class đã xác nhận |
| custom_food_name | VARCHAR(200) | NULL | Món ngoài danh sách |
| prediction_id | UUID | FK → prediction_history, NULL | Liên kết dự đoán gốc |
| portion_size | VARCHAR(20) | CHECK S/M/L/custom | Loại portion |
| estimated_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định/nhập tay |
| kcal_per_100g_snapshot | NUMERIC(10,2) | NULL, CHECK >= 0 | Kcal tham chiếu tại lúc lưu |
| estimated_kcal | NUMERIC(10,2) | NULL, CHECK >= 0 | Tổng kcal ước tính |
| nutrition_source_snapshot | VARCHAR(255) | NULL | Nguồn tại lúc lưu |
| calculation_method | VARCHAR(30) | NOT NULL | reference_formula/manual/unknown |
| is_user_corrected | BOOLEAN | NOT NULL DEFAULT false | Đã sửa nhãn AI |
| notes | TEXT | NULL | Ghi chú món |
| created_at | TIMESTAMPTZ | NOT NULL | Created |
| updated_at | TIMESTAMPTZ | NOT NULL | Updated |

Check `(confirmed_food_class_id IS NOT NULL) OR (custom_food_name IS NOT NULL)`; `estimated_grams` có thể NULL nếu chưa có gram đáng tin. `estimated_kcal` chỉ tính được khi đủ dữ liệu; custom manual kcal vẫn được nhưng phải lưu `calculation_method='manual'`.

### 12.10. Quan hệ và thứ tự tạo bảng

1. `users`, `food_classes`, `nutrition_foods`, `model_versions`.
2. `food_nutrition_mappings`.
3. `prediction_history`, `meal_logs`.
4. `meal_items`.
5. Indexes, seed và constraints.

### 12.11. Indexes

- `users(email)` unique nếu email không null.
- `food_classes(dataset_key)` unique.
- `model_versions(run_id)` unique.
- `meal_logs(user_id, consumed_on DESC)`.
- `meal_items(meal_log_id)`.
- `prediction_history(user_id, created_at DESC)`.
- `food_nutrition_mappings(food_class_id)`.
- Partial unique indexes cho default nutrition mapping và production model như mục trên.

### 12.12. Migrations và dữ liệu seed

- `alembic revision --autogenerate ...` rồi review migration, `alembic upgrade head`.
- Seed idempotent: demo user; food_classes lấy từ manifest/class mapping, nutrition references đã có nguồn, default mapping đã duyệt.
- Không seed toàn bộ 101 món với kcal tự đoán. Cho lớp không mapping → nutrition unavailable.
- Test DB riêng; không chạy test làm xóa database thực.

---
