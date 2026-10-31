# Food Recognition & Dietary Assistant

Bộ khung dự án theo [PLAN.md](PLAN.md), mục 15. MVP dùng Python 3.11,
Streamlit, PostgreSQL, SQLAlchemy/Alembic và pipeline ML offline.

Hiện đã tạo cấu trúc thư mục, module dành sẵn, cấu hình mẫu và năm trang
Streamlit thông báo trạng thái. Chưa triển khai nhận diện, CRUD nhật ký,
schema database hoặc huấn luyện. Chưa có kết quả ML và dữ liệu nutrition.

## Cấu trúc

```text
configs/                    Cấu hình dữ liệu, POC, ba mô hình và inference
src/food_assistant/
  app.py                    Điểm chạy Streamlit
  config.py                 Cấu hình môi trường và đường dẫn
  ui/pages/                 Home, Recognize, Diary, Dashboard, About
  ui/components/            Cropper, prediction card, nutrition editor
  domain/                   Enums và vị trí dành cho DTO/schema
  services/                 Image, inference, nutrition, meal, dashboard, registry
  ml/                       Dataset, transforms, features, training và evaluation
  persistence/              Database base, ORM, repository, seed
  utils/                    Image validation, logging, paths
scripts/                    Điểm vào pipeline; hiện chỉ có --help và báo chưa triển khai
migrations/                 Alembic scaffold; chưa có business schema/revision
tests/                      Vị trí dành cho unit, integration và smoke tests
data/                       Raw, processed, splits, local_eval, nutrition CSV
artifacts/                  Model releases, production pointer và báo cáo
storage/                    Uploads và tệp tạm (không commit)
diagrams/                   12 Mermaid source trích từ kế hoạch
docs/                       Thiết kế, checklist và tài liệu thí nghiệm
```

## Chạy giao diện khung trên PowerShell

Chạy từ thư mục chứa README này:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m streamlit run src/food_assistant/app.py
```

Các trang hiện chỉ thông báo trạng thái, chưa cần PostgreSQL hoặc model để mở.
Dependencies trong `pyproject.toml` là ràng buộc khởi đầu; chưa kiểm chứng toàn bộ
môi trường và chưa có lockfile. Khi làm ML, cài thêm `-e ".[ml,dev]"`;
cần chọn cặp Torch/Torchvision và bản CPU/CUDA phù hợp rồi khóa môi trường đã thử.

## Chuẩn bị PostgreSQL

Đổi `POSTGRES_PASSWORD` và mật khẩu tương ứng trong `DATABASE_URL` ở `.env`, rồi chạy:

```powershell
docker compose up -d postgres
docker compose ps
```

Compose chỉ bind PostgreSQL vào `127.0.0.1`. Database test phải tạo riêng theo
`TEST_DATABASE_URL`. Chưa chạy migration/seed ở bước bộ khung: Alembic sẽ báo
chưa triển khai schema, tránh báo thành công khi không có bảng nghiệp vụ.

## Kiểm tra và bước tiếp theo

```powershell
python scripts/smoke_test.py
```

Lệnh trên kiểm tra cấu trúc, cú pháp Python, TOML và các placeholder quan trọng;
không thay thế kiểm thử nghiệp vụ. Các tệp `tests/` chưa có test thực thi;
`pytest` hiện sẽ trả mã 5 vì không có test được thu thập.

Theo dõi [checklist triển khai](docs/tasks.md). Không tạo
`artifacts/production/current.json` cho đến khi có model được huấn luyện,
đánh giá và lựa chọn bằng kết quả thực. Nutrition CSV chỉ có header;
không điền kcal thiếu nguồn. Bộ sơ đồ là thiết kế dự kiến, chưa render SVG/PNG.

Năng lượng chỉ là ước tính theo khẩu phần/khối lượng tham khảo, không đo trực tiếp từ ảnh.
