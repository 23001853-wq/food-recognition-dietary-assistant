# Food Recognition & Dietary Assistant

**Tiếp tục từ trạng thái hiện tại:** đọc [RUN_COMMANDS.md](RUN_COMMANDS.md).
Tác vụ dài được bạn chủ động chạy từng bước trong Terminal; script tổng hợp mặc định chỉ in lệnh.

**POC AI thật hiện tại:** production `efficientnet_b0/poc10-rtx3050-v1` đã có checkpoint
và kết quả test được lưu (Accuracy 89.80%). Phiên bổ sung Nutrition không huấn luyện
hoặc đánh giá lại model. Hướng dẫn training độc lập giữ tại [POC_REAL_AI.md](POC_REAL_AI.md).

**Dinh dưỡng tự động cho 10 lớp:** xem [NUTRITION_GUIDE.md](NUTRITION_GUIDE.md) để
nạp catalog USDA, khởi động lại ứng dụng và kiểm tra Hamburger/Kem/Pizza. Gram S/M/L
là quy ước thiết kế có ghi chú; không đo khối lượng bằng ảnh. Không cần migration mới.

Ứng dụng local một người dùng: **Streamlit → FastAPI → PostgreSQL**, nhận ảnh/crop,
xác nhận món và khẩu phần, ước tính dinh dưỡng có nguồn, lưu nhật ký và xem dashboard.
Pipeline HOG + SVM, ResNet18 và EfficientNet-B0 chạy offline, tách khỏi ứng dụng.

**Trạng thái thực tế:** đã chạy end-to-end với Mock AI trên Windows và Docker Compose;
cả ba container healthy, HTTP smoke đã đạt. Food-101 đã tải đủ 101.000 ảnh, POC 10 lớp
đã tạo manifest 6.000 train / 1.500 val / 2.500 test. Trạng thái huấn luyện và kết quả
thật được cập nhật tại [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) và
[MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md). Mock và smoke training
không được dùng làm kết quả thực nghiệm.

## Chạy nhanh bằng Docker

Từ thư mục chứa README này, tạo `.env` từ `.env.example` nếu chưa có, thay mật khẩu
và giữ mật khẩu URL khớp `POSTGRES_PASSWORD`. Không ghi đè `.env` đang sử dụng.

```powershell
docker compose up -d --build
docker compose ps
```

- Giao diện: http://127.0.0.1:8501
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Trong chế độ phát triển, chọn **mock** ở trang Nhận diện. Tải ảnh JPEG/PNG,
crop một món nếu cần, phân tích, xác nhận/sửa món, chọn portion/gram rồi lưu.
Cả 10 lớp POC có thực phẩm USDA tham chiếu; món ngoài danh mục chưa có nguồn hiển thị
“Chưa có dữ liệu”. Các công thức khác nhau có thể khác dinh dưỡng; xem ghi chú ánh xạ.
Trang Nhật ký hỗ trợ thêm thủ công, sửa/xóa có xác nhận; dashboard lấy dữ liệu thật.

Image mặc định dành cho mock/runtime nhẹ. Để thêm CPU real inference vào Docker:

```powershell
docker compose build --build-arg INSTALL_ML=true backend
docker compose up -d
```

Biến thể ML chỉ được xác nhận chạy khi có ghi nhận kiểm thử tương ứng trong báo cáo.
`APP_ENV=production` yêu cầu `ALLOW_MOCK_AI=false` và model release thật đã kiểm chứng.

## Chạy backend/frontend bằng PowerShell

Python 3.11. Tạo venv riêng, cài dependencies theo lock phù hợp:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -c requirements-lock.txt -e ".[ml,dev]"
docker compose up -d postgres
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe scripts/seed_database.py
.venv/Scripts/python.exe -m uvicorn food_assistant.api.main:app --host 127.0.0.1 --port 8000
# Terminal khác:
.venv/Scripts/python.exe -m streamlit run src/food_assistant/app.py
```

Dừng backend/frontend container trước khi dùng cùng port trên host. Có thể dùng
PostgreSQL cài trực tiếp trên Windows bằng DATABASE_URL tương ứng; không thay SQLite.
Môi trường thực thi hiện dùng torch CPU dù máy có RTX 3050 4GB. Runtime lock và full
ML/dev lock được ghi riêng; bản venv phát triển hiện tái sử dụng các gói hệ thống đã có.

## Kiểm thử

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src scripts tests migrations
.venv/Scripts/python.exe scripts/smoke_test.py --model mock
```

Integration tests dùng TEST_DATABASE_URL riêng có tên kết thúc `_test`; không reset DB
ứng dụng. [TEST_REPORT.md](TEST_REPORT.md) ghi kết quả đã chạy. Smoke HTTP tạo/xóa
chỉ bữa thử nghiệm của chính nó; ảnh người dùng không được lưu mặc định.

## Dữ liệu và mô hình

```powershell
.venv/Scripts/python.exe scripts/download_food101_parallel.py --workers 8
.venv/Scripts/python.exe scripts/create_splits.py
.venv/Scripts/python.exe scripts/verify_dataset.py
.venv/Scripts/python.exe scripts/run_poc_experiment.py --run-id poc-v1-cpu-budget-v2-primal --skip-existing
```

Lệnh cuối chỉ xem trước, không train; cần `--execute` khi bạn chủ động muốn chạy cả pipeline.
Không ghi đè split/run folder đã đóng băng. Xem hướng dẫn trước khi chạy lại thí nghiệm.
Food-101 được người dùng xác nhận phù hợp quy định môn học; license trong archive giới
hạn scientific fair use. Không phân phối lại ảnh. POC không phải benchmark full 101 lớp.
Release chỉ được chọn sau so sánh thực tế; không mặc định EfficientNet thắng.

## Tài liệu và mã nguồn

| Tài liệu | Nội dung |
|---|---|
| [PLAN.md](PLAN.md) | Kiến trúc v2 được người dùng yêu cầu; bản v1 được bảo tồn trong docs |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Từng phase, bằng chứng chạy, blocker |
| [DATASET_GUIDE.md](DATASET_GUIDE.md) | Download, manifest, nhãn, ảnh Việt và leakage |
| [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) | Train/resume/evaluate/artifact release |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API và JSON contracts |
| [DATABASE_GUIDE.md](DATABASE_GUIDE.md) | ORM, migrations, snapshot và seed |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Windows, Docker và vận hành |
| [diagrams](diagrams/README.md) | Mermaid nguồn; chưa render SVG/PNG |

`src/food_assistant`: api, domain, services, persistence, ml, ui và utils.
`scripts/`: CLI dữ liệu/train/evaluate/deploy smoke. `configs/`: cấu hình tái lập.
`contracts/`: Pydantic JSON Schema và ví dụ mock. `tests/`: unit/integration/smoke.
Dataset, pretrained weights, checkpoints, uploads và secrets không commit Git.

Năng lượng chỉ là ước tính theo khối lượng tham khảo/người dùng nhập, không đo từ ảnh.
