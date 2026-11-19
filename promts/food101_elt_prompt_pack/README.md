# Bộ prompt chuyển đổi Food Recognition & Dietary Assistant sang ELT

**Phiên bản prompt pack:** 1.0 — 17/11/2026  
**Repository:** https://github.com/23001853-wq/food-recognition-dietary-assistant  
**Ngôn ngữ prompt:** Tiếng Việt  
**Đối tượng dùng:** Codex, Cursor, GitHub Copilot Chat hoặc AI coding assistant có quyền đọc và chỉnh sửa repository.

## Mục tiêu

Chuẩn hóa quy trình dữ liệu của dự án theo **Extract → Load → Transform** mà **không sử dụng Data Warehouse**, không tải lại Food-101 khi dữ liệu hiện có đã được xác thực, không làm hỏng `data/raw/food-101/`, và không thay đổi manifest POC `data/splits/v1/` hoặc cách các model đang sử dụng POC.

Bộ prompt này là **hướng dẫn để AI viết code**, không phải code ELT đã triển khai. Chưa có phase nào được xác nhận hoàn thành trên máy local chỉ vì bộ prompt này được tạo.

## Nội dung

| Thứ tự | Prompt | Vai trò |
|---|---|---|
| 0 | `prompts/00_MASTER.md` | Điều phối, giới hạn và thứ tự triển khai |
| 0a | `prompts/00_GUARDRAILS.md` | Quy tắc an toàn bắt buộc, dùng ở mọi phase |
| 1 | `prompts/01_PHASE_0_BASELINE.md` | Xác minh hiện trạng, môi trường, dữ liệu, test |
| 2 | `prompts/02_PHASE_1_ARCHITECTURE.md` | Thiết kế hợp đồng E/L/T, cấu hình, state |
| 3 | `prompts/03_PHASE_2_DATA_SAFETY.md` | Safe load, staging, atomic write, lock |
| 4 | `prompts/04_PHASE_3_EXTRACT.md` | Download retry, resume, integrity |
| 5 | `prompts/05_PHASE_4_TRANSFORM.md` | Quality gate, manifest, PyTorch transforms |
| 6 | `prompts/06_PHASE_5_ORCHESTRATION.md` | CLI runner, skip/resume, reports |
| 7 | `prompts/07_PHASE_6_TESTING_RELEASE.md` | Failure testing, regression, tài liệu |
| 8 | `prompts/08_REVIEW_PR.md` | Kiểm tra chất lượng thay đổi ở cuối mỗi phase |
| 9 | `prompts/09_ROLLBACK_RECOVERY.md` | Kế hoạch phục hồi nếu triển khai lỗi |

Thêm `references/REPO_SNAPSHOT.md` (mốc đối chiếu GitHub), `references/ACCEPTANCE_MATRIX.md` (tiêu chí nghiệm thu), `references/EXECUTION_LOG_TEMPLATE.md` (mẫu ghi tiến độ) và `prompt_manifest.json`.

## Cách sử dụng tốt nhất

1. Mở repository trên máy, chuyển sang nhánh mới, ví dụ `feature/food101-elt` (kiểm tra `git status` trước để không làm mất công việc chưa commit).
2. Mở AI coding assistant tại **thư mục gốc repository**.
3. Gửi nội dung của `prompts/00_MASTER.md` và `prompts/00_GUARDRAILS.md` để thiết lập bối cảnh. Nếu công cụ không nhớ giữa các hội thoại, gửi lại guardrails ở đầu mỗi phase.
4. Gửi **một** prompt phase, từ `01_PHASE_0_BASELINE.md` đến `07_PHASE_6_TESTING_RELEASE.md`. Không ghép tất cả phase để AI làm trong một lần.
5. Sau mỗi phase, sử dụng `prompts/08_REVIEW_PR.md` để rà soát diff và test. Chỉ chuyển phase khi đã có bằng chứng đạt tiêu chí trong `references/ACCEPTANCE_MATRIX.md`.
6. Ghi kết quả vào bản sao `references/EXECUTION_LOG_TEMPLATE.md` ngoài repo hoặc một file log riêng được phép tạo.

## Mẫu câu gửi đầu mỗi phase

```text
Bạn đang làm việc tại thư mục gốc repo https://github.com/23001853-wq/food-recognition-dietary-assistant.
Hãy tuân thủ 00_GUARDRAILS.md và thực hiện CHỈ phase trong prompt tiếp theo.
Bắt đầu bằng việc đọc mã nguồn thực tế và `git status`; phát hiện sai khác với prompt thì báo cáo và điều chỉnh kế hoạch trước khi sửa.
Sau khi sửa, chạy các test khả thi, cung cấp bằng chứng, liệt kê file đổi và dừng đúng cổng nghiệm thu.
Không download dataset thật, không chạy training, không sửa `data/raw/food-101/` hoặc `data/splits/v1/`.
```

Sau đó dán nội dung prompt của phase tương ứng.

## Các lệnh kiểm tra tham khảo trên PowerShell

Chỉ chạy từ thư mục gốc repo và xác minh `.venv-311` đang tồn tại:

```powershell
.\.venv-311\Scripts\python.exe --version
.\.venv-311\Scripts\python.exe -m pip show pytest torch torchvision Pillow
.\.venv-311\Scripts\python.exe -m pytest tests/unit/test_poc_preprocessing.py tests/unit/test_ml_pipeline.py -q
git status --short
```

Nếu thiếu `pytest`, cài đúng dependency phát triển của dự án trong môi trường **được cho phép thay đổi**, không tự đổi Python phiên bản hoặc gỡ môi trường cũ:

```powershell
.\.venv-311\Scripts\python.exe -m pip install -e ".[ml,dev]"
```

Lệnh runner ELT bên dưới **chỉ là mục tiêu sau Phase 5** — không thể dùng ngay từ đầu:

```powershell
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --dry-run
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --resume
```

## Các quyết định đã thống nhất

- Không cần PostgreSQL/Data Warehouse/Airflow/Spark cho pipeline ảnh chỉ để gọi là ELT.
- Ưu tiên `Python + filesystem + Pillow + PyTorch/torchvision` và test `pytest`.
- Giữ nguyên manifest `v1` của POC; nếu thay đổi hợp đồng dữ liệu, phát hành `v2` hoặc phiên bản mới, không sửa lặng lẽ phiên bản cũ.
- Transform ảnh on-the-fly trong `Dataset`/`DataLoader` là hợp lệ; không bắt buộc ghi thêm 101.000 ảnh biến đổi.
- Chất lượng dữ liệu phải có trạng thái `PASS`, `FAIL` hoặc `INCOMPLETE`, không `PASS` chỉ vì file có tồn tại.

## Cảnh báo về dữ liệu đã có

`data/splits/v1/dataset_metadata.json` trên nhánh `main` mô tả **10.000 ảnh / 10 lớp**, không phải manifest toàn bộ Food-101. Dữ liệu gốc **101.000 ảnh / 101 lớp** từng được báo cáo có sẵn trên **máy local**, nhưng không thể chứng minh trạng thái đĩa local chỉ từ GitHub. Các prompt bắt AI kiểm tra trạng thái thực tế trước khi quyết định skip/download/verify.
