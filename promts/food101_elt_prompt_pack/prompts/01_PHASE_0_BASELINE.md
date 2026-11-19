# PHASE 0 — Baseline: xác lập hiện trạng có chứng cứ

## Mục tiêu

Biết chính xác code, môi trường, dataset và kiểm thử trên máy local đang ở trạng thái nào **mà không chỉnh sửa dữ liệu**. Phase này chỉ audit và sửa nhỏ lỗi setup nếu thật sự cần, không refactor architecture/downloader.

## Đọc và kiểm tra

- Đọc `README.md`, `pyproject.toml`, `configs/data.yaml`, `configs/dataset_approval.json`, `data/splits/v1/dataset_metadata.json` (nếu có), `.gitignore`.
- Xem `scripts/download_food101.py`, `scripts/download_food101_parallel.py`, `scripts/verify_dataset.py`, `scripts/create_splits.py`, `src/food_assistant/ml/datasets.py`, `src/food_assistant/ml/transforms.py`, `src/food_assistant/ml/cli.py`.
- Xem `tests/unit/test_poc_preprocessing.py`, `tests/unit/test_ml_pipeline.py` và các test liên quan download/dataset đang có. File audit `verify_food101.py` hoặc `food101_verifier.py` từ tài liệu cũ **có thể không có trên `main`**, chỉ kiểm tra nếu local thực sự có.
- `git status --short`, `git branch --show-current`, phiên bản Python qua `.\.venv-311\Scripts\python.exe --version` (hoặc đường dẫn venv thực tế), `python -m pip show pytest torch torchvision Pillow`, `python -m pytest --version`.
- Mục tiêu Python là 3.11 (pyproject yêu cầu `>=3.11,<3.12`); không tự xóa venv cũ nếu đang có nhiều Python.

## Bài test cần thực hiện khi môi trường cho phép

```powershell
.\.venv-311\Scripts\python.exe --version
.\.venv-311\Scripts\python.exe -m pip show pytest torch torchvision Pillow
.\.venv-311\Scripts\python.exe -m pytest tests/unit/test_poc_preprocessing.py tests/unit/test_ml_pipeline.py -q
```

Nếu thiếu `pytest` hoặc `torchvision`, báo `BLOCKED` do môi trường; đưa lệnh cài `python -m pip install -e ".[ml,dev]"` để người dùng thực hiện hoặc xin phép trước khi thay đổi venv. Không giả lập kết quả test.

## Kiểm tra dữ liệu KHÔNG phá hủy

1. Phân biệt raw dataset và POC manifest. Kiểm tra tồn tại `data/raw/food-101/`, các official meta files; đọc metadata/report đã có, chỉ đếm/kiểm nhanh file nếu chi phí hợp lý.
2. Nếu có report full audit đáng tin, xác định thời điểm, phạm vi, fingerprint/checksum. Không coi bản report cũ là kết quả audit hiện tại nếu không chứng minh được tính tương thích.
3. Kiểm tra manifest `v1` có 10.000 ảnh/10 lớp theo metadata; phát hiện `dataset_metadata.json.root` có đường dẫn tuyệt đối cũ: ghi rủi ro portablity, KHÔNG sửa v1.
4. Thử import real torchvision transforms và DataLoader bằng fixture/synthetic data; không cần training, không cần tải weights.
5. Liệt kê file bị Git track trong data/raw hoặc báo cáo có nguy cơ lộ dữ liệu; không xóa.

## Đầu ra

Tạo **chỉ khi được phép ghi docs** `artifacts/reports/elt/baseline.md` hoặc tương đương, không đụng dữ liệu. Báo cáo gồm:

- SHA/branch, Git status, Python 3.11 venv/dependencies và các lỗi cài đặt.
- Phạm vi data local được xác thực (không biết thì ghi `UNKNOWN`).
- Unit test: lệnh, pass/fail/blocked, traceback ngắn.
- Map luồng code hiện tại, các hàm có thể tái sử dụng.
- Danh sách thiếu sót để chuyển Phase 1–6, mức rủi ro.

## Cổng nghiệm thu

- [ ] Môi trường Python 3.11 + dependencies cần cho test hoạt động (hoặc blocker được nêu rõ).
- [ ] Chạy thành công unit test liên quan và real transform / không dùng fallback giả; nếu không thì chưa PASS.
- [ ] Không thay đổi raw Food-101, manifests hoặc downloader.
- [ ] Có bảng bằng chứng, không claim dataset full 101k chỉ dựa vào GitHub.

**Dừng sau Phase 0. Không tự bắt đầu Phase 1.**
