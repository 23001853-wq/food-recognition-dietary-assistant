# Snapshot repository làm căn cứ viết prompt

**Ngày kiểm tra:** 17/11/2026  
**Nhánh:** `main`  
**Git tree SHA lấy từ API GitHub:** `2ead0dbd472a97978205ecb907e8a342b43f9fc9`  
**Repository:** https://github.com/23001853-wq/food-recognition-dietary-assistant

## Điều đã trực tiếp đọc từ GitHub

| Đường dẫn | Ghi nhận |
|---|---|
| `scripts/download_food101.py` | `download` hỗ trợ `.part` và Range cơ bản; `safe_extract` mở file đích `wb`; metadata marker và đếm ảnh được dùng cho skip. |
| `scripts/download_food101_parallel.py` | Multi-threaded 32 MiB chunks, retry một số lỗi, reuse chunk theo kích thước, combine archive trực tiếp. |
| `src/food_assistant/ml/datasets.py` | `audit_manifest` kiểm checksum/cross-split duplicate; `prepare_food101` tạo train/val/test; `ManifestDataset` đọc manifest, áp dụng transform. |
| `src/food_assistant/ml/transforms.py` | Có `preprocessing`, `build_transform` dựa trên torchvision. |
| `src/food_assistant/ml/cli.py` | `create_splits`, `verify_dataset` và các công cụ ML khác. |
| `src/food_assistant/utils/process_lock.py` | Có khóa cho training, không đồng nghĩa đã có khóa chung cho downloader. |
| `configs/data.yaml` | Root dữ liệu, manifest_dir v1, seed 42, validation_fraction 0.2. |
| `configs/dataset_approval.json` | Academic approval, nguồn ETH, `official_sha256: null`. |
| `data/splits/v1/dataset_metadata.json` | POC: 10.000 ảnh/10 lớp, train 6000, val 1500, test 2500; đường root tuyệt đối từ môi trường khác. |
| `pyproject.toml` | Python `>=3.11,<3.12`, extras `ml`, `dev`, pytest và Ruff cấu hình. |
| `scripts/verify_dataset.py`, `scripts/create_splits.py` | Entry-point gọi hàm CLI trong `food_assistant.ml.cli`. |

## Giới hạn của snapshot

- Không đọc/kiểm tra ổ đĩa local của người dùng. Không có quyền kết luận thực tế hiện có 101.000 ảnh trên máy.
- Không chạy pytest/torch/torchvision ở môi trường local. Không có bằng chứng phase nào đã PASS.
- Tài liệu cuộc trò chuyện trước cho biết đã có full Food-101 và từng gặp lỗi `.venv-311` thiếu `pytest`, nhưng phải kiểm tra lại.
- File `verify_food101.py`, `food101_verifier.py`, `test_food101_verifier.py` được nhắc trong kế hoạch cũ nhưng không thấy trong tree nhánh `main` đã kiểm tra; có thể là các file local hoặc từ nhánh khác.
- Snapshot là tham chiếu, không phải chỉ thị bỏ qua code cập nhật về sau. Mỗi prompt đều yêu cầu đọc code local trước.
