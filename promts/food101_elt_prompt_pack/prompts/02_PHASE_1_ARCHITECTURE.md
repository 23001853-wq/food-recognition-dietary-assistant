# PHASE 1 — ELT Architecture: hợp đồng stage trước khi code

## Tiền điều kiện

Đọc báo cáo Phase 0 và xác nhận phạm vi dữ liệu local; nếu Phase 0 chưa đạt, chỉ viết thiết kế tạm với BLOCKED rõ ràng, không phát hành mã runner. Tuân thủ `00_GUARDRAILS.md`.

## Mục tiêu

Định nghĩa boundary giữa `Extract`, `Load` và `Transform` theo cấu trúc repository có thật, phù hợp Python + filesystem. **Không thêm Data Warehouse.** Dùng `configs/data.yaml`/`configs/classes_poc.yaml` làm tham chiếu và tránh thay đổi semantics manifest v1.

## Các bước bắt buộc

1. **Đọc code hiện hữu:** hai downloader, `safe_extract`, `prepare_food101`, `audit_manifest`, `ManifestDataset`, `build_transform`, CLI verify/splits, `process_lock.py`.
2. **Vẽ flow thực tế và flow mục tiêu**, ghi lại rõ stage nào skip được, stage nào cần file đầu vào, stage nào bị chặn bởi approval/quality gate.
3. **Định nghĩa stage contract** dạng bảng có cột: stage, input, output, schema, fingerprint, precondition, success proof, fail states, skip/resume rule, side effects, rollback. Phải phân biệt raw full dataset và POC 10 lớp.
4. **Định nghĩa nguồn chân lý:** raw dataset path + metadata/verification, archive + origin metadata, immutable manifest version, config snapshot. Sự tồn tại của folder/file không đủ.
5. **Định nghĩa run state:** `PENDING/RUNNING/SKIPPED/PASS/FAIL/INCOMPLETE/BLOCKED` (hoặc enum được giải thích), chuyển trạng thái hợp lệ, run_id, config_hash, input fingerprint, output hashes, timestamps, error categories, skip_reason.
6. **Định nghĩa config** trong `configs/food101_elt.yaml`, tối thiểu: path raw/staging/archive/manifest/report, approval, class selection, split version, seed, val_fraction, download policy (safe default: không download), retry/backoff, quality policy, lock timeouts, test-only fixtures; kiểm tra giá trị và path containment.
7. **Đề xuất module** `src/food_assistant/data_pipeline/` gồm `config.py`, `extract.py`, `load.py`, `quality.py`, `transform.py`, `state.py`, `atomic.py`, `pipeline.py`, nhưng ưu tiên code ít file hơn nếu hợp lý; không sao chép toàn bộ logic ML đang có.
8. **Thiết kế tính portable:** manifest v1 metadata hiện có thể trỏ tới đường dẫn tuyệt đối trên máy khác; manifest mới dùng path tương đối hoặc root override có kiểm soát, đồng thời vẫn đọc được v1.
9. **Thiết kế đường lui:** nếu migration bị dừng, POC hiện có vẫn chạy như trước.

## Phạm vi sửa code

- Có thể tạo `docs/food101_elt_architecture.md` và `configs/food101_elt.yaml` với test parsing/schema.
- Chưa viết runner thật, downloader mới, chưa migrate data, chưa sửa `data/splits/v1`.

## Test cần có

- Config load + schema validation + path validation.
- Từ chối config sai (seed không hợp lệ, output trùng raw/v1, path nguy hiểm, version không hỗ trợ).
- Không có side effects khi chỉ load/validate config.

## Cổng nghiệm thu

- [ ] Data contract E/L/T đủ rõ để triển khai độc lập.
- [ ] Có chính sách xử lý existing raw/partial downloads, state, invalid cache, idempotency.
- [ ] File config parse thành công, test validation PASS.
- [ ] Không động vào raw, v1, model artifacts.

**Chỉ hoàn thành Phase 1; không sửa downloader hoặc chạy download.**
