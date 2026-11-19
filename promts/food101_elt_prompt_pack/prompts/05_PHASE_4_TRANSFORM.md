# PHASE 4 — Transform: quality gate, immutable manifest & preprocessing

## Tiền điều kiện

Phases 0–3 đạt cổng nghiệm thu; dữ liệu gốc read-only. Không chạy full rehash trên 101.000 ảnh chỉ để kiểm thử thay đổi nếu chưa có yêu cầu; ưu tiên fixture nhỏ, và phân biệt full validation với kiểm thử logic.

## Đối chiếu code hiện có

- `src/food_assistant/ml/datasets.py::audit_manifest`: check path containment, checksums, duplicate crossing splits, capture-group leakage, mapping contiguous.
- `prepare_food101`: dùng original train/test của Food-101; chia validation từ train bằng stratified split; hiện xuất file CSV trực tiếp vào output.
- `ManifestDataset`: đọc v1 manifest, đối chiếu labels, decode ảnh và EXIF transpose.
- `src/food_assistant/ml/transforms.py`: `preprocessing`, `build_transform` dùng torchvision.
- `src/food_assistant/ml/cli.py` có `create_splits` và `verify_dataset`.
- `data/splits/v1/dataset_metadata.json` trên `main`: 10.000 ảnh, 10 lớp, split 6000/1500/2500; có absolute Windows path từ môi trường cũ.

## Yêu cầu triển khai

1. Thêm **quality gate** theo phạm vi khai báo: `full Food-101` (101 lớp/101k ảnh) **hoặc** `POC selection` (class list chỉ định). Có metrics verified, errors/warnings và trạng thái `PASS/FAIL/INCOMPLETE`. Không đánh PASS cho incomplete audit.
2. Reuse `audit_manifest` và image verification, tăng kiểm tra schema/rows/label IDs/checksums/EXIF/orientation theo policy; xử lý duplicate exact ở train/val/test theo chính sách leakage (cross-split fail, within-split có thể report). Không tự loại ảnh mà không có chính sách và report rõ.
3. Giữ **official Food-101 test split**; chỉ tạo validation từ official train theo seed, stratify và deterministic ordering. Không chọn test ảnh để chỉnh config/tuning.
4. Manifest version **mới**: tạo CSV/labels/metadata/quality report trong staging, xác minh row count, classes, uniqueness, hashes, input fingerprint, config hash, rồi publish nguyên phiên bản một lần. Nếu version đích đã tồn tại và hợp lệ => reuse; nếu invalid/conflict => stop, không overwrite.
5. Không chỉnh `data/splits/v1/*`; đảm bảo model/training code đang đọc v1 vẫn chạy. Với phiên bản mới, giảm phụ thuộc vào absolute root cũ bằng root-relative contract hoặc explicit root override; test tương thích POC v1.
6. Kiểm tra torchvision `train augmentation + Normalize` và `eval Resize/CenterCrop/Normalize` tương thích config/model weights hiện có. Kiểm shape, dtype, label/EXIF/paths bằng DataLoader thật, không fallback nếu claim acceptance.
7. Preprocessing tiếp tục **on-the-fly**, không bắt buộc materialize 101.000 ảnh transform. Nếu cần cache, cache phải theo fingerprint/version, có invalidation.
8. Bất kỳ audit cache/checkpoint nào cũng cần schema/fingerprint/completion; có option full recheck khi người dùng yêu cầu. Không suy đoán đã verify ảnh nếu chỉ đếm file.
9. Giữ bài toán POC đang có: 10 lớp/10k ảnh; nếu làm full Food-101, cần config/split version khác và test riêng, không đè POC.

## Test bắt buộc

- Fixture 2–3 classes có official meta synthetic; split reproducible cùng seed, test không rò rỉ.
- Duplicate across train/val hoặc train/test => FAIL.
- Missing/corrupt image hoặc wrong checksum => FAIL hoặc INCOMPLETE đúng policy, không PASS.
- Publisher crash giữa các file => không có phiên bản manifest được công bố dở.
- Rerun cùng inputs/config => same manifest fingerprint / skip hợp lệ; config khác nhưng same output version => conflict.
- `ManifestDataset` đọc v1 schema cũ; transforms train/eval thật từ torchvision cho shape/dtype/labels hợp lệ.
- Path traversal trong manifest bị từ chối.

## Cổng nghiệm thu

- [ ] Test quality gate và manifest staged publish PASS.
- [ ] Split deterministic, no leakage, immutable version.
- [ ] DataLoader/torchvision test PASS và không sửa raw.
- [ ] Không làm thay đổi mapping v1 hoặc model consumer cũ.

**Dừng sau Phase 4.**
