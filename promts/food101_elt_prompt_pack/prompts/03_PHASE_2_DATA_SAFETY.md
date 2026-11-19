# PHASE 2 — Data Safety & Load an toàn

## Tiền điều kiện

Đã có hợp đồng stage và config từ Phase 1. Dữ liệu Food-101 local và manifest POC là read-only. Không dùng dataset thật để test sự cố ghi.

## Mục tiêu

Khắc phục nguy cơ `safe_extract` ghi đè file ảnh tốt, writer xuất report/metadata bị dở, checkpoint tin dữ liệu thiếu bằng chứng, và cạnh tranh giữa hai tiến trình. Bổ sung Load an toàn trước khi cải tiến downloader.

## Code liên quan thực tế

- `scripts/download_food101.py`: `safe_extract` giải nén trực tiếp và dùng `destination.open('wb')` — có nguy cơ ghi đè ảnh đích.
- `scripts/download_food101_parallel.py`: sử dụng lại `safe_extract` và trực tiếp ghi download metadata.
- `src/food_assistant/ml/datasets.py`: `write_json` ghi trực tiếp, `prepare_food101` công bố các CSV rồi JSON ngay vào output.
- `src/food_assistant/utils/process_lock.py`: cơ chế training lock có sẵn; có thể học cách khóa đa nền tảng nhưng cần khóa data-specific riêng.

## Yêu cầu triển khai

1. Viết hàm **atomic_write_json/text/csv** dùng tempfile trong cùng directory/filesystem; flush + fsync nếu cần; tạo file parent an toàn; cleanup temp sau lỗi; không phá file trước đó. Kiểm tra portable Windows.
2. Viết **safe staged extraction**: archive giả lập giải nén vào directory staging mới, không có symlink/hardlink, reject absolute/traversal/duplicate entries/special entries, kiểm tra file count/meta/integrity theo contract; không ghi trực tiếp đích raw đang tồn tại.
3. **Publish bằng đường dẫn mới** khi target chưa tồn tại và dữ liệu được xác thực; nếu raw target có sẵn, validate+skip hoặc reject rõ ràng. Không rename đè thư mục nonempty trên Windows; có thể dùng versioned directory + atomic pointer nếu cần. Nếu validation thất bại giữ raw cũ nguyên vẹn.
4. Thêm **resource-scoped lock** đối với archive/raw/manifest writers; cân nhắc dùng `msvcrt` trên Windows, `fcntl` trên Unix hoặc dependency đã có. Không tái dùng lock training theo cách làm thay đổi training behavior.
5. Xác thực **checkpoint schema/version/source fingerprint/input hash/expected files/completed flag**. Từ chối checkpoint stale, thiếu field, sửa tay, corrupted; không đánh PASS theo marker đơn thuần.
6. Với report/metadata, chỉ ghi `complete = true` sau validation; fail state và diagnostic tách biệt, không bao giờ làm giả success marker.
7. Giữ CLI script cũ tương thích nhưng đường write phải qua cơ chế an toàn; mọi external download tests bằng local archive/HTTP mock.

## Failure tests tối thiểu

- Tar với `../`, path tuyệt đối, symlink, hardlink, device, duplicate member => bị từ chối.
- Giải nén lỗi ở 50% => raw cũ không đổi; staging được xử lý an toàn.
- Có đích raw cũ với ảnh tốt => không mở `wb` để thay ảnh.
- Crash trước `os.replace` file JSON => report cũ không bị truncate.
- Hai process cùng publish => tối đa một writer thành công, bên kia báo rõ conflict/skip.
- Cache/checkpoint JSON sai schema, checksum không trùng, incomplete => không được skip/PASS.
- Test trên `tmp_path`, archive nhỏ tự tạo, không ghi `data/raw/food-101/`.

## Cổng nghiệm thu

- [ ] Tất cả failure tests liên quan safety PASS.
- [ ] Không có thao tác ghi trực tiếp vào raw hiện có khi chạy lại.
- [ ] Reporter/metadata sử dụng atomic write.
- [ ] POC v1 không đổi, manifest reader và unit tests cũ vẫn đạt.
- [ ] Giải thích rõ phương thức publish directory trên Windows.

**Dừng sau Phase 2; chưa refactor retry/resume ở Phase 3.**
