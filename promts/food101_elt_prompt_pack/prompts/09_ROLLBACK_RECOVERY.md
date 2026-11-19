# RECOVERY PROMPT — Phục hồi an toàn khi migration ELT gặp lỗi

Vai trò: Incident responder chuyên dữ liệu. Mục tiêu là phục hồi trạng thái hoạt động cũ **không xóa dữ liệu**, không tự sửa raw Food-101. Không chạy `git reset --hard`, `git clean`, `rm -rf`, đổi branch có nguy cơ ghi đè file chưa commit.

## Trình tự

1. Đọc `git status --short`, log/reports của run thất bại và state/metadata. Xác định phase, stage và thời điểm lỗi; tạm dừng mọi writer khác để tránh race.
2. Phân loại: code/config bug, partial download, staging incomplete, manifest publish incomplete, invalid checkpoint, corrupted output, lock conflict, incompatible manifest root.
3. Xác định **tập dữ liệu bất biến** cần giữ: `data/raw/food-101/`, `data/splits/v1/`, existing models/reports. Ghi fingerprint/checksum từ bằng chứng sẵn có, tuyệt đối không alter.
4. Nếu có staging/part còn dở: **không xóa ngay**. Liệt kê kích thước, metadata, nguồn; cách xử lý mặc định là quarantine vào đường dẫn an toàn riêng mới (chỉ khi không ghi đè và được phép). Chỉ dọn dẹp sau khi đã phân tích và có chấp thuận.
5. Nếu lỗi state/checkpoint: không sửa tay thành `PASS`, tạo state `FAIL/INCOMPLETE` và tái kiểm theo schema. Nếu file cũ bị ghi hỏng, xem bản sao/commit/backup trước, không suy đoán nội dung.
6. Nếu model consumer bị ảnh hưởng, phục hồi config trỏ tới manifest v1 theo cách không làm mất thay đổi đang có; ưu tiên reversible patch/branch, không hard reset.
7. Test lại trên fixtures synthetic; dùng `--dry-run` để xác nhận đường xử lý, không tự chạy download.

## Output

- Root cause có bằng chứng / giả thuyết chưa xác nhận.
- File/data có nguy cơ, file an toàn đã chứng minh.
- Hướng recovery không phá dữ liệu, thay đổi tối thiểu, checkpoint phải kiểm tra lại.
- Lệnh đề xuất, chỉ chạy các lệnh read-only trước.
- Trạng thái `RECOVERED / BLOCKED / PARTIAL` cùng tests đã chạy thật.

**Không tự động rollback bằng lệnh xóa hoặc force Git.**
