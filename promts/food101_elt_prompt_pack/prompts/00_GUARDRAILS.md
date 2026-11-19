# QUY TẮC AN TOÀN — áp dụng cho MỌI prompt

Bạn PHẢI tuân theo các bất biến sau:

## Dữ liệu và tương thích

1. **Không xóa, sửa, di chuyển, giải nén đè, thay thế** `data/raw/food-101/` hiện hữu. Nếu cần kiểm thử cơ chế ghi dữ liệu, sử dụng `tmp_path`, archive nhỏ giả lập hoặc một staging directory **mới**.
2. **Không ghi đè** `data/splits/v1/` (manifest, labels, quality report, metadata) hoặc các artifacts/model cũ. Với thay đổi schema, tạo **phiên bản mới**. Không thay đổi mapping nhãn hoặc official test split của POC hiện có.
3. **Không tải lại Food-101 thật**, không chạy huấn luyện hay download weights, không gọi mạng, trừ khi có yêu cầu rõ ràng từ người dùng. Bất kỳ test downloader nào phải dùng HTTP mock / local fake server.
4. **Không khẳng định 101.000 ảnh đã có hoặc đã kiểm thử trên máy local** nếu chưa thực sự quan sát được. Phân biệt `full Food-101 = 101.000 ảnh / 101 lớp` với `POC v1 = 10.000 ảnh / 10 lớp`.
5. Không đổi/sửa xóa môi trường Python cũ hoặc các dependency không liên quan. Mục tiêu là Python **3.11** qua `.venv-311`, theo `pyproject.toml`; nếu thiếu dependency, báo cáo và chỉ cài sau khi được phép.
6. Không tác động model training, API, DB nghiệp vụ/UI ngoài yêu cầu cụ thể; đảm bảo đọc được manifest cũ của `ManifestDataset`.

## Bảo vệ dữ liệu và vận hành

7. Không dùng sự tồn tại file hoặc đếm ảnh làm bằng chứng duy nhất của `PASS`; phải có hợp đồng, fingerprint/schema, integrity check tương ứng và trạng thái đầy đủ.
8. **Atomic publish:** ghi file tạm trong cùng filesystem, flush/fsync nếu phù hợp, rồi publish atomically; với thư mục, dùng thư mục staging có tên riêng, kiểm tra trước và chỉ publish khi đích chưa tồn tại. Không giả định `os.replace` có thể thay thư mục đích không rỗng trên Windows.
9. **Locks:** mọi writer vào cùng archive/manifest/raw directory phải dùng resource-scoped cross-process lock. Lock cho downloader không được vô tình chiếm khóa của training; test hai tiến trình cạnh tranh.
10. **Retry/resume:** kiểm tra HTTP status, chính xác `Content-Range`, expected offset, total length, ETag/Last-Modified nếu có; file `.part`/chunk đủ kích thước nhưng không đủ bằng chứng thì không được tin mù quáng.
11. **Security khi giải nén:** từ chối path traversal, absolute paths, links, device/special entries, tên trùng gây ghi đè và đường dẫn vượt staging; kiểm tra giới hạn để tránh archive bomb khi thích hợp.
12. **Checkpoint:** versioned schema, metadata nguồn, fingerprint, cấu hình, độ hoàn tất; cache/checkpoint lỗi phải fail closed. State `INCOMPLETE` không được chuyển thành PASS khi resume.
13. **Manifest:** chính sách duplicate crossing splits là fail; official test phải giữ nguyên; chia validation chỉ từ original train, deterministic seed; tránh data leakage.
14. **Quality gate:** có `PASS | FAIL | INCOMPLETE`, có giải thích và metric kiểm tra. Khi gate không đạt thì không chạy dependent Transform/Validate.
15. **Idempotency:** chạy lại không thay đổi đầu ra đã xác thực; `--dry-run` không ghi dữ liệu/trạng thái hoặc tạo lock/staging mang tính đột biến.

## Kỹ thuật làm việc

16. Trước khi sửa, báo `git status --short` và liệt kê file thay đổi sẵn; không `git reset --hard`, `git clean -fdx`, `rm -rf`, force-push, amend commit hay tự merge.
17. Thay đổi tối thiểu, theo style Python 3.11 và Ruff/pytest của repo; tránh tạo abstraction không cần thiết. Có test minh chứng **failure mode** cho code an toàn.
18. Mọi lệnh test phải ghi lại `RUN/PASS/FAIL/NOT RUN`. Không sáng tác output, số lượng ảnh hoặc test passing.
19. Nếu yêu cầu xung đột với bất biến trên, **dừng hành động có thể phá dữ liệu**, nêu xung đột và chọn giải pháp không phá hủy.
20. Không commit dữ liệu Food-101, file tải về, checkpoint môi trường, khóa truy cập hoặc log chứa thông tin nhạy cảm; kiểm tra `.gitignore` và `git diff --stat`.
