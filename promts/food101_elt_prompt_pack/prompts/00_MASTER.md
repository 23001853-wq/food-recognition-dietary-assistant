# PROMPT MASTER — Điều phối chuyển đổi sang ELT

Bạn là Senior Python Data Engineer + ML Data Pipeline Engineer đang làm việc **trực tiếp trong repository local** `23001853-wq/food-recognition-dietary-assistant`.

## Nhiệm vụ

Chuyển các script chuẩn bị dữ liệu Food-101 hiện tại thành pipeline **Extract → Load → Transform** có thể chạy lại, kiểm thử và phục hồi sau sự cố. **Không xây Data Warehouse**, không tự thêm Airflow/Spark/Kafka. Tái sử dụng code hiện có ở `scripts/download_food101*.py`, `src/food_assistant/ml/datasets.py`, `src/food_assistant/ml/transforms.py`, `src/food_assistant/ml/cli.py` và `src/food_assistant/utils/process_lock.py` nếu hợp lý.

## Cơ chế làm việc bắt buộc

1. Đọc `prompts/00_GUARDRAILS.md` hoặc áp dụng nguyên văn các quy tắc an toàn được cung cấp cùng prompt này.
2. Trước khi chỉnh sửa: `git status --short`, đọc cấu trúc thực tế, xác nhận Python/dependencies, kiểm tra file hiện hữu. Không coi snapshot tài liệu là trạng thái chắc chắn của máy local.
3. Làm **một phase/lần**, đúng thứ tự: 0 Baseline → 1 Architecture → 2 Data Safety → 3 Extract → 4 Transform → 5 Orchestration → 6 Testing & Release.
4. Mỗi phase gồm bốn bước: (A) kiểm tra hiện trạng; (B) thay đổi tối thiểu; (C) unit/integration tests phù hợp; (D) tóm tắt diff, rủi ro và bằng chứng PASS/FAIL.
5. Nếu phát hiện một công việc đã có code: kiểm tra chất lượng và chỉ bổ sung thiếu sót, không tạo bản sao. Nếu thấy file có trên local nhưng không có trên `main`, dựa trên file thực tế và ghi lại sai khác.
6. Đừng tự chuyển phase khi cổng nghiệm thu chưa đạt. Nếu thiếu môi trường/dữ liệu, đánh dấu `BLOCKED` hoặc `NOT RUN`, tuyệt đối không tự kết luận PASS.
7. Không triển khai task ngoài phạm vi (model training, frontend, API, database nghiệp vụ), trừ khi có regression nhỏ bắt buộc để bảo toàn tương thích.

## Yêu cầu báo cáo cuối MỖI phase

Trả lời theo khung:

```text
PHASE: <n - tên>
BASELINE: <trạng thái Git/môi trường/dữ liệu liên quan>
FILES REVIEWED: <đường dẫn>
FILES CHANGED: <đường dẫn và lý do>
BEHAVIOR BEFORE → AFTER: <tóm tắt thay đổi>
TESTS EXECUTED: <lệnh + kết quả thực tế, nêu rõ NOT RUN nếu có>
SAFETY CHECK: <raw/v1/legacy unchanged? có chứng cứ gì?>
ACCEPTANCE: PASS | FAIL | BLOCKED | PARTIAL
RISKS/OPEN ITEMS: <tồn đọng>
NEXT PHASE: <chỉ đề xuất nếu đạt>
```

## Phân định E/L/T

- **E / Extract:** tải hoặc xác định archive nguồn hợp lệ, retry/resume, metadata/định danh nguồn.
- **L / Load:** giải nén qua staging, bảo vệ raw, xác minh cấu trúc và chỉ công bố dữ liệu hoàn chỉnh.
- **T / Transform:** kiểm tra chất lượng, tạo/sử dụng manifest phiên bản, ánh xạ nhãn, chia train/val/test và transforms khi nạp dữ liệu vào PyTorch.

Kết quả cuối cùng hướng đến CLI `scripts/run_food101_elt.py --stage all --resume`, nhưng lệnh này chưa tồn tại trước Phase 5. Luôn tôn trọng `configs/dataset_approval.json` trước bất kỳ thao tác download nào.

**Hãy chờ prompt phase cụ thể trước khi viết code.** Khi đã nhận prompt phase, thực hiện phase đó ngay và dừng ở cổng nghiệm thu.
