# PHASE 6 — Testing & Release: nghiệm thu ELT và bàn giao

## Tiền điều kiện

Các phase trước đã hoàn thành bằng chứng; chỉ review/fix/regression/docs trong phạm vi ELT. Không xóa dataset để "test lại từ đầu".

## Nhiệm vụ

1. **Rà soát test suite**: unit tests cho config, schema/state, atomic JSON/CSV, safe tar, lock, retry/backoff, HTTP Range, chunk resume, checksum, quality gate, manifest, transforms/loader, runner.
2. **Integration tests** với dataset + tar giả lập nhỏ và local HTTP mock; không network tới nguồn thật, không tải 101k ảnh. Xác nhận E → L → T/skip/resume/report đầy đủ.
3. **Failure injection tests:** kill/interruption giữa extract, merge, manifest publish, state update; stale checkpoint, corrupted chunk same size, ETag change, two-process writer, quality incomplete. Kiểm tra raw cũ/v1 không đổi.
4. **Regression:** chạy `tests/unit/test_ml_pipeline.py`, `tests/unit/test_poc_preprocessing.py` và test training-consumer/data-related phù hợp; kiểm kiểu dữ liệu, nhãn và manifest hash cũ không bị thay đổi.
5. **Real dataset validation read-only, chỉ khi được cho phép và dữ liệu local sẵn:** chạy kiểm tra verification theo phạm vi config và ghi log công bố. Nếu không có data hoặc chưa được cho phép, báo `NOT RUN`; không lấy fixture test để khẳng định full dataset PASS.
6. **Style/quality:** `python -m pytest ...`, `python -m ruff check ...` nếu các thư viện đã có; xem coverage các lỗi an toàn, không chỉ test happy path.
7. **Docs:** cập nhật `README.md`/`DATASET_GUIDE.md` hoặc tài liệu mới `docs/food101_elt_runbook.md` giải thích architecture E/L/T, thư mục, stage contract, CLI, resume/retry, failure recovery, artifact schema, không cần DWH, các hạn chế và an toàn thao tác.
8. **Hướng dẫn bàn giao:** bảng mapping script cũ → module mới, command setup Windows 3.11, dry run, pipeline run có approval, verify reports, troubleshooting, rollback.
9. **Report nghiệm thu:** liệt kê testcase, kết quả thật, ngày/commit, phạm vi dataset tested, số ảnh chính xác theo scope, sha/fingerprint, known issues. Đánh PASS chỉ nếu điều kiện thật sự được chứng minh.

## Lệnh tham khảo

```powershell
.\.venv-311\Scripts\python.exe -m pytest tests/unit -q
.\.venv-311\Scripts\python.exe -m pytest tests/integration -q
.\.venv-311\Scripts\python.exe -m ruff check scripts src tests
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --dry-run
```

Không mặc định chạy integration tests đòi database thật (repo có các integration test PostgreSQL); tách rõ kiểm thử ELT synthetic khỏi nhóm cần dịch vụ bên ngoài.

## Tiêu chí cuối cùng

- [ ] Ứng dụng/POC hiện hành đọc được v1 không regression.
- [ ] Failure tests an toàn dữ liệu PASS và có bằng chứng.
- [ ] ELT synthetic end-to-end PASS.
- [ ] Không download dataset thật hoặc thay đổi raw/v1 trong quá trình test.
- [ ] CLI/documentation/rollback hướng dẫn tái lập được.
- [ ] Nếu real dataset validation chưa chạy, phải công bố hạn chế thay vì nói "đã bàn giao production-ready".

**Chỉ kết luận RELEASABLE khi các tiêu chí bắt buộc đã đạt; nếu không, tổng kết PARTIAL/BLOCKED có đề xuất cụ thể.**
