# PHASE 5 — Orchestration: một lệnh chạy ELT có thể resume

## Tiền điều kiện

Phases 1–4 đã có config, contracts, safe Load, safe Extract và Transform. Không viết runner bằng cách gọi shell script tùy tiện gây side effects mà không kiểm tra trạng thái.

## Mục tiêu

Tạo `scripts/run_food101_elt.py` dùng các module đã kiểm thử, cung cấp CLI rõ ràng để inspect → extract → load → quality → transform → validate → report, có skip/resume/dry-run và run logs. Không training, không DWH/Airflow.

## CLI mục tiêu

```powershell
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --help
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --dry-run
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --resume
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage transform --resume
```

Các tham số cụ thể có thể cải tiến nhưng phải giải thích và test tương thích trong docs.

## Yêu cầu triển khai

1. Đọc `configs/food101_elt.yaml`; hỗ trợ `--config`, `--stage`, `--dry-run`, `--resume`, `--report-dir` hoặc tương đương. Validate arguments trước khi thay đổi dữ liệu.
2. **DAG:** `inspect → extract (nếu cần) → load (nếu cần) → quality gate → transform → validate → report`. `--stage transform` vẫn phải kiểm prerequisite quality; không bypass gate.
3. **Dry run đúng nghĩa:** đọc config/trạng thái, in stage plan và reason (`RUN/SKIP/BLOCKED`), không tạo archive, staging, state/locks có tác dụng đột biến hay log write nếu không được thiết kế rõ. Test snapshot file tree trước/sau.
4. **Resume:** chỉ reuse stage khi input fingerprint/config hash/version/schema/output validation hợp lệ; checkpoint stale/incomplete/corrupt => revalidate/recompute an toàn hoặc stop. Không chỉ `if output.exists()`.
5. **Offline-first:** đã có raw verified thì skip tải/giải nén; nếu thiếu và download chưa được cấp phép, `BLOCKED` thay vì tự kết nối tới ETH.
6. State theo `run_id`, `stage_id`, timestamps, attempts, status, error type/reason, artifact fingerprints, duration, skip_reason. Atomic writes và lock; không ghi secrets.
7. **Failure propagation:** quality FAIL/INCOMPLETE chặn transform và downstream; output report có kết luận thất bại, process exit code != 0. Không báo tổng pipeline PASS khi stage phụ thuộc bị fail.
8. **Rerun deterministic:** cùng input/config không tạo manifest mới nếu không cần; rerun không sửa raw/v1. Chạy lại fail stage từ vị trí an toàn sau interruption.
9. **Reporting:** xuất JSON machine-readable + Markdown human-readable dưới `artifacts/reports/elt/<run-id>/`, không chạm report đã công bố, có tổng kết E/L/T, số ảnh đúng phạm vi, warnings.
10. Không trộn logic model training vào pipeline; output cuối chỉ cần DataLoader validate sẵn sàng cho model consumer.

## Test bắt buộc

- End-to-end synthetic E → L → T khi nguồn có sẵn/mock.
- Existing raw verified => E/L skipped với reasons; không network.
- Raw missing và download policy off => BLOCKED, không gọi network.
- `--dry-run` không sửa filesystem.
- Quality gate FAIL/INCOMPLETE => không chạy T.
- Resume từ trạng thái interrupted/partially completed.
- Chạy lại 2 lần: outputs immutable, fingerprints giống, no redundant work.
- Hai runner cùng resource: lock và conflict đúng.

## Cổng nghiệm thu

- [ ] CLI `--help`, `--dry-run`, `--resume` hoạt động.
- [ ] End-to-end synthetic + failure propagation tests PASS.
- [ ] Dữ liệu legacy còn nguyên; không có auto-download/train.
- [ ] Mỗi stage có chứng cứ, logs và report.

**Dừng sau Phase 5, không tự chạy full Food-101 hoặc training.**
