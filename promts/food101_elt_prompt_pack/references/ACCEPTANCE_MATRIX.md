# Acceptance matrix — điều kiện chuyển phase

Không đánh dấu [x] chỉ vì AI đưa ra code. Phải có test command + thực tế kết quả, hoặc ghi `BLOCKED/NOT RUN`.

| Phase | Bằng chứng bắt buộc để PASS | Không được làm |
|---|---|---|
| 0 Baseline | Python 3.11, dependencies, unit test, real torchvision transform, Git/data inventory | Không sửa raw/manifest/download |
| 1 Architecture | E/L/T stage contracts, config, state/skip/resume policy, config tests | Không viết runner, không download |
| 2 Data Safety | atomic write, staged extraction, concurrent lock và failure tests | Không giải nén trực tiếp raw cũ |
| 3 Extract | mocked HTTP edge tests, retry/resume/chunk integrity, archive/lock semantics | Không download Food-101 thật |
| 4 Transform | manifest immutable, quality gate, no leakage, DataLoader tests, v1 regression | Không sửa manifest v1, raw |
| 5 Orchestration | `--help`, `--dry-run`, `--resume`, synthetic E2E, skip/report/error propagation | Không tự chạy training, network |
| 6 Release | test suite + failure injection + docs, reproducibility + rollback guidance | Không gọi fixture test là validation full dataset |

## Mẫu bảng kết quả để điền

| Phase | Status | Commit/branch | Tests thực chạy | Git diff reviewed | Blocker | Ngày |
|---|---|---|---|---|---|---|
| 0 | NOT RUN | — | — | — | — | — |
| 1 | NOT RUN | — | — | — | — | — |
| 2 | NOT RUN | — | — | — | — | — |
| 3 | NOT RUN | — | — | — | — | — |
| 4 | NOT RUN | — | — | — | — | — |
| 5 | NOT RUN | — | — | — | — | — |
| 6 | NOT RUN | — | — | — | — | — |

Gợi ý trạng thái: `PASS`, `FAIL`, `PARTIAL`, `BLOCKED`, `NOT RUN`. Không tự cập nhật thành PASS khi thiếu log.
