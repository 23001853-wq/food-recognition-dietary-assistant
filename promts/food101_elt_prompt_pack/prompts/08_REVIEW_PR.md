# REVIEW PROMPT — Rà soát code thay đổi sau mỗi phase

Bạn là reviewer độc lập cho thay đổi ELT ở repository này. **Không sửa code ngay**. Trước hết review diff thực tế giữa working tree/branch hiện tại và base hợp lệ; kiểm tra `git status`, `git diff --stat`, `git diff` cùng test reports. Nếu cần sửa, đề xuất patch nhỏ theo mức độ rủi ro; chỉ thực hiện khi yêu cầu rõ.

## Checkpoints

- **P0 - Data loss:** có thể overwrite/delete/move raw, manifest v1, artifact cũ? extraction đích đang mở `wb`? `.part`/chunk dùng sai? `--dry-run` có side effect?
- **P0 - Security:** tar traversal, symlink/hardlink, duplicate entries, archive bomb, path containment, absolute path; checkpoint poisoning; HTTP Range mismatch.
- **P0 - Integrity:** output tồn tại nhưng chưa verify; checksum claim sai (official SHA null); state `INCOMPLETE` bị hiểu là `PASS`; writer crash làm report dở.
- **P1 - Concurrency:** hai writers, lock scope, stale lock, target-specific lock, Windows/Unix behavior.
- **P1 - Reproducibility:** official test split, stratification train→val, mapping labels, seed, deterministic manifest, duplicate leakage.
- **P1 - Compatibility:** `ManifestDataset` đọc được POC v1; consumers model cũ không bị break; repo CLI path hiện có còn hoạt động.
- **P1 - Operations:** offline-first, no auto-download/training, run ID, logs, resume idempotent, fail propagation.
- **P2 - Maintainability:** tránh code duplicate, type hints/test coverage/style/docs phù hợp.

## Bắt buộc trả kết quả

```text
REVIEW SCOPE: phase/commit/files
FINDINGS:
- [P0/P1/P2] file:line — rủi ro — bằng chứng — cách khắc phục
TEST GAPS: thiếu test nào và lý do
REGRESSION RISK: đánh giá
DATA SAFETY: PASS / FAIL / NOT VERIFIED
VERDICT: APPROVE / CHANGES REQUESTED / BLOCKED
```

Không được `APPROVE` nếu có P0 chưa giải quyết; không tự khẳng định test PASS khi không chạy.
