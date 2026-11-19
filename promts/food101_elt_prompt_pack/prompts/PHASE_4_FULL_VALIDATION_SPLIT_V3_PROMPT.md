# PHASE 4 — Full Validation on Cleaned Split v3

## Vai trò và mục tiêu

Bạn là kỹ sư Data Engineering phụ trách xác minh và nghiệm thu **phiên bản split v3 đã được làm sạch** của dự án `food-recognition-dietary-assistant`.

**Mục tiêu:** Mở rộng bằng chứng nghiệm thu của Phase 4 từ POC 10 lớp/10.000 ảnh sang **phạm vi thực tế được khai báo của split v3**, ưu tiên full Food-101 nếu v3 thật sự chứa 101 lớp. Kiểm tra v3 và lineage/evidence của nó, không làm sạch lại từ đầu và không tự coi v3 là PASS chỉ vì tên gọi “cleaned”.

Repository: `https://github.com/23001853-wq/food-recognition-dietary-assistant`

**Đây là bước nghiệm thu bổ sung Phase 4, KHÔNG phải Phase 4.5 (ML Image Processing) hoặc Phase 5 (Orchestration). Dừng sau bước này.**

## Trạng thái và ràng buộc đã biết

- Phase 3 đã PASS theo báo cáo Extract; không được tải dữ liệu thật. Academic approval/download vẫn phải được tôn trọng.
- Phase 4 đã báo cáo PASS **chỉ đối với POC** tại `data/splits/poc-v2/`: 10 lớp, 10.000 ảnh, train/val/test = 6.000/1.500/2.500; 111 regression tests PASS; lần chạy lại SKIPPED với fingerprint phù hợp.
- Phase 4 tạo `src/food_assistant/data_pipeline/transform.py`, `scripts/transform_food101.py`, `configs/food101_elt_poc_v2.yaml`, cùng report POC.
- Evidence raw được ghi nhận trong báo cáo trước: `source_fingerprint = f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`; `inventory_sha256 = b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`.
- Báo cáo lịch sử ghi **FAIL leakage của official train/test trên full dataset gốc**, trong khi kiểm tra POC không phát hiện exact duplicates xuyên split.
- Các bước Phase 2.x đã tạo/đánh giá split v2/v3. Theo mục đích dự án, **v3 là split đã làm sạch**, nhưng phải tự đọc bằng chứng/contract v3 để xác định chính xác policy, scope, các ảnh loại bỏ và trạng thái audit.
- Workspace có thể chứa thay đổi chưa commit; giữ nguyên toàn bộ thay đổi tồn tại trước khi triển khai.

**Không suy diễn rằng v3 có đủ 101 lớp, 101.000 ảnh hoặc không còn leakage chỉ từ tên thư mục.** Dùng số lượng và điều kiện hợp lệ do v3 contract/cleanup policy quy định.

## Bước 0 — Đọc và đối chiếu trạng thái thực tế (bắt buộc)

1. Đọc guardrails, master ELT plan và báo cáo Phase 0–4, đặc biệt Phase 2.5/2.6/2.7 liên quan làm sạch, provenance, policy cho duplicate/near-duplicate và split v3.
2. Tìm **đúng vị trí thực tế** của v3 trong worktree (có thể là `data/splits/v3/` hoặc đường dẫn khác). Liệt kê các file hiện hữu và định dạng; không tự giả định manifest schema của v3 giống `poc-v2`.
3. Kiểm tra implementation hiện tại của:
   - `src/food_assistant/data_pipeline/transform.py`
   - `scripts/transform_food101.py`
   - `src/food_assistant/ml/datasets.py` (`audit_manifest`, `ManifestDataset`)
   - verifier, split builder v2/v3, config loader, atomic writer, process/resource lock, tests liên quan.
4. Xác định rõ: v3 là full Food-101 hay subset; số lớp; số image IDs mỗi split; nguồn official meta; ảnh được loại/giữ/chuyển split theo policy nào; bằng chứng nào đã hoàn tất và bằng chứng nào chỉ là fixture/synthetic.
5. Chỉ sau khi có thông tin này mới quyết định có thể nghiệm thu **full scope** hay phải xuất `INCOMPLETE` với lý do cụ thể.

## Phạm vi được phép

- **Đọc** raw Food-101, split v1/v2/v3, `poc-v2`, evidence/checkpoints, existing configs và reports.
- **Tạo** config riêng và báo cáo run-specific cho full-v3 validation; có thể tạo manifest mới trong một **đường dẫn version mới chưa tồn tại** khi chất lượng đã PASS và policy cho phép.
- **Sửa tối thiểu** logic Phase 4 để hỗ trợ đọc v3 + quality gate trong phạm vi riêng; reuse audit, publisher/locking và deterministic serialization hiện có.
- **Không chỉnh sửa, xóa, di chuyển hoặc ghi đè:** `data/raw/**`, `data/splits/v1/**`, `data/splits/v2/**`, `data/splits/v3/**`, `data/splits/poc-v2/**`, model artifacts, source validation history.
- Không tự loại thêm ảnh, đổi nhãn, resplit, reassign train/val/test hay chỉnh official test cho “đủ PASS”. Nếu v3 có bất thường, báo cáo đúng sự thật và dừng trước publish.
- Không download, giải nén lại, train model, chạy inference benchmark, hoặc chuyển sang Phase 4.5/5.

## Yêu cầu implementation

### 1. Full v3 preflight — read-only, không publish

- Hỗ trợ CLI `--dry-run`/`--preflight` hoặc cờ tương đương để kiểm tra v3 mà không thay đổi raw/split/manifest đã công bố.
- Kiểm tra schema/metadata, version, class list, mapping contiguous, unique IDs, relative paths, path containment, expected split names, source lineage.
- Tính và báo cáo **counts thực tế**: class count, total rows, train/val/test counts, per-class counts; đối chiếu với **expected counts của v3 policy**, không mặc định 101.000 hoặc 750/250 khi v3 có loại bỏ hợp lệ.
- Đối chiếu v3 với official Food-101 train/test IDs và **approved cleanup policy**: ảnh test nào được giữ/loại, ảnh train nào được giữ/loại; không tự mặc định v3 giữ nguyên mọi official test ID nếu chính sách v3 đã mô tả ngoại lệ.
- Phân biệt `PASS` (phạm vi đủ và evidence hợp lệ), `FAIL` (vi phạm contract) và `INCOMPLETE` (thiếu evidence, không đủ scope hoặc không thể kết luận). Không báo PASS cho subset nếu claim là full 101 lớp.

### 2. Leakage check — xác minh split v3, không làm sạch lại

- Reuse cơ chế exact-duplicate SHA-256 và capture-group leakage đã có; nếu có bằng chứng near-duplicate/semantic dedup từ Phase 2.x thì đọc và đối chiếu theo policy, không tự tuyên bố near-duplicate-free nếu chỉ kiểm exact bytes.
- Kiểm tra duplicate/capture group **xuyên train/val/test** theo rule v3; report chi tiết image IDs, split pairs, hashes/group IDs (nếu tìm thấy).
- Duplicate nằm **trong cùng split** phải thống kê và xử lý theo policy đã duyệt; không tự biến chúng thành lỗi hoặc tự xóa.
- Xác minh quy tắc test isolation: không dùng test để chọn ngưỡng dedup, chọn classes, tuning config hoặc training decisions.
- Nếu evidence cũ trên full *official uncleaned* báo `FAIL`, **không áp FAIL đó máy móc cho v3**; kiểm v3 theo danh sách đã làm sạch và provenance. Đồng thời **không xóa/che giấu** historical FAIL.

### 3. Integrity & evidence reuse

- Reuse checkpoint/inventory/fingerprint Phase 2.5 và audit evidence v3 nếu chứng minh được: schema version, source identity, root scope, manifest/split fingerprint, completion, validation policy/config hash và counts đều tương thích.
- Ghi rõ phạm vi chứng cứ: `verified_from_existing_evidence`, `newly_checked`, `not_checked` cho các phần kiểm tra.
- Không full rehash 101.000 ảnh chỉ để test code; dùng fixture nhỏ cho unit/regression tests.
- **Không suy ra ảnh đã được hash/decoded đầy đủ chỉ từ `file count`, `mtime`, marker hoặc tên report**. Nếu thiếu evidence cần thiết, đánh `INCOMPLETE` và xuất hướng dẫn full recheck như một hành động **opt-in riêng**; không tự chạy full scan tốn tài nguyên.
- Các lỗi missing/corrupt images, checksum mismatch, source/root mismatch, EXIF/decode không đạt policy phải phân loại đúng `FAIL/INCOMPLETE`; cảnh báo EXIF/TIFF phải được lưu, không tự coi ảnh hỏng nếu vẫn qua policy.
- Nếu raw fingerprint được đọc từ evidence chứ không tính lại trong run, trình bày đúng là “reused prior evidence”, không mô tả thành raw full re-verification mới.

### 4. New full-v3 manifest publication — chỉ sau PASS

- Nếu preflight/quality gate đạt PASS **trên đúng full scope v3**, có thể build một phiên bản manifest mới riêng (ví dụ `data/splits/full-v3-validated/`, nhưng kiểm tra tên version/đường dẫn với project conventions trước).
- **Không sao chép/biến đổi toàn bộ ảnh**; manifest trỏ an toàn đến ảnh gốc hợp lệ. Giữ chính xác assignments của v3, **không stratified split lại**.
- Dùng staging, checksum, uniqueness checks, source fingerprint, config hash, run metadata, policy provenance và atomic publish dưới shared lock; không tạo phiên bản đích “nửa chừng”.
- Run lại cùng input/config/version => verify + `SKIPPED`; input/config khác trên version đích đã có => `CONFLICT/FAIL`, không overwrite.
- Nếu v3 không đủ bằng chứng hoặc FAIL => **không publish một manifest được đánh dấu PASS**; cho phép ghi báo cáo chẩn đoán ở report directory tách biệt theo run ID, an toàn/atomic.
- Bảo đảm consumer contract phù hợp `ManifestDataset` cho manifest mới mà không làm ảnh hưởng v1 hoặc model cũ. Root-relative paths hoặc explicit root override, không phụ thuộc absolute Windows path cũ.

### 5. Báo cáo có bằng chứng, không chỉ in PASS

Báo cáo full-v3 cần tối thiểu:

- `run_id`, timestamp UTC, source commit/dirty state (nếu khả dụng), scope yêu cầu và scope thực tế.
- `input_split_version`, path nguồn v3, lineage/version of cleanup policy.
- `class_count`, `image_count`, split counts, per-class counts, removed/retained counts so với official meta, chênh lệch có giải thích.
- `source_fingerprint`, `inventory_sha256` và nguồn evidence (newly verified hay reused), manifest/config/policy hash.
- `leakage_exact_cross_split_count`, capture-group count, số lượng same-split duplicates; nếu có near-duplicate audit phải nêu rõ phạm vi và mức bằng chứng.
- `images_verified`, `images_skipped_by_evidence`, `images_not_verified`, cùng lý do; schema/decode/EXIF warning counters.
- `status`: `PASS`, `FAIL` hoặc `INCOMPLETE`; `errors`, `warnings`, `evidence_paths`; `published_version` nếu có.
- Phân biệt rõ status cho **POC** và **full-v3**; không dùng PASS POC như bằng chứng PASS full.

## Tests bắt buộc — chỉ synthetic/fixture nhỏ khi đủ

1. Synthetic v3 (2–3 classes) với cleanup ledger hợp lệ => quality gate PASS theo fixture scope, output deterministic.
2. Fixture v3 chứa exact duplicate xuyên train/test hoặc train/val => FAIL.
3. Duplicate nội bộ train => report và áp dụng đúng policy, không tự xóa.
4. Capture-group cross-split => FAIL khi policy cấm.
5. Official metadata reconciliation: removed image hợp lệ có policy + provenance => PASS cho expected count; thiếu lý do => FAIL/INCOMPLETE phù hợp contract.
6. Scope mismatch: 10 lớp nhưng được yêu cầu `full 101` => INCOMPLETE, tuyệt đối không PASS.
7. Missing/corrupt image, wrong image checksum, unsafe path traversal hoặc mapping sai => FAIL/INCOMPLETE đúng policy.
8. Evidence cache sai fingerprint, thiếu completion, schema khác => không reuse, không PASS giả.
9. Rerun cùng input/config/version => same output fingerprint và `SKIPPED`; config/source khác nhưng cùng output version => conflict.
10. Crash/exception trong staging/publish => không có version công bố dở; source split và outputs cũ không đổi.
11. `ManifestDataset` đọc được manifest mới bằng DataLoader smoke thực, kiểm shape/label tối thiểu; backward compatibility v1/POC không đổi.
12. Lock contention hoặc concurrent publishers cùng target => một writer, không overwrite/corruption.
13. Report `FAIL/INCOMPLETE` vẫn ghi rõ evidence và không sinh manifest “PASS”.

## Thứ tự chạy

1. **Inventory read-only:** khám phá hiện trạng v3, đọc prior evidence và lập bảng `expected vs actual`.
2. **Preflight:** chạy quality gate có giới hạn an toàn; in scope, counts, leakage và trạng thái. Nếu evidence thiếu, dừng tại `INCOMPLETE`, không tự full rehash.
3. **Implementation/tests** tối thiểu cần thiết để hỗ trợ full-v3 audit; lint + targeted tests + regression tests.
4. **Publish only if PASS:** tạo manifest immutable phiên bản mới bằng staging + atomic publish; kiểm tra idempotent rerun.
5. **Post-run safety:** đối chiếu hash/fingerprint/file inventory liên quan trong phạm vi nhẹ, hoặc reuse baseline đủ tin cậy; xác nhận không thay đổi raw, split v1/v2/v3, POC v2 hoặc model artifacts.
6. Viết report tại `artifacts/reports/elt/phase_4_full_v3_validation.md` hoặc vị trí report phù hợp; nếu đã tồn tại, không ghi đè bừa bãi và tuân thủ project policy.

## Cổng nghiệm thu

- [ ] Đã xác minh scope thực tế của v3, không giả định v3 = full 101 lớp nếu không có evidence.
- [ ] Đã đối chiếu counts với policy v3 và reconciliation với official metadata.
- [ ] Đã kiểm leakage trên **v3** hoặc reuse evidence hoàn chỉnh gắn đúng v3; historical official leakage FAIL vẫn được bảo lưu.
- [ ] Integrity/quality gate có trạng thái trung thực PASS/FAIL/INCOMPLETE và đường dẫn evidence.
- [ ] Nếu PASS full scope: immutable manifest mới publish atomically, rerun SKIPPED, conflict không overwrite.
- [ ] Nếu FAIL/INCOMPLETE: có report chẩn đoán, không publish manifest PASS.
- [ ] Test synthetic + regression PASS, compatibility v1/POC PASS; không network, không full scan vô cớ.
- [ ] Không thay đổi raw, split v1/v2/v3, `poc-v2`, model artifacts.

## Định dạng báo cáo kết thúc (bắt buộc)

```text
PHASE: 4 — Full Validation on cleaned split v3
BASELINE: ...
FILES REVIEWED: ...
V3 SCOPE DISCOVERED: ...
INPUT EVIDENCE: ...
FILES CHANGED: ...
BEHAVIOR BEFORE → AFTER: ...
TESTS EXECUTED: ...
VALIDATION RESULTS: expected vs actual / class & split counts / leakage / integrity / evidence scope
PUBLISH RESULT: PASS+PUBLISHED / PASS+SKIPPED / FAIL+NOT PUBLISHED / INCOMPLETE+NOT PUBLISHED
SAFETY CHECK: ...
ACCEPTANCE: PASS / FAIL / INCOMPLETE (nêu rõ là full-v3 hay subset)
RISKS / OPEN ITEMS: ...
NEXT PHASE: 4.5 — ML Image Processing & Preprocessing (chỉ đề xuất, không triển khai)
```

**DỪNG SAU BƯỚC FULL VALIDATION NÀY. Không tự động chuyển sang Phase 4.5 hoặc Phase 5.**
