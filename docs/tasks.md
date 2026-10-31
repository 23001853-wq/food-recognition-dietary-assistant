# Checklist triển khai

- [x] Đọc kế hoạch và tạo cấu trúc theo mục 15.
- [x] Tạo cấu hình mẫu, trang Streamlit khung, module placeholders và tài liệu.
- [ ] Xác minh toàn bộ dependency trong venv và tạo lockfile.
- [ ] Triển khai ORM, migrations và test DB riêng.
- [ ] Triển khai dữ liệu, ML, services, repositories và UI theo roadmap dưới đây.

## 20. ROADMAP TRIỂN KHAI THEO PHASE / TASK / DEPENDENCY

> Timeline gợi ý **10 tuần** cho nhóm nhỏ hoặc sinh viên, phụ thuộc GPU và khối lượng báo cáo. Đây là kế hoạch, không phải cam kết thời gian xử lý. Diagramming agent thể hiện thứ tự phụ thuộc, không biến số tuần thành số liệu bắt buộc.

### Phase 0 — Project setup & design (Tuần 1)

- [ ] P0-01: Tạo repository, `PLAN.md`, README skeleton, .gitignore.
- [ ] P0-02: Khóa phạm vi MVP, định nghĩa actors/user stories/use cases.
- [ ] P0-03: Tạo venv, dependency constraints, lint & pytest.
- [ ] P0-04: Chọn storage strategy và database schema.
- [ ] P0-05: Thiết kế sơ đồ context/container/ERD/user-flow bản đầu.
- [ ] P0-06: Thiết lập PostgreSQL + test DB + Alembic.

**Exit criteria:** Repo chạy test rỗng, DB migrate thành công, sơ đồ không có component Future trong MVP.

### Phase 1 — Dataset & EDA (Tuần 2)

- [ ] P1-01: Tải Food-101 từ nguồn đáng tin.
- [ ] P1-02: Verify số class, số ảnh/split, ảnh lỗi, thống kê ảnh.
- [ ] P1-03: Chọn POC classes bằng config; lưu list và rationale.
- [ ] P1-04: Tạo stratified train_internal/val_internal từ original train.
- [ ] P1-05: Lock original test set, manifest paths và seed.
- [ ] P1-06: Visual EDA các lớp, balanced counts, ảnh lỗi/noisy labels.
- [ ] P1-07: Test split leakage và class consistency.

**Exit criteria:** Có manifests reproducible, EDA và invariant train/val/test không overlap.

### Phase 2 — Traditional baseline (Tuần 3)

- [ ] P2-01: Implement HOG extractor, resize/preprocess.
- [ ] P2-02: Build linear SVM pipeline, train trên POC.
- [ ] P2-03: Tune C trên val, log config và seed.
- [ ] P2-04: Export trusted model artifact + label map.
- [ ] P2-05: Evaluate once on locked test sau khi chọn config.
- [ ] P2-06: Confusion matrix, per-class F1, failure samples.

**Exit criteria:** Baseline có metric thật, artifact load/predict thành công.

### Phase 3 — CNN comparisons (Tuần 4–5)

- [ ] P3-01: Implement dataset loader/transforms và model factory.
- [ ] P3-02: ResNet18 transfer-learning head warmup.
- [ ] P3-03: Fine-tune selected layers, val early stopping.
- [ ] P3-04: EfficientNet-B0 pretrained head warmup.
- [ ] P3-05: Fine-tune EfficientNet, log val metrics.
- [ ] P3-06: Evaluate locked test và benchmark inference.
- [ ] P3-07: So sánh 3 methods, chọn production artifact.
- [ ] P3-08: Nếu đủ tài nguyên, mở rộng từ POC lên full 101, train/eval mới, không trộn kết quả POC/full.

**Exit criteria:** Báo cáo so sánh minh bạch, production pointer valid.

### Phase 4 — PostgreSQL + nutrition curation (Tuần 6)

- [ ] P4-01: Viết SQLAlchemy ORM + migration theo mục 12.
- [ ] P4-02: Tạo repository, unit of work, transaction rollback.
- [ ] P4-03: Seed demo_user + food class list theo artifact.
- [ ] P4-04: Curate một tập nutrition records **có nguồn**.
- [ ] P4-05: Implement `NutritionService` + formula + NULL behavior.
- [ ] P4-06: Implement meal CRUD và dashboard aggregation.
- [ ] P4-07: Integration tests trên DB test.

**Exit criteria:** CRUD và analytics chạy, nguồn dinh dưỡng không bị bịa.

### Phase 5 — Streamlit application (Tuần 7)

- [ ] P5-01: Tạo navigation/pages và app configuration.
- [ ] P5-02: Image upload, robust validation, preview, crop.
- [ ] P5-03: Load production model cached; inference top-k.
- [ ] P5-04: Label confirmation/unknown handling.
- [ ] P5-05: Portion editor + kcal estimate + disclaimer.
- [ ] P5-06: Save meal via service transaction.
- [ ] P5-07: Diary CRUD và Dashboard charts.
- [ ] P5-08: Error, loading, no-data, rerun state, save idempotency.

**Exit criteria:** End-to-end demo hoạt động với PostgreSQL và model thật.

### Phase 6 — Evaluation & robustness (Tuần 8)

- [ ] P6-01: Test các ảnh người dùng/chụp thật riêng.
- [ ] P6-02: Mixed dish/crop testing; ghi các giả định.
- [ ] P6-03: Error taxonomy, confusion matrix, qualitative cases.
- [ ] P6-04: Latency P50/P95 trên máy demo.
- [ ] P6-05: Validate protection: invalid files, missing model, DB failures.
- [ ] P6-06: Kiểm tra dataset license, image privacy, nutrition provenance.

**Exit criteria:** Có robustness report, không tuyên bố quá giới hạn.

### Phase 7 — Docs, diagrams, delivery (Tuần 9–10)

- [ ] P7-01: Chốt README: cài đặt, data, train, evaluate, run.
- [ ] P7-02: Chốt report/hypothesis/methods/results/limitations.
- [ ] P7-03: Xuất 12 sơ đồ Mermaid theo phần 23; render SVG/PNG nếu có công cụ.
- [ ] P7-04: Review diagram với implementation thực, cập nhật edge/ERD.
- [ ] P7-05: Screenshots và demo script 3–5 phút.
- [ ] P7-06: Full test suite, migration from blank DB, smoke test fresh clone.
- [ ] P7-07: Zip source code + docs + configs + artifact checklist; không nhúng secrets/raw dataset trái phép.

**Exit criteria:** Người khác cài và chạy được theo README (khi có model artifact/dataset), sơ đồ tương ứng code thực.

### 20.1. Critical path

`P0 setup → P1 split → P2/P3 training & eval → production artifact → P5 inference UI`.

`P0 DB schema → P4 migrations/seed/services → P5 save diary/dashboard`.

`P2/P3 experiment metrics + P5 app screenshots + P6 analysis → P7 final report/diagrams`.

### 20.2. Parallelizable work

- UI mockups có thể làm song song dataset EDA; không cần model giả được công bố là thật.
- PostgreSQL/schema/meal CRUD có thể làm song song CNN training.
- Nutrition curation có thể làm độc lập, nhưng mapping cần có food class list ổn định.
- Diagram draft có thể tạo sớm, final diagram phải verify theo code cuối.

---
