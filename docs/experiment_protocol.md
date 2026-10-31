# Experiment Protocol

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

## 8. DATASET: FOOD-101 VÀ QUẢN LÝ DỮ LIỆU

### 8.1. Đặc điểm đã xác minh

- **101 class, 101.000 ảnh**: mỗi class 750 ảnh train, 250 ảnh test.
- Dataset gốc có `train = 75.750`, `test = 25.250`.
- Ảnh train có thể chứa label noise/chất lượng không đồng đều; test được kiểm tra nhãn thủ công.
- `torchvision.datasets.Food101` hỗ trợ `split='train'` và `split='test'`; một số nền tảng khác gọi test chuẩn là `validation`. Trong **project này dùng tên nội bộ chuẩn `test`** để tránh nhầm.
- Food-101 **không** cung cấp nhãn gram, kcal, bounding box hay mask chuẩn để train portion/detection trực tiếp.
- Kiểm tra license và điều khoản chia sẻ ảnh/dữ liệu trước khi đóng gói/distribute; không đưa hàng GB ảnh vào Git.

### 8.2. Chiến lược chia tập

**Stage A — POC subset (10–20 class):** Chọn class có trong Food-101 (phải lưu `classes_poc.yaml`); từ `train` gốc, chia stratified `train_internal` và `val_internal`; dùng đúng các class tương ứng trong test gốc để test cuối. Không tùy ý chọn ảnh dễ làm mất tính đại diện. Báo cáo rõ là **subset result**, không ghi là Food-101 full benchmark.

**Stage B — Full 101:** Tương tự với toàn bộ classes. Đề xuất `80%/20%` *trong train gốc* cho train/validation. Với 750 ảnh/class: **600 train + 150 val/class**, cộng **250 test/class** giữ nguyên. Toàn bộ 101 class tương ứng: 60.600 train, 15.150 val, 25.250 test. Không đưa augmented image sang val/test.

### 8.3. Manifest và tránh leakage

- Sinh `data/splits/{version}/train.csv`, `val.csv`, `test.csv`: columns `image_path`, `class_id`, `class_name`, `source_split`, `sha256` (hash nếu ngân sách xử lý cho phép).
- Chia với seed cố định (ví dụ 42); ghi lại timestamp, script version, class order, dataset source/version.
- Duplicate exact/near-duplicate cross-split: kiểm tra tối thiểu exact hash; nếu phát hiện lẫn giữa splits, ghi sự cố và quy tắc xử lý nhất quán. Không âm thầm sửa benchmark test.
- Dùng augmentation *chỉ* ở train: resized random crop, horizontal flip phù hợp, nhẹ brightness/contrast; tránh rotation/color biến đổi quá mạnh làm sai món.
- Val/test dùng deterministic resize + center crop và normalization đúng pretrained weights.
- Đảm bảo 3 model dùng cùng tập class và các file test; với SVM scale ảnh riêng theo HOG là được, nhưng không thay đổi tập mẫu/nhãn.

### 8.4. Domain-shift evaluation

- Tạo `data/local_eval/` với ảnh tự chụp, có quyền sử dụng, đủ nhãn và điều kiện ánh sáng/góc chụp được ghi chú.
- Ghi metadata: `source`, `camera_optional`, `lighting`, `distance`, `dish`, `background`, `mixed_dish`, `reviewed_label`.
- Chạy test **riêng**, không gộp local_eval vào kết quả Food-101 test.
- Nếu đưa ảnh local vào huấn luyện thì phải tạo phiên bản thử nghiệm mới, giữ tập kiểm thử độc lập; không vừa train vừa khoe test trên cùng ảnh.

### 8.5. Nhãn món và mapping DB

- `class_id` = integer liên tục `0..C-1` theo class map *đúng artifact model*, không tự suy theo thứ tự bảng PostgreSQL.
- `food_classes.dataset_key`: tên gốc Food-101 (ví dụ `pizza`); `display_name_vi` có thể dịch để hiển thị.
- `nutrition_food_id` **không** mặc định = `class_id`. Nhiều món có nhiều công thức, kcal khác nhau; mapping cần curator xem xét.
- Nếu không tìm được nutrition reference phù hợp, hệ thống chỉ hiển thị nhãn và yêu cầu nhập thủ công.

## 9. THIẾT KẾ CÁC PHƯƠNG PHÁP AI

### 9.1. Baseline A: HOG + SVM (MUST)

Pipeline: Image RGB → grayscale → resize cố định (ví dụ 128×128) → HOG (orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)) → StandardScaler (nếu dùng) → LinearSVC → predict class.

- Fit mọi scaler và classifier **chỉ trên train_internal**; validation để tuning.
- Chọn `C` trên validation, không thử hyperparameter bằng test.
- HOG chủ yếu mô tả hình dạng/biên, thường hạn chế với món ăn giống texture/color.
- Lưu `hog_svm.joblib`, `hog_config.json`, `class_to_idx.json`; chỉ load artifact tự tạo/tin cậy do nguy cơ code execution từ pickle/joblib.
- Đo prediction latency trên cùng máy, cùng protocol.

### 9.2. CNN B: ResNet18 Transfer Learning (MUST để so sánh sâu)

- Load `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`.
- Thay classifier head `model.fc = Linear(in_features, C)`.
- Stage 1: freeze backbone, train head; Stage 2: tùy GPU, unfreeze last blocks hoặc entire network với LR thấp hơn.
- Loss: CrossEntropyLoss; optimizer AdamW hoặc SGD momentum; scheduler tùy thực nghiệm.
- Save best checkpoint theo **val macro-F1** hoặc val loss; metric lựa chọn phải định nghĩa trước.
- Preprocess train/validation/inference phù hợp thông số weights (đặc biệt normalization).
- Lưu `resnet18_best.pth`, config, label map, preprocess metadata.

### 9.3. CNN C: EfficientNet-B0 Transfer Learning (SHOULD, ứng viên ưu tiên)

- Load `efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`.
- Thay `model.classifier[-1]` thành Linear(out_features=C).
- Stage 1 train head, Stage 2 fine-tune các block cuối/toàn mạng tùy GPU.
- Chú ý preprocessing của weights (resize 256, center crop 224, bicubic, ImageNet mean/std); **đóng gói preprocess config cùng model**.
- Save best checkpoint theo cùng criterion như ResNet18.
- Không tự khẳng định EfficientNet-B0 sẽ tốt hơn ResNet18 trên Food-101; phải đo thực tế.

### 9.4. Hyperparameters (đề xuất khởi đầu, không cố định)

| Parameter | ResNet18 | EfficientNet-B0 | Ghi chú |
|---|---|---|---|
| Input | 224×224 | 224×224 | Theo transform cụ thể weights |
| Batch size | 16–64 | 16–64 | Tùy VRAM/CPU |
| Epoch | 10–25 giai đoạn đầu | 10–25 | Có early stopping |
| Initial LR head | 1e-3 (thử) | 1e-3 (thử) | Tuning trên val |
| Fine-tune LR | 1e-4 đến 1e-5 (thử) | 1e-4 đến 1e-5 | Nhỏ hơn head |
| Optimizer | AdamW | AdamW | Ghi version và config |
| Weight decay | 1e-4 (thử) | 1e-4 (thử) | Không lấy test tuning |
| Criterion | CrossEntropyLoss | CrossEntropyLoss | Có thể class weights nếu cần |
| AMP | Nếu CUDA phù hợp | Nếu CUDA phù hợp | Benchmark memory |
| Checkpoint selection | Val macro-F1 | Val macro-F1 | Thống nhất |
| Random seed | 42 + ghi lại | 42 + ghi lại | Có thể chạy nhiều seeds nếu khả thi |

### 9.5. Artifact contract

Mỗi model release gồm:

```text
artifacts/models/<model_name>/<run_id>/
├── model.pth                   # PyTorch state_dict (CNN); SVM dùng model.joblib
├── model_config.json           # architecture, num_classes, input_size, run_id
├── class_to_idx.json           # dataset_key -> integer class id
├── idx_to_class.json           # integer -> dataset_key
├── preprocessing.json          # resize/crop/interpolation/normalize
├── train_config.yaml           # seed, split, hyperparameters
├── metrics_val.json            # validation metrics
├── metrics_test.json           # test metrics (chỉ chạy khi kết thúc tuning)
├── source_commit.txt           # Git commit
└── checksum.sha256             # integrity khi deploy
```

`artifacts/production/current.json` chỉ đến artifact **được chọn có chứng cứ**, app không hardcode đường dẫn checkpoint tùy tiện. `num_classes`, head dimension và label map phải khớp. `torch.load` chỉ đọc artifact tin cậy; ưu tiên `weights_only=True` khi khả dụng và dùng `state_dict` thay vì serialize toàn bộ module.

### 9.6. Protocol chọn production model

1. Freeze class set/split.
2. Train baseline HOG + SVM, ResNet18 và EfficientNet-B0.
3. Tune theo train/val; checkpoint theo cùng metric.
4. Chạy test **một lần cho các ứng viên đã chốt**, tránh lặp vòng tối ưu dựa trên test.
5. So sánh trên cùng test: Top-1, macro-F1, per-class F1, inference latency P50/P95, artifact size; tham khảo GPU/CPU.
6. Chọn production model theo điểm tổng hợp chất lượng và khả năng triển khai; ghi rationale.
7. Tạo production pointer và smoke test full app.

### 9.7. Có nên dùng mô hình lớn hơn?

Không nên thay bằng model cực lớn ngay đầu dự án. Nếu POC có dư GPU và mục tiêu học thuật cần mở rộng, có thể thử EfficientNet-B2/ConvNeXt-Tiny **sau** khi đã hoàn thành so sánh bắt buộc. Kết quả Top-1 ImageNet chỉ giúp sàng lọc backbone, không đủ quyết định hiệu quả Food-101.

---

## 18. PHƯƠNG PHÁP ĐÁNH GIÁ VÀ BÁO CÁO HỌC THUẬT

### 18.1. Metrics tối thiểu

- `Top-1 Accuracy = N(correct top-1) / N(total)`.
- `Precision_c = TP_c / (TP_c + FP_c)` khi denominator khác 0.
- `Recall_c = TP_c / (TP_c + FN_c)`.
- `F1_c = 2·Precision_c·Recall_c / (Precision_c + Recall_c)` khi mẫu số khác 0.
- `Macro-F1 = mean_c(F1_c)` trên **tất cả** class mục tiêu, `zero_division` cấu hình rõ.
- Confusion matrix: rows = true labels; columns = predicted labels.
- Optional: Top-3 Accuracy, per-class precision/recall, model size, P50/P95 latency, parameter count, wall-clock train time, ECE/calibration.

### 18.2. Quy trình lập bảng so sánh

| Experiment | Dataset | Classes | Top-1 | Macro-F1 | P50 ms | Size MB | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| HOG + SVM | Food-101 split v1 | C | TBD | TBD | TBD | TBD | Baseline |
| ResNet18 fine-tuned | Food-101 split v1 | C | TBD | TBD | TBD | TBD | CNN |
| EfficientNet-B0 fine-tuned | Food-101 split v1 | C | TBD | TBD | TBD | TBD | Candidate |

Tất cả `TBD` chỉ được điền bằng script đọc metric thực, không tự bịa hay lấy số accuracy ImageNet thế vào.

### 18.3. Fair comparison requirements

- Cả ba dùng cùng các file train/val/test cho cùng class list.
- Dùng đúng test chuẩn Food-101 tương ứng subset/full, giữ nguyên nhãn.
- Dùng validation cho tuning và model selection.
- Với SVM, trình bày preprocessing khác CNN là khác biệt bản chất phương pháp, **không** được đổi data split.
- Báo hardware CPU/GPU và protocol warmup/batch-size đo inference; tránh so latency trên máy khác nhau.
- Báo number of runs/seeds (một seed vẫn chấp nhận nếu giới hạn tài nguyên, nhưng nêu hạn chế).
- Nếu chọn model dựa trên test nhiều lần thì đó là test leakage; sử dụng validation cho lựa chọn.

### 18.4. Error analysis bắt buộc

Phân loại tối thiểu 5 nhóm lỗi:

1. `VISUAL_SIMILARITY`: lớp trông giống nhau, ví dụ cấu trúc/texture tương tự.
2. `MIXED_DISH`: nhiều món trên cùng ảnh hoặc thành phần trộn.
3. `PLATING_VARIATION`: bố trí trang trí/đĩa/góc chụp khác.
4. `LIGHTING_OCCLUSION`: thiếu sáng, che khuất, out of focus.
5. `DOMAIN_SHIFT`: món Việt hoặc ảnh điện thoại người dùng khác Food-101.

Chọn bộ ca điển hình *có ảnh và quyền trình bày*; mỗi ca ghi `true_label`, `predicted_label`, model score, nguyên nhân giả thuyết, biện pháp khả thi và giới hạn. Có thể thêm Grad-CAM để minh họa định tính sau MVP nhưng không bắt buộc và không dùng như bằng chứng nhân quả.

### 18.5. Báo cáo tổng kết (đề cương)

1. Introduction & problem statement.
2. Related work: classical CV vs CNN transfer learning.
3. Dataset, sampling, data quality and preprocessing.
4. Methods: HOG+SVM, ResNet18, EfficientNet-B0.
5. Experimental setup (split, seed, hardware, training config).
6. Results: Top-1, macro-F1, confusion matrix, per-class discussion.
7. Error analysis and mixed dish/plating/domain shift.
8. Application architecture, PostgreSQL and user journey.
9. Portion estimation assumptions and nutrition source limitations.
10. Privacy, security, reproducibility, limitations, future work.
11. Demo screenshots + references.

---
