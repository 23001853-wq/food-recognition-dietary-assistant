# PHASE 4.5 — ML Image Processing, Preprocessing & Near-Duplicate Analysis

## VAI TRÒ VÀ MỤC TIÊU

Bạn là **Senior Computer Vision Engineer / ML Engineer**, phụ trách hoàn thiện phần xử lý ảnh và tiền xử lý ảnh của dự án:

Repository: https://github.com/23001853-wq/food-recognition-dietary-assistant

Làm việc trên repository và worktree hiện tại.

### Mục tiêu

Hoàn thiện, kiểm thử và lập báo cáo hệ thống xử lý ảnh phục vụ bài toán nhận diện món ăn, đáp ứng các yêu cầu thực hành Image Processing và tích hợp an toàn với ML pipeline hiện hữu.

Phạm vi gồm:

1. Image representation và color-space transformations.
2. Pixel-level transformations.
3. Geometric transformations.
4. Filtering và denoising.
5. Mathematical morphology.
6. Edge detection và region processing.
7. **Near-duplicate image detection bằng pHash, Hamming Distance và SSIM.**
8. Image preprocessing và augmentation cho Machine Learning.
9. Tích hợp với dataset manifest, DataLoader và các mô hình hiện hữu.
10. Kiểm thử, minh họa trước/sau, đánh giá định lượng và báo cáo kỹ thuật.

**Không triển khai lại các công việc Data Engineering đã được nghiệm thu trong Phase 4.**

Không tự động chuyển sang Phase 5, huấn luyện mô hình hoặc thay đổi dữ liệu đã công bố.

---

# 0. BASELINE — TRẠNG THÁI DỰ ÁN

## 0.1. Kết quả Phase 4 gần nhất

Theo báo cáo Phase 4 — Full Validation on Cleaned Split v3:

| Thành phần | Kết quả được báo cáo |
|---|---|
| Dataset | Food-101 |
| Tổng ảnh gốc | 101.000 |
| Tổng ảnh split v3 | 100.990 |
| Số lớp | 101 |
| Train | 68.166 |
| Validation | 7.574 |
| Test | 25.250 |
| Exact duplicate xuyên split | 0 |
| Exact duplicate nội bộ split | 0 |
| Regression/synthetic tests | 62 passed |
| Ruff | PASS |
| Full-v3 acceptance | PASS theo exact-duplicate policy |
| Publication | PASS+SKIPPED |

Split v3 đã được làm sạch theo cleanup policy và exclusion ledger.

Official test được giữ nguyên.

Đã loại 10 image references so với metadata gốc: 2 từ quá trình xử lý v2 và 8 bổ sung ở v3.

**Lưu ý:** Các số liệu trên là baseline được báo cáo, không phải bằng chứng thay thế cho việc đọc manifest và báo cáo thực tế trong worktree.

## 0.2. Evidence và giới hạn

Đọc báo cáo:

`artifacts/reports/elt/phase_4_full_v3_validation_final.md`

Đối chiếu:

- Config `configs/food101_elt_full_v3.yaml`.
- Manifest và metadata v3.
- Cleanup policy và exclusion ledger.
- Source fingerprints và manifest fingerprints.
- Các báo cáo Phase 2.5, 2.6, 2.7 và Phase 4.

Giới hạn đã biết:

- Exact duplicate được kiểm tra bằng SHA-256 và evidence hiện có.
- Capture-group IDs chưa có trong manifest nên chưa kết luận được capture-group leakage.
- Near-duplicate chưa được audit trên full-v3.
- Không có đầy đủ số liệu cảnh báo EXIF/TIFF lịch sử.

Không được diễn giải `PASS` của exact-duplicate thành khẳng định toàn bộ ảnh đã không còn near-duplicate.

## 0.3. Điều kiện để bắt đầu

Xác minh manifest v3 thực sự tồn tại và đúng với báo cáo.

**Không yêu cầu tạo manifest mới chỉ vì Phase 4 trả về `PASS+SKIPPED`.**

Nếu manifest v3 hiện hữu đã được xác thực và phù hợp consumer contract, được phép dùng trực tiếp ở chế độ read-only.

Nếu thiếu evidence cần thiết, báo `INCOMPLETE` cho phần tích hợp thực tế, nhưng vẫn có thể triển khai và kiểm thử các hàm xử lý ảnh độc lập bằng synthetic fixtures.

## 0.4. Bảo vệ dữ liệu

Nghiêm cấm chỉnh sửa, xóa, ghi đè hoặc phân chia lại:

- `data/raw/**`
- `data/splits/v1/**`
- `data/splits/v2/**`
- `data/splits/v3/**`
- `data/splits/poc-v2/**`
- Mọi manifest đã công bố.
- Model weights và model artifacts.
- Official train/test metadata.
- Báo cáo nghiệm thu lịch sử.

Không tự download dataset, cài package, tải pretrained weights, full rehash 101.000 ảnh, train model hoặc chạy full inference benchmark.

Chỉ sử dụng fixture nhỏ hoặc một tập mẫu có kiểm soát cho các thao tác minh họa.

Giữ nguyên mọi thay đổi đã tồn tại trong working tree trước khi bắt đầu.

---

# 1. KHẢO SÁT IMPLEMENTATION HIỆN TẠI

Trước khi viết code, đọc và phân tích các thành phần:

- `src/food_assistant/ml/transforms.py`
- `src/food_assistant/ml/datasets.py`
- `src/food_assistant/ml/hog_features.py`
- `src/food_assistant/ml/train_utils.py`
- `src/food_assistant/services/model_registry.py`
- `tests/unit/test_poc_preprocessing.py`

Cùng các config, tests, notebook và module xử lý ảnh liên quan.

Đặc biệt xem xét:

- `preprocessing()`
- `build_transform()`
- `ManifestDataset`
- `ImageOps.exif_transpose()`
- RGB conversion
- HOG feature extraction
- `preprocessing.json`
- `Release.predict()`
- Train/eval/inference transform configuration

Lập bảng:

| Chức năng | File/hàm hiện tại | Trạng thái | Hành động |
|---|---|---|---|
| RGB conversion | ... | ALREADY / PARTIAL / MISSING | ... |
| Normalization | ... | ... | ... |
| Image augmentation | ... | ... | ... |
| Filtering | ... | ... | ... |
| Near-duplicate detection | ... | ... | ... |

Sử dụng các trạng thái:

- `ALREADY IMPLEMENTED`
- `PARTIAL`
- `MISSING`
- `MUST PRESERVE`

**Ưu tiên tái sử dụng implementation hiện hữu. Không nhân đôi pipeline hoặc thay đổi hành vi legacy khi không cần thiết.**

Trước khi sửa, xác định rõ kiến trúc dự kiến và các file cần thay đổi.

---

# 2. PHÂN CHIA TRÁCH NHIỆM TRONG ELT

## 2.1. Data Engineering — Phase 4

Chịu trách nhiệm:

- Data integrity.
- Missing/corrupt images.
- SHA-256 exact duplicate detection.
- Split policy.
- Train/validation/test leakage.
- Dataset manifest.
- Versioning.
- Lineage.
- Immutable publication.
- Quality gate.

Phase 4.5 chỉ sử dụng kết quả của các bước này.

## 2.2. Computer Vision — Phase 4.5

Chịu trách nhiệm:

- Image representation.
- Color transformations.
- Pixel operations.
- Filtering.
- Geometric operations.
- Morphology.
- Edge detection.
- Near-duplicate detection demonstration.
- ML preprocessing.
- Image augmentation.
- Integration và compatibility tests.

Near-duplicate analysis của Phase 4.5 chỉ được thực hiện ở chế độ **experimental, read-only**.

Nếu phát hiện các cặp ảnh đáng ngờ trong v3, ghi thành báo cáo phục vụ Data Quality review.

**Không tự động xóa ảnh, đổi nhãn, gộp nhóm hoặc tạo lại split.**

Nếu cần làm sạch dataset sau audit, đề xuất một tác vụ Data Quality riêng với sự phê duyệt của người dùng.

---

# 3. IMAGE REPRESENTATION & COLOR SPACES

## 3.1. Những thao tác bắt buộc

Triển khai hoặc tái sử dụng:

1. RGB → Grayscale.
2. RGB → HSV.
3. HSV → RGB.
4. RGB ↔ BGR khi cần tương thích OpenCV.
5. HWC ↔ CHW.
6. `uint8` → `float32`.
7. Chuyển pixel range 0–255 sang 0–1.
8. Chuyển đổi đầu vào grayscale/RGBA/P sang RGB khi phù hợp model contract.

Module có thể sử dụng:

`src/food_assistant/ml/image_ops.py`

Hoặc vị trí khác phù hợp kiến trúc hiện tại.

## 3.2. Yêu cầu kỹ thuật

- Nêu rõ channel ordering.
- Không nhầm RGB và BGR.
- Xử lý EXIF orientation trước các phép biến đổi phụ thuộc hướng.
- Không double-normalize.
- Xác định rõ shape, dtype, range và clipping.
- Có input validation và docstring.
- Không âm thầm thay đổi preprocessing của model cũ.

## 3.3. Kiểm thử

Tạo synthetic images với giá trị pixel xác định.

Kiểm tra:

- Correct output channels.
- Shape.
- Data type.
- Value range.
- HSV/RGB round-trip với dung sai hợp lý.
- EXIF orientation.
- Grayscale/RGBA handling.

Không bật HSV/grayscale trong production classification pipeline nếu không có căn cứ kỹ thuật.

---

# 4. PIXEL-LEVEL TRANSFORMATIONS

## 4.1. Các phép bắt buộc

- Brightness adjustment.
- Contrast adjustment.
- Gamma correction.
- Normalization.
- Pixel clipping.
- Xử lý các giá trị pixel ngoài miền hợp lệ.

## 4.2. Yêu cầu

Phân biệt:

- Deterministic pixel transformations.
- Random photometric augmentation.
- Normalization phục vụ model.

Các random transforms chỉ được sử dụng trong train.

Validation/test/inference phải deterministic với cùng cấu hình và đầu vào.

Không tự thay mean/std hoặc normalization profile của model cũ.

## 4.3. Tests

- Uniform image.
- Grayscale image.
- Extreme pixel values.
- Invalid gamma.
- Non-finite float input khi được hỗ trợ.
- Shape/dtype preservation.
- Correct clipping.
- No double-normalization.

---

# 5. GEOMETRIC IMAGE PROCESSING

## 5.1. Các phép bắt buộc

- Resize.
- CenterCrop.
- RandomResizedCrop.
- HorizontalFlip.
- Optional Rotation.

Có thể tái sử dụng torchvision transforms hiện tại.

## 5.2. Yêu cầu

Giải thích:

- Interpolation.
- Aspect ratio.
- Crop size.
- Output shape.
- Crop có thể làm mất đối tượng món ăn.
- Ảnh hưởng của geometric augmentation tới bài toán nhận diện món ăn.

Không áp dụng augmentation ngẫu nhiên cho validation/test/inference.

Rotation chỉ được bật khi opt-in và có cấu hình rõ ràng.

## 5.3. Tests

Sử dụng synthetic geometric fixtures.

Kiểm tra:

- Resize dimensions.
- Crop boundaries.
- Flip semantics.
- Rotation.
- Output shape/dtype.
- Deterministic evaluation pipeline.

---

# 6. FILTERING & DENOISING

## 6.1. Các bộ lọc bắt buộc

- Gaussian Blur.
- Median Filter.

Có thể sử dụng OpenCV, PIL hoặc thư viện hiện có.

## 6.2. Thí nghiệm

Tạo synthetic images và thêm:

- Gaussian noise.
- Salt-and-pepper noise.

Áp dụng các bộ lọc tương ứng.

Tạo hình minh họa:

- Clean image.
- Noisy image.
- Gaussian filtered image.
- Median filtered image.

## 6.3. Đánh giá

Sử dụng ít nhất một metric định lượng phù hợp, ưu tiên MSE so với clean reference trên fixture.

So sánh các kết quả.

Không được kết luận Gaussian hay Median luôn tốt hơn trong mọi điều kiện.

Kiểm thử:

- Kernel validation.
- Sigma validation.
- Border behavior.
- Dtype/shape preservation.
- Filtering effect.

**Không tự động áp dụng denoising trên toàn bộ Food-101.**

Filtering chỉ được đưa vào production ML pipeline nếu có justification và validation thích hợp.

---

# 7. MATHEMATICAL MORPHOLOGY, EDGES & REGIONS

## 7.1. Mathematical morphology

Triển khai hoặc tái sử dụng:

- Erosion.
- Dilation.
- Opening.
- Closing.

Thực hiện demo trên synthetic binary masks.

Yêu cầu giải thích:

- Structuring element.
- Kernel size.
- Foreground/background convention.
- Tác dụng của từng phép.

## 7.2. Edge detection

Có ít nhất một trong hai:

- Sobel.
- Canny.

Khuyến khích minh họa cả hai nếu thư viện hiện tại hỗ trợ và không làm tăng độ phức tạp không cần thiết.

## 7.3. Region processing

Minh họa một bài toán nhỏ có liên quan:

- Thresholding.
- Binary mask.
- Connected components.
- Region extraction.

Không bắt buộc triển khai segmentation model.

## 7.4. Giới hạn

Food-101 là bài toán image classification.

Do đó không mặc định rằng morphology, edge detection hoặc segmentation sẽ cải thiện CNN.

Chỉ tích hợp các kỹ thuật này vào production nếu có bằng chứng thực nghiệm và sự chấp thuận phù hợp.

## 7.5. Tests

Không chỉ kiểm tra chương trình chạy thành công.

Cần kiểm tra các thuộc tính toán học:

- Erosion làm co foreground trong fixture phù hợp.
- Dilation làm mở rộng foreground.
- Opening loại các chi tiết nhỏ được thiết kế trong fixture.
- Closing lấp khoảng trống nhỏ theo cấu trúc kernel.
- Edge detector phát hiện biên tại vị trí dự kiến.

---

# 8. NEAR-DUPLICATE IMAGE DETECTION

**Đây là phần bổ sung bắt buộc của Phase 4.5.**

## 8.1. Mục tiêu

Sử dụng kiến thức xử lý ảnh để nhận diện các cặp ảnh trùng hoặc gần giống nhau về mặt thị giác, kể cả khi file có SHA-256 khác nhau.

Chỉ triển khai thành module nghiên cứu/đánh giá chất lượng dữ liệu ở chế độ read-only.

Không tự động sửa manifest hoặc split v3.

## 8.2. SHA-256 baseline

Giải thích rõ:

SHA-256 xác định file trùng nhau về byte, nhưng không đảm bảo nhận diện được ảnh đã:

- Resize.
- Recompress JPEG.
- Crop.
- Điều chỉnh độ sáng.
- Chuyển đổi định dạng.

Không chạy lại full SHA-256 scan của Food-101.

Có thể tính SHA-256 trên synthetic fixtures để minh họa sự khác nhau giữa exact duplicate và near-duplicate.

## 8.3. Perceptual Hash (pHash)

Triển khai hoặc tái sử dụng một phương pháp perceptual hashing có định nghĩa rõ ràng.

Khuyến nghị sử dụng DCT-based pHash nếu thư viện hiện có cho phép.

Yêu cầu:

- Chuẩn hóa ảnh đầu vào.
- Chuyển grayscale phù hợp.
- Tính perceptual hash.
- Cố định hash size.
- Tạo biểu diễn hash nhất quán.
- So sánh hai hash bằng Hamming Distance.

Không nhầm SHA-256 với pHash.

Không dùng pHash như bằng chứng tuyệt đối rằng hai ảnh là cùng một ảnh.

## 8.4. Hamming Distance

Viết hàm tính khoảng cách giữa hai perceptual hashes.

Kiểm tra:

- Hash giống nhau → distance = 0.
- Hash khác nhau → distance theo số bit khác.
- Reject incompatible hash lengths.
- Output deterministic.

Ngưỡng near-duplicate phải được cấu hình, không hard-code thành quy tắc đúng cho mọi dataset.

## 8.5. Structural Similarity Index — SSIM

Triển khai hoặc tái sử dụng SSIM từ thư viện đã có.

Yêu cầu:

- Kiểm tra điều kiện kích thước ảnh.
- Có quy tắc resize/alignment rõ ràng cho các cặp ứng viên.
- Xác định grayscale hoặc multichannel.
- Cấu hình `data_range` chính xác.
- Xử lý ảnh rất nhỏ hoặc không đủ điều kiện tính SSIM.
- Giải thích hạn chế của SSIM đối với crop, rotation và viewpoint changes.

SSIM dùng làm tín hiệu xác minh cặp ứng viên, không thay thế manual review.

## 8.6. Quy trình near-duplicate demonstration

```text
Small image fixtures
        |
        v
Calculate SHA-256 (comparison only)
        |
        v
Calculate pHash
        |
        v
Hamming Distance
        |
        v
Candidate Pairs
        |
        v
SSIM Verification
        |
        v
Suspicious Pair Report
        |
        v
Manual Review Recommendation
```

Không triển khai tìm kiếm mọi cặp ảnh trên 100.990 ảnh theo brute-force O(N²).

Không tự chạy full dataset audit.

## 8.7. Test fixtures

Tạo bộ fixture nhỏ có các trường hợp:

1. Hai file giống hệt nhau.
2. Ảnh gốc và ảnh resize.
3. Ảnh gốc và ảnh JPEG recompression.
4. Ảnh gốc và ảnh thay đổi brightness.
5. Ảnh gốc và ảnh crop nhẹ.
6. Ảnh gốc và ảnh flip/rotate.
7. Hai ảnh hoàn toàn khác nhau.
8. Hai ảnh tương tự về màu sắc nhưng khác nội dung.
9. Hai ảnh có nhãn khác nhau nhưng dễ gây nhầm lẫn.

Không mặc định mọi biến đổi đều được pHash/SSIM phát hiện đúng.

Cần ghi nhận kết quả thực tế.

## 8.8. Đánh giá định lượng

Với fixture có ground-truth pair labels:

- True Positive.
- False Positive.
- True Negative.
- False Negative.
- Precision.
- Recall.
- False-positive examples.
- False-negative examples.

Không suy rộng precision/recall từ synthetic fixture thành hiệu năng trên toàn Food-101.

Nếu ngưỡng được lựa chọn trên fixture, phải ghi rõ ngưỡng đó chỉ có giá trị trong thử nghiệm này.

Không dùng official test để lựa chọn threshold.

## 8.9. Optional v3 sample audit

Nếu có thể đọc manifest v3 an toàn, cho phép kiểm tra một tập train/validation nhỏ, được chọn theo quy tắc có thể tái lập.

Yêu cầu:

- Read-only.
- Giới hạn số lượng ảnh và cặp ảnh.
- Cấu hình sample size.
- Seed cố định.
- Không full scan.
- Không xóa ảnh.
- Không thay đổi nhãn.
- Không resplit.
- Không cập nhật exclusion ledger.
- Không tuyên bố near-duplicate-free cho toàn bộ v3.

Nếu phát hiện suspicious cross-split pairs:

Xuất report riêng với:

- Image IDs.
- Split assignments.
- pHash distance.
- SSIM score.
- Review status.
- Limitations.

**Chỉ đề xuất Data Quality re-audit riêng nếu cần. Không tự sửa dataset.**

---

# 9. ML IMAGE PREPROCESSING & AUGMENTATION

## 9.1. Model compatibility

Kiểm tra các pipeline hiện hữu:

- ResNet18.
- EfficientNet-B0.
- HOG/SVM.

Đọc cấu hình preprocessing đã được lưu cùng model artifacts.

Kiểm tra khả năng tương thích với torchvision pretrained transform metadata đã có tại môi trường local.

Không tải pretrained weights.

## 9.2. Train preprocessing

Tái sử dụng pipeline hiện có, ví dụ:

```text
Read Image
    |
EXIF Orientation
    |
RGB Conversion
    |
RandomResizedCrop
    |
Optional HorizontalFlip
    |
ToTensor
    |
Normalize
    |
Model Input
```

Không mặc định thay đổi hành vi production hiện tại.

Các phép thử nghiệm mới chỉ được bật qua opt-in profile/config version mới.

## 9.3. Validation/Test preprocessing

```text
Read Image
    |
EXIF Orientation
    |
RGB Conversion
    |
Resize
    |
CenterCrop
    |
ToTensor
    |
Normalize
    |
Model Input
```

Pipeline phải deterministic với cùng input/config.

Không sử dụng augmentation ngẫu nhiên trong validation/test.

Không tune parameters dựa trên official test.

## 9.4. Inference preprocessing

Kiểm tra preprocessing tại inference phải tương thích với profile của model đã huấn luyện.

Không làm thay đổi hành vi của `Release.predict()` hoặc model registry nếu không cần thiết.

Kiểm tra:

- RGB conversion.
- Image size.
- Interpolation.
- Crop.
- Normalization.
- Channel order.
- Tensor dtype.

## 9.5. HOG/SVM

Bảo vệ pipeline HOG hiện tại.

Không thay đổi:

- Grayscale conversion semantics.
- Resize.
- HOG configuration.
- Feature dimensionality.

Không áp dụng CNN mean/std normalization cho HOG nếu pipeline cũ không yêu cầu.

## 9.6. Preprocessing profiles

Nếu cần tạo preprocessing profile mới:

- Có profile ID và schema/version.
- Có config hash.
- Có input manifest fingerprint.
- Có library version information.
- Có seed nếu sử dụng stochastic transform.
- Có backward compatibility tests.

Không ghi đè `preprocessing.json` hiện có.

Không thay đổi model weights.

---

# 10. INTEGRATION VỚI SPLIT V3 VÀ DATALOADER

## 10.1. Manifest contract

Xác nhận Phase 4 v3 manifest tương thích `ManifestDataset`.

Nếu không tương thích, báo cụ thể nguyên nhân.

Không tự thay đổi manifest v3 để làm test PASS.

## 10.2. DataLoader smoke test

Thực hiện ít nhất một DataLoader test thật với synthetic manifest hoặc sample nhỏ hợp lệ từ v3.

Không mock toàn bộ Dataset/DataLoader trong integration test.

Kiểm tra:

- Batch shape theo model profile.
- Ví dụ `[B, 3, 224, 224]` nếu config quy định.
- Tensor dtype.
- Finite values.
- Label mapping.
- Label dtype.
- Image orientation.
- Path resolution.
- Correct normalization.
- Train/eval transform selection.

Không hard-code 224 khi profile khác quy định kích thước khác.

## 10.3. Reproducibility

- Controlled random seed.
- Deterministic evaluation.
- Config serialization.
- Transform version.
- Metadata về môi trường.
- Test synthetic có kết quả ổn định.

Không khẳng định reproducibility tuyệt đối trên mọi hardware/multi-worker environment nếu chưa có bằng chứng.

---

# 11. VISUALIZATION & DEMONSTRATION

Tạo các hình before/after phục vụ báo cáo môn học.

Ít nhất có các nhóm:

1. RGB / Grayscale / HSV.
2. Brightness / Contrast / Gamma.
3. Resize / Crop / Flip.
4. Gaussian Noise / Gaussian Blur.
5. Salt-and-Pepper / Median Filter.
6. Erosion / Dilation / Opening / Closing.
7. Edge Detection.
8. SHA-256 vs pHash comparison.
9. Near-duplicate candidates và SSIM scores.
10. Train augmentation examples.

Các hình minh họa phải có:

- Input/source.
- Transform name.
- Transform parameters.
- Seed khi áp dụng.
- Output.
- Caption hoặc giải thích.
- Đánh dấu demo-only nếu không phải production preprocessing.

Không bắt buộc xuất đủ mọi ảnh bằng nhiều file riêng; có thể tổ chức thành các bảng hình minh họa rõ ràng.

Không ghi ảnh vào raw hoặc thư mục split bất biến.

Lưu demonstration artifacts trong thư mục reports riêng của Phase 4.5.

---

# 12. BỘ TEST BẮT BUỘC

Sử dụng Python 3.11 trong `.venv-311` nếu môi trường đó tồn tại và phù hợp.

Tận dụng thư viện hiện có.

Không cài package mới khi chưa được phê duyệt.

Tạo hoặc cập nhật các test tại những vị trí phù hợp:

- `tests/unit/test_ml_image_ops.py`
- `tests/unit/test_ml_transforms_contract.py`
- `tests/unit/test_near_duplicate_detection.py`
- `tests/integration/test_ml_image_pipeline.py`

Tên file có thể điều chỉnh theo cấu trúc repository.

## 12.1. Core image operations

- RGB/BGR distinction.
- RGB/HSV conversion.
- RGB/Grayscale.
- HWC/CHW.
- uint8/float32.
- Pixel clipping.
- EXIF handling.
- Grayscale/RGBA input.
- Invalid input handling.

## 12.2. Pixel transformations

- Brightness.
- Contrast.
- Gamma.
- Normalization.
- Extreme values.
- Invalid parameters.

## 12.3. Geometry

- Resize.
- Crop.
- Flip.
- Optional rotation.
- Shape preservation.
- Interpolation behavior.

## 12.4. Filtering

- Gaussian Blur.
- Median.
- Gaussian noise.
- Salt-and-pepper noise.
- MSE comparison.
- Invalid kernels.

## 12.5. Morphology/edge

- Erosion.
- Dilation.
- Opening.
- Closing.
- Sobel/Canny.
- Region/mask correctness.

## 12.6. Near-duplicate detection

- Exact duplicate.
- Different SHA-256 but near-identical image.
- pHash reproducibility.
- Hamming Distance correctness.
- Resize/recompression.
- Brightness modifications.
- Cropping limitations.
- False-positive examples.
- False-negative examples.
- SSIM input compatibility.
- Invalid thresholds.
- Read-only behavior.
- No implicit mutation of v3.

## 12.7. ML preprocessing contract

- ResNet18.
- EfficientNet-B0.
- HOG/SVM regression.
- Train-only augmentation.
- Eval determinism.
- Inference consistency.
- Legacy preprocessing compatibility.

## 12.8. Integration and safety

- Real DataLoader smoke test.
- ManifestDataset compatibility.
- Correct labels and paths.
- No raw mutation.
- No split mutation.
- No model artifact mutation.
- No network access.
- No training.
- No full dataset rehash.

Nếu test bắt buộc không thể chạy, phải báo `INCOMPLETE`, không đánh dấu PASS giả.

---

# 13. IMPLEMENTATION RULES

1. Đọc repository và báo cáo trước khi sửa.
2. Không giả định cấu trúc file chỉ từ prompt.
3. Không sửa những phần đã hoạt động đúng.
4. Tái sử dụng PIL/OpenCV/NumPy/Torchvision/scikit-image khi đã có sẵn.
5. Không viết lại thuật toán thư viện nếu không phục vụ yêu cầu môn học.
6. Không làm thay đổi preprocessing của model đang hoạt động.
7. Mọi thử nghiệm mới dùng separate functions hoặc opt-in profiles.
8. Không tự thay đổi Phase 4 data quality policy.
9. Không xóa, di chuyển hoặc thay đổi ảnh trong v3.
10. Không tự chạy tác vụ tốn tài nguyên trên toàn bộ dataset.
11. Không tự commit hoặc push.
12. Không che giấu test failures hoặc exceptions.
13. Không mô tả demo thành benchmark ML.
14. Không tuyên bố cải thiện accuracy/F1 nếu chưa có experiment phù hợp.
15. Ghi nhận các giới hạn kỹ thuật trung thực.

---

# 14. THỨ TỰ TRIỂN KHAI

## Step 0 — Preflight

- Kiểm tra repository/worktree.
- Đọc báo cáo Phase 4.
- Xác minh manifest v3.
- Kiểm tra Python environment.
- Kiểm tra dependencies.
- Ghi nhận baseline Git status.

## Step 1 — Architecture Review

- Đọc transforms hiện có.
- Đọc Dataset và model consumers.
- Lập feature matrix.
- Xác định thay đổi tối thiểu.

## Step 2 — Core Image Processing

- Representation.
- Color spaces.
- Pixel operations.
- Geometry.
- Filtering.
- Morphology.
- Edges.

Triển khai và kiểm thử từng nhóm.

## Step 3 — Near-Duplicate Analysis

- pHash.
- Hamming Distance.
- SSIM.
- Synthetic fixtures.
- Metrics.
- Demonstration.

Không thay đổi split v3.

## Step 4 — ML Integration

- Preprocessing profiles.
- ResNet18/EfficientNet compatibility.
- HOG/SVM regression.
- DataLoader integration.
- Inference preprocessing consistency.

## Step 5 — Validation

Chạy:

- Unit tests.
- Integration tests.
- Existing regression tests.
- Ruff/lint nếu có.
- Determinism checks.
- Data-safety checks.

## Step 6 — Report

Tạo báo cáo kỹ thuật, metrics và before/after visualizations.

Kiểm tra Git diff và các files đã thay đổi.

Dừng sau Phase 4.5.

---

# 15. ACCEPTANCE GATES

Chỉ báo PASS tổng thể khi tất cả điều kiện bắt buộc tương ứng đã đạt.

## Gate A — Core Image Processing

- [ ] Color spaces PASS.
- [ ] Pixel operations PASS.
- [ ] Geometric transforms PASS.
- [ ] Filtering PASS.
- [ ] Morphology/edge demos PASS.

## Gate B — Near-Duplicate Detection

- [ ] pHash hoạt động.
- [ ] Hamming Distance đúng.
- [ ] SSIM hoạt động trong điều kiện hợp lệ.
- [ ] Synthetic duplicate tests PASS.
- [ ] False-positive/false-negative được báo cáo.
- [ ] Không chỉnh sửa dataset v3.
- [ ] Không tuyên bố full-v3 near-duplicate-free.

## Gate C — ML Compatibility

- [ ] ResNet18 preprocessing contract PASS.
- [ ] EfficientNet-B0 preprocessing contract PASS.
- [ ] HOG/SVM compatibility PASS.
- [ ] Train augmentation đúng phạm vi.
- [ ] Eval/inference deterministic.
- [ ] Legacy model artifacts không đổi.

## Gate D — Data Integration

- [ ] Đọc manifest v3 phù hợp.
- [ ] DataLoader thật hoạt động trên scope kiểm tra.
- [ ] Shape/dtype/labels hợp lệ.
- [ ] Không resplit hoặc thay đổi official test.

## Gate E — Data Safety

- [ ] Raw không đổi.
- [ ] Split v1/v2/v3 và poc-v2 không đổi.
- [ ] Published manifests không đổi.
- [ ] Model artifacts không đổi.
- [ ] Không network/download.
- [ ] Không full train/full rehash.

## Gate F — Evidence & Reporting

- [ ] Có before/after visualizations.
- [ ] Có quantitative filtering results.
- [ ] Có near-duplicate demo/metrics.
- [ ] Có regression test evidence.
- [ ] Có bảng liên hệ các kỹ thuật với yêu cầu xử lý ảnh môn học.
- [ ] Có giải thích rõ experimental vs production.

Phân biệt:

- `PASS`
- `FAIL`
- `INCOMPLETE`
- `BLOCKED`

Không đánh dấu toàn bộ Phase 4.5 là PASS nếu còn acceptance gate bắt buộc chưa đạt.

---

# 16. ĐẦU RA BẮT BUỘC

Tạo báo cáo:

`artifacts/reports/elt/phase_4_5_ml_image_processing.md`

Nếu file đã tồn tại, thực hiện theo quy tắc quản lý report của dự án, tránh ghi đè lịch sử không chủ đích.

Nội dung:

```text
PHASE: 4.5 — ML Image Processing & Near-Duplicate Analysis

BASELINE:
- Repository/branch
- Git dirty state
- Python and dependency versions
- Input dataset/split version
- Phase 4 evidence

FILES REVIEWED:

FILES CHANGED:

FEATURE MATRIX:
- Requirement
- Code location
- Existing/new
- Tests
- Status

IMAGE PROCESSING RESULTS:
- Representation
- Color spaces
- Pixel operations
- Geometry
- Filtering
- Morphology
- Edges

NEAR-DUPLICATE RESULTS:
- pHash implementation
- Hamming Distance
- SSIM
- Synthetic fixture counts
- TP/FP/TN/FN
- Precision/Recall
- Thresholds
- Known limitations
- Real-v3 sample scope, if any

ML PREPROCESSING:
- Train transform
- Eval transform
- Inference transform
- Model profiles
- Legacy compatibility

BEFORE/AFTER EVIDENCE:
- Paths
- Metrics
- Configurations
- Seed
- Experimental vs production

TESTS EXECUTED:
- Commands
- Passed
- Failed
- Skipped
- Incomplete
- Errors and fixes

DATA SAFETY:
- Raw
- v1/v2/v3
- POC v2
- Published manifests
- Model artifacts

ACCEPTANCE:
- Gate A
- Gate B
- Gate C
- Gate D
- Gate E
- Gate F
- Overall PASS/FAIL/INCOMPLETE

RISKS / OPEN ITEMS:

NEXT PHASE:
- None — STOP
```

Ngoài báo cáo, lưu các demo artifacts vào thư mục riêng tương ứng.

Báo cáo phải phân biệt rõ:

1. Tính đúng đắn của thuật toán trên synthetic fixtures.
2. Khả năng tích hợp với v3.
3. Các kiểm tra chưa thực hiện trên full dataset.
4. Các kỹ thuật đã tích hợp production.
5. Các kỹ thuật chỉ phục vụ nghiên cứu và minh họa.

---

# 17. ĐỊNH DẠNG PHẢN HỒI CUỐI

Sau khi triển khai, trả lời:

```text
PHASE: 4.5

BASELINE VERIFIED:

IMPLEMENTATION SUMMARY:

FILES REVIEWED:

FILES CHANGED:

IMAGE PROCESSING FEATURES:

NEAR-DUPLICATE DETECTION:

ML PIPELINE INTEGRATION:

BACKWARD COMPATIBILITY:

TEST RESULTS:

DATA SAFETY:

REPORTS AND ARTIFACTS:

ACCEPTANCE:
Gate A:
Gate B:
Gate C:
Gate D:
Gate E:
Gate F:
Overall:

RISKS / OPEN ITEMS:

NEXT PHASE: NONE — STOP
```

**QUAN TRỌNG:**

- Chỉ thực hiện Phase 4.5.
- Không thay đổi lại dữ liệu Food-101 v3.
- Không thực hiện full near-duplicate scan.
- Không tự động training.
- Không download.
- Không chuyển sang Phase 5.
- Không tự commit/push.

Hoàn thành implementation, tests và report, sau đó dừng.