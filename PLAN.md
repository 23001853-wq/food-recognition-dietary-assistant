# PLAN.md — FOOD RECOGNITION & PORTION-AWARE DIETARY ASSISTANT

> **Project 18 — Bản đặc tả tổng thể dành cho AI coding agent và diagramming agent**  
> **Phiên bản:** 1.0 · **Ngày:** 31/10/2026 · **Ngôn ngữ tài liệu:** Tiếng Việt  
> **Trạng thái:** Kế hoạch triển khai, chưa phải báo cáo kết quả huấn luyện  
> **Đích:** Xây dựng một hệ thống Computer Vision *có huấn luyện, đánh giá và so sánh phương pháp*, nhận diện loại món ăn từ ảnh; người dùng xác nhận món, chọn khẩu phần thô và ghi nhật ký ăn uống.  
> **Công nghệ đã chốt cho MVP:** Python 3.11, PyTorch/Torchvision, OpenCV, scikit-learn, EfficientNet-B0, ResNet18, HOG + SVM, Streamlit, PostgreSQL, SQLAlchemy, Alembic, pytest, Git.  
> **Nguyên tắc:** Mọi sơ đồ phải phản ánh đúng luồng xử lý thực tế, không thêm chức năng AI/ước lượng dinh dưỡng nào chưa được thiết kế hoặc kiểm chứng.

---

## 0. HƯỚNG DẪN CHO AGENT ĐỌC FILE NÀY

### 0.1. Nhiệm vụ của diagramming agent

Đọc **toàn bộ** `PLAN.md` trước khi vẽ. Dựa trên các phần 1–22, dựng hệ thống sơ đồ tại phần 23; ưu tiên sơ đồ kiến trúc MVP, training, inference, ERD, user journey, deployment. Các khối hoặc liên kết phải thống nhất tên dịch vụ, bảng dữ liệu, vai trò và phạm vi.

1. **Không biến một mô hình pretrained ImageNet thành mô hình nhận diện món ăn hoàn chỉnh.** Cần fine-tune trên Food-101 rồi mới dùng cho nhiệm vụ món ăn.
2. **Không nhầm PostgreSQL với dataset hoặc model:** PostgreSQL lưu dữ liệu nghiệp vụ, Food-101 là dữ liệu huấn luyện, EfficientNet/ResNet18 là mô hình học máy.
3. **Không khẳng định mô hình đạt accuracy/F1 bao nhiêu** khi chưa có kết quả đo; các chỉ số chỉ là mục tiêu đề xuất hoặc placeholder.
4. **Không coi estimated kcal là phép đo từ pixel.** Người dùng chọn `small/medium/large` hoặc nhập gram; dữ liệu dinh dưỡng lấy từ bảng tham chiếu có nguồn. Hệ thống cho phép sửa kết quả.
5. **Phân biệt `MVP` và `Future` rõ ràng:** Không vẽ YOLO, segmentation, barcode, React, FastAPI, mobile app hoặc tích hợp thiết bị vào kiến trúc MVP. Chỉ vẽ chúng trong sơ đồ mở rộng, nếu có.
6. **Một ảnh/crop → một nhãn món ăn chính** trong MVP. Với nhiều món, người dùng crop từng vùng, phân loại/lưu từng món riêng. Không giả định AI phát hiện mọi món tự động.
7. **Giữ độc lập training pipeline và inference pipeline.** Đào tạo offline tạo model artifact; ứng dụng chỉ tải artifact và infer, không train khi người dùng upload.
8. **Không được đưa dữ liệu test vào train/validation, kể cả qua xử lý hoặc augmentation.**
9. Diagramming agent xuất mỗi sơ đồ thành `*.mmd` (Mermaid source) và `*.svg` hoặc `*.png` nếu có công cụ dựng; nếu chưa render được thì xuất file Mermaid hợp lệ và báo rõ.
10. Đánh dấu nguồn dữ liệu, nơi lưu thông tin, ranh giới tin cậy, các bước xác nhận của người dùng và các nhánh lỗi.

### 0.2. Phân loại phạm vi

- **MUST**: Bắt buộc để đáp ứng đề hoặc làm hệ thống hoạt động đúng.
- **SHOULD**: Nên làm để hoàn thiện học thuật/sản phẩm; có thể cắt nếu thiếu thời gian.
- **COULD**: Nâng cao; **không** nằm trong phạm vi đầu tiên.
- **WON'T (MVP)**: Không triển khai ở bản đầu.

### 0.3. Nguồn chân lý (source of truth)

| Hạng mục | Quyết định của dự án |
|---|---|
| Primary dataset | Food-101, đúng split gốc `train` / `test` |
| Split validation | Tách *stratified* từ `train`, không chạm `test` |
| Traditional baseline | HOG features + linear SVM (scikit-learn) |
| CNN comparison | ResNet18 pretrained ImageNet-1K, fine-tune trên Food-101 |
| Candidate chính | EfficientNet-B0 pretrained ImageNet-1K, fine-tune trên Food-101 |
| Model production | Chọn sau khi so sánh macro-F1, Top-1, độ trễ, kích thước |
| Giao diện MVP | Streamlit (multipage) |
| Business layer | Python service/repository tách khỏi Streamlit |
| Database | PostgreSQL (nghiệp vụ); SQLAlchemy ORM + Alembic |
| File storage | Ảnh demo/user upload và model artifact trên filesystem ở MVP; chỉ lưu đường dẫn/metadata trong DB |
| Authentication | MVP chạy single-user local/demo, không có login; schema dành chỗ cho multi-user ở giai đoạn tiếp |
| Portion | Chọn `small`, `medium`, `large` hoặc tùy chọn nhập số gram bằng tay |
| Nutrition | Giá trị tham khảo có nguồn; công thức tính theo gram giả định hoặc gram người dùng nhập; ghi rõ **ước tính** |
| Mixed dish | Không auto-detect; crop thủ công từng vùng, nêu hạn chế |
| Chỉ số bắt buộc | Top-1 accuracy, macro-F1, confusion matrix, phân tích lỗi |
| Backend REST | FastAPI **Future**, không là điều kiện chạy MVP |
| UI React | **Future**, không là điều kiện chạy MVP |

---

## 1. BỐI CẢNH, BÀI TOÁN, KẾT QUẢ CẦN CÓ

### 1.1. Bài toán

Ảnh món ăn thay đổi theo góc chụp, ánh sáng, trang trí, cách đặt trên đĩa và nguyên liệu. Hệ thống cần dự đoán *loại món ăn* từ một ảnh hoặc vùng crop; hỗ trợ ghi nhật ký một cách tiện lợi. Nhận diện tên món **không đủ** để suy ra chính xác khối lượng hay calorie. Người dùng phải có bước xác nhận và chỉnh sửa.

### 1.2. Các đầu ra tối thiểu

1. Mô hình truyền thống HOG + SVM **đã train** và có kết quả test.
2. Mô hình ResNet18 **đã fine-tune** và có kết quả test.
3. Mô hình EfficientNet-B0 **đã fine-tune** và có kết quả test, hoặc ghi rõ lý do giới hạn tài nguyên nếu chưa train được.
4. Bảng so sánh công bằng trên **cùng tập test**, cùng tập class, gồm Top-1 Accuracy, macro-F1, thời gian inference, kích thước artifact.
5. Web demo: tải ảnh/crop → hiển thị top-3 → sửa/xác nhận món → chọn portion → kcal tham khảo → lưu nhật ký → dashboard.
6. PostgreSQL với migrations và seed dữ liệu thực phẩm có nguồn.
7. Phân tích định tính ảnh sai, mixed dishes, ảnh chụp thật và domain shift.
8. Báo cáo nêu giả định/giới hạn của calories, nguồn dữ liệu, quy trình tái lập thí nghiệm.
9. README hướng dẫn cài đặt/chạy demo, khóa phiên bản môi trường, kiểm thử.
10. Bộ sơ đồ hệ thống rõ ràng, thống nhất và sẵn dùng cho báo cáo/bảo vệ.

### 1.3. Tính đúng đắn trong phát biểu

- Nói **"dự đoán loại món ăn"**, không nói "nhận diện chắc chắn".
- Nói **"calories ước tính"**, không nói "đo calorie chính xác từ ảnh".
- `confidence_score` là điểm softmax, không mặc định là xác suất đã hiệu chuẩn; cần calibration nếu muốn diễn giải như độ tin cậy thực.
- Giá trị trên ImageNet của một pretrained backbone **không phải** kết quả đo trên Food-101.
- Nếu top-1 thấp, lớp không thuộc tập huấn luyện, hoặc ảnh có nhiều món: cho chọn lại/nhập thủ công.

## 2. ĐỐI CHIẾU ĐỀ GỐC VỚI THIẾT KẾ

| Yêu cầu đề | Mức độ | Module | Bằng chứng hoàn thành |
|---|---|---|---|
| Predict food category for each image/crop | MUST | `inference`, `recognize` | Demo, dự đoán top-1/top-3 |
| Portion categories small/medium/large nếu làm | SHOULD | `portion_service` | Giao diện, quy tắc gram, test |
| Compare at least one baseline with stronger method | MUST | `train_baseline`, `train_cnn`, `evaluate` | CSV, bảng so sánh, đường học |
| Application-oriented output | MUST | Streamlit pages | Ảnh màn hình, demo live |
| Meal diary/summary | MUST | `meal_service`, DB, dashboard | Bản ghi, biểu đồ theo ngày |
| Discuss mixed dishes/plating variation | MUST | error analysis | Bộ ảnh khó, case study |
| Food-region extraction or crop assumption | MUST trong pipeline | `crop_service` | Crop thủ công + ghi crop coordinates |
| Feature extraction / transfer learning | MUST | HOG/SVM, CNN | notebook/scripts + log |
| Food-category classification | MUST | `inference_service` | Top-1 + top-3 |
| Portion approximation | OPTIONAL gốc | `portion_service` | Mô hình định tính, giả định công khai |
| Meal summary visualization | MUST trong pipeline | `dashboard` | Biểu đồ tổng hợp |
| Food-101 or curated subset | MUST | `data/` | manifest, labels, split |
| Locally captured photos for domain shift | SHOULD | `evaluation/local_photos` | Kết quả riêng, không trộn test |
| Accuracy, macro-F1, qualitative analysis | MUST | `evaluate`, `error_analysis` | Bảng metric + ảnh lỗi |
| Realistic application limits | MUST | report | Mục limitations |

## 3. MỤC TIÊU, PHẠM VI VÀ TIÊU CHÍ THÀNH CÔNG

### 3.1. Goals

- Giải quyết đúng bài toán **image classification** trong Computer Vision, không phụ thuộc dịch vụ nhận diện thương mại.
- Thiết kế quy trình thí nghiệm có thể tái lập, so sánh được, tránh rò rỉ dữ liệu.
- Hỗ trợ UX có bước human-in-the-loop: người dùng xác nhận nhãn, portion, sửa gram và kcal trước khi lưu.
- Có nhật ký ăn uống và biểu đồ từ dữ liệu thật trong PostgreSQL.
- Chạy local trên Windows 11 hoặc Linux; có phương án huấn luyện trên Colab GPU và đem artifact về chạy local.

### 3.2. MVP (v1.0)

- Upload JPG/JPEG/PNG, hiển thị preview; giới hạn và kiểm tra ảnh.
- Crop thủ công vùng món ăn nếu cần; cho phép ảnh đơn không crop.
- Một inference trên mỗi ảnh/crop; top-3 class + scores.
- Sửa nhãn thủ công từ danh sách lớp đã hỗ trợ; có lựa chọn `unknown/other` không tính nutrition tự động.
- Portion S/M/L hoặc grams; lấy kcal/100g từ nguồn dinh dưỡng đã seed và mapping kiểm duyệt.
- Meal log: tạo, xem, sửa, xóa có xác nhận.
- Dashboard: tổng kcal ước tính hôm nay/7 ngày, breakdown theo meal type, lịch sử theo ngày.
- Phân tích 3 mô hình; có report results và confusion matrix.
- Single-user demo; không cần đăng ký tài khoản.

### 3.3. Chức năng nâng cao (sau khi MVP ổn định)

- Multi-user/login, JWT/session authentication, authorization theo user.
- Ảnh chứa nhiều món: YOLO detector hoặc segmentation, nếu có dữ liệu bounding box/mask thích hợp để huấn luyện.
- Segmentation/ước lượng diện tích, camera đa góc, scale reference để hỗ trợ portion thông minh hơn; vẫn cần đánh giá sai số.
- Thêm món Việt có gán nhãn riêng; đánh giá dữ liệu outside-domain.
- FastAPI REST + React hoặc mobile client.
- Tra cứu nutrition qua API đáng tin cậy có cache và mapping duyệt thủ công.
- Triển khai cloud, object storage, CI/CD nâng cao, monitoring đa người dùng.

### 3.4. Không thực hiện trong MVP

- Suy ra gram chính xác chỉ từ một ảnh RGB thiếu vật chuẩn.
- Khuyên dinh dưỡng cá nhân hóa theo bệnh lý hoặc dùng kết quả như lời khuyên y tế.
- Nhận diện toàn bộ nguyên liệu và vi chất từ pixel, đặc biệt món nước/hỗn hợp.
- Công bố accuracy vượt trội hoặc inference real time khi chưa benchmark.
- Sử dụng 101 nhãn Food-101 như 101 công thức dinh dưỡng đã được xác thực.

### 3.5. KPI đề xuất (mục tiêu, **không phải kết quả**)

| KPI | Mức nghiệm thu | Ghi chú |
|---|---|---|
| Hoạt động end-to-end | 100% demo path thành công với ảnh mẫu hợp lệ | Phải kiểm thử |
| Số lớp giai đoạn POC | 10–20 lớp có ghi version | Mục tiêu ban đầu |
| Số lớp mục tiêu cuối | 101 nếu đủ tài nguyên | Nêu rõ nếu chỉ subset |
| So sánh | >= 2 phương pháp; mục tiêu 3 | Same labels/test split |
| Bắt buộc về chỉ số | Accuracy + macro-F1 | Không ấn định % thành tích |
| Bảo toàn nhật ký | Create/Read/Update/Delete đúng và nhất quán | Integration test DB |
| Nutrition | Gắn nhãn ước tính, không giá trị không nguồn | Không giả định 1 ảnh=1 suất chuẩn |
| Inference latency | Đo P50/P95 theo hardware | Không đặt SLA nếu chưa đo |

---

## 4. STACK CHÍNH THỨC VÀ LÝ DO CHỌN

| Layer | Chọn | Vai trò | Ghi chú |
|---|---|---|---|
| Runtime | Python 3.11 | Pipeline, services, app | Khóa phiên bản package đã kiểm thử |
| Deep learning | PyTorch + Torchvision | Pretrained, fine-tune, inference | Lấy transform và label map khớp artifact |
| Conventional CV | OpenCV + scikit-image HOG | Feature extraction | Chuẩn hóa ảnh nhất quán |
| ML baseline | scikit-learn `LinearSVC` hoặc `SVC(kernel='linear')` | Baseline HOG + SVM | Quyết định cụ thể và lưu config |
| Metrics | scikit-learn | Accuracy, F1, confusion matrix | Report JSON/CSV |
| Image I/O | Pillow | RGB decode, EXIF orientation | Kiểm tra ảnh lỗi |
| Data | Food-101 | Training/evaluation | Split đúng gốc |
| UI | Streamlit | Demo web | Giao diện nhiều trang |
| Business logic | Python services | Orchestrate inference, meal, nutrition | Không đặt SQL trong code UI |
| Database | PostgreSQL | Persistent app data | 1 DB local dev cũng được |
| ORM | SQLAlchemy 2.x | Models, sessions, repository | Driver `psycopg` v3 |
| DB migrations | Alembic | Schema versioning | Không chỉnh DB bằng tay |
| Configuration | `.env` + `pydantic-settings` (hoặc `python-dotenv`) | Env config | `.env` không commit |
| Visualization | Pandas, Matplotlib/Plotly | Reports, dashboard | Không hardcode kết quả |
| Experiment logging | CSV/JSON + TensorBoard | Train/validation logs | Có thể thêm MLflow sau |
| Testing | pytest | Unit/integration/smoke | Tách DB test |
| Code quality | Ruff | Lint & format | Có pre-commit nếu muốn |
| Version control | Git | Code/versioning | Dataset + model weights không commit |
| Docker Compose | SHOULD | Local PostgreSQL | App có thể chạy trong venv |
| Backend API | FastAPI – Future | REST interface | Không cần cho MVP |

**Vì sao chưa quyết định model mạnh nhất?** EfficientNet-B0 là **ứng viên triển khai ưu tiên**, không phải kết luận thắng. Model cuối chỉ chọn sau kết quả trên *cùng dữ liệu, cùng protocol*. ResNet18 dùng pretrained ImageNet-1K, không phải sẵn được fine-tune cho Food-101. PostgreSQL lưu metadata/data nghiệp vụ, không huấn luyện từ PostgreSQL.

## 5. TỪ ĐIỂN KHÁI NIỆM

| Thuật ngữ | Giải thích ngắn, đúng phạm vi |
|---|---|
| Computer Vision | Kỹ thuật để máy tính phân tích, hiểu dữ liệu hình ảnh |
| Image Classification | Gán một nhãn chính cho ảnh/crop |
| Detection | Xác định vị trí (bounding box) và loại của nhiều đối tượng |
| Segmentation | Dự đoán mask/vùng pixel của đối tượng |
| Feature Extraction | Biến ảnh thành biểu diễn đặc trưng |
| HOG | Histogram of Oriented Gradients; mô tả phân bố hướng biên |
| SVM | Support Vector Machine; phân lớp trên đặc trưng đã trích |
| CNN | Mạng nơ-ron tích chập khai thác đặc trưng hình ảnh |
| Pretrained weights | Trọng số học từ tác vụ/dataset khác (ví dụ ImageNet) |
| Transfer Learning | Tái sử dụng biểu diễn từ mô hình pretrained |
| Fine-tuning | Huấn luyện thêm/điều chỉnh một phần hoặc toàn bộ mô hình trên Food-101 |
| Backbone | Mạng trích xuất đặc trưng trước classification head |
| Classification Head | Tầng ánh xạ đặc trưng sang số lớp đích |
| Epoch | Một vòng duyệt qua tập train |
| Batch | Nhóm ảnh huấn luyện trong một bước |
| Cross Entropy | Hàm mất mát cho bài toán phân lớp |
| Data Augmentation | Tăng biến thể dữ liệu train qua phép biến đổi |
| Overfitting | Model học quá sát train và kém trên dữ liệu mới |
| Domain Shift | Dữ liệu triển khai khác phân phối data huấn luyện |
| Top-1 Accuracy | Tỉ lệ top-1 đúng trên số mẫu |
| Macro-F1 | Trung bình F1 từng class (trọng số ngang nhau) |
| Confusion Matrix | Bảng đối chiếu nhãn thực và nhãn dự đoán |
| Softmax score | Điểm dự đoán sau softmax; không đồng nghĩa calibrated probability |
| Inference | Dự đoán từ model đã train, không cập nhật trọng số |
| Crop | Cắt một vùng ảnh chọn thủ công, đưa vào classifier |
| Calorie reference | kcal/100 g từ bảng thành phần thực phẩm có nguồn |
| Portion category | Nhãn S/M/L, ánh xạ khối lượng ước lượng có giả định |
| ORM | Ánh xạ bảng DB sang object Python |
| Migration | Thay đổi có phiên bản đối với schema database |
| Artifact | Model weights, config, class mapping, metrics được xuất ra |
| Human-in-the-loop | Người dùng có bước kiểm tra, sửa kết quả AI |

---

## 6. PERSONA, USE CASE, USER STORY

### 6.1. Actors

- **End User:** tải ảnh, crop, xác nhận món, nhập portion, lưu/xem/sửa meal logs.
- **Developer/Researcher:** chuẩn bị dataset, train, evaluate, version model, chạy demo và báo cáo.
- **Nutrition Data Curator (vai trò kiêm nhiệm):** kiểm tra mapping giữa food class và nutrition reference, ghi rõ nguồn, giả định.
- **External Dataset/Nutrition Source:** Food-101 và nguồn thành phần dinh dưỡng; không phải user của hệ thống.

### 6.2. User stories quan trọng

| ID | User story | Priority | Acceptance |
|---|---|---|---|
| US-01 | Là người dùng, tôi tải ảnh và xem dự đoán | MUST | Preview và top-3 hiển thị |
| US-02 | Tôi crop vùng món ăn để loại ảnh nền | MUST | Crop hợp lệ, có preview trước inference |
| US-03 | Tôi chọn nhãn khác khi AI dự đoán sai | MUST | Nhãn đã sửa được lưu, không ghi đè prediction raw |
| US-04 | Tôi chọn S/M/L hoặc gram để ước tính kcal | SHOULD | Bảng reference nguồn rõ ràng, hiện disclaimer |
| US-05 | Tôi lưu món vào bữa sáng/trưa/tối/phụ | MUST | Có record trong PostgreSQL |
| US-06 | Tôi xem, sửa, xóa nhật ký | MUST | Tổng cập nhật đúng sau sửa/xóa |
| US-07 | Tôi xem dashboard 1/7 ngày | MUST | Biểu đồ lấy từ DB, có trạng thái empty |
| US-08 | Tôi được cảnh báo nếu ảnh lỗi hoặc ngoài phạm vi | MUST | Không crash, có hướng dẫn sửa |
| US-09 | Là researcher, tôi train/eval công bằng | MUST | Config, metrics, seed, split manifest |
| US-10 | Tôi xem ca nhầm lẫn và ảnh domain shift | MUST | Case study + error taxonomy |
| US-11 | Tôi chụp ảnh có 2 món rồi crop từng món | SHOULD | Lưu được 2 `meal_items`, không tự động detect |
| US-12 | Tôi đăng nhập và lưu dữ liệu riêng | COULD | Chỉ Future |

### 6.3. Quy tắc nhập liệu

- Ảnh chấp nhận `JPEG/JPG/PNG` (MVP). Kiểm tra MIME *và* nội dung decode thật. File tối đa mặc định **10 MB** (mức thiết kế; có thể cấu hình). Giới hạn kích thước pixel, reject decompression bomb.
- Nếu crop: 4 giá trị chuẩn hóa `(x1,y1,x2,y2)` thuộc `[0,1]`, `x1<x2`, `y1<y2` và vùng đủ lớn.
- Người dùng chọn meal type: `breakfast | lunch | dinner | snack`.
- Chọn portion: `small | medium | large | custom`; với `custom` thì gram > 0, có giới hạn hợp lý/cảnh báo.
- Năng lượng có thể `NULL` nếu không có mapping đáng tin cậy; **không** mặc định 0 kcal.
- Người dùng có thể lưu món dù không có kcal; dashboard chỉ cộng các bản ghi có calories hợp lệ và nêu rõ số món thiếu dữ liệu.
- Không tự động lưu meal log khi vừa infer. Chỉ lưu sau khi user bấm **Xác nhận và lưu**.

## 7. END-TO-END JOURNEY / PRODUCT FLOW

1. Người dùng mở trang `Recognize`.
2. Chọn ảnh → UI kiểm tra extension/size → Pillow verify/decode → chuẩn hóa orientation và RGB.
3. Hiển thị preview. Nếu ảnh chứa nhiều món, yêu cầu crop từng món thủ công.
4. Khi nhấn `Analyze`, gửi ảnh/crop vào `InferenceService`.
5. Service tải *production model artifact* đã phê duyệt (cache) cùng class mapping, preprocess chính xác.
6. Chạy `torch.inference_mode()`, logits → softmax → top-3.
7. Nếu score top-1 thấp hơn ngưỡng **đã hiệu chỉnh trên validation**, hiển thị trạng thái không chắc chắn, hướng dẫn chọn thủ công. Ngưỡng chưa xác định cho đến khi có validation.
8. UI hiển thị tên món từ food class mapping; user chọn/chỉnh nhãn. Nếu `other/unknown`, skip nutrition auto.
9. UI cho chọn portion S/M/L hoặc custom grams. NutritionService lấy `kcal_per_100g` và gram giả định từ reference mapping đã duyệt.
10. Hiển thị `estimated_kcal` và công thức cùng cảnh báo về sai số.
11. User chọn ngày/meal type, xác nhận. MealService mở DB transaction, lưu record, đường dẫn ảnh (nếu được phép giữ ảnh) và provenance model.
12. UI thông báo lưu thành công, chuyển đến nhật ký hoặc tiếp tục thêm món thứ hai.
13. Tại `Diary`, người dùng có thể sửa/xóa; hệ thống tính lại biểu đồ từ DB.
14. `Dashboard` tổng hợp theo ngày và bữa, biểu đồ dựa trên những record có kcal khác `NULL`.
15. Không inference lại khi xem nhật ký; không train trong luồng người dùng.

---

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
## 10. NUTRITION / PORTION: THIẾT KẾ VÀ GIỚI HẠN

### 10.1. Nguồn calories và provenance

- Nên tạo `nutrition_foods` từ nguồn kiểm chứng được (ví dụ USDA FoodData Central cho thực phẩm phù hợp), hoặc bảng tham chiếu có dẫn nguồn rõ, ngày truy xuất, đơn vị, phiên bản; cho phép món không có mapping.
- USDA có dữ liệu dinh dưỡng theo thực phẩm/thành phần và API/download; **không** tự động suy ra một class Food-101 tương đương một entry USDA. Đối với pizza, cơm trộn, pasta…, công thức và tỷ lệ nguyên liệu biến thiên lớn.
- Trường `kcal_per_100g` lưu `NUMERIC`, nullable, >= 0; không nhập số vô căn cứ. Nếu không đủ bằng chứng: `NULL`.
- Mỗi food class có thể map nhiều nutrition profiles; MVP chọn một **representative profile** đã curator duyệt, có lý do và độ tin cậy định tính.
- Người dùng có thể sửa `estimated_grams` hoặc `estimated_kcal`; lưu lại `source_type` (`reference` / `manual`) để phân biệt.

### 10.2. Định nghĩa khẩu phần

- Với từng `nutrition_food`, cấu hình `small_grams`, `medium_grams`, `large_grams` (có nguồn hay quy ước công khai), không áp một giá trị gram chung cho mọi món.
- Nếu profile thiếu gram cho kích thước đã chọn: yêu cầu nhập custom grams, không tự tạo số.
- Giá trị `estimated_grams` là **assumed_mass**, không phải image-predicted mass.
- **Công thức:** `estimated_kcal = (kcal_per_100g × estimated_grams) / 100`.
- Mọi màn hình hiển thị ít nhất một câu: “Năng lượng chỉ là ước tính theo khẩu phần/khối lượng tham khảo, không đo trực tiếp từ ảnh.”
- Với image nhiều món, không nhân calories của toàn ảnh lên mỗi crop; chỉ tính từng item được xác nhận, tránh cộng trùng.

### 10.3. Example chỉ minh họa công thức (không dùng seed thật)

```text
Reference: 250 kcal / 100 g (con số giả định để giải thích)
Input: medium = 180 g (khối lượng giả định)
Estimated energy: (250 × 180) / 100 = 450 kcal
```

Không hiển thị những con số ví dụ này như giá trị dinh dưỡng của món ăn thực tế, không hardcode chúng trong seed mặc định.

### 10.4. Quy trình kiểm duyệt mapping

1. Lấy tên class từ `idx_to_class.json` của model.
2. Tìm nutrition entry phù hợp; kiểm tra tên, cách chế biến, đơn vị và nguồn.
3. Reviewer ghi lựa chọn `food_class_id → nutrition_food_id` và lý do.
4. Nhập hoặc trích giá trị `kcal_per_100g`; kiểm tra phạm vi hợp lý và nguồn.
5. Nhập S/M/L grams **nếu có**; nếu không, đặt NULL.
6. Lưu `provenance_url`, `retrieved_at`, `mapping_note`, `reviewed_by` (nếu có).
7. Khi đổi mapping sau này, snapshot giá trị đã dùng trên meal item cũ để nhật ký lịch sử không tự biến đổi.

### 10.5. Dashboard nguyên tắc thống kê

- `Daily total estimated kcal = SUM(meal_items.estimated_kcal WHERE item hợp lệ)`.
- `NULL` không cộng; có nhãn “N món chưa có thông tin năng lượng”.
- Không dùng score AI để nhân/chiết giảm calorie.
- Trục ngày theo local timezone cấu hình (`Asia/Ho_Chi_Minh` cho demo), lưu UTC trong DB nếu lưu timestamps; `consumed_on` là ngày lịch người dùng chọn.
- Phân biệt kcal "ước tính theo nguồn tham khảo" với mục tiêu dinh dưỡng riêng của người dùng (ngoài phạm vi MVP).

---

## 11. KIẾN TRÚC HỆ THỐNG

### 11.1. Kiến trúc MVP chuẩn — Modular Monolith

```text
┌───────────────────────────────────────────────────────────────┐
│ Browser                                                       │
│  Streamlit UI: Recognize / Diary / Dashboard / About          │
└───────────────────────────┬───────────────────────────────────┘
                            │ internal Python calls
                            v
┌───────────────────────────────────────────────────────────────┐
│ Application Service Layer (Python)                            │
│  ImageService | InferenceService | NutritionService            │
│  MealService  | DashboardService | ModelRegistry              │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
                v                               v
┌──────────────────────────────┐    ┌───────────────────────────┐
│ AI Runtime                   │    │ PostgreSQL                │
│ production artifact + PyTorch│    │ meal, classes, nutrition  │
└──────────────────────────────┘    └───────────────────────────┘
                │
                v
┌──────────────────────────────┐
│ Local file storage           │
│ image uploads / artifacts    │
└──────────────────────────────┘

Separate offline pipeline:
Food-101 → split/preprocess → train 3 methods → evaluate → model artifacts
                                               |
                                               v
                                  artifacts/production/current.json
```

**Đặc điểm:** Streamlit chỉ điều phối presentation, gọi trực tiếp service Python; không có FastAPI/HTTP nội bộ trong MVP. SQLAlchemy nằm sau repository layer. InferenceService chỉ đọc model artifact và class mapping, không truy cập train dataset khi xử lý yêu cầu.

### 11.2. Module dependencies

```text
ui/pages  -> services         -> repositories -> SQLAlchemy/PostgreSQL
           -> inference       -> model_loader -> model artifacts
           -> image_processing-> Pillow/OpenCV
           -> nutrition       -> repositories
           -> dashboard       -> repositories
training/scripts -> datasets/transforms/models -> checkpoints & reports
```

- UI không gọi `torch.load`, `session.execute`, hoặc truy vấn SQL trực tiếp.
- `training/` không import Streamlit.
- Inference module không được sửa meal records.
- Unit test service chạy với fake repositories; integration test chạy PostgreSQL test DB.
- Có một session DB **trên mỗi transaction**, không giữ shared mutable Session trong cache toàn cục.

### 11.3. Tương lai (không thuộc MVP)

`React/Mobile → FastAPI (REST) → Application Services → PyTorch inference + PostgreSQL + Object Storage`. Cần authentication, authorization, API versioning, rate limiting, file retention và deployment đa instance. **Không** dùng sơ đồ này làm hiện trạng v1.0.

### 11.4. Runtime deployment MVP

- Windows 11 / Linux host; Python virtual environment chạy Streamlit.
- PostgreSQL chạy local service hoặc Docker container (Docker Compose được khuyên dùng).
- `models/` và `artifacts/` là thư mục local có quyền đọc; uploads nằm ở thư mục riêng không nằm trong web root.
- `.env` chứa `DATABASE_URL`, storage config, timezone, model pointer; không commit credentials.
- Cấu hình port Streamlit mặc định 8501, PostgreSQL mặc định 5432 (có thể đổi); DB không phơi công khai qua Internet ở demo.
- Single-user local/demo; nếu deploy public cần thêm authentication và privacy controls trước.

---

## 12. DATABASE POSTGRESQL — ENTITY, COLUMN, RULE

### 12.1. Nguyên tắc

- Dùng PostgreSQL làm DB duy nhất cho ứng dụng nghiệp vụ.
- Tạo schema qua SQLAlchemy ORM + Alembic migrations.
- Mọi ID nghiệp vụ là UUID; có thể dùng `uuid.uuid4()` từ Python cho tính tương thích.
- Timestamps dùng `TIMESTAMP WITH TIME ZONE`; phân biệt `created_at`, `updated_at`.
- `NUMERIC(10,2)` cho gram/kcal, hạn chế lỗi tiền số float trong thống kê đơn giản.
- DB lưu đường dẫn ảnh tương đối, metadata, không lưu binary toàn bộ Food-101 vào DB.
- **Single-user MVP:** tạo `demo_user` cố định qua seed; `user_id` trong `meal_logs` để chuẩn bị multi-user nhưng chưa có tính năng đăng nhập. Không tuyên bố dữ liệu đã cách ly an toàn giữa nhiều user.
- `prediction_history` là lịch sử inference (nếu người dùng đồng ý lưu); `meal_items` là bản ghi đã xác nhận. Không nhầm prediction chưa xác nhận với món đã ăn.

### 12.2. Bảng `users`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | User ID |
| display_name | VARCHAR(120) | NOT NULL | Demo user name |
| email | VARCHAR(255) | NULL, UNIQUE | Không bắt buộc MVP |
| created_at | TIMESTAMPTZ | NOT NULL | Created UTC |

### 12.3. Bảng `food_classes`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Food class UUID |
| dataset_key | VARCHAR(150) | UNIQUE, NOT NULL | `pizza`, `sushi`, ... |
| display_name_en | VARCHAR(200) | NOT NULL | Food class English |
| display_name_vi | VARCHAR(200) | NULL | Tên hiển thị Việt |
| is_active | BOOLEAN | DEFAULT true | Cho phép chọn trong UI |
| created_at | TIMESTAMPTZ | NOT NULL | Created |

**Không** đặt `class_idx` trên `food_classes` như chân lý độc lập; class index thuộc **model release**, có thể đổi khi train lại. Lấy từ `model_class_labels` hoặc artifact mapping.

### 12.4. Bảng `nutrition_foods`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Profile UUID |
| name | VARCHAR(200) | NOT NULL | Tên mục dinh dưỡng |
| kcal_per_100g | NUMERIC(10,2) | NULL, CHECK >= 0 | Kcal tham khảo |
| source_name | VARCHAR(200) | NOT NULL | Ví dụ USDA FDC |
| source_record_id | VARCHAR(120) | NULL | External FDC ID nếu có |
| provenance_url | TEXT | NULL | Link nguồn |
| retrieved_at | TIMESTAMPTZ | NULL | Ngày truy xuất |
| small_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định cho S |
| medium_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định cho M |
| large_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định cho L |
| mapping_note | TEXT | NULL | Giả định/công thức |
| reviewed | BOOLEAN | DEFAULT false | Đã review mapping/nguồn |
| created_at | TIMESTAMPTZ | NOT NULL | Created |
| updated_at | TIMESTAMPTZ | NOT NULL | Last updated |

### 12.5. Bảng `food_nutrition_mappings`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Mapping UUID |
| food_class_id | UUID | FK → food_classes | Class của model |
| nutrition_food_id | UUID | FK → nutrition_foods | Profile lựa chọn |
| is_default | BOOLEAN | NOT NULL DEFAULT false | Profile mặc định |
| confidence_note | TEXT | NULL | Ghi chú mapping |
| created_at | TIMESTAMPTZ | NOT NULL | Created |

Unique `(food_class_id, nutrition_food_id)`; tạo **partial unique index** đảm bảo tối đa một row `is_default=true` mỗi `food_class_id`. Chỉ sử dụng auto lookup khi `nutrition_foods.reviewed=true` và `kcal_per_100g IS NOT NULL`.

### 12.6. Bảng `model_versions`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Release UUID |
| model_name | VARCHAR(80) | NOT NULL | `resnet18`, `efficientnet_b0`, `hog_svm` |
| run_id | VARCHAR(100) | UNIQUE, NOT NULL | Training run |
| num_classes | INT | CHECK > 0 | Số class |
| artifact_dir | TEXT | NOT NULL | Artifact local path |
| metrics_json | JSONB | NULL | Các metrics đã công bố |
| is_production | BOOLEAN | NOT NULL DEFAULT false | Model chạy demo |
| created_at | TIMESTAMPTZ | NOT NULL | Created |

Chỉ tối đa một production model trong DB (`partial unique index` trên constant `is_production=true`), đồng bộ với production pointer file theo thủ tục release. **MVP ưu tiên pointer file là nguồn model runtime**, DB chỉ để audit.

### 12.7. Bảng `prediction_history`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Prediction event |
| user_id | UUID | FK → users | Demo user |
| model_version_id | UUID | FK → model_versions | Provenance |
| image_path | TEXT | NULL | Nếu user cho phép lưu ảnh |
| image_sha256 | VARCHAR(64) | NULL | Optional fingerprint |
| crop_box_json | JSONB | NULL | Normalized crop coordinates |
| topk_json | JSONB | NOT NULL | `[{dataset_key, score}, ...]` |
| top1_score | NUMERIC(7,6) | NULL | Raw softmax score |
| uncertain | BOOLEAN | NOT NULL DEFAULT false | Theo threshold calibrated |
| created_at | TIMESTAMPTZ | NOT NULL | Inference time |

`prediction_history` không bắt buộc ghi với mỗi infer; có thể thiết lập `SAVE_PREDICTION_HISTORY=false` mặc định để hạn chế lưu ảnh/data. Nếu lưu prediction, yêu cầu về privacy áp dụng cho cả metadata.

### 12.8. Bảng `meal_logs`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Meal log UUID |
| user_id | UUID | FK → users | Owner |
| consumed_on | DATE | NOT NULL | Ngày ăn (địa phương) |
| meal_type | VARCHAR(20) | CHECK enum values | breakfast/lunch/dinner/snack |
| notes | TEXT | NULL | Ghi chú bữa |
| created_at | TIMESTAMPTZ | NOT NULL | Created |
| updated_at | TIMESTAMPTZ | NOT NULL | Updated |

Một meal log có nhiều `meal_items`. Có thể tạo nhiều logs cùng `consumed_on + meal_type` nếu thực tế ăn nhiều bữa phụ; không unique cặp ngày/bữa.

### 12.9. Bảng `meal_items`

| Column | Type | Constraint | Ý nghĩa |
|---|---|---|---|
| id | UUID | PK | Meal item UUID |
| meal_log_id | UUID | FK → meal_logs (ON DELETE CASCADE) | Bữa chứa item |
| confirmed_food_class_id | UUID | FK → food_classes, NULL | Class đã xác nhận |
| custom_food_name | VARCHAR(200) | NULL | Món ngoài danh sách |
| prediction_id | UUID | FK → prediction_history, NULL | Liên kết dự đoán gốc |
| portion_size | VARCHAR(20) | CHECK S/M/L/custom | Loại portion |
| estimated_grams | NUMERIC(10,2) | NULL, CHECK > 0 | Gram giả định/nhập tay |
| kcal_per_100g_snapshot | NUMERIC(10,2) | NULL, CHECK >= 0 | Kcal tham chiếu tại lúc lưu |
| estimated_kcal | NUMERIC(10,2) | NULL, CHECK >= 0 | Tổng kcal ước tính |
| nutrition_source_snapshot | VARCHAR(255) | NULL | Nguồn tại lúc lưu |
| calculation_method | VARCHAR(30) | NOT NULL | reference_formula/manual/unknown |
| is_user_corrected | BOOLEAN | NOT NULL DEFAULT false | Đã sửa nhãn AI |
| notes | TEXT | NULL | Ghi chú món |
| created_at | TIMESTAMPTZ | NOT NULL | Created |
| updated_at | TIMESTAMPTZ | NOT NULL | Updated |

Check `(confirmed_food_class_id IS NOT NULL) OR (custom_food_name IS NOT NULL)`; `estimated_grams` có thể NULL nếu chưa có gram đáng tin. `estimated_kcal` chỉ tính được khi đủ dữ liệu; custom manual kcal vẫn được nhưng phải lưu `calculation_method='manual'`.

### 12.10. Quan hệ và thứ tự tạo bảng

1. `users`, `food_classes`, `nutrition_foods`, `model_versions`.
2. `food_nutrition_mappings`.
3. `prediction_history`, `meal_logs`.
4. `meal_items`.
5. Indexes, seed và constraints.

### 12.11. Indexes

- `users(email)` unique nếu email không null.
- `food_classes(dataset_key)` unique.
- `model_versions(run_id)` unique.
- `meal_logs(user_id, consumed_on DESC)`.
- `meal_items(meal_log_id)`.
- `prediction_history(user_id, created_at DESC)`.
- `food_nutrition_mappings(food_class_id)`.
- Partial unique indexes cho default nutrition mapping và production model như mục trên.

### 12.12. Migrations và dữ liệu seed

- `alembic revision --autogenerate ...` rồi review migration, `alembic upgrade head`.
- Seed idempotent: demo user; food_classes lấy từ manifest/class mapping, nutrition references đã có nguồn, default mapping đã duyệt.
- Không seed toàn bộ 101 món với kcal tự đoán. Cho lớp không mapping → nutrition unavailable.
- Test DB riêng; không chạy test làm xóa database thực.

---

## 13. SERVICE CONTRACTS VÀ DATA CONTRACTS

### 13.1. Services (Python functions/classes, **không phải REST API trong MVP**)

| Service | Hàm/contract | Input | Output |
|---|---|---|---|
| `ImageService` | `validate_and_decode(bytes)` | bytes + mime/name | PIL RGB hoặc error |
| `CropService` | `crop(image, box)` | RGB + normalized box | RGB crop |
| `ModelRegistry` | `load_production_model()` | current.json | model + preprocess + label map |
| `InferenceService` | `predict(image, top_k=3)` | PIL RGB | `PredictionResult` |
| `NutritionService` | `lookup(food_class_id)` | UUID | optional reference |
| `NutritionService` | `estimate(food, portion, grams=None)` | profile + selected portion | kcal estimate hoặc unavailable |
| `MealService` | `create_meal_with_items(...)` | validated draft | saved IDs |
| `MealService` | `update_item(item_id, patch)` | patch | saved record |
| `MealService` | `delete_item(item_id)` | UUID | deletion result |
| `MealService` | `list_diary(date_range)` | dates + demo user | meal logs/items |
| `DashboardService` | `summary(date_range)` | dates + demo user | aggregated kcal + missing_count |
| `ExperimentService` (offline) | `evaluate(checkpoint, test_manifest)` | model + data | metrics + charts |

### 13.2. InferenceResult schema (Python TypedDict/Pydantic)

```json
{
  "model_name": "efficientnet_b0",
  "run_id": "example-run-id",
  "top_k": [
    {"dataset_key": "pizza", "class_idx": 0, "score": 0.91},
    {"dataset_key": "omelette", "class_idx": 1, "score": 0.05},
    {"dataset_key": "lasagna", "class_idx": 2, "score": 0.04}
  ],
  "uncertain": false,
  "inference_ms": 42.5,
  "crop_box": null
}
```

**Toàn bộ value ở ví dụ trên là dữ liệu minh họa, không phải số đo model đã train.** `class_idx` bắt buộc tra đúng `idx_to_class.json`; không lấy ID lớp từ bảng DB. `score` chỉ là model softmax, chỉ coi là probability nếu đã calibration.

### 13.3. MealCreateDraft (logic nội bộ)

```json
{
  "consumed_on": "2026-10-31",
  "meal_type": "lunch",
  "notes": "",
  "items": [
    {
      "confirmed_dataset_key": "pizza",
      "custom_food_name": null,
      "portion_size": "medium",
      "estimated_grams": null,
      "prediction_id": null,
      "calculation_method": "reference_formula"
    }
  ]
}
```

**Lưu ý:** `estimated_grams=null` chỉ hợp lệ ở draft; `MealService` phải resolve từ portion mapping hoặc chuyển sang unknown. Không tự ghi `estimated_kcal` giả; giá trị xác định ở server/service từ reference + snapshot sau validation.

### 13.4. Error codes nội bộ

| Code | Khi nào | UI behavior |
|---|---|---|
| `INVALID_IMAGE` | File không giải mã được | Yêu cầu ảnh khác |
| `IMAGE_TOO_LARGE` | Vượt byte/pixel limit | Hướng dẫn giảm ảnh |
| `INVALID_CROP` | Box vô lý/ngoài ảnh | Quay lại crop |
| `MODEL_NOT_READY` | Chưa có production artifact | Disable analyze, hướng dẫn setup |
| `ARTIFACT_MISMATCH` | Num classes/label map không khớp | Báo lỗi hệ thống, không infer |
| `LOW_CONFIDENCE` | Score dưới ngưỡng đã chọn | Cho sửa nhãn, không tự lưu |
| `NUTRITION_NOT_FOUND` | Thiếu mapping/nguồn | Không tự tính kcal |
| `INVALID_PORTION` | Không có gram hoặc gram không hợp lệ | Yêu cầu chỉnh portion |
| `DB_UNAVAILABLE` | Không kết nối DB | Không giả báo đã lưu |
| `SAVE_FAILED` | Transaction lỗi | Rollback và thông báo thử lại |
| `UNSUPPORTED_FOOD` | Out of taxonomy | Chọn `other`, ghi thủ công |

### 13.5. Optional API sau MVP (chỉ để agent lập sơ đồ tương lai)

- `POST /api/v1/predictions`: multipart image + optional crop → top-k.
- `GET /api/v1/foods`: danh sách food classes.
- `GET /api/v1/foods/{id}/nutrition`: profiles đã duyệt.
- `POST /api/v1/meals`: tạo meal + items.
- `GET /api/v1/meals?from=&to=`: nhật ký.
- `PATCH /api/v1/meals/{id}` hoặc `/items/{id}`: chỉnh.
- `DELETE /api/v1/meals/{id}`: xóa.
- `GET /api/v1/dashboard?from=&to=`: summary.

**Không coi các route này là đã được cài đặt** và không vẽ network call tới FastAPI trong MVP.

---

## 14. UI/UX — THIẾT KẾ TỪNG MÀN HÌNH

### 14.1. Layout chung

- Header: `Food Recognition & Dietary Assistant`.
- Sidebar: `Home`, `Recognize Food`, `Meal Diary`, `Dashboard`, `Model Info / About`.
- Main: status/error message rõ; ngày giờ local; disclaimer kcal khi có số liệu.
- UI responsive ở mức Streamlit cho desktop/laptop; mobile optimization Future.
- Theme: nền sáng, xanh lá/xanh teal điểm nhấn, hình món ăn rõ, không dùng màu chỉ để truyền thông tin.

### 14.2. Home

- Nêu mục tiêu và giới hạn (đây là demo hỗ trợ ghi nhật ký, không phải phép đo dinh dưỡng chính xác).
- CTA: `Nhận diện món ăn`.
- Statistics từ DB: số bữa đã ghi, món đã ghi (không tự bịa dữ liệu).
- Hiển thị current model version và số class hỗ trợ.

### 14.3. Recognize Food — màn hình chính

**Components:** uploader, image preview, crop selector, `Analyze`, results top-3, predicted label selector, portion selector, optional grams field, nutrition source, estimated kcal, date+meal type, `Xác nhận và lưu`.

**States:**

1. `EMPTY`: chưa có ảnh.
2. `IMAGE_READY`: ảnh hợp lệ, có thể crop.
3. `ANALYZING`: đang inference, khóa nút lặp.
4. `RESULT_READY`: có top-k và draft chưa lưu.
5. `NEEDS_CONFIRMATION`: score thấp/unsupported; yêu cầu sửa/chọn khác.
6. `SAVED`: DB đã commit.
7. `ERROR`: file/model/DB lỗi; có nút thử lại.

**Rule:** Nếu người dùng đổi ảnh hoặc crop sau inference thì invalidate prediction cũ trước khi lưu; không lấy nhãn của ảnh A để lưu ảnh B. Nút save disable khi chưa confirm label/meal type. Dùng `st.session_state` để giữ state khi Streamlit rerun.

### 14.4. Meal Diary

- Filter by date range, `meal_type`, name; danh sách theo ngày.
- Mỗi item hiển thị: food name (confirmed), portion, gram nếu có, kcal hoặc `Không có dữ liệu`, nguồn và thời gian.
- Chỉnh food/gram/portion/kcal có validation; khi đổi nutrition profile, hỏi xác nhận recalculation.
- Xóa item có confirm; nếu xóa item cuối, có thể giữ meal log rỗng hoặc xóa meal log theo policy đã chọn (MVP: xóa meal log rỗng cùng transaction).
- Mọi cập nhật tính lại dashboard khi reload; không nhân đôi bản ghi.

### 14.5. Dashboard

- Tổng ước tính kcal theo ngày/tập ngày.
- Biểu đồ cột 7 ngày gần nhất; phân bổ kcal theo loại bữa (stacked/column/pie tùy dữ liệu).
- Đếm số bữa/số món có/không có kcal; tooltip nguồn dữ liệu.
- Empty state rõ nếu chưa có nhật ký.
- Không gợi ý mục tiêu giảm cân hay diagnostic health.

### 14.6. Model Info / About

- Tên model, run id, số class, test metrics thật nếu file metrics tồn tại.
- Cấu trúc phương pháp so sánh (HOG+SVM, ResNet18, EfficientNet-B0).
- Nêu cảnh báo món hỗn hợp, ánh sáng/góc chụp, khác biệt văn hóa, confidence không chắc chắn.
- Dataset attribution, nutrition provenance, data privacy và model limitations.

### 14.7. Wireframe mô tả

```text
+----------------------------------------------------------------+
| Food Recognition & Dietary Assistant          Model: EfficientNet|
+--------------------+-------------------------------------------+
| Sidebar            | [Tải ảnh lên]                             |
| Home               | [Preview / Crop area]                     |
| Recognize          | [PHÂN TÍCH]                               |
| Meal Diary         | Top 3: [food A] [food B] [food C]         |
| Dashboard          | Confirmed food [dropdown]                |
| About              | Portion [S] [M] [L] [Custom grams]       |
|                    | kcal: [estimate/unavailable] + source     |
|                    | Date [..]  Meal type [..]                |
|                    | [XÁC NHẬN VÀ LƯU]                        |
+--------------------+-------------------------------------------+
```

---

## 15. CẤU TRÚC THƯ MỤC CODE — SẴN CHO AGENT

```text
food-recognition-dietary-assistant/
├── PLAN.md                         # Tài liệu này
├── README.md                       # Setup, training, inference, demo
├── pyproject.toml                  # Dependencies, Ruff, pytest config
├── .env.example                    # Không chứa secret
├── .gitignore
├── docker-compose.yml              # PostgreSQL local (SHOULD)
├── .streamlit/
│   └── config.toml                 # theme/upload limit
├── alembic.ini
├── migrations/
│   ├── env.py
│   └── versions/
├── configs/
│   ├── data.yaml                   # Dataset root, split seed, class list
│   ├── hog_svm.yaml                # Baseline config
│   ├── resnet18.yaml               # Train config
│   ├── efficientnet_b0.yaml        # Train config
│   ├── inference.yaml              # Threshold, top_k, device
│   └── classes_poc.yaml            # Chọn 10–20 classes ở POC
├── data/
│   ├── raw/                        # Food-101 archive/files (GITIGNORED)
│   ├── processed/                  # Derived training data (GITIGNORED)
│   ├── splits/
│   │   └── v1/                    # Manifest CSV, file paths
│   ├── local_eval/                 # Ảnh ảnh chụp mới có quyền sử dụng
│   └── nutrition/
│       ├── source_catalog.csv      # Nguồn nutrition, không bịa kcal
│       └── curated_mapping.csv     # Mapping đã kiểm duyệt
├── artifacts/
│   ├── models/
│   │   ├── hog_svm/
│   │   ├── resnet18/
│   │   └── efficientnet_b0/
│   ├── production/
│   │   └── current.json            # Pointer → winning model artifact
│   └── reports/
│       ├── experiment_comparison.csv
│       ├── confusion_matrices/
│       ├── learning_curves/
│       ├── failure_cases/
│       └── local_domain_shift/
├── storage/
│   ├── uploads/                    # User images (optional retention)
│   └── temp/                       # Temporary crop/image, cleaned
├── src/
│   └── food_assistant/
│       ├── __init__.py
│       ├── config.py
│       ├── app.py                   # Streamlit entrypoint
│       ├── ui/
│       │   ├── pages/
│       │   │   ├── home.py
│       │   │   ├── recognize.py
│       │   │   ├── diary.py
│       │   │   ├── dashboard.py
│       │   │   └── about.py
│       │   └── components/
│       │       ├── image_cropper.py
│       │       ├── prediction_card.py
│       │       └── nutrition_editor.py
│       ├── domain/
│       │   ├── enums.py             # meal_type, portion, status
│       │   └── schemas.py           # Internal DTO/Pydantic schemas
│       ├── services/
│       │   ├── image_service.py
│       │   ├── inference_service.py
│       │   ├── nutrition_service.py
│       │   ├── meal_service.py
│       │   ├── dashboard_service.py
│       │   └── model_registry.py
│       ├── ml/
│       │   ├── datasets.py
│       │   ├── transforms.py
│       │   ├── hog_features.py
│       │   ├── model_factory.py
│       │   ├── train_utils.py
│       │   ├── metrics.py
│       │   └── error_analysis.py
│       ├── persistence/
│       │   ├── database.py
│       │   ├── orm_models.py
│       │   ├── repositories.py
│       │   └── seed.py
│       └── utils/
│           ├── image_validation.py
│           ├── logging.py
│           └── paths.py
├── scripts/
│   ├── download_food101.py
│   ├── verify_dataset.py
│   ├── create_splits.py
│   ├── train_hog_svm.py
│   ├── train_resnet18.py
│   ├── train_efficientnet_b0.py
│   ├── evaluate_models.py
│   ├── analyze_failures.py
│   ├── select_production_model.py
│   ├── seed_database.py
│   └── smoke_test.py
├── tests/
│   ├── unit/
│   │   ├── test_image_validation.py
│   │   ├── test_crop.py
│   │   ├── test_label_mapping.py
│   │   ├── test_nutrition.py
│   │   ├── test_inference.py
│   │   └── test_meal_service.py
│   ├── integration/
│   │   ├── test_db_migrations.py
│   │   ├── test_meal_repository.py
│   │   └── test_diary_dashboard.py
│   └── smoke/
│       └── test_end_to_end.py
├── diagrams/
│   ├── 01-system-context.mmd
│   ├── 02-container-mvp.mmd
│   ├── 03-component-mvp.mmd
│   ├── 04-data-training.mmd
│   ├── 05-inference-sequence.mmd
│   ├── 06-meal-save-sequence.mmd
│   ├── 07-erd-postgresql.mmd
│   ├── 08-user-journey.mmd
│   ├── 09-state-recognize.mmd
│   ├── 10-deployment.mmd
│   ├── 11-roadmap.mmd
│   └── 12-future-architecture.mmd
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    ├── experiment_protocol.md
    ├── evaluation_report.md
    ├── nutrition_assumptions.md
    ├── deployment.md
    ├── limitations.md
    └── diagram_requirements.md
```

### 15.1. Quy tắc file/version

- Không commit `.env`, dataset gốc, model weights nặng, user uploads, DB dump chứa dữ liệu cá nhân.
- Commit seed/mapping *có nguồn và được phép chia sẻ*, train config, metric thực, documentation, diagram source.
- `pyproject.toml`: dùng version constraints tương thích với môi trường đã thử; có `requirements-lock.txt` hoặc lockfile phù hợp khi chốt môi trường. Không bịa version cài được khi chưa kiểm thử.
- Dùng path từ `config.py` với pathlib; tránh hardcode ổ `C:\...` làm mất khả năng chạy Colab/Linux.
- Model artifact phải tách khỏi code, có checksum và release version.

---

## 16. ENVIRONMENT VÀ HƯỚNG DẪN TRIỂN KHAI KỸ THUẬT

### 16.1. Local development

1. Cài Python 3.11, Git, PostgreSQL (hoặc Docker Desktop nếu dùng Compose), VS Code.
2. Tạo virtual environment.
3. Cài dependency từ `pyproject.toml`/lockfile sau khi agent tạo.
4. Copy `.env.example` thành `.env`, đặt `DATABASE_URL`.
5. Start PostgreSQL; kiểm tra readiness.
6. `alembic upgrade head` để tạo schema.
7. Chạy seed thực phẩm và demo user (chỉ nguồn đáng tin).
8. Tải dataset Food-101 vào `data/raw/`.
9. Tạo manifest bằng `create_splits.py`; chạy kiểm tra leakage.
10. Train baseline và CNN (trên local GPU hoặc Colab).
11. Evaluate và đặt production model pointer.
12. Chạy `streamlit run src/food_assistant/app.py` **sau khi đảm bảo package import path hoạt động** (nên cài project ở editable mode).
13. Upload ảnh và demo đủ luồng.

### 16.2. Ví dụ `.env.example` (không để password thật)

```dotenv
APP_ENV=development
DATABASE_URL=postgresql+psycopg://food_user:CHANGE_ME@localhost:5432/food_assistant
APP_TIMEZONE=Asia/Ho_Chi_Minh
MODEL_POINTER=artifacts/production/current.json
UPLOAD_DIR=storage/uploads
TEMP_DIR=storage/temp
MAX_UPLOAD_MB=10
MAX_IMAGE_PIXELS=20000000
SAVE_UPLOADED_IMAGES=false
SAVE_PREDICTION_HISTORY=false
INFERENCE_DEVICE=auto
```

**Lưu ý:** `DATABASE_URL` trên chỉ là mẫu. Phải đặt secret thật riêng, không commit. `INFERENCE_DEVICE=auto` chọn CUDA nếu khả dụng, nếu không dùng CPU; code phải bắt lỗi GPU thiếu RAM và có fallback/notification hợp lý, không âm thầm đổi model.

### 16.3. Docker Compose local

- 01 PostgreSQL service `postgres`; optional Adminer/pgAdmin chỉ ở môi trường dev (không bắt buộc).
- Mount named volume cho DB, healthcheck, không dùng mật khẩu mẫu thực tế.
- App Streamlit có thể chạy host venv; nếu containerize Streamlit, `DATABASE_URL` phải dùng hostname service `postgres` thay vì localhost.
- Có quy trình migration và backup trước deploy, không phụ thuộc DB GUI.

### 16.4. Training on Google Colab (khi không có NVIDIA GPU)

- Mount Drive hoặc tải dataset theo nguồn đã duyệt; không xáo trộn split.
- Đặt seed, class config, source commit, package versions.
- Tạo checkpoints và metric trong run folder.
- Download/sync **artifact đã tạo** sang máy chạy Streamlit; kiểm tra SHA256.
- Đo latency inference **trên máy demo**, tách khỏi tốc độ train của Colab.

---

## 17. TEST PLAN — UNIT / INTEGRATION / E2E / ML EVALUATION

### 17.1. Unit tests

| Test ID | Test case | Expected |
|---|---|---|
| UT-01 | JPEG/PNG hợp lệ | Decode RGB thành công |
| UT-02 | Dữ liệu giả PNG nhưng file thực không phải ảnh | Reject |
| UT-03 | File rỗng/corrupt/quá lớn | Error rõ ràng |
| UT-04 | Ảnh EXIF rotation | Orientation đúng |
| UT-05 | Crop coordinates ngoài [0,1] | Reject |
| UT-06 | Crop box rất nhỏ/trống | Reject |
| UT-07 | `idx_to_class` không khớp num_classes | Stop inference |
| UT-08 | Top-k có k > số class | Clamp/validate theo policy |
| UT-09 | Kcal known+gram known | Formula đúng |
| UT-10 | Kcal unknown | Kết quả `None`, không là 0 |
| UT-11 | Portion S/M/L không có grams mapping | Prompt custom grams |
| UT-12 | Sửa món sau khi infer | Lưu confirmed label riêng raw prediction |
| UT-13 | Đổi ảnh làm invalidate prediction | Không lưu nhãn cũ |
| UT-14 | User nhập negative grams/kcal | Reject |
| UT-15 | Không có dataset/model artifact | UI thông báo `MODEL_NOT_READY` |

### 17.2. Integration tests (PostgreSQL test DB)

| Test ID | Test case | Expected |
|---|---|---|
| IT-01 | Alembic upgrade từ DB trống | Mọi bảng/constraint tạo đúng |
| IT-02 | Seed chạy lại | Không tạo record trùng |
| IT-03 | Create meal + 2 items | 1 log, 2 items đúng FK |
| IT-04 | Save failure middle transaction | Rollback toàn transaction |
| IT-05 | Update portion | Recalculate kcal đúng snapshot policy |
| IT-06 | Delete 1 item | Tổng ngày cập nhật |
| IT-07 | Delete last item | Xóa empty meal theo policy |
| IT-08 | Unmapped food | Lưu không kcal, dashboard missing_count +1 |
| IT-09 | Concurrent writes cơ bản | Không double save do nút lặp; DB vẫn nhất quán |
| IT-10 | Unavailable DB | Không báo lưu thành công |

### 17.3. ML tests

- Same manifest và class order cho các model.
- No overlap giữa train/val/test.
- Preprocessing inference match artifact.
- Fixed seed cho split; log seed huấn luyện.
- Model output dimension = `num_classes`; tất cả class ids decode hợp lệ.
- Evaluation chạy `model.eval()` và `torch.inference_mode()`; không có augmentation ngẫu nhiên ở test.
- Metrics không lấy từ validation và gán nhãn là test.
- Confusion matrix đúng kích thước `C×C` với C class.
- Domain shift report riêng và có giải thích bias.

### 17.4. End-to-end scenarios

1. Ảnh hợp lệ có 1 món → infer → chỉnh nhãn → chọn S/M/L → lưu → diary → dashboard.
2. Ảnh 2 món → crop món A → lưu → crop món B → lưu → diary hiển thị 2 items.
3. Ảnh ngoài Food-101 → confidence/unknown hoặc user chọn khác → lưu manual item không giả calories.
4. Upload ảnh hỏng → UI không crash và DB không có log mới.
5. DB mất kết nối ở bước save → rollback/no success toast.
6. Nutrition source thiếu → cho lưu item nhưng hiển thị `Không có dữ liệu kcal`.
7. Thay model artifact → app reload model đúng, class mapping không chéo với checkpoint cũ.

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

## 19. RỦI RO, GIỚI HẠN VÀ BIỆN PHÁP

| Risk | Impact | Mitigation / Diagram note |
|---|---|---|
| Food-101 train label noise | Medium | Dataset audit; honest error analysis |
| Ảnh nhiều món không có bounding boxes | High | Crop thủ công mỗi món, không vẽ auto detector MVP |
| Món ngoài 101 classes | High | Unknown/manual correction, domain shift eval |
| ImageNet pretrained chưa chuyên cho food | Medium | Fine-tune trên Food-101 |
| Thiếu GPU / VRAM | High | Bắt đầu 10–20 class, batch nhỏ, Colab option |
| Nutrition không tương đương class | High | Curated mapping, nguồn, NULL nếu không chắc |
| Ước lượng gram chỉ bằng S/M/L | High | Gắn nhãn giả định, user input, không nói đo từ ảnh |
| SVM train trên full dataset chậm | Medium | POC subset, optimize extraction/batch |
| Test leakage | High | Immutable split, locked test protocol |
| Model + label map mismatch | High | Artifact contract + startup validation |
| DB mất kết nối | Medium | Transaction/rollback/error message |
| User image private | High | Default không lưu raw image; delete temp; quyền truy cập |
| CSV/SQL injected input | Medium | ORM parameter binding + validation |
| Exposed PostgreSQL | High | Local bind/internal network, strong credentials |
| Out-of-date package | Medium | Pin compatible lock after verification, update policy |
| Nhiều Streamlit reruns | Medium | `session_state`, resource cache, idempotent save handling |
| Độ tin cậy softmax sai lệch | Medium | Nêu raw score, optional temperature scaling val |
| Quyền sử dụng dataset/ảnh local | Medium | Kiểm tra attribution, source/terms |

### 19.1. Bảo vệ dữ liệu

- Default `SAVE_UPLOADED_IMAGES=false`: chỉ giữ ảnh trong memory/temporary khi infer, xóa sau phiên hoặc khi hoàn tất, trừ khi user đồng ý lưu.
- Nếu lưu ảnh thì lưu đường dẫn ngẫu nhiên UUID, không dùng trực tiếp filename user nhập; kiểm tra path traversal.
- File upload limit, dimension limit, decode validation; không dùng ảnh như lệnh hệ thống.
- Model artifacts trusted-only; không load pickle/joblib nhận từ user.
- PostgreSQL credentials trong environment, least privilege; không ghi password/PII vào log.
- Demo local không có auth, **không** mở public trong tình trạng chứa dữ liệu cá nhân.
- Có chức năng xóa nhật ký; nếu có image retention thì xóa ảnh tương ứng theo policy.

---

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

## 21. QUY TẮC THỰC HIỆN CHO CODING AGENT

1. Đọc `PLAN.md`, tạo task checklist; implement theo thứ tự dependency, không code mọi thứ trong một file Streamlit.
2. Với mỗi module, viết contract và tests trước hoặc đồng thời code.
3. Mọi dữ liệu kết quả ML phải lấy từ `metrics_*.json` đã sinh, không hardcode.
4. Chỉ sử dụng **đúng pretrained weights** khai báo và fine-tune; ghi checksum/version.
5. Train/test split + label map được ghi lại; verify trước mỗi run.
6. Khi chưa có artifact chạy được, hiển thị `MODEL_NOT_READY`, không mock prediction như kết quả thật.
7. Khi chưa có nutrition source cho món, trả `None` và yêu cầu input; không đoán calories.
8. ORM DB sessions có transaction và rollback rõ; migrations đi cùng code.
9. Mọi service phải import độc lập Streamlit để unit test được.
10. Thêm typing, exceptions, structured logs, input validation.
11. Không cho user upload file model thực thi; model artifact phải trusted.
12. Cuối mỗi phase, cập nhật README/diagram, running commands và test output.
13. Nếu GPU/dataset hoặc credentials chưa có, vẫn triển khai phần không phụ thuộc; báo blocker và không bịa tiến độ/metric.
14. Không tự ý thêm REST/microservice khi thiết kế MVP là modular monolith.
15. Khi agent thay đổi kiến trúc đã chốt, sửa `PLAN.md`, ADR/README và sơ đồ đồng bộ.

## 22. DEFINITION OF DONE / ACCEPTANCE CHECKLIST

### 22.1. ML / Research

- [ ] Dataset source/class list/splits rõ ràng, không leakage.
- [ ] HOG+SVM đã train và evaluate.
- [ ] ResNet18 fine-tuned và evaluate.
- [ ] EfficientNet-B0 fine-tuned và evaluate, hoặc nêu giới hạn nguồn lực.
- [ ] Cùng test benchmark với đúng class set.
- [ ] Accuracy, macro-F1, confusion matrix, per-class failures.
- [ ] Test metric không bị thay bởi ImageNet weight metric.
- [ ] Model selection có rationale và artifact class mapping khớp.
- [ ] Phân tích mixed dishes/plating/domain shift.

### 22.2. App / DB

- [ ] Upload/crop/analyze top-3 với model thực.
- [ ] Manual correction và unknown food.
- [ ] Portion S/M/L/custom grams và nutrition reference provenance.
- [ ] Không suy diễn gram/kcal chính xác chỉ từ ảnh.
- [ ] Meal CRUD và dashboard trên PostgreSQL.
- [ ] Không duplicate saves, rollback khi lỗi.
- [ ] Lưu snapshot nutrition tại thời điểm ghi nhật ký.
- [ ] Database migration từ DB trắng thành công.
- [ ] Không yêu cầu FastAPI để chạy MVP.
- [ ] Test coverage có unit/integration/E2E quan trọng.

### 22.3. Docs / Diagrams / Delivery

- [ ] Có 12 diagram sources `.mmd` hoặc source tương đương.
- [ ] Tên field/bảng trong ERD khớp ORM/migrations.
- [ ] Sơ đồ MVP không chứa kỹ thuật Future làm như đã triển khai.
- [ ] Model train/inference tách biệt rõ.
- [ ] Dữ liệu ảnh/labels/outputs đi đúng hướng.
- [ ] README chứa setup/train/evaluate/run trên một máy mới.
- [ ] Không commit secrets, raw private images, model nguy hiểm.
- [ ] Báo cáo khoa học và demo minh chứng actual results.

---
## 23. BỘ SƠ ĐỒ MERMAID ĐỂ DIAGRAMMING AGENT DỰNG TRỰC TIẾP

### 23.0. Quy chuẩn xuất sơ đồ

**Thư mục đích:** `diagrams/`. Đối với mỗi `DD-title` dưới đây, xuất `DD-title.mmd` và, nếu render được, `DD-title.svg` hoặc `DD-title.png` với cùng nội dung. Có thể dùng draw.io/Figma/PlantUML để trình bày lại đẹp hơn nhưng **không thay đổi ý nghĩa kiến trúc**.

**Style:** sạch, tươi sáng, nhất quán; `UI` xanh nhạt, `AI` tím nhạt, `Database` xanh lá/xanh teal, `Storage` vàng nhạt, `External` xám, `Error/Warning` đỏ nhạt; không mã hóa ý nghĩa chỉ bằng màu. Thể hiện nhãn tiếng Việt và tên technical English khi cần.

**Nguyên tắc tính trung thực:** Sơ đồ Mermaid dưới đây là **đặc tả mục tiêu** của hệ thống dự kiến, chưa phải bằng chứng hệ thống đã xây dựng. Diagramming agent không thêm tỷ lệ accuracy mẫu, danh tính user, hoặc kcal giả định lên sơ đồ sản phẩm. Mọi cạnh biểu diễn một hướng dữ liệu hoặc thao tác hợp lệ. Nếu không chắc, ưu tiên sơ đồ mục tiêu đã nêu và ghi chú là `planned`.

**Danh mục 12 diagram bắt buộc cho bộ tài liệu:**

| ID | File | Diagram type | Trả lời câu hỏi |
|---|---|---|---|
| D01 | `01-system-context.mmd` | System context | Ai dùng hệ thống, dữ liệu tới từ đâu? |
| D02 | `02-container-mvp.mmd` | C4-like container flowchart | Thành phần runtime MVP nào tồn tại? |
| D03 | `03-component-mvp.mmd` | Component/service | Các service và ranh giới trách nhiệm? |
| D04 | `04-data-training.mmd` | Flowchart | Train/evaluate/select model ra sao? |
| D05 | `05-inference-sequence.mmd` | Sequence | Upload/crop/predict ra top-k như thế nào? |
| D06 | `06-meal-save-sequence.mmd` | Sequence | Confirm/estimate/save/rollback thế nào? |
| D07 | `07-erd-postgresql.mmd` | ER diagram | Các bảng, khóa ngoại, quan hệ? |
| D08 | `08-user-journey.mmd` | Flowchart | Người dùng đi từ ảnh đến dashboard? |
| D09 | `09-state-recognize.mmd` | State diagram | Các trạng thái và nhánh lỗi của màn hình? |
| D10 | `10-deployment.mmd` | Deployment flowchart | Chạy local có những node nào? |
| D11 | `11-roadmap.mmd` | Gantt | Phụ thuộc và lịch làm theo phase? |
| D12 | `12-future-architecture.mmd` | Future architecture | Kiến trúc sau MVP khác thế nào? |

### D01 — System Context

```mermaid
flowchart LR
    U["Người dùng cuối"]
    R["Developer / Researcher"]
    C["Nutrition Curator"]
    SYS["Food Recognition & Dietary Assistant<br/>MVP local/demo"]
    FOOD["Food-101<br/>dataset ảnh và nhãn"]
    NUT["Nguồn dinh dưỡng<br/>bảng tham khảo có xuất xứ"]

    U -->|"Upload, crop, xác nhận, lưu và xem bữa ăn"| SYS
    SYS -->|"Dự đoán món, kcal ước tính, nhật ký"| U
    R -->|"Train, evaluate, chọn model release"| SYS
    FOOD -->|"Ảnh/nhãn dùng cho offline training"| R
    NUT -->|"Dữ liệu tham chiếu kcal/100g"| C
    C -->|"Review nutrition mapping/portion"| SYS
```

**Không có mũi tên Food-101 → inference request trực tiếp:** training là quy trình offline; runtime sử dụng trained artifact.

### D02 — Container Architecture MVP

```mermaid
flowchart TB
    USER["User / Browser"]
    subgraph APP["Python Application - Modular Monolith"]
        UI["Streamlit UI<br/>Home / Recognize / Diary / Dashboard / About"]
        SVC["Application Services<br/>Image / Inference / Nutrition / Meal / Dashboard"]
        ORM["Repositories + SQLAlchemy<br/>Transaction / Unit of Work"]
        PRED["PyTorch Runtime<br/>EfficientNet-B0 or selected model"]
    end
    DB[("PostgreSQL<br/>App records")]
    FS[("Filesystem<br/>Model artifacts + optional user uploads")]
    OFF["Offline pipeline<br/>Food-101 → train → evaluate"]

    USER <-->|"Streamlit interaction"| UI
    UI -->|"Internal Python calls"| SVC
    SVC -->|"CRUD and queries"| ORM
    ORM <-->|"SQL"| DB
    SVC -->|"Classify crop"| PRED
    PRED -->|"Read checkpoint + label map"| FS
    OFF -->|"Publish approved model release"| FS
    SVC -->|"Temporary file read/write if enabled"| FS
```

**Ghi chú:** Không có FastAPI, React, remote LLM, Kafka hoặc YOLO trong sơ đồ này.

### D03 — Component Diagram MVP

```mermaid
flowchart LR
    UI["Streamlit Pages"]
    IMG["ImageService"]
    CROP["CropService"]
    INF["InferenceService"]
    MODEL["ModelRegistry"]
    NUT["NutritionService"]
    MEAL["MealService"]
    DASH["DashboardService"]
    REPO["Repository Layer"]
    DB[("PostgreSQL")]
    ART[("Production Artifact")]

    UI --> IMG
    UI --> CROP
    UI --> INF
    UI --> NUT
    UI --> MEAL
    UI --> DASH
    IMG --> CROP
    INF --> MODEL
    MODEL --> ART
    NUT --> REPO
    MEAL --> NUT
    MEAL --> REPO
    DASH --> REPO
    REPO --> DB
```

**Rule:** `InferenceService` trả top-k chứ không ghi thẳng meal log. Chỉ `MealService` xử lý save sau user confirmation.

### D04 — Training and Model Selection Pipeline

```mermaid
flowchart TD
    A["Food-101 original<br/>train / test"] --> B["Validate images, labels, counts"]
    B --> C["Select POC classes or 101 full"]
    C --> D["Stratified split original train<br/>train_internal / val_internal"]
    D --> L["Locked test from original test"]
    D --> E["Shared train/val manifest"]

    E --> F1["HOG extraction<br/>train Linear SVM"]
    E --> F2["ResNet18 ImageNet weights<br/>replace head + fine-tune"]
    E --> F3["EfficientNet-B0 ImageNet weights<br/>replace head + fine-tune"]

    F1 --> V1["Tune with validation"]
    F2 --> V2["Checkpoint by validation"]
    F3 --> V3["Checkpoint by validation"]

    V1 --> G["Freeze candidate settings"]
    V2 --> G
    V3 --> G
    L --> H["Final test comparison"]
    G --> H
    H --> I["Top-1, macro-F1, confusion matrix<br/>latency, artifact size"]
    I --> J{"Select production model<br/>on measured evidence"}
    J --> K["Publish model + config + label map"]
    K --> M["artifacts/production/current.json"]

    D -.->|"Test never used for tuning"| H
```

**Bắt buộc:** Giữ `test` độc lập đến bước H; POC subset và full 101 là **hai thí nghiệm khác nhau**.

### D05 — Inference Sequence (không lưu bữa ăn)

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit Recognize
    participant IS as ImageService
    participant CS as CropService
    participant IN as InferenceService
    participant MR as ModelRegistry
    participant FS as Model Artifact FS

    U->>UI: Upload JPG/PNG
    UI->>IS: validate_and_decode(bytes)
    alt Invalid/too large image
        IS-->>UI: INVALID_IMAGE / IMAGE_TOO_LARGE
        UI-->>U: Hiển thị lỗi, yêu cầu ảnh khác
    else Valid image
        IS-->>UI: PIL RGB image
        UI-->>U: Preview and crop controls
        opt User selects crop
            U->>UI: Choose crop box
            UI->>CS: crop(image, normalized_box)
            CS-->>UI: Cropped RGB
        end
        U->>UI: Click Analyze
        UI->>IN: predict(RGB image/crop, top_k=3)
        IN->>MR: load_production_model()
        MR->>FS: Read model, preprocess, label mapping
        FS-->>MR: Trusted release files
        MR-->>IN: Model + transforms + labels
        alt Missing/inconsistent model
            IN-->>UI: MODEL_NOT_READY / ARTIFACT_MISMATCH
            UI-->>U: Báo lỗi, không tự tạo kết quả
        else Model ready
            Note over IN: eval + inference_mode<br/>resize/normalize match artifact
            IN-->>UI: Top-3 scores, model run_id, uncertain flag
            UI-->>U: Show predictions; require confirmation
        end
    end
```

**Note:** `ModelRegistry` có thể cache, không mở checkpoint từ disk trên mọi request. Sơ đồ mô tả dependency logic, không bắt buộc disk read lại mỗi lần.

### D06 — Nutrition Estimation and Meal Save Sequence

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit Recognize
    participant NS as NutritionService
    participant MS as MealService
    participant RP as Repositories
    participant DB as PostgreSQL

    U->>UI: Chọn/sửa món đã dự đoán
    U->>UI: Chọn S/M/L hoặc grams
    UI->>NS: lookup_and_estimate(confirmed_food, portion)
    NS->>RP: Find reviewed default nutrition profile
    RP->>DB: SELECT nutrition profile and mapping
    DB-->>RP: Profile or none
    RP-->>NS: Profile or none
    alt No reviewed source / missing grams
        NS-->>UI: kcal unavailable, request manual grams
        UI-->>U: Thông báo giới hạn, cho chỉnh tay
    else Valid reference + grams
        Note over NS: estimated_kcal = kcal_per_100g × grams / 100
        NS-->>UI: Estimated kcal + provenance + assumptions
        UI-->>U: Hiển thị kcal ước tính và nguồn
    end
    U->>UI: Confirm and Save
    UI->>MS: create_meal_with_items(validated_draft)
    MS->>RP: Begin transaction
    RP->>DB: INSERT meal_log and meal_items
    alt DB error
        DB-->>RP: Error
        RP->>DB: ROLLBACK
        RP-->>MS: SAVE_FAILED
        MS-->>UI: Error
        UI-->>U: Chưa lưu, có thể thử lại
    else Commit success
        RP->>DB: COMMIT
        DB-->>RP: Committed
        RP-->>MS: Saved IDs
        MS-->>UI: Success + IDs
        UI-->>U: Đã lưu, có thể xem Diary
    end
```

**Invariant:** Khi kcal không có nguồn thì vẫn lưu được `estimated_kcal=NULL`; `manual` kcal phải có dấu vết người dùng nhập.

### D07 — PostgreSQL ER Diagram

```mermaid
erDiagram
    users ||--o{ meal_logs : owns
    users ||--o{ prediction_history : requests
    model_versions ||--o{ prediction_history : generates
    meal_logs ||--|{ meal_items : contains
    prediction_history o|--o| meal_items : referenced_by
    food_classes o|--o{ meal_items : confirmed_as
    food_classes ||--o{ food_nutrition_mappings : mapped_by
    nutrition_foods ||--o{ food_nutrition_mappings : matched_to

    users {
        uuid id PK
        string display_name
        string email UK
        timestamptz created_at
    }
    food_classes {
        uuid id PK
        string dataset_key UK
        string display_name_en
        string display_name_vi
        boolean is_active
        timestamptz created_at
    }
    nutrition_foods {
        uuid id PK
        string name
        decimal kcal_per_100g
        string source_name
        string source_record_id
        string provenance_url
        timestamptz retrieved_at
        decimal small_grams
        decimal medium_grams
        decimal large_grams
        boolean reviewed
        string mapping_note
    }
    food_nutrition_mappings {
        uuid id PK
        uuid food_class_id FK
        uuid nutrition_food_id FK
        boolean is_default
        string confidence_note
    }
    model_versions {
        uuid id PK
        string model_name
        string run_id UK
        int num_classes
        string artifact_dir
        json metrics_json
        boolean is_production
    }
    prediction_history {
        uuid id PK
        uuid user_id FK
        uuid model_version_id FK
        string image_path
        string image_sha256
        json crop_box_json
        json topk_json
        decimal top1_score
        boolean uncertain
        timestamptz created_at
    }
    meal_logs {
        uuid id PK
        uuid user_id FK
        date consumed_on
        string meal_type
        string notes
        timestamptz created_at
        timestamptz updated_at
    }
    meal_items {
        uuid id PK
        uuid meal_log_id FK
        uuid confirmed_food_class_id FK
        uuid prediction_id FK
        string custom_food_name
        string portion_size
        decimal estimated_grams
        decimal kcal_per_100g_snapshot
        decimal estimated_kcal
        string nutrition_source_snapshot
        string calculation_method
        boolean is_user_corrected
        timestamptz created_at
        timestamptz updated_at
    }
```

**ERD note:** `model_versions` quản lý release; `class_to_idx` *thuộc model artifact* chứ không phải bảng nghiệp vụ. `prediction_history` và `confirmed_food_class_id` đều có thể NULL theo nghiệp vụ. Mermaid `||--|{ meal_items` biểu thị bản ghi bữa ăn có thể có từ một món sau khi commit; ứng dụng không lưu meal log rỗng cuối transaction. Diagramming agent phải review optionality khi triển khai ORM.

### D08 — User Journey / Activity Flow

```mermaid
flowchart TD
    START(["Open Recognize"]) --> UP["Upload image"]
    UP --> VAL{"Valid image?"}
    VAL -- No --> ERR["Show error and retry"]
    ERR --> UP
    VAL -- Yes --> PREV["Preview / optional manual crop"]
    PREV --> ANA["Analyze crop with trained model"]
    ANA --> READY{"Model available?"}
    READY -- No --> MODELERR["Show model setup error"]
    READY -- Yes --> TOP["Display top-3 + raw scores"]
    TOP --> CONF["User confirms/corrects food"]
    CONF --> PORT["Choose portion S/M/L or custom gram"]
    PORT --> NUT{"Reviewed nutrition mapping and grams?"}
    NUT -- Yes --> KC["Calculate estimated kcal + source"]
    NUT -- No --> UNK["Kcal unavailable / manual entry"]
    KC --> META["Choose date + meal type"]
    UNK --> META
    META --> SAVE["Confirm and save"]
    SAVE --> DBS{"DB committed?"}
    DBS -- No --> DBERR["Show save failed, retry"]
    DBERR --> SAVE
    DBS -- Yes --> DIARY["Meal Diary"]
    DIARY --> DASH["Dashboard"]
    DIARY --> MORE{"Another dish in same photo?"}
    MORE -- Yes --> PREV
    MORE -- No --> END(["Finish"])
    DASH --> END
```

### D09 — Recognize Screen State Machine

```mermaid
stateDiagram-v2
    [*] --> EMPTY
    EMPTY --> IMAGE_READY: upload valid image
    EMPTY --> ERROR: invalid image
    IMAGE_READY --> ANALYZING: click Analyze
    IMAGE_READY --> EMPTY: clear image
    ANALYZING --> RESULT_READY: model returns top-k
    ANALYZING --> ERROR: model/inference error
    RESULT_READY --> NEEDS_CONFIRMATION: low score or unknown
    RESULT_READY --> RESULT_READY: user edits label/portion
    NEEDS_CONFIRMATION --> RESULT_READY: user confirms corrected food
    RESULT_READY --> SAVED: DB commit success
    RESULT_READY --> ERROR: save error
    RESULT_READY --> IMAGE_READY: change image or crop
    SAVED --> IMAGE_READY: analyze another crop
    SAVED --> EMPTY: clear/reset
    ERROR --> IMAGE_READY: retry with valid image
    ERROR --> EMPTY: reset
```

**Quy tắc:** `RESULT_READY → IMAGE_READY` khi thay crop để bỏ kết quả model trước; `SAVED` chỉ sau PostgreSQL commit.

### D10 — Local Deployment Diagram

```mermaid
flowchart TB
    B["Web Browser<br/>http://localhost:8501"]
    subgraph HOST["Developer Laptop / Local Machine"]
        ST["Python venv<br/>Streamlit app & services"]
        RT["PyTorch inference<br/>CPU or CUDA device"]
        FILES[("Local Storage<br/>artifacts + optional uploads")]
        subgraph DBENV["PostgreSQL Environment"]
            PG[("PostgreSQL<br/>port 5432 - local/internal")]
            VOL[("Persistent DB volume/data")]
        end
        TRAIN["Offline training scripts<br/>CPU/GPU or external Colab"]
    end
    B <-->|"User interactions"| ST
    ST --> RT
    RT -->|"Load release"| FILES
    ST <-->|"SQLAlchemy / psycopg"| PG
    PG --> VOL
    TRAIN -->|"Release trained model to file storage"| FILES

    NOTES["No public access, no auth in single-user MVP"]
    ST -.-> NOTES
```

**Nếu training trên Colab**, diễn giải `TRAIN` là máy/cloud khác; chỉ copy release artifacts cần thiết sang host demo và xác minh checksum. PostgreSQL có thể là service local hoặc Compose container, không vẽ một DB cloud mặc định.

### D11 — Roadmap Gantt

```mermaid
gantt
    title Project 18 - 10 week implementation plan (proposed)
    dateFormat YYYY-MM-DD
    axisFormat %d/%m
    section Foundation
    Setup and architecture          :a1, 2026-11-02, 7d
    Dataset and EDA                :a2, after a1, 7d
    section ML Experiments
    HOG and SVM baseline           :a3, after a2, 7d
    ResNet18 and EfficientNet      :a4, after a3, 14d
    section App and Database
    PostgreSQL and nutrition       :a5, after a3, 7d
    Streamlit MVP                  :a6, after a4, 7d
    section Validation
    Robustness and error analysis  :a7, after a6, 7d
    Report diagrams and delivery   :a8, after a7, 14d
```

**Note:** Mốc 2026-11-02 chỉ là điểm bắt đầu minh họa sau ngày viết tài liệu, **không phải deadline đã được thầy xác nhận**. Agent có thể vẽ P4 Database song song CNN training; sơ đồ Gantt trên mô tả dependency tối thiểu và mang tính tham khảo.

### D12 — Future Architecture (KHÔNG PHẢI MVP)

```mermaid
flowchart LR
    USER["Web / Mobile Users"]
    FE["React or Mobile Client<br/>Future"]
    API["FastAPI REST<br/>Future"]
    AUTH["Auth & Authorization<br/>Future"]
    SV["Application Services"]
    AI["Model Inference Runtime"]
    DB[("PostgreSQL")]
    OBJ[("Object Storage<br/>Future")]
    OBS["Monitoring / Audit<br/>Future"]
    USER --> FE
    FE -->|"HTTPS / JSON / multipart"| API
    API --> AUTH
    API --> SV
    SV --> AI
    SV --> DB
    SV --> OBJ
    API --> OBS
    AI --> OBS
```

**Cảnh báo in đậm trong output:** `FUTURE ARCHITECTURE — NOT IMPLEMENTED IN MVP`. Không vẽ YOLO/segmentation trong sơ đồ tương lai như tính năng đã sẵn có nếu chưa có dữ liệu detection/segmentation.

---

## 24. MAPPING SƠ ĐỒ ↔ MODULE ↔ TEST

| Diagram | Modules | Tests/checks |
|---|---|---|
| D01 context | All, data sources | Coverage yêu cầu đề |
| D02 container | UI, services, persistence, ML | Smoke end-to-end |
| D03 component | Python services/repo | Unit/Integration contract |
| D04 training | `scripts/train_*`, `ml/*` | Split leakage, metrics validity |
| D05 inference sequence | `recognize`, `image_service`, `inference_service` | UT-01..08, E2E-01 |
| D06 meal save sequence | `nutrition_service`, `meal_service`, ORM | UT-09..14, IT-03..10 |
| D07 ERD | `orm_models.py`, migrations | IT-01, IT-02, referential tests |
| D08 user journey | UI and business services | E2E all scenarios |
| D09 UI state | Streamlit `st.session_state` | image-change, retry, save cases |
| D10 deployment | `.env`, Compose, app startup | Fresh clone smoke test |
| D11 roadmap | Phase/task list | Phases exit criteria |
| D12 future architecture | No MVP modules | Explicitly labeled Future |

## 25. CHECKLIST RÀ SOÁT SƠ ĐỒ CỦA AGENT

- [ ] Đủ 12 diagram names như mục 23.0.
- [ ] Mermaid source syntactically valid hoặc nêu chính xác lỗi chưa render.
- [ ] Trên container diagram, app là modular monolith với Streamlit + Python services.
- [ ] `Food-101` chỉ nằm trong training/data pipeline; runtime đọc artifact.
- [ ] `EfficientNet-B0` là candidate, **không** tự tuyên bố là winner.
- [ ] Có cả ResNet18 và HOG+SVM trong comparison.
- [ ] DB là PostgreSQL, ORM SQLAlchemy, migration Alembic.
- [ ] Nutrition data được map có nguồn; không AI estimate gram chính xác từ 1 ảnh.
- [ ] Có bước user correction trước khi lưu.
- [ ] Một ảnh/crop → một class chính; multi-dish bằng manual crop.
- [ ] Mọi meal save chạy transaction và rollback khi lỗi.
- [ ] ERD chính xác FK, nullable và default mapping.
- [ ] Ký hiệu nguồn external/dataset/model artifact khác database.
- [ ] Future architecture được tách biệt, gắn nhãn Future/Not in MVP.
- [ ] Không có số accuracy/kcal/runs giả trong sơ đồ kết quả.
- [ ] Mỗi sơ đồ có tiêu đề, legend nếu cần, tên file và note quan trọng.

---

## 26. OUTPUT REQUEST — PROMPT CÓ THỂ ĐƯA NGUYÊN VĂN CHO AGENT TẠO SƠ ĐỒ

> **Vai trò:** Bạn là software architect kiêm diagramming agent. Hãy đọc toàn bộ file `PLAN.md` của dự án `food-recognition-dietary-assistant`. Dựng 12 sơ đồ D01–D12 trong mục 23, lấy Mermaid blocks làm nền tảng nhưng được phép sắp xếp lại hình thức để dễ đọc. Không thêm công nghệ hoặc hành vi trái với các quyết định và phạm vi MVP trong mục 0, 4, 8–14. Với mỗi sơ đồ, lưu `diagrams/<exact_name>.mmd` và bản `.svg` (nếu khả thi); tạo `diagrams/README.md` mô tả đối tượng, ranh giới MVP/Future và link xem sơ đồ. Ưu tiên kiến trúc MVP (Streamlit → services → PostgreSQL và PyTorch artifact), pipeline học máy Food-101 → HOG/SVM vs ResNet18 vs EfficientNet-B0, ERD PostgreSQL, sequence upload/infer/confirm/save, state machine và deployment. Đánh dấu rõ limitation: portion không đo từ ảnh; nutrient values phải có nguồn; không tự nhận diện nhiều món. Tự kiểm tra tính thống nhất tên bảng/service và không bịa kết quả accuracy. Nếu chỉnh sửa ý nghĩa sơ đồ, ghi phần thay đổi và lý do để chủ dự án duyệt.

### 26.1. Output folder expected

```text
diagrams/
├── 01-system-context.mmd
├── 01-system-context.svg
├── 02-container-mvp.mmd
├── 02-container-mvp.svg
├── ...
├── 12-future-architecture.mmd
├── 12-future-architecture.svg
└── README.md
```

Nếu agent không có runtime render Mermaid, **chỉ tạo `.mmd` hợp lệ** và không làm giả `.svg/.png`. Nếu có thể, xuất thêm một trang tổng hợp 3–4 sơ đồ quan trọng cho slide thuyết trình, nhưng đó là deliverable phụ.

---

## 27. NGUỒN THAM KHẢO & TRA CỨU KỸ THUẬT

> Các link sau là tài liệu gốc để kiểm chứng lựa chọn kỹ thuật; tại thời điểm viết chưa có training result nào được giả định.

1. Food-101 homepage (ETH Zurich): https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
2. Torchvision Food101 dataset API (split `train`/`test`): https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html
3. Food-101 dataset details (TFDS, 101.000 ảnh / split, noisy train): https://www.tensorflow.org/datasets/catalog/food101
4. PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
5. Torchvision model weights & comparison: https://docs.pytorch.org/vision/stable/models.html
6. EfficientNet-B0 weights/transforms: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
7. Streamlit file uploader documentation: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
8. Streamlit model/resource caching: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
9. SQLAlchemy PostgreSQL dialect + psycopg: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
10. Alembic migration documentation: https://alembic.sqlalchemy.org/en/latest/
11. USDA FoodData Central datasets: https://fdc.nal.usda.gov/download-datasets/
12. USDA FoodData Central API guide: https://fdc.nal.usda.gov/api-guide/

## 28. ASSUMPTIONS VÀ CÁC ĐIỂM CÓ THỂ CẬP NHẬT

**A01:** Dùng PostgreSQL thay SQLite cho app.  
**A02:** Bản đầu single-user local/demo, chưa cần authentication.  
**A03:** UI Streamlit trực tiếp gọi Python services; FastAPI/React là Future.  
**A04:** HOG+SVM bắt buộc làm baseline; ResNet18 là CNN so sánh; EfficientNet-B0 là ứng viên chính.  
**A05:** Train trên POC 10–20 classes trước; nâng lên 101 khi GPU/thời gian cho phép.  
**A06:** S/M/L hoặc custom grams là ước tính dựa trên mapping; không có vision portion regression.  
**A07:** Tất cả kcal references cần nguồn rõ; không hardcode số tùy ý.  
**A08:** Mixed dishes xử lý manual crop; không có segmentation bắt buộc.  
**A09:** Không giả lập kết quả ML cho báo cáo chính thức.  
**A10:** Ngày trong Gantt chỉ minh họa, người quản lý có thể thay.

Nếu quyết định đổi stack (FastAPI ngay từ MVP, multi-user, tập món Việt, model ViT/CLIP...), **phải cập nhật PLAN.md và regenerate diagrams** thay vì chồng thêm sơ đồ mâu thuẫn.

---

**END OF PLAN.md — Source of truth cho triển khai và tạo sơ đồ dự án Project 18.**
