# Nutrition & Portion Engine — POC 10 lớp

Catalog đã đối chiếu đúng 10 nhãn trong `data/splits/v1/dataset_manifest.csv` và
`labels.json`. Không thay đổi class ID, checkpoint hay production pointer.

## Nguồn và phạm vi dữ liệu

Ưu tiên SR Legacy: 7 tham chiếu. Ba món tổng hợp còn lại sử dụng FNDDS 2021–2023.
Các con số năng lượng và macro được trích trực tiếp từ bản phát hành chính thức của
[USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets/), theo 100 g;
không tính năng lượng bằng cách suy ngược từ macro. Xem
[tài liệu dữ liệu USDA](https://fdc.nal.usda.gov/data-documentation/) để phân biệt
SR Legacy và dữ liệu khảo sát FNDDS.

| Nhãn manifest | Tham chiếu FDC | kcal/100 g | S/M/L (g) | Giới hạn ánh xạ |
|---|---:|---:|---|---|
| apple_pie | 175011 | 237 | 80/125/180 | Bánh táo thương mại, bột tăng cường vi chất |
| caesar_salad | 2709591 | 77 | 100/200/300 | Romaine, **không sốt**; thêm sốt làm thay đổi kết quả |
| cheesecake | 172711 | 321 | 60/100/150 | Cheesecake thương mại |
| chicken_curry | 2706437 | 107 | 150/250/350 | Cà ri gà, không gồm cơm ăn kèm |
| french_fries | 170698 | 312 | 70/120/180 | Khoai tây thức ăn nhanh chiên dầu thực vật |
| hamburger | 170694 | 263 | 100/150/220 | Một miếng thịt cỡ thường, có gia vị; không mặc định có phô mai |
| ice_cream | 167575 | 207 | 50/100/150 | Kem vanilla; không đại diện mọi loại kem |
| omelette | 172185 | 154 | 60/120/180 | Trứng nguyên quả tráng chín; không đại diện mọi loại nhân |
| pizza | 170317 | 268 | 100/200/300 | Pizza phô mai đế thường đông lạnh đã nướng |
| sushi | 2708959 | 94 | 90/180/270 | Sushi NFS, không xác định loại; nhân và sốt có thể khác |

**Toàn bộ gram S/M/L là giả định thiết kế để ghi nhật ký**, không phải khẩu phần
chính thức USDA hoặc kết quả đo bằng AI. Người dùng cần sửa gram khi biết khối lượng
thực tế. Các ánh xạ là thực phẩm đại diện do người triển khai đối chiếu mô tả nguồn,
không phải bằng chứng mọi ảnh trong lớp Food-101 có cùng thành phần dinh dưỡng.

10/10 nhãn có tham chiếu kcal và ba macro; không còn lớp POC nào thiếu tham chiếu trong
catalog. Các biến thể công thức, món ngoài danh mục, sốt và món ăn kèm chưa được tự
phân tách. Thiếu nguồn hoặc gram thì trạng thái `unavailable`, kcal `null`, không tự
điền 0. Khẩu phần Custom bắt buộc nhập gram trên giao diện trước khi lưu.

## Cách tính và lưu

`estimated_kcal = kcal_per_100g * grams / 100`, làm tròn hai chữ số `ROUND_HALF_UP`.
Protein, carbohydrate và fat tính tương tự; macro thiếu vẫn là `null`.

- `reference_formula` / `source_type=reference`: công thức theo nguồn tham chiếu.
- `manual` / `source_type=user_input`: kcal do người dùng nhập; không suy ra macro,
  không sửa bảng USDA. 0 kcal chỉ hợp lệ khi người dùng chủ động nhập 0.
- `unknown` / `source_type=unavailable`: chưa có đủ dữ liệu để tính năng lượng.
- Snapshot của nhật ký giữ nguồn, phiên bản, mã FDC, dữ liệu/100 g, gram, nguồn gram
  và ghi chú tại thời điểm lưu. Nạp catalog không cập nhật các nhật ký cũ.
- Dashboard cộng các kcal đã biết và báo số món thiếu riêng. Tổng 0 của tập rỗng
  hoặc phép cộng bỏ qua `null` không phải một phép đo 0 kcal cho món thiếu; giao diện
  hiện “Chưa có dữ liệu” khi tất cả món trong khoảng ngày đều thiếu kcal.

Schema PostgreSQL hiện tại đủ dùng, **không thêm migration**.

## Cập nhật trên Windows 11

Các lệnh chạy từ đúng repository bên trong. Không cần tải lại dataset hay training.
Nếu backend/frontend đang chạy trực tiếp, dừng hai terminal đó bằng `Ctrl+C` trước
khi khởi động lại; giữ PostgreSQL đang hoạt động. Không chạy thêm bản thứ hai trên
cùng cổng. Lệnh nạp dữ liệu lấy kết nối từ `.env` hiện có, không ghi đè file này.

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv\Scripts\python.exe scripts/seed_nutrition.py
```

Lệnh này chỉ bổ sung profile/mapping dinh dưỡng đã rà soát. Chạy lại an toàn:
`profiles_added=0`, `mappings_added=0` khi không có dữ liệu mới. Giữ nguyên default
mapping có sẵn; nếu có default khác, `defaults_preserved` thông báo số xung đột được
giữ nguyên. Profile cùng nguồn/mã/phiên bản mà khác số liệu sẽ báo lỗi và rollback,
không ghi đè. Không reset database và không cập nhật/xóa nhật ký.

Terminal backend (môi trường CUDA đã cài trước đó):

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv-cuda\Scripts\python.exe -m uvicorn food_assistant.api.main:app --host 127.0.0.1 --port 8000
```

Terminal frontend:

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv-cuda\Scripts\python.exe -m streamlit run src/food_assistant/app.py --server.address 127.0.0.1 --server.port 8501
```

Giữ cấu hình production EfficientNet-B0 hiện tại. Có thể dùng `.venv` cho seed và
kiểm chứng HTTP; `.venv-cuda` cho backend dùng GPU. Không cần cài lại PyTorch.

Nếu đang dùng **backend/frontend Docker**, dùng phương án riêng sau thay cho hai
lệnh chạy trực tiếp. Build bật ML để không mất khả năng real inference; container
chạy AI trên CPU. Chỉ chọn các service của Compose tại repository này:

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
docker compose build --build-arg INSTALL_ML=true backend
docker compose up -d --no-deps backend frontend
docker compose exec backend python scripts/seed_nutrition.py
```

Build có thể tải dependency lớn; đây là lệnh cho người dùng tự chạy khi cần. Không
dùng `down -v`, prune hay tác động tới container/volume dự án khác. Backend Compose
vốn chạy migration hiện có và seed lúc khởi động; không có migration mới của tính
năng này. Nếu PostgreSQL của dự án chưa chạy, khởi động riêng service đó bằng
`docker compose up -d postgres` trước.

## Kiểm chứng tự động và giao diện

Terminal khác, sau khi backend đã khởi động lại:

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv\Scripts\python.exe scripts/check_nutrition.py --base-url http://127.0.0.1:8000
```

Script chỉ GET, không gọi AI, không tạo nhật ký, không đánh giá model. Nó kiểm tra
12 trường hợp: Hamburger/Kem/Pizza × S/M/L/Custom 123.45 g; so sánh đúng nguồn,
phiên bản, gram và công thức. Có thể kiểm tra toàn bộ lớp bằng `--foods`:

```powershell
& .\.venv\Scripts\python.exe scripts/check_nutrition.py --foods apple_pie caesar_salad cheesecake chicken_curry french_fries hamburger ice_cream omelette pizza sushi
```

Mở http://127.0.0.1:8501 → Nhận diện món ăn → chọn production → tải ảnh → Phân tích.
Xác nhận/sửa món lần lượt Hamburger, Kem, Pizza. Ở khẩu phần Vừa (M):

| Món | Gram tham khảo | Kcal ước tính |
|---|---:|---:|
| Hamburger | 150 | 394.50 |
| Kem vanilla | 100 | 207.00 |
| Pizza phô mai | 200 | 536.00 |

Đổi S/L, bật nhập gram hoặc chọn Custom: kcal/macro phải đổi ngay. Khi chuyển món,
gram và kcal nhập tay của món trước được xóa khỏi form. Thử nhập 50 g Pizza:
134.00 kcal. Thử tự nhập 123 kcal: preview hiển thị nguồn `user_input`, không hiển
thị macro suy đoán. Xác nhận, lưu rồi xem Nhật ký và Dashboard. Với món khác không
có nguồn, giữ kcal chưa xác định hoặc chủ động nhập kcal.

“Giá trị dinh dưỡng chỉ là ước tính theo dữ liệu tham khảo, không phải đo trực tiếp từ ảnh”.

## Tái tạo catalog và kiểm thử dành cho phát triển

Không bắt buộc chạy lại để dùng ứng dụng: catalog đã được tạo sẵn trong Git.
`configs/nutrition_poc.json` chỉ chứa ID nguồn, mô tả chính xác, ghi chú và gram giả
định; không tự nhập số kcal/macronutrient vào cấu hình. Parser kiểm tra mô tả và
đơn vị kcal/g, nguồn/phiên bản và manifest. SHA256 của hai archive trong
`data/nutrition/coverage.json` là checksum tính tại máy, không mạo nhận checksum
chính thức. Archive ảnh Food-101 không liên quan tới thao tác này.

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_nutrition.py
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_nutrition.py tests/unit/test_nutrition_catalog.py tests/unit/test_inference.py tests/integration/test_automatic_nutrition.py tests/integration/test_diary_dashboard.py tests/integration/test_frontend_flow.py tests/integration/test_meal_repository.py tests/unit/test_meal_service.py tests/smoke/test_end_to_end.py -q
```

`prepare_nutrition.py` đọc archive USDA cục bộ (`usda_sr_legacy_2018.zip` và
`usda_fndds_2021_2023.zip`), không tự tải gì. Thiếu archive thì báo lỗi rõ ràng.
Các test tích hợp chỉ dùng `TEST_DATABASE_URL` tên database kết thúc `_test`,
transaction được rollback. Smoke real dùng một ảnh tổng hợp trên CPU, không dùng
tập test Food-101, không suy ra accuracy. Không cần GPU hoặc training cho bộ test này.

Mở rộng lên 101 lớp bằng manifest/labels đã chốt, bổ sung mapping được kiểm chứng
vào cấu hình và chạy chuẩn bị catalog. Lớp chưa có mapping được báo `unavailable`;
không tự chọn một món gần giống làm nguồn thật.
