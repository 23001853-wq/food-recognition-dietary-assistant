# Nutrition Assumptions

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

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
