# Giao diện Healthy Food · Project 18

Giao diện dùng mint/trắng, điểm cam nhẹ, ảnh món ăn gốc ở Home, trang nhận diện có
tải ảnh/chụp ảnh và Dashboard dùng dữ liệu nhật ký thực tế. Ảnh `3.jpg` được giữ nguyên
byte; cách đặt ảnh thay đổi bằng CSS responsive. Hai ảnh mẫu giao diện chỉ dùng tham khảo.

## Mở phiên bản mới trên Windows

Không cần migration hoặc seed. Giữ PostgreSQL đang chạy. Dừng terminal backend và
frontend cũ bằng `Ctrl+C` rồi chạy trong **hai terminal riêng** từ thư mục repository:

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv-cuda\Scripts\python.exe -m uvicorn food_assistant.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv-cuda\Scripts\python.exe -m streamlit run src/food_assistant/app.py
```

Backend cần khởi động lại để Dashboard nhận các trường tổng hợp macro mới. `.env`,
production model và pointer không thay đổi. Mở http://127.0.0.1:8501 rồi tải lại trang.
Nếu đang dùng Docker thay vì chạy trực tiếp, cập nhật image bằng quy trình có ML đã
ghi trong NUTRITION_GUIDE.md; không chạy thêm backend trên cùng cổng và không xóa volume.

## Kiểm tra giao diện và camera

1. Home: banner có ảnh gốc, chữ trên vùng nền sáng dễ đọc. “Bắt đầu nhận diện” mở trang
   Nhận diện. Thu nhỏ cửa sổ: ảnh và nội dung xếp dọc, không che chữ.
2. Nhận diện: thử Tải ảnh JPEG/PNG, xem preview/crop, Phân tích và xác nhận món.
3. Chọn Chụp ảnh, cấp quyền camera trong trình duyệt, chụp một món và Phân tích.
   Camera dùng cùng kiểm tra định dạng/kích thước và model production hiện tại.
4. Đổi nguồn ảnh hoặc bỏ ảnh: kết quả cũ và trạng thái xác nhận/lưu phải được xóa.
   Chụp/tải lại rồi xác nhận món/khẩu phần trước khi lưu vào Nhật ký.
5. Dashboard: chọn khoảng ngày có bữa ăn. Xem bốn card, biểu đồ năng lượng, tỷ trọng
   theo loại bữa và các macro theo ngày. Hover/chạm biểu đồ để xem số liệu.
6. Đối chiếu macro với Nhật ký. Món chưa có macro không đóng góp số 0 giả; từng card
   báo số món có/thiếu dữ liệu. Món nhập kcal thủ công không tự sinh macro.
7. Thử khoảng ngày không có nhật ký: không hiển thị dữ liệu mẫu hoặc mục tiêu giả.
8. Kiểm tra trên màn hình khoảng 390px và máy tính: sidebar thu gọn được, card/cột
   xếp dọc theo Streamlit, bảng rộng có thể cuộn ngang, biểu đồ theo chiều rộng.

Camera trình duyệt cần localhost hoặc HTTPS và quyền truy cập camera. Trên điện thoại,
truy cập một địa chỉ LAN HTTP thường không đủ để mở camera; dùng HTTPS hoặc dùng
tải ảnh từ thư viện/chụp bằng camera hệ điều hành qua bộ chọn tệp. Không mở cơ chế
bỏ qua bảo mật trình duyệt. Việc kiểm thử tự động đã giả lập dữ liệu JPEG từ camera;
quyền camera và phần cứng thật cần kiểm tra trên thiết bị.

## Dữ liệu và giới hạn hiển thị

Card dinh dưỡng là tổng các giá trị **đã lưu** trong khoảng ngày, không phải khuyến nghị
hay mục tiêu một ngày. Tỷ trọng tròn là kcal giữa các loại bữa, không phải mức hoàn
thành mục tiêu sức khỏe. Không nội suy ngày chưa có dữ liệu; từng macro có độ bao phủ
riêng. Các giá trị dinh dưỡng vẫn chỉ là ước tính từ thực phẩm/khẩu phần tham chiếu.

Các file chính: `ui/theme.py`, `ui/pages/recognize.py`, `ui/pages/dashboard.py`,
`ui/assets/food_photography.jpg`, `services/dashboard_service.py`.
Kết quả kiểm thử thực tế tại TEST_REPORT.md; không huấn luyện hoặc đánh giá lại model.
