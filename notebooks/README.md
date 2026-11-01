# Notebook báo cáo giữa kỳ — Project 18

Mở [midterm_image_processing_demo.ipynb](midterm_image_processing_demo.ipynb).
Bản tại `notebooks/midterm_image_processing_demo.ipynb` bên trong thư mục này là
bản tương thích đường dẫn ban đầu, có nội dung/output giống hệt sau lần chạy bàn giao.
Dùng bản chính ở trên khi chỉnh sửa; script chạy sẽ đồng bộ lại bản tương thích.

## Chạy trong VS Code trên Windows 11

1. Mở notebook, chọn **Select Kernel → Python Environments → .venv (Python 3.11)**.
   Interpreter là `.venv\Scripts\python.exe` trong repository bên trong.
2. Chọn Restart Kernel rồi Run All. Notebook tự tìm repository từ thư mục notebook,
   repository hoặc workspace cha. Hướng dẫn chi tiết cũng nằm trong ô Markdown đầu.
3. Không cần GPU, Docker, PostgreSQL hay cài lại PyTorch. Các thư viện Jupyter,
   OpenCV, NumPy, Matplotlib, scikit-image, PyTorch/Torchvision đã có trong môi trường
   CPU được dùng để kiểm chứng. Thiếu ảnh/artifact thì báo lỗi, không tự tải dữ liệu.

Chạy độc lập từ PowerShell nếu không muốn mở giao diện Jupyter:

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv\Scripts\python.exe scripts/run_midterm_notebook.py
& .\.venv\Scripts\python.exe scripts/verify_midterm_notebook.py
```

Runner dùng chính interpreter đang gọi, kernel tạm và cache IPython/Matplotlib trong
workspace. Mỗi lần Run All tạo thư mục kết quả mới, không ghi đè các lần cũ.
`build_midterm_notebook.py` chỉ dùng để tái tạo template; không chạy lại khi đã chỉnh
notebook thủ công vì thao tác `--overwrite` sẽ thay nội dung notebook và xóa output cũ.

## Nội dung đã chạy

- RGB, ma trận ảnh, tách kênh, grayscale, HSV/Lab, histogram.
- Resize HOG; quy trình CNN validation/inference thực tế; crop, flip, float [0,1]
  và Normalize ImageNet; phân biệt StandardScaler trên vector HOG.
- Gaussian/Salt-and-Pepper noise có seed cố định, Gaussian/Median Filter,
  MSE/PSNR/SSIM và nhận xét định lượng từ lần chạy.
- Gradient, HOG 8.100 chiều, kiểm tra vector visualize khớp chính xác hàm production.
- Nạp SVM đã có, dự đoán 4 ảnh validation; đọc lại metric cũ, không fit/evaluate mới.
- Bảng/biểu đồ so sánh ba mô hình và tổng kết phạm vi giữa kỳ bằng tiếng Việt.

Chưa triển khai hình thái, phân đoạn và SIFT: bài minh họa chọn hướng nhận dạng cổ
điển HOG theo yêu cầu giữa kỳ. Không khẳng định bộ lọc cải thiện accuracy; muốn kết
luận cần thí nghiệm độc lập trên train/validation và phiên bản mô hình mới.

## Hình và bảng bàn giao

Lần chạy cuối được kiểm chứng:
`artifacts/reports/midterm_visualizations/run_20261031T144001_929466Z/`.
Các lần trước là quá trình kiểm tra/hoàn thiện, được giữ nguyên, không dùng thay bản cuối.

| File PNG 300 dpi | Nội dung |
|---|---|
| 01_anh_validation.png | Bốn ảnh validation và nhãn |
| 02_rgb_va_kenh_mau.png | RGB, ba kênh và grayscale |
| 03_hsv_lab_tung_kenh.png | Ba kênh HSV và Lab |
| 04_histogram.png | Histogram RGB và ảnh xám |
| 05_bien_doi_hinh_hoc.png | Resize, CNN crop, crop thủ công, lật ngang |
| 06_chuan_hoa_pixel.png | Miền pixel [0,1] và Normalize ImageNet |
| 07_nhieu_gaussian.png | Nhiễu Gaussian, Gaussian/Median Filter |
| 08_nhieu_salt_pepper.png | Nhiễu xung và kết quả lọc |
| 09_so_sanh_psnr.png | PSNR thực nghiệm trên ảnh minh họa |
| 10_gradient_hog.png | Ảnh xám, gradient và HOG |
| 11_vector_hog.png | Vector 8.100 đặc trưng |
| 12_pipeline_hog_svm.png | Sơ đồ pipeline cổ điển |
| 13_so_sanh_mo_hinh.png | Accuracy, Macro-F1, p50 CPU từ báo cáo đã lưu |

Hai bảng: `noise_metrics.csv`, `saved_model_comparison.csv`.
`execution_summary.json` ghi seed, thư viện, ảnh/nhãn/split/checksum, nguồn metric,
danh sách hình và kết quả kiểm tra file bảo vệ. Không lưu nội dung `.env`.

Ảnh nguồn: Food-101 cục bộ của dự án, nguồn ETH Zurich đã được người dùng xác nhận
cho phép sử dụng. Bốn ID: hamburger/1010695, ice_cream/102595, pizza/1040878,
chicken_curry/104874, tất cả là validation. Ảnh hamburger được chọn thủ công theo nội
dung để tránh dùng tượng quảng cáo làm ảnh món ăn chính, không chọn theo dự đoán model.

## Kiểm chứng thực tế

23 ô gồm 13 code + 10 Markdown; 13 code chạy đầu–cuối không lỗi; 13 ảnh nhúng và
13 PNG xuất hợp lệ; hai bản notebook giống nhau. HOG visualization khớp vector thật;
Normalize CNN khớp preprocessing; nhiễu tái lập đúng seed; các file model/config/pointer
và `.env` được bảo vệ giữ nguyên hash trước/sau. Ruff cho ba script đạt.

Không chạy training, không đọc ảnh test hoặc chạy lại Evaluate, không đổi pipeline
production, không kết nối database và không sửa dữ liệu bữa ăn.
