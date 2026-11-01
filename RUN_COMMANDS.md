# Lệnh PowerShell — tiếp tục từ repository hiện tại

Ưu tiên mới cho AI thật POC/GPU: [POC_REAL_AI.md](POC_REAL_AI.md).
Các bước CPU cũ bên dưới được giữ làm tham khảo; dùng `scripts/poc_ai.ps1`
cho CNN CUDA và so sánh HOG cũ với hai CNN có run ID riêng.

Cập nhật 31/10/2026. Tác vụ dài do bạn chủ động chạy. Chạy từng bước trong Terminal VS Code;
đợi lệnh hiện tại kết thúc trước khi chuyển bước. Script không tự mở terminal hay chạy nền.

## 1. Vào đúng repository, xem trạng thái trước

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv\Scripts\python.exe scripts/status_project.py --docker
```

Trạng thái đã kiểm tra trong phiên này:

- Dataset Food-101 hoàn chỉnh: 101.000 ảnh. Không cần tải lại.
- POC v1: 6.000 train, 1.500 validation, 2.500 test; manifest và nhãn đã có.
- PostgreSQL, backend và frontend Docker đều healthy; port 5432/8000/8501.
- HOG + SVM run `poc-v1-cpu-budget-v2-primal` đang chạy (PID thực tại lần kiểm tra: 10308;
  PID wrapper 2992). **Không chạy thêm TrainHog khi tiến trình này còn hoạt động.**
- Hai orchestrator 11868/15760 đã dừng; SVM được giữ lại, CNN không tự khởi chạy tiếp.
- Có installer pip cũ, PID 3540/26680. Status phân biệt pip với pytest; chưa tự dừng installer.
  Không chạy pip install chồng khi installer còn hoạt động.
- Hai thư mục HOG v1/v2 là run dở, chưa có artifact sealed/trained; chưa có CNN thực,
  test metrics chính thức hoặc production pointer. PID là ảnh chụp tại thời điểm kiểm tra,
  có thể hết hiệu lực; luôn xem status mới trước khi thao tác.

Tệp `.tar.gz.part`, các chunk và archive đã ghép vẫn được giữ để truy vết/resume.
Sự tồn tại của chúng không có nghĩa dataset còn tải dở. Không tự xóa dữ liệu này.

Nếu PowerShell chặn `.ps1`, thay `& .\scripts\project.ps1 ...` bằng
`powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\project.ps1 ...`.
Lệnh này chỉ áp dụng policy cho tiến trình PowerShell đó.

## 2. Môi trường và dữ liệu: hiện đã chuẩn bị

Venv/dependencies đã có; không cần cài lại để tiếp tục. Chỉ khi dựng môi trường mới,
và sau khi xác nhận không có pip đang hoạt động:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[ml,dev]"
```

Chỉ khi chuyển máy hoặc dataset chưa hoàn chỉnh:

```powershell
& .\scripts\project.ps1 -Step DownloadData
& .\scripts\project.ps1 -Step PrepareData
```

Kiểm tra dữ liệu lại khi bạn muốn chạy một tác vụ kiểm tra dài:

```powershell
& .\scripts\project.ps1 -Step VerifyData
```

PrepareData giữ nguyên manifest v1 nếu đã có. Muốn đổi class/split phải dùng version/output mới.
Food-101 đã được xác nhận phù hợp quy định môn học; archive chỉ cho scientific fair use.

## 3. Theo dõi HOG hiện tại; không chạy trùng

```powershell
Get-Content .\artifacts\reports\poc-v1-cpu-budget-v2-primal\train_hog_svm.log -Tail 20
& .\.venv\Scripts\python.exe scripts/status_project.py
```

Không có log mới trong lúc `LinearSVC.fit` không đồng nghĩa đã dừng. Xem PID/CPU và
model_config.trained/checksum thay vì đoán từ log. Khóa mới chặn cả tiến trình legacy
chưa có lock file. Không ghi đè run dở.

Lệnh huấn luyện HOG độc lập cho một run mới **chỉ khi không còn HOG đang chạy**:

```powershell
& .\scripts\project.ps1 -Step TrainHog -RunId poc-v1-cpu-manual-v3
```

Nếu HOG v2 hoàn thành, tiếp tục dùng RunId v2 ở các bước sau. Nếu bạn chọn run mới,
dùng cùng RunId đó cho cả ba mô hình và các bước Evaluate/Select.

## 4. Huấn luyện CNN tuần tự, sau HOG

Máy có RTX 3050 nhưng PyTorch hiện là CPU build. Các lệnh dưới dùng CPU, không giả định CUDA:

```powershell
& .\scripts\project.ps1 -Step TrainResNet -RunId poc-v1-cpu-budget-v2-primal
& .\scripts\project.ps1 -Step TrainEfficientNet -RunId poc-v1-cpu-budget-v2-primal
```

Lệnh tương đương để chỉnh cấu hình trực tiếp:

```powershell
& .\.venv\Scripts\python.exe -u scripts/train_resnet18.py --run-id poc-v1-cpu-budget-v2-primal --device cpu --head-epochs 1 --fine-tune-epochs 1 --batch-size 4
& .\.venv\Scripts\python.exe -u scripts/train_efficientnet_b0.py --run-id poc-v1-cpu-budget-v2-primal --device cpu --head-epochs 1 --fine-tune-epochs 1 --batch-size 4
```

Đây là ngân sách POC ngắn đã khai báo trước test, không phải benchmark full 101 lớp.
Trọng số ImageNet đã cache trong artifacts/pretrained. Không chạy cả hai dạng lệnh;
chọn wrapper hoặc lệnh Python tương đương. Run folder có sẵn bị từ chối trước khi đọc dataset.

Checkpoint resume (CNN, dùng run mới):

```powershell
& .\.venv\Scripts\python.exe -u scripts/train_resnet18.py --run-id poc-resnet-resumed-v1 --resume artifacts/models/resnet18/poc-v1-cpu-budget-v2-primal/resume.pth --device cpu --head-epochs 1 --fine-tune-epochs 2 --batch-size 4
```

Chỉ resume checkpoint thực sự có sẵn. HOG chưa có checkpoint resume giữa một lần fit;
không dùng `--resume` cho HOG. Giữ manifest/class map cũ; không dùng test để chọn epoch.

## 5. Đánh giá sau khi cả ba ứng viên đã chốt

```powershell
& .\scripts\project.ps1 -Step Evaluate -RunId poc-v1-cpu-budget-v2-primal
& .\scripts\project.ps1 -Step Select -RunId poc-v1-cpu-budget-v2-primal
```

Evaluate kiểm tra đủ ba artifact trained/non-smoke trước khi mở test; giữ lại test evaluation
đã có thay vì chạy lại. Select kiểm tra cùng manifest, bộ nhãn, dataset version và số mẫu;
chọn theo validation macro-F1, latency để phá hòa. Nó xuất
artifacts/reports/experiment_comparison.csv và artifacts/production/current.json từ dữ liệu thực.
Không tự tạo pointer hoặc điền accuracy bằng tay.

## 6. Chạy ứng dụng với code mới

Container hiện vẫn chạy snapshot image trước một số sửa code mới. Khi bạn muốn cập nhật
runtime mock, tự chạy:

```powershell
& .\scripts\project.ps1 -Step ComposeUp
& .\scripts\project.ps1 -Step MockSmoke
```

Image mặc định chưa chứa PyTorch. Sau khi có real production artifact, có thể dùng host venv
để kiểm tra real inference mà không cần build image ML:

```powershell
docker compose stop backend frontend
& .\scripts\project.ps1 -Step Backend
# Terminal VS Code thứ hai, cùng thư mục repository:
& .\scripts\project.ps1 -Step Frontend
# Terminal thứ ba, sau khi API sẵn sàng và production pointer đã được tạo:
& .\scripts\project.ps1 -Step RealSmoke
```

Lựa chọn Docker có CPU ML (build lớn, do bạn chủ động chạy):

```powershell
docker compose build --build-arg INSTALL_ML=true backend
docker compose up -d
& .\scripts\project.ps1 -Step RealSmoke
```

UI http://127.0.0.1:8501; API docs http://127.0.0.1:8000/docs.
Không reset DB. Không chạy host backend/frontend chồng lên các container đang chiếm cùng port.

## 7. Kiểm thử do bạn chủ động chạy

```powershell
& .\scripts\project.ps1 -Step Tests
# Hoặc chia nhóm, không chạy cả suite và từng nhóm đồng thời:
& .\.venv\Scripts\python.exe -m pytest -m "not integration and not ml" -q
& .\.venv\Scripts\python.exe -m pytest -m integration -q
& .\.venv\Scripts\python.exe -m pytest -m ml -q
& .\.venv\Scripts\python.exe -m ruff check src scripts tests migrations
```

Toàn bộ test dùng TEST_DATABASE_URL riêng cho integration, ảnh tổng hợp cho ML smoke.
Full suite chưa được chạy lại sau các sửa đổi mới nhất; xem TEST_REPORT.md cho kết quả từng lượt.

## Script tổng hợp: chỉ xem trước mặc định

```powershell
& .\.venv\Scripts\python.exe scripts/run_poc_experiment.py --run-id poc-v1-cpu-budget-v2-primal --skip-existing
```

Lệnh trên chỉ in danh sách lệnh, không train/test. Chỉ thêm `--execute` nếu bạn chủ động
muốn chạy toàn bộ pipeline sau khi không còn tiến trình trùng. Với tình trạng hiện tại,
nên dùng từng bước độc lập ở trên.
