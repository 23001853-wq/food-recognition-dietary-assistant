# AI thật — Food-101 POC 10 lớp, RTX 3050 Ti 4 GB

Ngày 31/10/2026. Chỉ bạn chạy các tác vụ dài trong Terminal VS Code.
Không mở rộng 101 lớp trước khi hoàn thành các bước dưới đây.

## 1. Trạng thái và dữ liệu đã kiểm tra

```powershell
Set-Location 'D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant'
& .\.venv\Scripts\python.exe scripts/status_project.py
& .\.venv\Scripts\python.exe scripts/audit_poc.py --output data/splits/v1/poc_contract_audit.json
```

Audit nhanh đã chạy: 10 lớp, 6.000 train / 1.500 validation / 2.500 test, thứ tự
class ID cố định 0..9, khớp class config/labels/manifest và các CSV con. Tập test POC
khớp chính xác các ảnh test chính thức của 10 lớp. Train/val chỉ chứa ảnh train gốc.
Manifest SHA256: `fcac93e2ca9eb3e179bbb76636d813bcbbffa37b9f9e841a21401d671603bd5a`.

Audit này không giải mã/hash lại toàn bộ ảnh. Khi bạn muốn kiểm tra sâu (tác vụ dài):

```powershell
& .\.venv\Scripts\python.exe scripts/verify_dataset.py --manifest-dir data/splits/v1
& .\.venv\Scripts\python.exe scripts/audit_near_duplicates.py --manifest-dir data/splits/v1
```

Giữ nguyên manifest; không tạo lại split hoặc thay class order. Một cặp hamburger
trùng nội dung đã phát hiện nằm trong train, không có hash trùng giữa các split.
data/processed trống là có chủ đích: RGB/HOG/CNN transforms chạy trong bộ nhớ.

## 2. HOG đang chạy: không khởi chạy lại

Run hiện tại: `poc-v1-cpu-budget-v2-primal`; PID worker đã quan sát là 10308, wrapper 2992.
PID có thể đổi/kết thúc, nên xem status mới. Không có orchestrator tự chạy tiếp CNN.

```powershell
Get-Content .\artifacts\reports\poc-v1-cpu-budget-v2-primal\train_hog_svm.log -Tail 20
& .\.venv\Scripts\python.exe scripts/status_project.py
```

LinearSVC chạy trên CPU, không được tăng tốc bởi việc cài CUDA. Hiện chưa có metric
validation/test hay checkpoint sealed của run này. `fit` lâu không đồng nghĩa đã xong.
Không cài đè Torch/CUDA vào .venv đang chạy HOG. Không ghi đè run dở.

Nếu sau này bạn chủ động chạy HOG từ đầu và không còn HOG đang hoạt động:

```powershell
& .\.venv\Scripts\python.exe -u scripts/train_hog_svm.py --run-id poc10-hog-new-v1
```

Code mới ghi rõ C đang fit, thời gian/iterations/convergence vào svm_trials.json và
giữ best_candidate.joblib sau mỗi C hoàn tất. Đó chưa phải release hoàn chỉnh;
chỉ `trained=true` cùng checksum.sha256 mới đủ điều kiện đánh giá.
Các cải tiến này không thay đổi code đã nạp trong worker cũ đang chạy.

Nếu bạn quyết định thay lượt fit 128px quá chậm bằng baseline nhẹ hơn, đã có profile
`configs/poc/hog_svm_64.yaml`: 1.764 đặc trưng thay vì 8.100, max_iter 500/tol .01.
Chưa chạy hoặc đo accuracy của profile này. Chỉ chạy sau khi bạn tự xử lý lượt HOG cũ,
không còn HOG hoạt động; khóa sẽ chặn nếu vẫn còn tiến trình. Ví dụ:

```powershell
& .\scripts\poc_ai.ps1 -Step TrainHog -HogRunId poc10-hog64-v1
```

Không xóa/ghi đè run 128px. Nếu dùng baseline 64px, ở mọi bước Freeze/Evaluate/Select
phải thêm `-HogRunId poc10-hog64-v1`. Tập ảnh và nhãn vẫn giữ nguyên; khác biệt HOG
preprocessing được lưu trong artifact và phải nêu trong báo cáo. Không tune theo test.

## 3. Tạo môi trường CUDA riêng

Đã xác minh phần cứng: RTX 3050 Ti Laptop 4096 MiB, driver 566.36.
Venv hiện tại là torch 2.9.1+cpu / torchvision 0.24.1+cpu, CUDA chưa dùng được.

Các lệnh sau tải package lớn, bạn chủ động chạy. Chỉ tạo venv nếu chưa có:

```powershell
if (-not (Test-Path .\.venv-cuda\Scripts\python.exe)) { py -3.11 -m venv .venv-cuda }
& .\.venv-cuda\Scripts\python.exe -m pip install --timeout 300 --retries 10 "torch==2.9.1+cu126" "torchvision==0.24.1+cu126" --index-url https://download.pytorch.org/whl/cu126
& .\.venv-cuda\Scripts\python.exe -m pip install -c requirements-lock.txt -e ".[ml,dev]"
& .\.venv-cuda\Scripts\python.exe scripts/check_training_device.py --require-cuda --probe
```

Cặp phiên bản và index CUDA 12.6 lấy từ [hướng dẫn PyTorch chính thức](https://pytorch.org/get-started/previous-versions/#v2-9-1).
Phải khóa cả hậu tố `+cu126`: số phiên bản không có hậu tố có thể để pip giữ lại bản CPU.
Nếu đã cài đủ `[ml,dev]` nhưng kiểm tra vẫn trả `+cpu`, chỉ thay hai wheel trong .venv-cuda:

```powershell
& .\.venv-cuda\Scripts\python.exe -m pip install --timeout 300 --retries 10 --force-reinstall --no-deps "torch==2.9.1+cu126" "torchvision==0.24.1+cu126" --index-url https://download.pytorch.org/whl/cu126
& .\scripts\poc_ai.ps1 -Step CUDA
```

`--no-deps` ở lệnh sửa áp dụng khi dependencies còn lại đã được cài thành công.
Không chạy lại lệnh tạo venv hoặc đổi .venv CPU đang chạy HOG.
AMP dùng [torch.amp autocast/GradScaler](https://docs.pytorch.org/docs/2.9/amp.html).
Chưa cài hoặc kiểm chứng CUDA trên máy này trong lượt làm việc; chỉ đi tiếp khi
`cuda_available=true` và `probe_passed=true`. Không tự fallback CPU trong profile CUDA.

Sau khi cài thành công, có thể ghi lock riêng cho môi trường CUDA:

```powershell
& .\.venv-cuda\Scripts\python.exe scripts/lock_environment.py --extras ml dev --output requirements-cuda-lock.txt
```

## 4. Huấn luyện từng CNN

HOG được giữ nguyên. Khi chuyển đến CNN, chạy từng lệnh, đợi hoàn tất rồi mới chạy lệnh sau:

```powershell
& .\scripts\poc_ai.ps1 -Step TrainResNet
& .\scripts\poc_ai.ps1 -Step TrainEfficientNet
```

Tương đương (chỉ chọn một cách gọi, không chạy cả hai cách):

```powershell
& .\.venv-cuda\Scripts\python.exe -u scripts/train_resnet18.py --config configs/poc/resnet18_rtx3050.yaml --run-id poc10-rtx3050-v1
& .\.venv-cuda\Scripts\python.exe -u scripts/train_efficientnet_b0.py --config configs/poc/efficientnet_b0_rtx3050.yaml --run-id poc10-rtx3050-v1
```

Profile: batch 4, accumulation 4 (effective batch 16 khi các batch đủ), AMP, head 2 epochs,
fine-tune tối đa 8 epochs, early stopping patience 3, chọn best checkpoint theo val macro-F1.
Pretrained ImageNet V1 được fine-tune thực sự; không coi pretrained là food-trained.
Transforms: RGB/EXIF → random resized crop + flip chỉ ở train; val/test/inference
resize 256 + center crop 224 + ImageNet normalize. ResNet dùng bilinear, EfficientNet dùng bicubic.

4 GB là cấu hình khởi đầu, chưa phải cam kết vừa VRAM trong mọi trạng thái máy.
Nếu OOM, giảm batch_size xuống 2, accumulation lên 8 trong bản config mới, dùng run ID mới;
không âm thầm đổi model hoặc sử dụng test để chỉnh cấu hình. Checkpoint cũ được giữ.

Resume CNN khi đã có resume.pth (không dùng cho HOG):

```powershell
& .\.venv-cuda\Scripts\python.exe -u scripts/train_resnet18.py --config configs/poc/resnet18_rtx3050.yaml --run-id poc10-resnet-resume-v1 --resume artifacts/models/resnet18/poc10-rtx3050-v1/resume.pth
```

Head epochs, nhãn và preprocessing phải khớp. Muốn kéo dài training chỉ tăng fine-tune epochs
trước khi freeze/test. Resume lưu model, optimizer, scaler, best weights, history và RNG.
Nếu dùng run ID resume, truyền `--artifacts` trực tiếp ở bước freeze/select với đúng ba đường dẫn;
wrapper mặc định giả định hai CNN cùng run ID poc10-rtx3050-v1.

## 5. Chốt cả ba ứng viên rồi mới đọc test

Chỉ làm khi cả HOG và hai CNN có trained=true, không smoke và checksum đầy đủ:

```powershell
& .\scripts\poc_ai.ps1 -Step Freeze
& .\scripts\poc_ai.ps1 -Step Evaluate
& .\scripts\poc_ai.ps1 -Step Select
```

Mặc định wrapper dùng HOG `poc-v1-cpu-budget-v2-primal`, CNN `poc10-rtx3050-v1`.
Có thể đổi bằng -HogRunId/-CnnRunId nếu dùng run khác. Freeze giữ nguyên checksum
checkpoint, labels, preprocessing và val metrics trước test; không ghi đè protocol đã chốt.
Evaluate dùng **CPU cho cả ba** để đo latency theo cùng môi trường triển khai,
batch 1, warmup 3 ảnh validation, cùng 2.500 ảnh test và cùng mapping.
Không chạy lại completed test hoặc tune theo test; checkpoint thay đổi sau freeze bị từ chối.

Mỗi run xuất:

- training_history.csv, model_config.json, train_config.yaml, preprocessing.json;
- class_to_idx.json, idx_to_class.json, labels.json, source_state.json và checksum;
- evaluation_test/metrics.json, classification_report.csv, confusion_matrix.csv/png;
- evaluation_metadata.json, predictions.json, evaluation_summary.md, evaluation_complete.json;
- metrics_test.json ở gốc artifact sau khi evaluation hoàn tất.

Select xuất artifacts/reports/experiment_comparison.csv/.md và production/current.json.
Chọn theo val macro-F1, dùng CPU latency phá hòa; không mặc định EfficientNet-B0 thắng.
Test metrics phục vụ báo cáo cuối. Phân tích lỗi vẫn cần xem ảnh, không tự gán nguyên nhân.

## 6. Chuyển FastAPI/Streamlit sang model thật sau khi có kết quả

Kiểm tra release trước; không thay đổi .env:

```powershell
& .\.venv-cuda\Scripts\python.exe scripts/activate_real_model.py
```

Sau khi bạn xem bảng metric và chọn chấp nhận kết quả:

```powershell
& .\.venv-cuda\Scripts\python.exe scripts/activate_real_model.py --apply
```

Script yêu cầu release POC thật đã đánh giá đủ 2.500 test, có frozen protocol và completion
marker. Nó sao lưu .env (được Git ignore), đặt ALLOW_MOCK_AI=false, DEFAULT_MODEL=production,
ALLOW_SMOKE_MODELS=false, INFERENCE_DEVICE=cpu. Không reset DB hoặc tự restart service.
CPU inference giúp tách việc demo khỏi VRAM dành cho train. Registry/UI đưa production lên đầu;
API trả is_mock=false, đúng nhãn/version và ghi provenance khi lưu nhật ký.

Để dùng code mới và model thật trên host:

```powershell
docker compose stop backend frontend
& .\.venv-cuda\Scripts\python.exe -m uvicorn food_assistant.api.main:app --host 127.0.0.1 --port 8000
# Terminal thứ hai, cùng repository:
& .\.venv-cuda\Scripts\python.exe -m streamlit run src/food_assistant/app.py
```

PostgreSQL container giữ nguyên. Không chạy chồng container và host trên cùng port.
Image Docker mặc định là runtime mock nhẹ; phương án image ML nằm trong DEPLOYMENT_GUIDE.md.

## 7. Kiểm tra end-to-end bằng ảnh validation thật

Terminal thứ ba, chọn một ảnh pizza validation (không lấy ảnh test để thử/tune thủ công):

```powershell
$pocMetadata = Get-Content .\data\splits\v1\dataset_metadata.json -Encoding UTF8 -Raw | ConvertFrom-Json
$pocPhoto = Import-Csv .\data\splits\v1\dataset_manifest.csv | Where-Object { $_.split -eq 'val' -and $_.class_key -eq 'pizza' } | Select-Object -First 1
$pocImagePath = Join-Path $pocMetadata.root $pocPhoto.relative_path
& .\.venv-cuda\Scripts\python.exe scripts/smoke_test.py --model production --image $pocImagePath
```

Smoke kiểm tra upload → real predictor → người dùng xác nhận pizza → USDA reference →
lưu DB → diary/dashboard → sửa/xóa bữa do chính test tạo. Nó không khẳng định prediction
phải đúng trên một ảnh; accuracy chỉ lấy từ evaluation chính thức.

Mở http://127.0.0.1:8501, kiểm tra không còn lựa chọn mock, model production được chọn,
kết quả có version thực, xác nhận/sửa món và lưu nhật ký. Ghi nhận cả các dự đoán sai.

Full suite do bạn chạy khi muốn (không chạy đồng thời với training nặng):

```powershell
& .\.venv-cuda\Scripts\python.exe -m pytest -q
```

Hiện chưa có kết quả thực nghiệm chính thức để activate hoặc kiểm thử real end-to-end.
Không đánh dấu hoàn thành các bước 4–7 trước khi chính các lệnh này chạy thành công.
