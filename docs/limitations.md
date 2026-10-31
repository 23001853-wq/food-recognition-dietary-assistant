# Limitations

Thiết kế dự kiến từ PLAN.md; bộ khung chưa triển khai nghiệp vụ.

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
