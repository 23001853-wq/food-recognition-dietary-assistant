# PHASE 3 — Extract: downloader có Retry, Resume, Integrity & Lock

## Tiền điều kiện

Phase 2 đã nghiệm thu staging/atomic write/lock. Bất kỳ download thật đều bị cấm trong phase test này; dùng mock responses, fake local HTTP server nếu cần.

## Điểm yếu cần giải quyết

- `scripts/download_food101.py::download` có `.part` và Range nhưng chưa có retry/backoff, kiểm soát 200-vs-206 chặt, validation nguồn/offset/chunk.
- `scripts/download_food101_parallel.py` có retry giới hạn cho vài exception nhưng có thể tin chunk chỉ dựa kích thước và ghép file ra đường dẫn cuối trực tiếp.
- `configs/dataset_approval.json` hiện đặt `official_sha256: null`; không được nói đã kiểm SHA256 chính thức nếu không có expected digest đáng tin.

## Yêu cầu implementation

1. Tách HTTP/download logic thành hàm testable `extract.py` hoặc module phù hợp; giữ `scripts/download_food101.py` và `scripts/download_food101_parallel.py` như CLI wrappers tương thích. Tránh tạo hai logic retry khác nhau không nhất quán.
2. **Retry:** classify transient errors (timeouts, resets, một số HTTP 429/5xx), permanent errors (404, authorization, unsupported Range), exponential backoff có jitter và max attempts; hỗ trợ `Retry-After` hợp lý nếu có; không retry vô hạn.
3. **Sequential resume:** lưu `.part` có metadata nguồn, URL/ETag/Last-Modified/expected total; Range từ offset; bắt buộc validate status `206` và exact `Content-Range` start/end/total; nếu server trả `200`/Range sai/`416`, chỉ reset vào scratch mới sau khi chứng minh an toàn, không nối dữ liệu sai và không phá file `.part` cũ không kiểm soát.
4. **Parallel resume:** xác minh chunk boundaries và nguồn, ETag/length, checksum khi có; chunk đủ size nhưng có metadata không phù hợp hoặc không có bằng chứng integrity phải revalidate/re-fetch; reject chunk vượt expected size; test chunk corrupt same-size.
5. **Combine:** ghép ra temporary file, xác minh total length và toàn bộ archive digest; khi không có official sha256, chỉ ghi `local_sha256`, `official_checksum_verified=false`; tar readability và expected content chỉ là các loại integrity evidence khác, không thay thế official checksum.
6. **Concurrency:** lock chung cho cùng archive/source + target giữa sequential/parallel, test nhiều process. Nếu lock bận phải fail fast hoặc chờ trong giới hạn config.
7. **Idempotence:** nếu raw dataset được xác thực theo hợp đồng từ Phase 1/2, skip Extract an toàn. Nếu chỉ có marker cũ hoặc count 101000 mà chưa có validation phù hợp, không tự gắn `PASS`; có thể yêu cầu verify/INCOMPLETE.
8. Logging theo run id; không log credential hoặc token; metadata ghi atomically. Tôn trọng academic approval và không tự request network.

## Test matrix bắt buộc (mock)

| Tình huống | Kết quả cần đạt |
|---|---|
| 500, timeout rồi 206 hợp lệ | Retry giới hạn, resume chính xác |
| 404/authorization | Không retry vô hạn, báo lỗi dễ hiểu |
| Server ignore Range, trả 200 | Không append `.part` sai |
| `Content-Range` start/end/total không đúng | Reject response |
| ETag đổi trong lúc resume | Invalidate/restart an toàn |
| Chunk same-size nhưng corrupt | Không reuse không kiểm chứng |
| Lỗi ở giai đoạn ghép | Không publish archive sai |
| 2 downloader cùng target | Lock bảo đảm một writer |
| Đã có raw hợp lệ | Skip, không động vào raw/download |

## Cổng nghiệm thu

- [ ] Test retry/resume/network edge cases PASS.
- [ ] Không thể coi chunk file có size chuẩn là valid mà không có evidence.
- [ ] Sequential/parallel chia sẻ safety contract và lock.
- [ ] Không download thật; raw và v1 không đổi.

**Dừng sau Phase 3.**
