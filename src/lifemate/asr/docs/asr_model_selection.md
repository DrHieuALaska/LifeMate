# Quyết định model ASR cho LifeMate

## Quyết định

**Chọn `vinai/PhoWhisper-medium` làm ASR mặc định hiện tại để tích hợp. Mức chắc chắn: vừa.** Model dẫn đầu WER trong 5 ứng viên chính, có CER thấp nhất cả 8 model thực tế trong ZIP và thành công 991/991 audio. Đây là quyết định theo cấu hình đã thử, chưa chứng minh tốt nhất ngoài LifeMate. PhoASR-small là đối thủ gần nhất, cần giữ để đối chiếu sau khi xác minh điều kiện sử dụng và lỗi quan trọng.

## Kết quả đã đối chiếu

File thực tế có **8 model**, nên bảng giữ đủ 8; dấu * chỉ 5 ứng viên chính. Validation gồm **991 audio/309 hội thoại: 309 final, 682 history**, tổng 14.376 token reference theo khoảng trắng. Benchmark 100 audio là tập con, không cộng thêm hoặc coi là test độc lập.

| Model | WER % | CER % | OK/991 | p50/p95 ms | RTF | VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|
| PhoWhisper-medium* | 29.00 | 19.75 | 991 | 944/1715 | 0.210 | 1.76 |
| PhoASR-small | 28.41 | 19.95 | 991 | 455/846 | 0.102 | 0.62 |
| Qwen3-ASR-1.7B* | 29.79 | 20.94 | 991 | 909/1807 | 0.207 | 4.25 |
| Whisper-v3-turbo* | 33.47 | 23.66 | 990 | 435/601 | 0.092 | 1.88 |
| PhoWhisper-small | 34.84 | 23.76 | 991 | 479/915 | 0.108 | 0.62 |
| Qwen3-ASR-0.6B* | 36.97 | 25.75 | 991 | 881/1754 | 0.200 | 1.91 |
| wav2vec2-250h* | 37.79 | 22.67 | 991 | 39/49 | 0.008 | 0.72 |
| Whisper-small | 49.62 | 33.89 | 987 | 452/829 | 0.101 | 0.64 |

WER/CER tính lại từ raw predictions: corpus edit distance/tổng độ dài reference. WER theo khoảng trắng không phải tách từ vựng tiếng Việt; CER bỏ khoảng trắng. NFC, lowercase, dấu câu/ký hiệu thành khoảng trắng, giữ dấu và chữ số. Không đồng nhất số chữ với chữ số. Lỗi/thiếu được chấm như câu rỗng; turbo lỗi 1, Whisper-small lỗi 4, không có mẫu thiếu. Bảng chỉ trên ca thành công và final/history ở metrics CSV (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git).

Tất cả chạy một **T4, FP16, batch 1, eager, Transformers 5.14.1/torch 2.11.0+cu128**. Latency gồm đọc audio đến decode, không gồm load/warm-up. VRAM là PyTorch peak reserved, gồm weights, không phải tổng GPU memory; RAM process và load/warm-up nằm trong chi tiết (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git).

## Vì sao chọn như vậy?

- **Chất lượng và ổn định:** PhoWhisper-medium có WER final/history **29,59%/28,48%**, không lỗi pipeline. So Qwen 1.7B, WER thấp hơn **0,79 điểm phần trăm**, nhưng CI 95% **[-2.13; 0.53]** chưa tách biệt chắc chắn. CER thấp hơn, VRAM **1,76 thay vì 4,25 GiB**; Qwen 1.7B có **21/991** output chứa chữ ngoài Latin, bản 0.6B có **33/991**. Đây là kiểm tra ký tự, không phải nhãn ngôn ngữ/giọng.
- **PhoASR-small cạnh tranh thực sự:** WER thấp hơn medium **0,59 điểm**, p95 nhanh hơn khoảng **2,03 lần** và dùng ít VRAM hơn. Nhưng CI của **PhoASR − medium** là **[-1.56; 0.38] điểm**, chưa đủ kết luận thắng hoặc tương đương. PhoASR ít lỗi token hơn trên **380** câu, medium trên **327**, hòa **284**. [Model card](https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small) ghi BSD-3-Clause-Clear kèm Qualcomm Responsible AI License và intended use nghiên cứu/giáo dục; cần chốt quyền dùng, không tự suy cấm thương mại. PhoWhisper ghi [BSD-3-Clause](https://huggingface.co/vinai/PhoWhisper-medium). Chọn medium hiện tại không có nghĩa medium thắng PhoASR mọi tiêu chí.
- **Tốc độ không đủ để chọn:** wav2vec2 rất nhanh nhưng WER cao hơn medium **8,79 điểm**; turbo nhanh hơn nhưng cao hơn **4,47 điểm**. Chưa có yêu cầu latency sản phẩm để đánh đổi thêm lỗi lấy tốc độ. Không tạo trọng số tổng hợp tùy ý.

## Giới hạn quyết định

Paired bootstrap **10.000 lần, seed 2026, gom theo 309 hội thoại**; 28 cặp (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) là phân tích khám phá, CI chưa điều chỉnh so sánh nhiều cặp. Không có speaker ID để chứng minh tách người nói. Validation đã dùng chọn model nên là **development validation**, không phải test cuối độc lập.

Annotation gốc đều trống; chưa nghe audio, chưa đo lỗi theo giọng hay downstream. ID/path thuộc train, checksum cache khớp bản audit cũ và các model dùng cùng reference; ZIP không có ba CSV cache gốc để kiểm tra lại split/development hoặc chất lượng gold. Timing một lượt trên Kaggle không suy ra latency CPU/máy triển khai. Config/revision đầy đủ (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) lưu bằng chứng.

## Bước tiếp theo

- Pin PhoWhisper-medium revision `55a7e3eb6c906de891f8f06a107754427dd3be79`, `vi/transcribe`, greedy beam 1, tối đa 400 token; giao `text_model` khi status ok cho agent, normalization chỉ dùng scoring.
- Nghe/chấm lại 44 ca review, ưu tiên 12 ca tìm lỗi, rồi thử downstream trên các ca critical. Nếu PhoASR không kém về lỗi quan trọng và điều kiện dùng phù hợp, cân nhắc chuyển mặc định để giảm latency; không cần chạy lại toàn bộ 991 audio.
- **Chưa fine-tune hoặc thêm fallback model ngay.** Chỉ fine-tune trên development sau khi xác nhận lỗi lặp và gold. Không dùng reference/WER từng câu hoặc confidence tự bịa để kích hoạt fallback; khi lỗi kỹ thuật, trả lỗi để agent yêu cầu nói lại.
