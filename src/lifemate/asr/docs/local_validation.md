# Kiểm tra local được gì?

## Kiểm tra bản tích hợp ngày 30/10/2026

Trên Windows/Python 3.14, bản tích hợp đã cài editable với extras
`eda,asr-analysis`: **60 tests repo chung và 33 tests ASR đạt, không skip**.
Hai CLI `scripts/run_dev_benchmark.py --help` và
`scripts/explore_lifemate.py --help` chạy được. Notebook benchmark được build
lại từ module và kiểm tra khớp source. Adapter được kiểm tra bằng backend giả:
giữ raw text, từ chối output lỗi/rỗng, load một lần, CPU/GPU defaults và import
không kéo ML dependencies vào core. Không tải checkpoint, chạy inference thật
hoặc đo lại WER/latency trong lần tích hợp này.

## Bằng chứng trước tích hợp

**Đã kiểm tra cả logic local và kết quả Kaggle được cung cấp.** Ngày 29/10/2026: **27 test tổng hợp đạt** (22 pipeline + 5 phân tích độc lập). Bộ kết quả thực tế có 8 model, cùng 991 audio validation; WER/CER tính lại khớp bảng gốc. Quyết định: [asr_model_selection](asr_model_selection.md), bằng chứng: analysis (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git).

| Đã kiểm tra | Giới hạn |
|---|---|
| Audit 2 ZIP thực, tính lại WER/CER, bootstrap 10.000 theo hội thoại | Không chạy inference mới, không nghe audio; CSV/số đo chỉ trong cấu hình đã ghi |
| Scoring, normalization, lỗi/missing, journal, giữ review, bootstrap | Dữ liệu giả; không load checkpoint |
| ZIP theo danh sách cho phép, checksum, cell ZIP độc lập | Không kèm audio/weights/environment trong fixture |
| Trạng thái model, kể cả checkpoint lỗi; bảng chọn theo giới hạn/Pareto | Không chứng minh checkpoint chạy được hoặc chất lượng tốt trên Kaggle |
| Cache thư mục/ZIP, audio chọn lọc từ ZIP, chạy năm cell model mặc định cùng ID | Fixture cache 6.969 dòng + 100 entry audio giả; model giả, không phải inference thật |
| Qwen native: waveform, language vi, bỏ prefix, parse transcription và EOS | Model/audio giả; không truyền reference/context vào processor |
| Tải/fingerprint checkpoint Qwen | Có cả `chat_template.jinja`; thay template local làm đổi fingerprint, tránh resume nhầm |
| Sửa kiểm tra EOS của Whisper | Fixture xác nhận chuỗi gốc có EOS được chấp nhận, thiếu EOS vẫn lỗi; đã gọi thử `ASR.transcribe` bằng model/audio giả |
| Ghép template annotation khi chấm điểm | Giữ duration/kind/sample_uid của scoring; chạy đủ năm model giả với template thật do code tạo. Bản vá session giữ nguyên prediction và ghi nguồn scoring vào ZIP |
| Danh mục ZIP gốc (28/10) | 6.969 entry WAV thực, 1.105.219.092 byte; có manifest train/public test/sample submission. Chưa mở audio/manifest |

**Chưa có ba CSV cache gốc local** để kiểm tra lại join/split so với nguồn. Reference trong hai output ZIP đã đối chiếu giữa các model. `analysis/audit_verification.json` là bản ghi cũ, không phải bằng chứng xác minh độc lập lần này; trường `raw_audio_available_here=false` mô tả môi trường cũ. Không đọc environment/state hoặc chạy lại toàn bộ notebook audit.

## Tự kiểm tra lại

PowerShell tại gốc repo LifeMate (đã cài `pip install -e ".[asr-analysis]"`), dùng Python có NumPy/Pandas/RapidFuzz:

```powershell
python -m unittest discover -s src/lifemate/asr/tests -v
python -m py_compile src/lifemate/asr/asr_benchmark.py
```

Sau khi sửa module, chạy `python -m lifemate.asr.tools.build_notebook` để đồng bộ notebook standalone, rồi chạy lại tests. Mỗi code cell có tên chức năng và metadata để kiểm tra không phụ thuộc số thứ tự cell.

Có thêm kiểm tra API thực bằng **model cực nhỏ khởi tạo ngẫu nhiên**, không tải checkpoint/dữ liệu: Whisper generate giữ EOS, Qwen generate/bỏ prefix, wav2vec2 forward greedy và processor Qwen đã pass trên CPU với Transformers 5.14.1, torch 2.13.0+cpu. Lệnh dưới cần cùng dependencies notebook; không dùng kiểm tra này để suy WER hoặc VRAM pretrained:

```powershell
python src/lifemate/asr/tests/check_native_models.py --check-processor
```

Khi có đúng cache follow-up, thay đường dẫn mẫu:

```powershell
python -c "from lifemate.asr.asr_benchmark import load_cache, select_rows; r,j,b,h=load_cache(r'D:\path\lifemate_asr_followup'); print(len(r),len(j),len(b)); print({s:len(select_rows(j,b,s)) for s in ['smoke','benchmark','validation']}); print(h)"
```

Mốc cần đối chiếu: 6.969 reference, 5.036 dòng split; stage 8/100/991 audio. So checksum với `analysis/audit_verification.json`. `load_cache` kiểm tra ID/join, train-only, benchmark thuộc validation, nhóm hội thoại không giao và khớp path/reference; **không kiểm tra lại waveform, mọi hash overlap hoặc tách biệt người nói**. Cache khác phải xét nguồn gốc, không tự đổi số đếm hay chia lại split.

## Nếu cần thí nghiệm tiếp

- [ ] Không cần chạy lại toàn bộ benchmark/validation: dữ liệu số học hiện đủ. Chỉ chạy lại nếu thay model/config hoặc phát hiện lỗi mới.
- [ ] Nghe lại 44 ca đã review văn bản, ưu tiên 12 ca tìm lỗi; gán nhãn critical trước khi fine-tune hoặc chốt downstream.
- [ ] Đo trên môi trường triển khai thật nếu khác T4; không suy số đo CPU từ GPU.
- [ ] Kiểm tra resume/output thật; xác minh luật, license và tài nguyên trước khi chốt model.

Runtime local ghi nhận ngày 28/10: Python 3.14.6, torch 2.13.0+cpu, Transformers 5.14.1; không có CUDA, thiếu soundfile. Log Kaggle người dùng gửi ngày 29/10 cho thấy T4, torch 2.11.0+cu128, Transformers 4.57.1, đủ 100 path, Whisper load/inference được nhưng bị code cũ báo lỗi EOS hàng loạt. Hai ZIP và notebook được cung cấp sau đó đã cho phép kiểm tra metric/metadata của cả 8 model; chưa nghe audio hoặc đo downstream. Bộ phân tích độc lập không chạy lại inference.

Nguyên nhân và cách sửa: [mã Whisper Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/generation_whisper.py) loại EOS khỏi tensor segment mặc định; `return_dict_in_generate=True` với short-form trả chuỗi gốc trong `sequences`. Pipeline 2.0 vẫn kiểm tra chuỗi này, hỗ trợ EOS dạng số/danh sách và pad riêng. Không tăng token limit hoặc bỏ kiểm tra để che lỗi.

Cách chạy và lấy ZIP: [README](README.md).
