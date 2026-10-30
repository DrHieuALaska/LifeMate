# LifeMate ASR — chọn model phù hợp

**Mặc định đề xuất hiện tại: `vinai/PhoWhisper-medium`.** Kết quả validation 991 audio: WER 29,00%, CER 19,75%, không lỗi/thiếu; mức chắc chắn vừa. PhoASR-small là đối thủ gần nhất, nhanh hơn và WER thấp nhẹ nhưng chưa tách biệt chắc chắn, cần xác minh lỗi quan trọng/điều kiện dùng.

Đọc **[asr_model_selection.md](asr_model_selection.md)** để gửi nhóm. Bằng chứng CSV/JSON và cách tính lại (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) giữ đầy đủ 8 model thực có trong hai ZIP, không sửa predictions gốc.

ASR xử lý audio final/history → transcript; agent hiểu ngữ cảnh và hành động. Notebook vẫn mặc định so 5 ứng viên chính (PhoWhisper-medium, turbo, Qwen 0.6B/1.7B, wav2vec2), có 3 model bổ sung tùy chọn. Đây là cấu hình để tái lập/so sánh, không tự chọn lại mặc định tích hợp.

## Nếu cần chạy lại trên Kaggle

- [ ] Import `lifemate_asr_benchmark.ipynb` trong `src/lifemate/asr/notebook/` vào Kaggle, đặt **private**.
- [ ] Add Input dataset gốc và output audit follow-up. Cache cần đúng ba file: `references_enriched.csv`, `split_manifest.csv`, `benchmark_manifest.csv`.
- [ ] Bật GPU và Internet nếu điều lệ cho phép. Ba mount của bạn đã điền sẵn trong cell cấu hình; chỉ sửa nếu tên input thay đổi.
- [ ] Mở **kernel mới**, giữ `STAGE='smoke'` và 5 tên mặc định trong `SELECTED_MODELS`, rồi **Run All**. Mỗi model chạy cùng 8 audio; cần `n_ok=8`, không lỗi/missing trước khi đi tiếp.
- [ ] Đổi `STAGE='benchmark'`, chạy lại từ bước 2 trở xuống, giữ `OUTPUT_DIR`: so 100 audio. Sau đó chạy `validation` cho các ứng viên cạnh tranh trên 991 audio nếu cache hợp lệ, review lỗi rồi mới chọn.
- [ ] Xem `comparison_status.csv`: benchmark cần `n_expected=100`, `n_missing=0`; số lỗi phải được báo và chấm như transcript rỗng. `ready` đòi không lỗi chạy, không có nghĩa chất lượng tốt. Không xóa ca lỗi để làm đẹp WER.
- [ ] Sau mỗi stage, tải **`/kaggle/working/lifemate_asr_benchmark_outputs.zip`**, đổi tên để phân biệt benchmark/validation và lưu private. Các bảng tổng hợp được ghi lại theo stage mới; không chỉ giữ ZIP cuối.

| Cấu hình | Cách đặt |
|---|---|
| `FOLLOWUP_INPUT` | Đã đặt `/kaggle/input/notebooks/jamel27/zalo-lifemate-audit-follow-up`; tự lấy CSV trong `lifemate_asr_followup/`, thư mục mount hoặc ZIP follow-up |
| `CACHE_DIR` | Mặc định `None` để tự lấy từ follow-up; chỉ đặt thủ công khi cần chỉ rõ thư mục chứa trực tiếp ba CSV |
| `AUDIT_INPUT` | Đã đặt `/kaggle/input/notebooks/jamel27/zalo-lifemate-audit`; chỉ ghi nhận, không dùng thay cache follow-up đã sửa |
| `DATASET_ROOT` | Thư mục **cha** của `train+public_test`; mặc định `/kaggle/input/datasets/jamel27/zalo-aichallange-lifemate-dataset` |
| `OUTPUT_DIR` | Mặc định `/kaggle/working/lifemate_asr_benchmark` |
| `DEVICE`, `PRECISION` | GPU: `'cuda:0'`, `'float16'`; CPU: `'cpu'`, `'float32'` |
| `STAGE` | `'smoke'`: kiểm tra 8 audio; `'benchmark'`: so sơ bộ 100 audio; `'validation'`: đánh giá 991 audio nếu cache hợp lệ |
| `INSTALL`, `OFFLINE` | Mặc định `True`, `False`: cài stack Transformers 5.14.1 và tải weights; giữ torch/CUDA Kaggle |
| `MAX_LATENCY_P95_MS`, `MAX_VRAM_GB` | Mặc định `None`; chỉ điền khi đã biết giới hạn triển khai, không tự bịa ngưỡng |

**Ba CSV cache gốc chưa được cung cấp local:** phân tích hiện tại dùng reference/ID trong ZIP kết quả. Chỉ cần lấy lại đúng output follow-up cũ khi chạy lại inference hoặc kiểm tra lại split gốc. Không chia lại split hoặc chạy lại toàn bộ EDA để thay cache. Các đường dẫn Kaggle phải kiểm tra theo mount thực tế.

Notebook dùng audio đã mount; nếu chỉ có `train+public_test.zip` hoặc `data.zip`, tự lấy đúng audio đã chọn vào `WORK_DIR` ngoài thư mục output. ZIP tên khác thì đặt `DATASET_ZIP` chính xác. Cache ZIP chỉ lấy ba CSV, audio ZIP chỉ lấy entry cần thiết, chặn environment/metadata và đường dẫn vượt thư mục đích; không giải nén toàn dataset.

Notebook có 16 bước: input bước 4; **PhoWhisper-medium bước 7, turbo bước 8, Qwen 0.6B/1.7B bước 9–10, wav2vec2 bước 11**; tùy chọn bước 12–14; so sánh bước 15; ZIP bước 16. Các model chạy tuần tự, cùng ID, lưu kết quả sau từng model.

## Nếu lỗi hoặc muốn chạy tiếp

- **Thư viện:** cell đầu cài stack đã dùng trong output được phân tích, giữ torch/CUDA Kaggle. Nếu đã import trước khi đổi package, restart kernel. Số đo chỉ áp dụng runtime ghi trong kết quả, không bảo đảm mọi image Kaggle khác.
- **Bản cũ:** lỗi EOS và trùng cột `duration` đã sửa. Bước 3 phải in `lifemate-model-selection-2.0`. Đổi runtime/code tạo run mới, không trộn prediction cũ vào phép so sánh mới. [Bản vá scoring](../tools/fix_annotation_scoring.py) chỉ dành cho session notebook cũ, không cần dùng trong bản 2.0.
- **Checkpoint:** `revision='main'` được khóa thành SHA trong `<name>_checkpoint.json` khi tải đầu tiên; giữ file này để tái lập. Offline: đặt `OFFLINE=True`, `INSTALL=False`, mount đủ packages và checkpoint (config, processor/tokenizer, weights), sửa `local_path` trong `CANDIDATES`. Checkpoint local được hash nội dung.
- **Resume:** Save Version/output private trước khi hết session. Session mới mount output cũ, đặt `RESUME_DIR`, để `OUTPUT_DIR` trống; copy xong notebook tự đặt lại `RESUME_DIR=None`. Giữ nguyên dataset, cache, code và cấu hình. Thay thư viện/phần cứng có thể tạo run mới; resume không tính lại hash WAV.
- **Chạy lại lỗi:** bật `RETRY_ERRORS=True`. **Chỉ chấm lại:** đặt `RUN_INFERENCE=False`, chọn đúng stage, chạy cell đánh giá. Nếu prediction đổi sau retry, review thủ công cần đối chiếu lại.
- **Cứu output:** cell ZIP cuối chạy riêng được nếu output còn trên đĩa. Trong kernel mới, sửa `zip_source` nếu bạn dùng đường dẫn khác mặc định.

## ZIP có gì?

| File | Dùng để làm gì? |
|---|---|
| `comparison_status.csv`, `metrics.csv` | Tiến độ, WER/CER, lỗi, tốc độ và tài nguyên |
| `model_selection.csv` | Các run đủ dữ liệu/giới hạn triển khai và nhóm không bị vượt cả WER lẫn latency; vẫn cần review để chọn |
| `predictions.jsonl`, `per_audio.csv` trong từng run | Dự đoán từng audio; CSV ghép reference để tìm lỗi |
| `paired_bootstrap.csv` | Khoảng tin cậy chênh lệch WER giữa các run đủ dữ liệu |
| `benchmark_request.json`, `input_provenance.json`, `selection_*.csv`, checkpoint lock, config/runtime/session, `dependency_check.txt` | Model, revision, nguồn cache/audio, ID dữ liệu, cấu hình và môi trường chạy |
| `benchmark_source.py`, `EXPORT_MANIFEST.json` | Mã nguồn đã dùng, danh sách file và checksum của ZIP |
| `model_errors.jsonl`, CSV annotation/review | Lỗi chạy và biểu mẫu chấm tay; chỉ có khi được tạo |

ZIP loại audio gốc, environment và weights. **Giữ private:** predictions, reference và annotation chứa dữ liệu cuộc thi. Không gửi audio ra API bên ngoài.

Đọc [local_validation](local_validation.md) để biết phạm vi kiểm tra; [research_report](research_report.md) giữ nguồn nghiên cứu và giao thức; [asr_model_selection](asr_model_selection.md) là quyết định từ kết quả thực nghiệm.
