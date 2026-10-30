# Chọn model ASR cho LifeMate

**Quyết định hiện tại: chọn PhoWhisper-medium làm ASR mặc định để tích hợp**, mức chắc chắn vừa. Bộ output thực tế có 8 model, không chỉ 5; PhoASR-small là đối thủ gần nhất. Xem [bản cô đọng gửi nhóm](asr_model_selection.md) và audit/metrics tính lại (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git). Tài liệu này giữ cơ sở chọn ứng viên và giao thức tái lập.

## Các model đã đưa vào notebook

Năm model chạy mặc định trong notebook; kết quả gửi về đã chạy cả 8 model trên cùng dữ liệu/runtime. Lý do chọn ứng viên dưới đây là giả thuyết ban đầu, không thay thế số đo LifeMate trong báo cáo quyết định.

| Checkpoint | Vai trò, bằng chứng và giới hạn |
|---|---|
| [vinai/PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium) | Whisper chuyên biệt Việt, huấn luyện thêm trên 844 giờ đa giọng. Kiểm tra lợi ích của medium; dữ liệu paper khác LifeMate. BSD-3-Clause |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Whisper multilingual với decoder rút gọn; thử cân bằng chất lượng/tốc độ. Hỗ trợ Việt không chứng minh giữ tốt từ miền hoặc English trong LifeMate. Giữ license theo checkpoint/gói thực tải |
| [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | ASR chuyên dụng, hỗ trợ Việt; kiểm tra phương án Qwen ít tham số hơn. Apache-2.0; số đo T4/VRAM đã có trong báo cáo quyết định |
| [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Đo lợi ích tăng quy mô so với 0.6B và chi phí đi kèm. Apache-2.0; đã chạy được trên T4 trong cấu hình đo, không suy model lớn hơn luôn thắng |
| [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) | CTC greedy không LM; card công bố 13K giờ Việt không nhãn + 250 giờ có nhãn, khuyên audio <10 s. CC-BY-NC-4.0 cần làm rõ quyền dùng |

Qwen native có từ Transformers 5.13; notebook dùng 5.14.1, không cài package `qwen-asr` dùng stack 4.x. Bản `-hf` là checkpoint chuyển sang runtime native, không được tính như một model học mới khác bản gốc. Qwen dùng encoder âm thanh + decoder Qwen; chưa xác minh đủ chi tiết dữ liệu huấn luyện/phân bố giọng Việt để suy hiệu quả LifeMate. Xem [model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) và [technical report](https://arxiv.org/abs/2601.21337); report này chưa được xác minh venue/ranking, không tính vào bốn paper CORE bên dưới.

| Model bổ sung đã có kết quả; notebook giữ tùy chọn | Lý do giữ/giới hạn |
|---|---|
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | Đối chiếu multilingual cỡ nhỏ với kết quả cũ |
| [vinai/PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) | So chi phí small/medium, BSD-3-Clause |
| [Qualcomm-AI-Research/PhoASR-whisper-small](https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small) | Model Việt mới, card công bố 3.000 giờ và cảnh báo ít giọng miền Trung. BSD-3-Clause-Clear **kèm Qualcomm Responsible AI License**, mục đích research/education; cần xác minh quyền dùng trước khi đưa vào quyết định triển khai |

Đã xem thêm [Parakeet Vietnamese](https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese): card có yêu cầu xác nhận license/quyền truy cập, runtime NeMo khác stack hiện tại và LM ngoài tùy chọn. Chưa đưa vào notebook này vì chưa xác nhận quyền tải/runtime; đây không phải kết luận Parakeet kém chất lượng.

Điều lệ pretrained/external data và eligibility vẫn chưa xác minh. Không dùng model chỉ vì mới/nhiều tham số; một model chung là bước đầu, chưa có bằng chứng để chia theo giọng.

## Dữ liệu và cấu hình so sánh

**Đã kiểm tra:** danh mục ZIP gốc có 6.969 WAV; hai ZIP kết quả xác nhận benchmark 100 và validation 991 audio cùng reference/role giữa 8 model, không có ID/path public test. Ba CSV cache gốc chưa có local để tái kiểm tra development-disjoint/hash audio. Quy mô train/development/public test dưới đây vẫn theo audit cũ; benchmark/validation đã được kiểm tra từ output mới.

| Tập | Audio / hội thoại | Cách dùng |
|---|---|---|
| Train | 5.036 / 1.545 | Nguồn chia development và validation |
| Development | 4.045 / 1.236 | Chỉ dùng tập này để cập nhật trọng số nếu fine-tune |
| Validation | 991 / 309; 309 final + 682 history | Chọn/đánh giá ứng viên; giữ split cũ |
| Benchmark | 100 / 81; 48 final + 52 history | Tập con validation cố định, không chọn theo prediction |
| Public test | 1.933 / 634 | Không dùng để chọn model hoặc tune theo leaderboard |

Cache cũ ghi tổng 9,5913 giờ, mono 16 kHz PCM_16, dài nhất 22,3335 s; signal audit chỉ xét 2.000 file, có 39 cờ nhiều quiet frames, chưa chứng minh noise/SNR. Benchmark có 18% câu >10 s, validation khoảng 4,2%: điểm 100 audio không đại diện hoàn toàn cho validation. Chưa có speaker ID đáng tin để chứng minh hai tập khác người nói.

Bản ghi cũ báo đã sửa nhầm split do tên thư mục `train+public_test` và loại sample submission khỏi dữ liệu. Khi có cache, đối chiếu SHA256 trong `analysis/audit_verification.json`, join bằng `ref_uid`, kiểm tra nhóm hội thoại/hash overlap; không ghép theo thứ tự dòng hoặc chia random lại. Hướng dẫn: [hướng dẫn ASR](../README.md).

Cấu hình nằm trong `src/lifemate/asr/notebook/lifemate_asr_benchmark.ipynb` và `src/lifemate/asr/asr_benchmark.py`:

- Seed 2026, một GPU, batch 1, FP16 trên CUDA/FP32 trên CPU. Stack mới: Transformers 5.14.1, HF Hub 1.24.0, tokenizers 0.22.2, safetensors 0.8.0, soundfile 0.13.1, psutil 7.1.0, librosa 0.11.0; giữ torch/CUDA của Kaggle. Hai ZIP mới có đủ 8 model trên T4/FP16; bộ tính lại đã đối chiếu metric và log lỗi thật. Không mở rộng kết luận sang phần cứng khác.
- Whisper: `vi/transcribe`, beam 1, `do_sample=False`, tối đa 400 token, không timestamps. Qwen: `language=vi`, beam 1, `do_sample=False`, tối đa 512 token, bỏ prefix input và dùng phần transcription sau `<asr_text>`. CTC: greedy, không LM ngoài. Không đưa reference/prompt/hotword/context vào inference; kiểm tra EOS thật để phát hiện output chưa kết thúc. Khác số token tối đa vì tokenizer/kiến trúc khác, đều ghi trong config.
- `revision='main'` khóa thành SHA khi tải; checkpoint local dùng hash nội dung. Giữ lock/config/runtime và mã nguồn xuất trong ZIP. Thay runtime, quantization hoặc decoding là cấu hình mới.
- Audio phải mono 16 kHz; sai định dạng báo lỗi, không tự cắt/resample. Whisper >30 s báo lỗi; CTC >10 s cảnh báo nhưng xử lý toàn câu. Giới hạn token/thiếu EOS phải báo lỗi.

## Đọc kết quả và quyết định

| Chỉ số | Cách hiểu và giới hạn |
|---|---|
| WER/CER | Tổng số phép sửa / tổng token hoặc ký tự reference; càng thấp càng tốt. WER tách khoảng trắng; CER bỏ khoảng trắng. Có thể >100% nếu thêm nhiều lời |
| Failure rate | `(n_error + n_missing) / n_selected`; load lỗi lưu riêng, audio chưa có record là missing |
| Latency/RTF | Latency gồm đọc audio đến decode, đồng bộ GPU; không gồm download/load/warm-up. RTF = tổng thời gian xử lý / tổng thời lượng audio, tính trên ca thành công |
| RAM/VRAM | RAM là RSS process lấy mẫu mỗi 10 ms, có thể bỏ sót peak ngắn; VRAM là PyTorch peak allocated/reserved, gồm weights. Không phải tổng bộ nhớ hệ thống |
| Bootstrap | Notebook: 2.000 lần; phân tích độc lập: 10.000 lần, seed 2026. Cùng lấy mẫu theo hội thoại, CI 95% của delta WER; chứa 0 không chứng minh thắng hoặc tương đương |

Normalization version `nfc_lower_punct_symbol_to_space_keep_accents_digits_v1`: NFC, lowercase, dấu câu/ký hiệu → khoảng trắng, gộp whitespace; giữ dấu tiếng Việt và chữ số. Không coi `250000` tương đương “hai trăm năm mươi nghìn”; dấu thập phân cũng bị thay. Giữ text/reference gốc để review. Number normalization mới cần kiểm thử riêng và báo song song; scorer này chưa phải scorer chính thức.

Lỗi/missing được chấm như transcript rỗng, kèm điểm chỉ trên ca thành công. `n_ok=0` thì không có WER/CER hợp lệ, không bootstrap. Báo cáo độc lập so mọi active run, giữ cả ca lỗi trong mẫu số; so riêng final/history, ≤10 s/>10 s và trên cùng phần cứng/runtime. Batch 1 không chứng minh tốc độ xử lý batch lớn.

**Đề xuất chọn:** chạy đủ, xử lý lỗi pipeline, nghe review thông tin quan trọng, rồi cân bằng WER với tốc độ/bộ nhớ. `model_selection.csv` loại run thiếu/lỗi và vượt giới hạn latency/VRAM do bạn đặt; đánh dấu Pareto WER–latency (không có model khác vừa ít lỗi hơn vừa nhanh hơn). Đây là bảng hỗ trợ quyết định, không tự phong model thắng; không đặt trọng số tùy tiện khi chưa có yêu cầu agent. WER thấp nhưng hay đảo “không” hoặc sai tên chưa chắc phù hợp agent. Chỉ kết luận trong model/config đã thử; không tuyên bố tối ưu toàn cục.

## Bằng chứng nghiên cứu

Các nguồn dưới đã đối chiếu ngày 28/10/2026. Dùng **CORE2023** làm mốc ranking, không coi là ranking hiện hành duy nhất. Có 4 venue đạt A*/A; Tiny Papers chưa xác minh ranking riêng. Không dùng SJR/JCR vì không chọn bài tạp chí. **Số đo trong paper không phải kết quả LifeMate.**

**Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges** — Nguyen Van Dinh, Thanh Chi Dang, Luan Thanh Nguyen, Kiet Van Nguyen (EMNLP 2024). [Paper](https://aclanthology.org/2024.emnlp-main.426/), DOI [10.18653/v1/2024.emnlp-main.426](https://doi.org/10.18653/v1/2024.emnlp-main.426). EMNLP **CORE2023 A\***: [nguồn CORE](https://portal.core.edu.au/conf-ranks/448/).

ViMD: 102,56 giờ, 63 nhãn tỉnh. Table 7 trước fine-tune trên ViMD: wav2vec2-250h WER toàn quốc/miền Trung 17,47%/20,97%; PhoWhisper-base 18,61%/24,15%. Hỗ trợ thử CTC và đo theo giọng; không suy thứ hạng sang LifeMate hoặc bản small.

**PhoWhisper: Automatic Speech Recognition for Vietnamese** — Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen (ICLR 2024 **Tiny Papers track**). [Paper](https://arxiv.org/html/2406.02555v1), DOI bản arXiv [10.48550/arXiv.2406.02555](https://doi.org/10.48550/arXiv.2406.02555). **Ranking riêng của Tiny Papers: chưa xác minh; không tính là paper đạt CORE A\*.**

PhoWhisper-small có WER 11,08/6,33/15,93/32,96 trên CMV–Vi/VIVOS/VLSP T1/T2. Hỗ trợ thử model chuyên biệt Việt; chưa chứng minh từ địa phương hoặc code-switching. Không giả định được dùng dữ liệu riêng của tác giả.

**Robust Speech Recognition via Large-Scale Weak Supervision** — Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (ICML 2023). [Paper](https://proceedings.mlr.press/v202/radford23a.html), DOI bản arXiv [10.48550/arXiv.2212.04356](https://doi.org/10.48550/arXiv.2212.04356). ICML **CORE2023 A\***: [CORE](https://portal.core.edu.au/conf-ranks/1121/).

Whisper học từ 680K giờ đa ngôn ngữ/đa nhiệm có giám sát yếu. Hỗ trợ baseline tổng quát; chưa chứng minh nhận tốt mọi giọng Việt, tên riêng hoặc English phát âm kiểu Việt.

**wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** — Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli (NeurIPS 2020). [Paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html), DOI bản arXiv [10.48550/arXiv.2006.11477](https://doi.org/10.48550/arXiv.2006.11477). NeurIPS **CORE2023 A\***: [CORE](https://portal.core.edu.au/conf-ranks/98/).

Học biểu diễn không nhãn rồi fine-tune giúp giảm nhu cầu nhãn trên LibriSpeech tiếng Anh. Chỉ là cơ sở kiến trúc, không dự đoán điểm checkpoint tiếng Việt trên LifeMate.

**Don’t Stop Self-Supervision: Accent Adaptation of Speech Representations via Residual Adapters** — Anshu Bhatia, Sanchit Sinha, Saket Dingliwal, Karthik Gopalakrishnan, Sravan Bodapati, Katrin Kirchhoff (Interspeech 2023). [Paper](https://www.isca-archive.org/interspeech_2023/bhatia23_interspeech.html), DOI [10.21437/Interspeech.2023-2192](https://doi.org/10.21437/Interspeech.2023-2192). Interspeech **CORE2023 A**: [CORE](https://portal.core.edu.au/conf-ranks/494/).

HuBERT-large với adapter theo bốn giọng tiếng Anh giảm WER tương đối trung bình 22,7%, không phải 22,7 điểm phần trăm. Gợi ý adaptation sau baseline; chưa có thí nghiệm phương ngữ Việt.

Bằng chứng trực tiếp nhất về đa giọng Việt là ViMD; PhoWhisper hỗ trợ lựa chọn checkpoint. Chưa có nguồn đủ sát Vietnamese–English code-switching trong bộ này; đa ngôn ngữ không bảo đảm xử lý tốt chuyển ngôn ngữ trong câu.

## Chấm tay và bước sau khi so sánh pretrained

**Chưa có nhãn/metric entity gold.** Đã review định tính 32 câu theo metadata và 12 ca tìm lỗi từ văn bản; bảng có ID (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) tách rõ hai nhóm, chưa nghe audio. Đề xuất nghe/chấm lại panel 44 ca hiện có; nếu lập panel mới, chọn từ metadata trước khi xem prediction. Ca chọn thêm do thấy lỗi phải báo riêng. Không suy vùng miền từ transcript/regex. Nếu có thể, hai người chấm độc lập, ít nhất 30 audio chấm đôi, thống nhất bất đồng.

| File | Cách điền |
|---|---|
| `dialect_annotations.csv` | Tách nhãn cách phát âm (`accent_variation`), từ địa phương, English phát âm kiểu Việt, chuyển ngôn ngữ và noise. Dùng present/absent hoặc noticeable/not_noticeable theo cột; `accent_region`: northern/central/southern/mixed. Mọi cột cho phép `uncertain`; vùng chỉ là nhận xét khi nghe |
| `critical_gold_annotation.csv` | Tên, tiền/số, thời gian, địa điểm, thiết bị/app, phủ định; đối chiếu audio, giữ từ địa phương. Span JSON `start,end,text`, chỉ số ký tự Python, `end` loại trừ |
| `critical_model_review_template.csv` | Chấm correct/wrong/missing/added/uncertain; nhiều entity trong câu cần review từng lần xuất hiện |

Ô trống = chưa chấm; `uncertain` = đã nghe nhưng chưa chắc; `[]` trong gold = đã kiểm tra và không có entity. `reference_problem` dùng yes/no/uncertain, ghi chú thay vì tự sửa reference. `not_noticeable` không có nghĩa người nói không có accent. Báo `(wrong+missing)/số entity gold đã chấm` theo loại, mẫu số/uncertain riêng; entity thêm sai tính riêng trên số câu review. Phủ định phải xét việc đổi nghĩa; chưa có scorer entity tự động.

Fine-tuning **chỉ là kế hoạch có điều kiện**: pretrained đã được đánh giá đủ dữ liệu, lỗi pipeline đã sửa, review thấy lỗi lặp lại và luật cho phép. Chọn model theo kết quả, train chỉ trên development; đề xuất batch nhỏ, gradient accumulation, early stopping và checkpoint resume đầy đủ. Chưa đo giờ train/VRAM. Validation đã dùng chọn model/hyperparameter phải gọi là development validation; nếu cần đánh giá cuối độc lập, lập held-out có version từ development trước thí nghiệm mới, không âm thầm đổi split.

Trước khi nộp bài/triển khai thật: xác minh điều lệ pretrained/external data, license, Internet, phần cứng/quota, scorer và schema nộp bài. ASR chỉ trả transcript: `engine.transcribe(audio_path)` → status, text_model, text_normalized, model/revision, latency, warnings/error. Agent dùng text_model khi status ok; không tự suy confidence, thời gian/người nhận hoặc tool call.
