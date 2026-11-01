# Chọn model ASR cho LifeMate

## Cập nhật sau fine-tune — 01/11/2026

Hai model đã hoàn tất các run fine-tune/continuation. Cùng valid 504 audio:
hynt baseline 98 câu exact / WER 21,430% → best theo EM epoch 9 là 242 câu / WER 8,269%;
Sherpa baseline 85 câu / WER 19,636% → best epoch 3 là 240 câu / WER 8,059%.
Đợt train tiếp tăng hynt thêm 3 câu exact so với epoch 5 nhưng làm WER/CER tăng nhẹ;
Sherpa epoch 6–8 không vượt epoch 3. Chưa có model thắng toàn diện.

Xem [CSV aggregate và phạm vi bằng chứng](results/README.md) và [web nghe review local](../README.md).
Số liệu là validation chọn checkpoint, không phải test độc lập. Split mới đã tái sử dụng 991 mẫu lịch sử
trong nguồn train/valid, nên không dùng benchmark 991 cũ làm đánh giá độc lập sau fine-tune.
ZIP handoff không chứa weights/predictions; lần kiểm tra handoff xác nhận checksum, counters và state,
chưa xác minh inference checkpoint hay tích hợp hệ thống.

Phần bên dưới giữ nguyên các kết quả pretrained/991 và nhận định trước fine-tune để tái lập lịch sử.

**Cập nhật mới nhất 31/10/2026: Zipformer hynt 153M dẫn đầu chất lượng đã đo**, đạt **WER 15,34%, CER 10,65%, 991/991 OK** trong **12 checkpoint/cấu hình riêng biệt có prediction** trên cùng tập 991 mẫu cũ. Ưu tiên nghe review 153M/68M; **giữ 68M là ứng viên tích hợp nhẹ và nhánh xác minh fine-tune**, vì card 153M ghi CC-BY-NC-ND-4.0, chưa chốt quyền adaptation/triển khai. Hataphu 291M chưa có inference thành công, chưa xếp hạng. Đây là kết quả pretrained, **chưa có cải thiện do fine-tune**, chưa đổi adapter. Xem [quyết định và kế hoạch](asr_model_selection.md).

## Kết quả Zipformer 153M/291M và kiểm tra cùng tập mẫu

ZIP `lifemate_asr_zipformer_large_full_test_outputs.zip`, SHA256 `676a4613f186906ed7521fe2030933af9b7524168d04cfa7b3c733c94266c981`. Đã kiểm tra checksum **32 file**, prediction ID không trùng và tính lại WER/CER từ raw predictions bằng scorer local, không thực thi code trong ZIP. Bằng chứng local/ignored ở `artifacts/asr_zipformer_large_20261031/`: script `check_outputs.py`, `same_set_audit.json`, `hynt_153m_metrics.json`, `hynt_153m_subsets.json`, `paired_153m_vs_68m.json`.

| Model/config | WER % | CER % | Exact normalized | OK/error/missing | p50/p95 ms | RTF |
|---|---:|---:|---:|---|---:|---:|
| **hynt Zipformer 153M** | **15,34** | **10,65** | **260/991 = 26,24%** | **991/0/0** | 257/473 | 0,0571 |
| Sherpa Zipformer vi 68M, lượt trước | 18,79 | 14,07 | 177/991 = 17,86% | 991/0/0 | 159/315 | 0,0363 |
| PhoWhisper-large, lượt trước | 26,79 | 18,07 | 122/991 = 12,31% | 991/0/0 | 1359/2382 | 0,297 |
| hataphu Zipformer 291M | — | — | Chưa đánh giá | **0/0/991** | — | — |

153M có **2.205/14.376 word edits**, **4.953/46.494 char edits**, giảm **496 word edits, 3,45 điểm phần trăm WER (18,36% tương đối)** so 68M; thêm **83 câu exact normalized**. Paired bootstrap tính lại **10.000 lần, seed 2026, cluster theo 309 hội thoại**: delta WER **153M − 68M = −3,45 điểm**, CI95 **[−4,07; −2,81]**. 153M ít word edits hơn trên **432 câu**, 68M trên **173**, hòa **386**. CI khám phá chưa điều chỉnh nhiều so sánh, không chứng minh speaker-disjoint hoặc ưu thế theo giọng/critical/downstream. Chênh lệch do đổi pretrained checkpoint, không phải fine-tune; exact normalized chỉ là khớp toàn câu sau chuẩn hóa, WER không phải phần bù accuracy.

**Đã đối chiếu 4 ZIP** (validation lịch sử, large/Qwen, under2B, Zipformer lớn): toàn bộ bảng per-audio có cùng **991 ref_uid, reference, sample_uid, kind và duration**, gồm **309 hội thoại, 309 final/682 history**. Fingerprint mới vẫn là `2c9bb8c6966bfd116b2aabc6d565cfbf9af3d45ae154660727c4fe79c20478c6`. Model 291M được chọn cùng tập nhưng chưa có prediction. `full_test` vẫn là chạy hết development validation đã dùng chọn model, không phải tập test độc lập; smoke 8/benchmark 100 không được cộng thêm. Không có checksum waveform trong export để chứng minh byte audio giống nhau giữa các phiên; đã xác nhận danh tính mẫu/metadata/reference.

153M: `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h`, revision `8c3d474d3d74cb247f4b8c7b0f8b48598ac8c1c6`; encoder/decoder/joiner **epoch-11-avg-2, FP32 ONNX**, Sherpa-ONNX **2.0.1**, **CPU 2 threads, greedy_search, không LM/hotword/punctuation model**. Vocabulary 2.000, nguồn token `config.json` là text symbol/id. Process peak RSS **1,12 GiB**, VRAM không áp dụng. Bound trong `parameters.json` là **149.906.361 scalar tensor ONNX**, không phải số trainable chính xác; cộng LLM giả định đúng 4B/other=0 thành **4.149.906.361**, chưa chứng nhận eligibility. Latency khác phiên/runtime là tham khảo, chưa có máy đích hoặc training-memory proof.

291M khóa revision `9158a1e17503df848dacc974d837216d0d4df6d0`, đã qua tải checkpoint nhưng dừng ở validator với `Expected contiguous vocabulary IDs with <blk> at 0`, trước inference: **0 OK, 0 lỗi audio, 991 missing, comparison_ready=False**. WER/CER không hợp lệ; không gọi là sai 100% hoặc kết luận model kém. ZIP không chứa token file gốc nên chưa biết chính xác điều kiện con gây lỗi. [Notebook sửa riêng 291M](../notebook/zalo_lifemate_zipformer_291m_fixed_full_test.ipynb) giữ đủ output ID theo metadata ONNX, nguyên mapping; bỏ ép tên blank/ID phụ, dùng `tokens.txt` ở root và thư mục output mới. **32 kiểm tra giả lập liên quan qua, chưa có inference thật với bản sửa**; chưa dùng bản sửa để thay đổi số đo 153M đã xuất.

**Tác động tới lựa chọn:** [card hynt](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) ghi **CC-BY-NC-ND-4.0**; [card hataphu](https://huggingface.co/hataphu/zipformer-k2-rnn-lm-vi) ghi **MIT** và yêu cầu quyền tải, đối chiếu ngày 31/10/2026. Không mặc định WER dẫn đầu đồng nghĩa đủ quyền fine-tune/phát hành/triển khai. Trước mắt review 153M/68M trên cùng audio; giữ 68M cho hướng tích hợp nhẹ/xác minh trainability, large + LoRA dự phòng. Chờ output 291M trước khi đánh giá vai trò của nó. Chưa có kết quả adaptation LifeMate hoặc bằng chứng cải thiện sau train.

## Kết quả Ichigo/Sherpa 68M ở lượt trước

ZIP `lifemate_asr_under2b_full_test_outputs.zip`, SHA256 `cdcddb91206f09df5690010133075e25e318c93f5a12f7e8b506aee3d6684d72`. Phân tích độc lập local đã kiểm tra checksum **36 file**, source/worker hash, trạng thái, toàn bộ prediction ID và tính lại WER/CER từ raw predictions; không thực thi mã đính kèm. **991 ID/reference/nhóm/kind/duration/path/partition và ba cache hash khớp lượt large/Qwen**; input fingerprint chung `2c9bb8c6966bfd116b2aabc6d565cfbf9af3d45ae154660727c4fe79c20478c6`. Preflight báo 991 input hợp lệ. Bằng chứng ở `artifacts/asr_under2b_20261031/` gồm script `analyze.py`, audit JSON, metrics, exact match, chẩn đoán ký tự và paired comparisons; giữ local/ignored.

| Model/config | WER % | CER % | Exact normalized | OK/error/missing | p50/p95 ms | Runtime |
|---|---:|---:|---:|---|---:|---|
| **Sherpa Zipformer vi 68M** | **18,79** | **14,07** | **177/991 = 17,86%** | **991/0/0** | 159/315 | CPU, 2 threads |
| PhoWhisper-large, lượt trước | 26,79 | 18,07 | 122/991 = 12,31% | 991/0/0 | 1359/2382 | T4, FP16 |
| IchigoWhisper + Whisper-medium | 41,75 | 29,07 | 30/991 = 3,03% | **988/3/0** | 1078/1848 | T4, FP32 weights/FP16 decode |

Sherpa có **2.701/14.376 word edits**, **6.540/46.494 char edits**. WER final/history **19,23%/18,40%**, ≤10 s/>10 s **18,74%/19,30%**. So large: giảm **1.151 word edits**, **8,01 điểm phần trăm WER**, tương đương **29,88% tương đối**. Đây là lợi ích thay checkpoint pretrained, chưa phải adaptation. Exact normalized tăng từ 122 lên 177 câu; không đồng nhất “exact với reference” và “đúng ý nghĩa”. WER 18,79% không được đổi thành accuracy 81,21%.

Paired bootstrap **10.000 lần, seed 2026, cluster theo 309 hội thoại**: delta WER Sherpa − large **−8,01 điểm**, CI95 **[−9,10; −6,94]**; Sherpa − medium **−10,21**, **[−11,37; −9,10]**; Sherpa − PhoASR **−9,62**, **[−10,72; −8,53]**. Sherpa ít word edits hơn large trên **549 câu**, large tốt hơn **182**, hòa **260**. CI khám phá chưa điều chỉnh nhiều so sánh; không có speaker-disjoint proof, chưa đo theo giọng hoặc downstream. Không tạo router/oracle ensemble từ reference từng câu.

Ichigo có **6.002 word edits/13.515 char edits**, 3 ca history lỗi **`ICHIGO_TOKEN_LIMIT`** ở giới hạn 224 token. Lỗi được chấm như hypothesis rỗng và giữ trong mẫu số. **WER success-only 41,55%**, nên không quy toàn bộ chênh lệch cho 3 ca lỗi. `state=completed` chỉ là hoàn tất vòng lặp, **comparison_ready=False** ở tập all/history; không xếp run này như đã thành công đầy đủ. Không suy mọi biến thể/decoding Ichigo sẽ có cùng chất lượng.

Runtime hai model: Python **3.11.17**, numpy **1.26.4**, pandas **2.2.3**, môi trường tách riêng. Sherpa là ONNX CPU provider, **2 threads, greedy_search, không LM/hotword/punctuation model**, RTF **0,0363**, process peak RSS **0,72 GiB**; VRAM không áp dụng. Ichigo dùng T4/torch **2.5.1+cu124**, quantizer/Whisper weights FP32, decode FP16, `vi/transcribe`, greedy, `sample_len=224`, RTF **0,2369**, reserved VRAM **4,30 GiB**. Không quy latency CPU/GPU khác phiên thành tốc độ so sánh có kiểm soát hoặc latency điện thoại. Không có số đo bộ nhớ/tốc độ train.

Checkpoint Sherpa `csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20` revision `0fc3fea3ccd9c50b439755fa8a6aba546cb3a7d4`; Ichigo `Menlo/Ichigo-whisper-v0.1` revision `0df8a9479d1d0f6a32019c509be8735e2b04fde3`, file quantizer `merge-medium-vi-2d-2560c-dim64.pth` cộng Whisper-medium nền. `parameters.json` ghi Sherpa **67.146.958** tensor scalar trong ONNX (bound bảo thủ), Ichigo **784.983.104 parameters + 2.069.669 floating buffers = 787.052.773**. Không gọi bound là số trainable chính xác hoặc chỉ đếm tokenizer 22M. Với LLM giả định đúng 4B/other=0, tổng là **4.067.146.958 / 4.787.052.773**; không chứng nhận điều lệ từ giả định người chạy.

**Hướng fine-tune sau kết quả mới:** trước hết nghe chấm Sherpa trên cả các câu khác reference và một phần câu exact, đặc biệt tên/số/phủ định, rồi đánh giá agent/máy đích. Nếu gold xác nhận lỗi lặp cần adaptation, ưu tiên xác minh khả năng train Zipformer/RNN-T. [Sherpa chính thức](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#sherpa-onnx-zipformer-vi-2025-04-20-vietnamese) chỉ nguồn về [zzasdf/viet_iter3_pseudo_label](https://huggingface.co/zzasdf/viet_iter3_pseudo_label), card Apache-2.0. Thư mục [exp](https://huggingface.co/zzasdf/viet_iter3_pseudo_label/tree/main/exp) có `epoch-12-avg-8.pt`, `epoch-12.pt`, `pretrained.pt`, nhưng hiện mới xem danh mục: chưa nạp/khớp architecture, tokenizer, decode tương đương ONNX, recipe hoặc backward. ONNX inference tốt không tự chứng minh fine-tune được ngay.

Giữ **large + PEFT/LoRA** làm nhánh train dự phòng nếu chưa khôi phục được recipe Sherpa; medium là phương án Whisper ít tài nguyên hơn. Chưa ưu tiên train Ichigo từ run hiện tại. Không cần fine-tune chỉ để có bước huấn luyện: nếu Sherpa pretrained đạt nhu cầu sau review, có thể giữ nguyên. Chưa có output Cohere thành công hoặc omni trong ZIP mới để xếp hạng. Bảo toàn 991 mẫu ngoài gradient, đối chiếu development/split nhóm, chọn checkpoint bằng split nội bộ; toàn bộ giới hạn dữ liệu ở các phần dưới tiếp tục áp dụng.

## Kết quả PhoWhisper-large/Qwen ở lượt trước

ZIP người chạy gửi: `lifemate_asr_full_test_full_test_outputs.zip`, SHA256 `801508882ae11af53be330bee9803ba73f3e46e918ec2032f2f66f2ed00db273`. Phân tích local kiểm tra checksum 33 file, source/worker hash, trạng thái, độ phủ ID và tính lại metric từ raw predictions; không thực thi mã trong ZIP. Bằng chứng tái lập nằm local/ignored ở `artifacts/asr_full_test_20261031/` (`analyze.py`, `audit.json`, `metrics_recomputed.csv`, `paired_comparisons.csv`, `exact_match.csv`, `text_diagnostics.csv`). Giữ tách với phân tích lịch sử `analysis/asr_selection_20261029/`.

| Model | WER % | CER % | Exact normalized | OK/error/missing | p95 ms | Peak reserved GiB |
|---|---:|---:|---:|---|---:|---:|
| PhoWhisper-large, mới | **26,79** | **18,07** | 122/991 (12,31%) | 991/0/0 | 2382 | 3,64 |
| Qwen3-ASR-1.7B-hf, mới | 29,79 | 20,94 | 106/991 (10,70%) | 991/0/0 | 1956 | 4,28 |
| PhoWhisper-medium, cũ | 29,00 | 19,75 | 103/991 (10,39%) | 991/0/0 | 1715 | 1,76 |
| PhoASR-small, cũ | 28,41 | 19,95 | 123/991 (12,41%) | 991/0/0 | 846 | 0,62 |

**Kiểm tra khả năng đối chiếu:** mở lại ZIP validation cũ, kiểm tra checksum và so trực tiếp ID, reference, nhóm hội thoại, kind, duration cho medium/PhoASR/Qwen; cả 991 mẫu và ba cache hash khớp. Qwen mới có **991/991 transcript trùng Qwen cũ**, không phải bằng chứng độc lập. Lượt mới ghi Tesla T4, FP16, batch 1, eager, torch `2.11.0+cu128`, Transformers `5.14.1`, numpy `1.26.4`. Cùng loại GPU nhưng khác phiên/worker/dependency với lượt cũ; so latency xuyên lượt chỉ mang tính tham khảo. `pip check` vẫn có cảnh báo package nền Kaggle, dù 2 model đã chạy đủ. Không có số đo training trong ZIP.

Large có **3.852/14.376 word edits**, **8.402/46.494 char edits**; Qwen tương ứng **4.283/14.376**, **9.737/46.494**. Large WER final/history **26,56%/27,00%**, so Qwen **30,33%/29,32%**. So medium, large giảm 317 word edits, **2,21 điểm phần trăm WER**, tương đương **7,60% tương đối**. Đây là mức giảm nhờ đổi checkpoint pretrained, không gán cho fine-tune. WER không phải phần bù của “accuracy”; exact match là khớp toàn câu sau normalization, không đánh giá đúng ý nghĩa. Large không dẫn đầu mọi metric: PhoASR-small exact hơn large 1 câu, và nhanh/nhẹ hơn.

Paired bootstrap tính lại **10.000 lần, seed 2026, theo 309 hội thoại**: delta WER large − medium **−2,21 điểm**, CI95 **[−3,06; −1,36]**; large − PhoASR **−1,61**, **[−2,60; −0,62]**; large − Qwen **−3,00**, **[−4,25; −1,77]**. CI khám phá chưa điều chỉnh nhiều so sánh; không chứng minh người nói độc lập. Không có nhãn audio mới để kết luận ưu thế giọng miền Trung, entity hoặc downstream. Large có 0 output chứa ký tự ngoài Latin, Qwen 21: chỉ là kiểm tra ký tự, không phải nhãn hallucination/ngôn ngữ/giọng.

**Lựa chọn ở thời điểm chưa có Sherpa:** ưu tiên large + pilot PEFT/LoRA vì khi đó WER/CER thấp nhất; hiện giữ làm nhánh dự phòng theo cập nhật mới ở trên. Medium/PhoASR và Qwen giữ vai trò đối chứng; không suy khả năng cải thiện sau train chỉ từ thứ hạng pretrained.

Số tham số thực tế: large **1.543.304.960**, Qwen **2.038.052.480**. Nếu LLM thật đúng 4B và model khác bằng 0 như cấu hình người chạy điền, tổng lần lượt **5.543.304.960** và **6.038.052.480**: Qwen vượt mốc 6B theo giả định đó. Đây không phải xác nhận điều lệ/eligibility. VRAM inference 3,64 GiB của large không chứng minh đủ bộ nhớ để full fine-tune trên T4; LoRA cũng cần đo pilot, và không làm giảm số trọng số gốc trong hệ thống triển khai.

**Câu hỏi nhánh Whisper dự phòng:** “Fine-tune PhoWhisper-large trên audio/transcript LifeMate đã rà gold có giảm WER/CER và lỗi thông tin quan trọng so với large gốc trên dữ liệu không dùng train/tuning hay không?” Chưa có thí nghiệm trả lời. Quy trình nhánh Whisper đề xuất:

1. Nghe và chấm gold/lỗi lặp, ưu tiên tên, số, thời gian, phủ định, thiết bị/app, từ địa phương; không dùng prediction làm pseudo-gold mặc định.
2. Đối chiếu manifest thật của 4.045 audio development lịch sử với split nhóm; không mặc định tương ứng train của dev 193 sample hiện tại. Giữ 991 mẫu cũ ngoài gradient. Chia tập early stopping có version trong development theo người nói/hội thoại, kiểm tra trùng audio/hash. Thiếu speaker metadata thì ghi giới hạn group-disjoint. Không dùng holdout/public test để tune.
3. Pilot large + LoRA với batch 1, gradient accumulation, mixed precision phù hợp GPU, checkpoint/resume. Đo VRAM/loss/generate trước train dài; chưa ấn định giờ train, learning rate tối ưu hoặc mức tăng kỳ vọng. Nếu tài nguyên không đủ, chuyển pilot medium trên cùng giao thức. Tham khảo [fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper) và [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora); kiểm tra tương thích stack thực tế trước khi dùng code tutorial.
4. Chọn checkpoint bằng split nội bộ; so checkpoint gốc và fine-tuned bằng cùng scorer/decoding. Báo WER/CER, exact normalized, final/history, failure rate, lỗi critical đã chấm tay và latency/VRAM máy đích. 991 mẫu cũ chỉ là đối chiếu development đã dùng chọn model; cần tập đánh giá cuối chưa dùng để chọn/tune mới có kết luận tổng quát hóa.

## Các model đã đưa vào notebook

Lượt notebook lịch sử có 5 ứng viên mặc định và kết quả 8 checkpoint. Sau đó bổ sung **PhoWhisper-large**, **IchigoWhisper + Whisper-medium**, **Sherpa Zipformer vi 68M** và **hynt Zipformer 153M**, thành **12 checkpoint/cấu hình riêng biệt có prediction**. Hataphu 291M là ứng viên thứ 13 đã thử nhưng chưa có prediction; Qwen 1.7B chạy lại không tính thêm model. Ichigo có 3 lỗi, một số run lịch sử cũng lỗi; không coi cả 12 đều 991/991 OK. Cohere/omni chưa có output thành công trong các ZIP đã kiểm tra để xếp hạng. Lý do chọn ứng viên dưới đây không thay số đo LifeMate.

| Checkpoint | Vai trò, bằng chứng và giới hạn |
|---|---|
| [hynt/ZipFormer-150M-CR-CTC-RNNT-6000h](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) | Dẫn đầu WER/CER đã đo: 15,34%/10,65%, 991/991 OK. Card CC-BY-NC-ND-4.0; chưa chốt quyền fine-tune/triển khai, chưa xác minh training checkpoint/recipe |
| [hataphu/zipformer-k2-rnn-lm-vi](https://huggingface.co/hataphu/zipformer-k2-rnn-lm-vi) | Khoảng 291M theo card MIT, cần quyền tải. Lượt vừa chạy lỗi validator trước inference; chưa xếp hạng, chưa bật RNNLM |
| [csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20) | Ứng viên tích hợp nhẹ/đối chứng 153M: WER 18,79%, 991/991 OK, ONNX CPU 2 threads. Xác minh trainability từ checkpoint PyTorch/recipe nguồn trước pilot fine-tune |
| [Menlo/Ichigo-whisper-v0.1](https://huggingface.co/Menlo/Ichigo-whisper-v0.1) | Tokenizer/quantizer cộng Whisper-medium; đã đo WER 41,75%, 988 OK/3 lỗi, chưa ưu tiên. Card CC-BY-NC-SA-4.0; số tham số cả hệ ASR khác số tokenizer |
| [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) | Nhánh Whisper/LoRA dự phòng và đối chứng sau khi có Zipformer. Card BSD-3-Clause; checkpoint thực chạy `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`. Số đo LifeMate ở phần mới, không dùng kết quả paper thay thế |
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

**Đã kiểm tra:** danh mục ZIP gốc có 6.969 WAV; hai ZIP lịch sử xác nhận benchmark 100 và validation 991 audio cùng reference/role giữa 8 model. Ba ZIP full 991 gửi tiếp theo bổ sung large, Ichigo, Sherpa 68M và hynt 153M; tổng 12 checkpoint/cấu hình có prediction, hataphu 291M chưa inference. Bốn ZIP của tập 991 khớp ID/reference/nhóm/kind/duration. Ba CSV cache gốc chưa có local để tái kiểm tra development-disjoint/hash audio. Quy mô train/development/public test dưới đây vẫn theo audit cũ; các lượt 991 đã được kiểm tra từ output, không dùng tập public test để xếp hạng.

| Tập | Audio / hội thoại | Cách dùng |
|---|---|---|
| Train | 5.036 / 1.545 | Nguồn chia development và validation |
| Development | 4.045 / 1.236 | Chỉ dùng tập này để cập nhật trọng số nếu fine-tune |
| Validation | 991 / 309; 309 final + 682 history | Chọn/đánh giá ứng viên; giữ split cũ |
| Benchmark | 100 / 81; 48 final + 52 history | Tập con validation cố định, không chọn theo prediction |
| Public test | 1.933 / 634 | Không dùng để chọn model hoặc tune theo leaderboard |

Cache cũ ghi tổng 9,5913 giờ, mono 16 kHz PCM_16, dài nhất 22,3335 s; signal audit chỉ xét 2.000 file, có 39 cờ nhiều quiet frames, chưa chứng minh noise/SNR. Benchmark có 18% câu >10 s, validation khoảng 4,2%: điểm 100 audio không đại diện hoàn toàn cho validation. Chưa có speaker ID đáng tin để chứng minh hai tập khác người nói.

Bản ghi cũ báo đã sửa nhầm split do tên thư mục `train+public_test` và loại sample submission khỏi dữ liệu. Khi có cache, đối chiếu SHA256 trong `analysis/audit_verification.json`, join bằng `ref_uid`, kiểm tra nhóm hội thoại/hash overlap; không ghép theo thứ tự dòng hoặc chia random lại. Hướng dẫn: [hướng dẫn ASR](../README.md).

Cấu hình lịch sử ở [notebook ASRmodel](../notebook/lifemate_asr_test_ASRmodel.ipynb); lượt mới ở [full 991 HF](../notebook/zalo_lifemate_full_test.ipynb). Scorer/runtime dùng [asr_benchmark.py](../asr_benchmark.py), worker và source snapshot trong ZIP. Notebook [full 991 omni](../notebook/zalo_lifemate_omni_full_test.ipynb) đã chuẩn bị nhưng ZIP mới không chứa kết quả omni. `full_test` vẫn dùng nguyên 991 ID validation cũ, không chia lại dữ liệu và không biến tập đó thành test độc lập.

Giao thức lịch sử, giữ để đối chiếu:

- Seed 2026, batch 1. Stack HF lịch sử: Transformers 5.14.1, HF Hub 1.24.0, tokenizers 0.22.2, safetensors 0.8.0, soundfile 0.13.1, psutil 7.1.0, librosa 0.11.0; giữ torch/CUDA của Kaggle. Tám model lịch sử và large dùng T4/FP16; **không áp mô tả này cho mọi run mới**: Ichigo tách runtime torch 2.5.1, FP32 weights/FP16 decode; Sherpa ONNX CPU 2 threads, chi tiết ở phần cập nhật. Không mở rộng kết luận sang phần cứng khác.
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
