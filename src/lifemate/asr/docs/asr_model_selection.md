# Quyết định model ASR cho LifeMate

## Trạng thái sau fine-tune — 01/11/2026

Đã có kết quả fine-tune và continuation trên cùng 4.532 train / 504 valid.
Best theo EM rồi WER: **hynt epoch 9: 242/504, WER 8,269%, CER 6,231%**;
**Sherpa epoch 3: 240/504, WER 8,059%, CER 5,770%**.
hynt epoch 5 có WER thấp hơn epoch 9 (8,157%) nhưng chỉ 239 câu exact.
Các CSV tổng hợp, counters và định nghĩa ở [results](results/README.md).

Đây là validation dùng chọn checkpoint, không phải test độc lập; chưa có kiểm định để nói
chênh lệch 2 câu giữa hai model là chắc chắn. Dừng tăng epoch ở cấu hình hiện tại,
ưu tiên nghe đối chiếu và kiểm tra inference. Chưa chốt model triển khai, chưa có bằng chứng export ONNX/máy đích.
Adapter PhoWhisper cũ đã bỏ; package giữ interface ASR. Web đọc predictions có sẵn, không chạy model.

Các mục ngày 31/10 dưới đây giữ làm lịch sử pretrained/991 và kế hoạch trước khi có kết quả train;
các câu “chưa fine-tune” trong đó không mô tả trạng thái hiện tại.

## Quyết định cập nhật ngày 31/10/2026

**Zipformer hynt 153M dẫn đầu chất lượng đã đo: WER 15,34%, CER 10,65%, đủ 991/991 OK**, trong **12 checkpoint/cấu hình riêng biệt có prediction** trên tập 991 mẫu cũ. Ưu tiên nghe review đối chiếu **153M và Sherpa Zipformer vi 68M**. **Giữ 68M làm ứng viên tích hợp nhẹ và nhánh xác minh khả năng fine-tune**; 153M chưa được chốt cho triển khai/fine-tune vì card ghi CC-BY-NC-ND-4.0, cần làm rõ quyền sử dụng tương ứng. Adapter đang tích hợp chưa tự đổi theo tài liệu. **Hataphu 291M chưa có inference thành công, không xếp hạng.**

**Chưa fine-tune ngay:** ưu tiên kiểm tra gold/lỗi quan trọng và baseline Sherpa 68M trước. Nếu cần adaptation, xác minh checkpoint PyTorch và recipe Zipformer/RNN-T tương ứng 68M để pilot; giữ **PhoWhisper-large + LoRA** làm nhánh fine-tune dự phòng. Chưa có kết quả fine-tune chứng minh mức tăng. Ichigo không được ưu tiên từ cấu hình đã đo. Chưa có output Cohere thành công hoặc kết quả omni trong các ZIP đã kiểm tra để xếp hạng; chưa chốt model triển khai cuối trước khi đo máy đích và nghe review.

## Kết quả Zipformer 153M/291M mới nhất

Nguồn `lifemate_asr_zipformer_large_full_test_outputs.zip`, SHA256 `676a4613f186906ed7521fe2030933af9b7524168d04cfa7b3c733c94266c981`. Đã kiểm tra checksum **32 file**, độ phủ/không trùng prediction ID và tính lại WER/CER từ raw predictions; không thực thi code trong ZIP. Bằng chứng local/ignored: `artifacts/asr_zipformer_large_20261031/` (`check_outputs.py`, `same_set_audit.json`, `hynt_153m_metrics.json`, `hynt_153m_subsets.json`, `paired_153m_vs_68m.json`).

| Model/config | WER % | CER % | Exact normalized | OK/error/missing | p50/p95 ms |
|---|---:|---:|---:|---|---:|
| **hynt Zipformer 153M** | **15,34** | **10,65** | **260/991 = 26,24%** | **991/0/0** | 257/473 |
| Sherpa Zipformer vi 68M, lượt trước | 18,79 | 14,07 | 177/991 = 17,86% | 991/0/0 | 159/315 |
| PhoWhisper-large, lượt trước | 26,79 | 18,07 | 122/991 = 12,31% | 991/0/0 | 1359/2382 |
| hataphu Zipformer 291M | — | — | Chưa đánh giá | **0/0/991** | — |

153M có **2.205/14.376 word edits**, **4.953/46.494 char edits**; giảm **496 word edits, 3,45 điểm phần trăm WER (18,36% tương đối)** so 68M, thêm **83 câu exact normalized**. Paired bootstrap **10.000 lần, seed 2026, theo 309 hội thoại**: delta WER **153M − 68M = −3,45 điểm**, CI95 **[−4,07; −2,81]**; 153M ít word edits hơn trên **432 câu**, 68M tốt hơn **173**, hòa **386**. CI khám phá chưa điều chỉnh nhiều so sánh; không chứng minh ưu thế theo giọng, lỗi critical, downstream hay speaker-disjoint. Đây là lợi ích đổi checkpoint pretrained, chưa phải fine-tune; không diễn giải WER thành accuracy.

**991 mẫu có giống nhau không?** Đã mở lại **4 ZIP**: validation lịch sử, large/Qwen, under2B và Zipformer lớn. Toàn bộ bảng per-audio được kiểm tra có cùng **991 ref_uid, reference, sample_uid, kind và duration**, gồm **309 hội thoại**. Lượt 291M cũng được chọn đúng tập nhưng chưa có prediction. Fingerprint lượt mới vẫn là `2c9bb8c6966bfd116b2aabc6d565cfbf9af3d45ae154660727c4fe79c20478c6`. Đây vẫn là **development validation**, không phải test độc lập; smoke 8/benchmark 100 là các lượt nhỏ riêng. ZIP không chứa checksum waveform nên kiểm tra này xác nhận danh tính mẫu/metadata, chưa chứng minh byte audio giữa các phiên.

153M dùng `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h`, revision `8c3d474d3d74cb247f4b8c7b0f8b48598ac8c1c6`, ONNX epoch-11-avg-2 **FP32, CPU 2 threads, greedy_search, không LM/hotword/punctuation model**, Sherpa-ONNX **2.0.1**. RTF **0,0571**, process peak RSS **1,12 GiB**, VRAM không áp dụng. `config.json` là bảng token symbol/id, vocabulary 2.000. Bound lưu trong ONNX là **149.906.361 scalar**, không gọi là số trainable chính xác; cộng LLM giả định đúng 4B/other=0 thành **4.149.906.361**, chưa xác nhận eligibility. Latency giữa các phiên, đặc biệt CPU/GPU, chỉ để tham khảo; chưa có benchmark máy đích hoặc bộ nhớ training.

**291M lỗi trước inference:** log dừng tại `Expected contiguous vocabulary IDs with <blk> at 0`; đã qua bước tải checkpoint. `n_ok=0`, `n_missing=991`, `comparison_ready=False`, WER/CER không hợp lệ; không coi là model nhận dạng sai 100%. ZIP không có file token gốc để xác định chính xác nhánh điều kiện nào gây lỗi. Đã tạo [notebook sửa riêng 291M](../notebook/zalo_lifemate_zipformer_291m_fixed_full_test.ipynb): kiểm tra đủ output ID theo metadata ONNX, giữ nguyên mapping, không ép tên blank/ID phụ, dùng `tokens.txt` ở root và output mới. **32 kiểm tra giả lập liên quan đã qua; chưa chạy inference 291M thật với bản sửa.**

[Card hynt](https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) ghi **CC-BY-NC-ND-4.0**; [card hataphu](https://huggingface.co/hataphu/zipformer-k2-rnn-lm-vi) ghi **MIT**, cần quyền tải. Đối chiếu card ngày 31/10/2026. Chất lượng 153M tốt nhất trong các run đã đo không tự xác nhận quyền fine-tune/phát hành/triển khai; chưa lấy nó làm nhánh train mặc định. 291M có license khác nhưng chưa có kết quả để thay lựa chọn.

## Kết quả Ichigo/Sherpa 68M ở lượt trước

Nguồn `lifemate_asr_under2b_full_test_outputs.zip`, SHA256 `cdcddb91206f09df5690010133075e25e318c93f5a12f7e8b506aee3d6684d72`. Đã kiểm tra checksum **36 file**, source/worker hash, coverage/không trùng prediction ID và tính lại WER/CER từ raw prediction. Không thực thi code trong ZIP. **991 ref_uid, reference, sample_uid, kind, duration, path, partition và ba cache hash khớp lượt large/Qwen**, cùng `input_fingerprint` `2c9bb8c6966bfd116b2aabc6d565cfbf9af3d45ae154660727c4fe79c20478c6`. Preflight ghi 991 file hợp lệ. Chi tiết ở `artifacts/asr_under2b_20261031/`, giữ local/ignored.

| Model/config | WER % | CER % | Exact normalized | OK/error/missing | p50/p95 ms | Thiết bị |
|---|---:|---:|---:|---|---:|---|
| **Sherpa Zipformer vi 68M** | **18,79** | **14,07** | **177/991 = 17,86%** | **991/0/0** | 159/315 | CPU, 2 threads |
| PhoWhisper-large, lượt trước | 26,79 | 18,07 | 122/991 = 12,31% | 991/0/0 | 1359/2382 | T4 GPU |
| PhoASR-small, lịch sử | 28,41 | 19,95 | 123/991 = 12,41% | 991/0/0 | 455/846 | T4 GPU |
| PhoWhisper-medium, lịch sử | 29,00 | 19,75 | 103/991 = 10,39% | 991/0/0 | 944/1715 | T4 GPU |
| Qwen3-ASR-1.7B-hf, lượt trước | 29,79 | 20,94 | 106/991 = 10,70% | 991/0/0 | 1072/1956 | T4 GPU |
| IchigoWhisper + Whisper-medium | 41,75 | 29,07 | 30/991 = 3,03% | **988/3/0** | 1078/1848 | T4 GPU |

**Sherpa:** 2.701/14.376 word edits, 6.540/46.494 char edits; WER final/history **19,23%/18,40%**, >10 s **19,30%**. Giảm **1.151 word edits**, **8,01 điểm phần trăm WER**, tương đương **29,88% WER tương đối** so large. Đây là lợi ích **đổi pretrained checkpoint**, không phải fine-tune và không phải “đúng 81,21%”. Exact match vẫn chỉ là khớp reference sau chuẩn hóa.

Paired bootstrap **10.000 lần, seed 2026, theo 309 hội thoại**: delta WER **Sherpa − large = −8,01 điểm**, CI95 **[−9,10; −6,94]**; Sherpa − medium **−10,21**, **[−11,37; −9,10]**; Sherpa − PhoASR **−9,62**, **[−10,72; −8,53]**. Sherpa ít word edits hơn large trên **549 câu**, large tốt hơn trên **182**, hòa **260**. CI khám phá chưa điều chỉnh nhiều so sánh, không chứng minh speaker-disjoint; chưa có nhãn giọng hoặc downstream để kết luận các mặt đó.

**Ichigo:** 6.002 word edits, 13.515 char edits; 3 ca history bị `ICHIGO_TOKEN_LIMIT` ở giới hạn 224 token, chấm như hypothesis rỗng và giữ mẫu số. `state=completed` chỉ là đã đi hết lượt, **`comparison_ready=False`** vì còn lỗi. WER trên 988 ca thành công vẫn **41,55%**; chất lượng kém trong cấu hình này không chỉ do 3 lỗi. Không loại các ca lỗi để xếp hạng, cũng chưa suy mọi cấu hình Ichigo đều kém.

Runtime: cả hai Python **3.11.17**, numpy **1.26.4**, pandas **2.2.3**; môi trường tách riêng. Sherpa dùng CPU provider, **2 threads, greedy_search, không LM/hotword/punctuation model**, RAM process peak khoảng **0,72 GiB**, RTF **0,0363**; VRAM **không áp dụng**, không ghi thành 0. Ichigo dùng T4/torch **2.5.1+cu124**, weights float32, decoding FP16, `vi/transcribe`, greedy, `sample_len=224`, VRAM reserved **4,30 GiB**, RTF **0,2369**. Không coi latency CPU và GPU khác runtime là phép đo tốc độ tương đối có kiểm soát, chưa suy sang điện thoại/máy triển khai.

Checkpoint: Sherpa revision `0fc3fea3ccd9c50b439755fa8a6aba546cb3a7d4`; Ichigo revision `0df8a9479d1d0f6a32019c509be8735e2b04fde3`, dùng quantizer `merge-medium-vi-2d-2560c-dim64.pth` và OpenAI Whisper-medium nền. Bound được ghi: **67.146.958** scalar tensor ONNX cho Sherpa; Ichigo **784.983.104 parameters + 2.069.669 floating buffers = 787.052.773**. Đây là cách đếm bảo thủ được mô tả trong `parameters.json`, không gọi bound là số trainable chính xác, không chỉ đếm tokenizer 22M của Ichigo. Cả hai dưới 2B; cộng LLM giả định đúng 4B/other=0 lần lượt **4.067.146.958 / 4.787.052.773**, chưa phải xác nhận eligibility.

## Kết quả PhoWhisper-large/Qwen ở lượt trước

Nguồn: `lifemate_asr_full_test_full_test_outputs.zip`, SHA256 `801508882ae11af53be330bee9803ba73f3e46e918ec2032f2f66f2ed00db273`. Đã xác minh checksum **33 file** trong manifest, prediction đủ/không trùng ID, trạng thái hoàn tất và tính lại WER/CER từ raw predictions bằng mã local; không thực thi code trong ZIP. Cả hai model **991 OK, 0 error, 0 missing**; preflight ghi 991 input hợp lệ.

| Model, lượt mới | WER % | CER % | Exact normalized | p50/p95 ms | RTF | VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|
| PhoWhisper-large | **26,79** | **18,07** | 122/991 = 12,31% | 1359/2382 | 0,297 | 3,64 |
| Qwen3-ASR-1.7B-hf | 29,79 | 20,94 | 106/991 = 10,70% | 1072/1956 | 0,237 | 4,28 |

Mẫu số: **14.376 token, 46.494 ký tự không khoảng trắng**; large có **3.852 word edits/8.402 char edits**, Qwen **4.283/9.737**. WER final/history của large **26,56%/27,00%**, Qwen **30,33%/29,32%**. Exact raw lần lượt 0 và 18 câu; khác định dạng văn bản cũng làm mất exact raw. **WER 26,79% không có nghĩa “đúng 73,21%”**; exact normalized là khớp toàn câu sau chuẩn hóa, chưa phải đúng ý nghĩa/tool call.

Runtime mới: Tesla T4, FP16, batch 1, eager, torch `2.11.0+cu128`, Transformers `5.14.1`, numpy `1.26.4`. Latency từ đọc audio đến decode, không gồm tải/load/warm-up; VRAM là peak reserved của PyTorch, **không phải bộ nhớ training**. `pip check` có cảnh báo package nền Kaggle; hai lượt inference thực tế vẫn hoàn tất.

Checkpoint khóa: large `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`, `vi/transcribe`, greedy, tối đa 400 token; Qwen `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`, tiếng Việt cố định, greedy, tối đa 512 token.

Đã mở lại ZIP validation cũ, kiểm tra checksum và đối chiếu **ref_uid, reference, sample_uid, kind, duration** của 991 mẫu giữa lượt mới với medium/PhoASR/Qwen cũ; ba cache hash khớp. **Qwen mới có 991/991 transcript giống hệt Qwen cũ**. `full_test` chỉ là tên chạy đủ **development validation 991 audio/309 hội thoại** cũ, không phải test độc lập. Lượt cũ/mới khác phiên, worker và một số dependency; không xem tỷ lệ latency giữa lượt là phép đo triển khai có kiểm soát.

Paired bootstrap **10.000 lần, seed 2026, lấy mẫu theo 309 hội thoại**, trên cùng ID/reference:

| Delta WER, A − B | Điểm phần trăm | CI 95% |
|---|---:|---:|
| large − medium | **−2,21** | **[−3,06; −1,36]** |
| large − PhoASR-small | **−1,61** | **[−2,60; −0,62]** |
| large − Qwen 1.7B | **−3,00** | **[−4,25; −1,77]** |

Large giảm **317 word edits**, tương đương **7,60% WER tương đối** so medium: đây là lợi ích **đổi checkpoint pretrained**, chưa phải fine-tune. Large ít word edits hơn medium trên 398 câu, medium tốt hơn trên 288, hòa 305. CI khám phá chưa điều chỉnh nhiều so sánh; không có speaker ID để chứng minh tách người nói hoặc kết luận riêng giọng miền Trung. Bằng chứng mới ở `artifacts/asr_full_test_20261031/`, giữ local/ignored; số cũ bên dưới giữ nguồn riêng.

## Model nào đáng fine-tune?

| Ưu tiên | Cách dùng | Căn cứ và giới hạn |
|---|---|---|
| **Dẫn đầu pretrained: hynt Zipformer 153M** | Nghe review đối chiếu; chưa chọn làm nhánh fine-tune mặc định | WER 15,34%, CER 10,65%, 991/991 OK; bound 149.906.361. Card CC-BY-NC-ND-4.0: cần làm rõ quyền phù hợp trước adaptation/phát hành/triển khai; chưa kiểm chứng training checkpoint/recipe |
| **1. Sherpa Zipformer vi 68M** | Review/tích hợp pretrained trước; pilot fine-tune khi xác minh trainability | WER 18,79%, CER 14,07%, 991/991 OK; bound 67.146.958. Các file ONNX vừa đo phục vụ inference; cần checkpoint/recipe huấn luyện gốc, chưa có pilot train |
| **2. PhoWhisper-large** | Nhánh PEFT/LoRA dự phòng; baseline đối chứng Sherpa | WER 26,79%, CER 18,07%, 991/991 OK, 1.543.304.960 tham số. Chưa đo VRAM/giờ train hoặc mức tăng sau train |
| **3. PhoWhisper-medium** | Phương án train Whisper nhẹ hơn, baseline tích hợp | WER 29,00%; giữ làm đối chứng nếu pilot large không vừa GPU |
| **PhoASR-small** | Nhánh phụ sau khi chốt điều kiện dùng | WER 28,41%, nhanh/nhẹ; không còn là nhánh tiết kiệm ưu tiên trước Sherpa theo số đo mới. Card có license kèm điều kiện sử dụng, xem nguồn lịch sử bên dưới |
| **Chưa ưu tiên Qwen 1.7B** | Giữ làm đối chứng | WER/CER cao hơn large; số tham số thực tế 2.038.052.480, không chỉ 1,7B theo tên |
| **Chưa ưu tiên Ichigo** | Chỉ xem lại nếu có lý do sửa decoding/cấu hình | WER 41,75%, 3 lỗi token limit; không đủ điều kiện comparison_ready |
| **Chờ kết quả: hataphu Zipformer 291M** | Chạy notebook sửa, chưa xếp hạng hoặc chọn fine-tune | 0 prediction thành công; lỗi validator trước inference. Không suy chất lượng từ tên/quy mô model |

[Tài liệu Sherpa chính thức](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#sherpa-onnx-zipformer-vi-2025-04-20-vietnamese) trỏ tới [model nguồn Apache-2.0](https://huggingface.co/zzasdf/viet_iter3_pseudo_label). Thư mục [exp](https://huggingface.co/zzasdf/viet_iter3_pseudo_label/tree/main/exp) có `epoch-12-avg-8.pt`, `epoch-12.pt`, `pretrained.pt`: đã thấy danh mục public, **chưa tải/nạp/khôi phục architecture hoặc chạy backward**, chưa xác minh recipe/tokenizer và tương đương với bản ONNX. Không coi có file `.pt` là đủ bằng chứng fine-tune chạy được. [Ichigo card](https://huggingface.co/Menlo/Ichigo-whisper-v0.1) ghi CC-BY-NC-SA-4.0; số đo card không thay điểm LifeMate.

Nếu LLM thật có **đúng 4.000.000.000 tham số**, chưa có model khác: large + LLM = **5.543.304.960**, Qwen + LLM = **6.038.052.480**. Qwen vượt mốc 6B theo giả định này; số LLM/other trong ZIP do người chạy điền, `below_6B` không chứng nhận eligibility. LoRA giảm tham số cập nhật khi train, không loại trọng số model gốc khỏi hệ thống triển khai. Large có [model card BSD-3-Clause](https://huggingface.co/vinai/PhoWhisper-large); điều lệ cuộc thi vẫn cần xác minh riêng.

## Kế hoạch kiểm chứng fine-tune

1. **Nghe và xác nhận gold/lỗi lặp:** đối chiếu 153M/68M cùng audio, ưu tiên tên, tiền/số, thời gian, phủ định, thiết bị/app và từ địa phương. Không sửa reference theo prediction; không gọi 869 câu large chưa exact là “869 câu sai nội dung”. Lần phân tích này chưa nghe audio. Chốt quyền dùng model trước khi chọn nhánh train; không tự chuyển sang fine-tune 153M chỉ vì WER dẫn đầu.
2. **Khóa dữ liệu trước train:** giữ 991 mẫu hiện tại chỉ để đối chiếu kết quả cũ, tuyệt đối không đưa vào gradient. 4.045 audio development theo audit là nguồn train dự kiến, cần đối chiếu manifest thật và split nhóm; không mặc nhiên trùng train tương ứng dev 193 sample của nhóm. Tách tập early stopping trong development theo người nói/hội thoại có version, kiểm tra trùng audio/hash. Thiếu speaker metadata thì chỉ được nói group-disjoint. Không dùng holdout/public test để tune.
3. **Xác minh trainability Sherpa rồi mới pilot:** khớp architecture, tokenizer và checkpoint PyTorch gốc với ONNX đã đo; chạy thử load/decode và backward trên fixture trước dữ liệu thật. Sau khi gold/lỗi lặp được xác nhận, mới pilot adaptation Zipformer/RNN-T và xuất lại ONNX để so cùng runtime. Nếu chưa khôi phục được recipe, giữ Sherpa pretrained và dùng **large + PEFT/LoRA** làm nhánh train dự phòng; không chuyển mô hình chỉ để có bước fine-tune. Nhánh Whisper có thể thử batch 1/gradient accumulation/mixed precision, đo VRAM/loss/generate trước train dài. Chưa chạy, chưa đặt giờ train hoặc tỷ lệ tăng kỳ vọng. Tham khảo [fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper) và [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora); tutorial không phải bằng chứng LifeMate.
4. **So trước/sau cùng giao thức:** WER/CER, exact normalized, final/history, lỗi quan trọng đã chấm tay, failure rate và latency/VRAM trên máy đích. Chọn checkpoint bằng split nội bộ; không liên tục chọn theo 991 rồi gọi đó là test cuối. Chỉ giữ model fine-tuned nếu cải thiện trên dữ liệu không dùng train/tuning và không tăng lỗi quan trọng.

## Kết quả lịch sử ngày 29/10 — trước khi có PhoWhisper-large

Quyết định khi đó là medium trong 5 ứng viên chính. Phần dưới giữ bằng chứng và lập luận lịch sử; lựa chọn hiện tại theo cập nhật 153M/68M ở đầu tài liệu.

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

### Cơ sở chọn medium trong lần cũ

- **Chất lượng và ổn định:** PhoWhisper-medium có WER final/history **29,59%/28,48%**, không lỗi pipeline. So Qwen 1.7B, WER thấp hơn **0,79 điểm phần trăm**, nhưng CI 95% **[-2.13; 0.53]** chưa tách biệt chắc chắn. CER thấp hơn, VRAM **1,76 thay vì 4,25 GiB**; Qwen 1.7B có **21/991** output chứa chữ ngoài Latin, bản 0.6B có **33/991**. Đây là kiểm tra ký tự, không phải nhãn ngôn ngữ/giọng.
- **PhoASR-small cạnh tranh thực sự:** WER thấp hơn medium **0,59 điểm**, p95 nhanh hơn khoảng **2,03 lần** và dùng ít VRAM hơn. Nhưng CI của **PhoASR − medium** là **[-1.56; 0.38] điểm**, chưa đủ kết luận thắng hoặc tương đương. PhoASR ít lỗi token hơn trên **380** câu, medium trên **327**, hòa **284**. [Model card](https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small) ghi BSD-3-Clause-Clear kèm Qualcomm Responsible AI License và intended use nghiên cứu/giáo dục; cần chốt quyền dùng, không tự suy cấm thương mại. PhoWhisper ghi [BSD-3-Clause](https://huggingface.co/vinai/PhoWhisper-medium). Chọn medium hiện tại không có nghĩa medium thắng PhoASR mọi tiêu chí.
- **Tốc độ không đủ để chọn:** wav2vec2 rất nhanh nhưng WER cao hơn medium **8,79 điểm**; turbo nhanh hơn nhưng cao hơn **4,47 điểm**. Chưa có yêu cầu latency sản phẩm để đánh đổi thêm lỗi lấy tốc độ. Không tạo trọng số tổng hợp tùy ý.

### Giới hạn quyết định cũ

Paired bootstrap **10.000 lần, seed 2026, gom theo 309 hội thoại**; 28 cặp (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) là phân tích khám phá, CI chưa điều chỉnh so sánh nhiều cặp. Không có speaker ID để chứng minh tách người nói. Validation đã dùng chọn model nên là **development validation**, không phải test cuối độc lập.

Annotation gốc đều trống; chưa nghe audio, chưa đo lỗi theo giọng hay downstream. ID/path thuộc train, checksum cache khớp bản audit cũ và các model dùng cùng reference; ZIP không có ba CSV cache gốc để kiểm tra lại split/development hoặc chất lượng gold. Timing một lượt trên Kaggle không suy ra latency CPU/máy triển khai. Config/revision đầy đủ (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) lưu bằng chứng.

### Kế hoạch tích hợp ở lần cũ

- Pin PhoWhisper-medium revision `55a7e3eb6c906de891f8f06a107754427dd3be79`, `vi/transcribe`, greedy beam 1, tối đa 400 token; giao `text_model` khi status ok cho agent, normalization chỉ dùng scoring.
- Nghe/chấm lại 44 ca review, ưu tiên 12 ca tìm lỗi, rồi thử downstream trên các ca critical. Nếu PhoASR không kém về lỗi quan trọng và điều kiện dùng phù hợp, cân nhắc chuyển mặc định để giảm latency; không cần chạy lại toàn bộ 991 audio.
- **Chưa fine-tune hoặc thêm fallback model ngay.** Chỉ fine-tune trên development sau khi xác nhận lỗi lặp và gold. Không dùng reference/WER từng câu hoặc confidence tự bịa để kích hoạt fallback; khi lỗi kỹ thuật, trả lỗi để agent yêu cầu nói lại.
