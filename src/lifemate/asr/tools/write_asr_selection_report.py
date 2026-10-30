"""Render the dated team memo from the independently recomputed CSV evidence."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/asr_selection_20261029'
m=pd.read_csv(OUT/'metrics_recomputed.csv')
v=m[(m.stage=='validation')&(m.subset=='all')].set_index('model')
pairs=pd.read_csv(OUT/'paired_comparisons.csv')
pair=pairs[(pairs.stage=='validation')&(pairs.a=='phoasr_small')&(pairs.b=='phowhisper_medium')].iloc[0]
qpair=pairs[(pairs.stage=='validation')&(pairs.a=='phowhisper_medium')&(pairs.b=='qwen3_asr_17b')].iloc[0]
audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
review=pd.read_csv(OUT/'error_review.csv')
assert len(v)==8 and v.n.eq(991).all() and len(review)==44
labels={'phowhisper_medium':'PhoWhisper-medium*','whisper_turbo':'Whisper-v3-turbo*',
    'qwen3_asr_06b':'Qwen3-ASR-0.6B*','qwen3_asr_17b':'Qwen3-ASR-1.7B*','wav2vec2_vi':'wav2vec2-250h*',
    'phoasr_small':'PhoASR-small','phowhisper_small':'PhoWhisper-small','whisper_small':'Whisper-small'}
table='| Model | WER % | CER % | OK/991 | p50/p95 ms | RTF | VRAM GiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in ['phowhisper_medium','phoasr_small','qwen3_asr_17b','whisper_turbo','phowhisper_small','qwen3_asr_06b','wav2vec2_vi','whisper_small']:
    r=v.loc[name]
    table+=f'| {labels[name]} | {r.wer*100:.2f} | {r.cer_no_spaces*100:.2f} | {r.n_ok:.0f} | {r.latency_p50_ms:.0f}/{r.latency_p95_ms:.0f} | {r.rtf_weighted:.3f} | {r.vram_peak_reserved_bytes/1024**3:.2f} |\n'
memo=f'''# Quyết định model ASR cho LifeMate

## Quyết định

**Chọn `vinai/PhoWhisper-medium` làm ASR mặc định hiện tại để tích hợp. Mức chắc chắn: vừa.** Model dẫn đầu WER trong 5 ứng viên chính, có CER thấp nhất cả 8 model thực tế trong ZIP và thành công 991/991 audio. Đây là quyết định theo cấu hình đã thử, chưa chứng minh tốt nhất ngoài LifeMate. PhoASR-small là đối thủ gần nhất, cần giữ để đối chiếu sau khi xác minh điều kiện sử dụng và lỗi quan trọng.

## Kết quả đã đối chiếu

File thực tế có **8 model**, nên bảng giữ đủ 8; dấu * chỉ 5 ứng viên chính. Validation gồm **991 audio/309 hội thoại: 309 final, 682 history**, tổng 14.376 token reference theo khoảng trắng. Benchmark 100 audio là tập con, không cộng thêm hoặc coi là test độc lập.

{table}
WER/CER tính lại từ raw predictions: corpus edit distance/tổng độ dài reference. WER theo khoảng trắng không phải tách từ vựng tiếng Việt; CER bỏ khoảng trắng. NFC, lowercase, dấu câu/ký hiệu thành khoảng trắng, giữ dấu và chữ số. Không đồng nhất số chữ với chữ số. Lỗi/thiếu được chấm như câu rỗng; turbo lỗi 1, Whisper-small lỗi 4, không có mẫu thiếu. Bảng chỉ trên ca thành công và final/history ở metrics CSV (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git).

Tất cả chạy một **T4, FP16, batch 1, eager, Transformers 5.14.1/torch 2.11.0+cu128**. Latency gồm đọc audio đến decode, không gồm load/warm-up. VRAM là PyTorch peak reserved, gồm weights, không phải tổng GPU memory; RAM process và load/warm-up nằm trong chi tiết (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git).

## Vì sao chọn như vậy?

- **Chất lượng và ổn định:** PhoWhisper-medium có WER final/history **29,59%/28,48%**, không lỗi pipeline. So Qwen 1.7B, WER thấp hơn **0,79 điểm phần trăm**, nhưng CI 95% **[{qpair.wer_ci95_low_pp:.2f}; {qpair.wer_ci95_high_pp:.2f}]** chưa tách biệt chắc chắn. CER thấp hơn, VRAM **1,76 thay vì 4,25 GiB**; Qwen 1.7B có **21/991** output chứa chữ ngoài Latin, bản 0.6B có **33/991**. Đây là kiểm tra ký tự, không phải nhãn ngôn ngữ/giọng.
- **PhoASR-small cạnh tranh thực sự:** WER thấp hơn medium **0,59 điểm**, p95 nhanh hơn khoảng **2,03 lần** và dùng ít VRAM hơn. Nhưng CI của **PhoASR − medium** là **[{pair.wer_ci95_low_pp:.2f}; {pair.wer_ci95_high_pp:.2f}] điểm**, chưa đủ kết luận thắng hoặc tương đương. PhoASR ít lỗi token hơn trên **380** câu, medium trên **327**, hòa **284**. [Model card](https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small) ghi BSD-3-Clause-Clear kèm Qualcomm Responsible AI License và intended use nghiên cứu/giáo dục; cần chốt quyền dùng, không tự suy cấm thương mại. PhoWhisper ghi [BSD-3-Clause](https://huggingface.co/vinai/PhoWhisper-medium). Chọn medium hiện tại không có nghĩa medium thắng PhoASR mọi tiêu chí.
- **Tốc độ không đủ để chọn:** wav2vec2 rất nhanh nhưng WER cao hơn medium **8,79 điểm**; turbo nhanh hơn nhưng cao hơn **4,47 điểm**. Chưa có yêu cầu latency sản phẩm để đánh đổi thêm lỗi lấy tốc độ. Không tạo trọng số tổng hợp tùy ý.

## Giới hạn quyết định

Paired bootstrap **10.000 lần, seed 2026, gom theo 309 hội thoại**; 28 cặp (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) là phân tích khám phá, CI chưa điều chỉnh so sánh nhiều cặp. Không có speaker ID để chứng minh tách người nói. Validation đã dùng chọn model nên là **development validation**, không phải test cuối độc lập.

Annotation gốc đều trống; chưa nghe audio, chưa đo lỗi theo giọng hay downstream. ID/path thuộc train, checksum cache khớp bản audit cũ và các model dùng cùng reference; ZIP không có ba CSV cache gốc để kiểm tra lại split/development hoặc chất lượng gold. Timing một lượt trên Kaggle không suy ra latency CPU/máy triển khai. Config/revision đầy đủ (bằng chứng riêng trong `analysis/asr_selection_20261029/`, không đưa lên Git) lưu bằng chứng.

## Bước tiếp theo

- Pin PhoWhisper-medium revision `55a7e3eb6c906de891f8f06a107754427dd3be79`, `vi/transcribe`, greedy beam 1, tối đa 400 token; giao `text_model` khi status ok cho agent, normalization chỉ dùng scoring.
- Nghe/chấm lại 44 ca review, ưu tiên 12 ca tìm lỗi, rồi thử downstream trên các ca critical. Nếu PhoASR không kém về lỗi quan trọng và điều kiện dùng phù hợp, cân nhắc chuyển mặc định để giảm latency; không cần chạy lại toàn bộ 991 audio.
- **Chưa fine-tune hoặc thêm fallback model ngay.** Chỉ fine-tune trên development sau khi xác nhận lỗi lặp và gold. Không dùng reference/WER từng câu hoặc confidence tự bịa để kích hoạt fallback; khi lỗi kỹ thuật, trả lỗi để agent yêu cầu nói lại.
'''
(OUT/'asr_model_selection.private.md').write_text(memo,encoding='utf-8')
readme='''# Bằng chứng phân tích ASR — 29/10/2026

## Kiểm kê đầu vào

| Nguồn | Có gì / dùng để kiểm tra | Thiếu hoặc giới hạn |
|---|---|---|
| `lifemate_asr_benchmark_outputs.zip` | 100 audio, 8 active runs; predictions, reference trong per_audio, metrics, ID/config/revision/runtime, code và checksum | Có 4 thư mục run cũ không thuộc comparison_status; không gộp vào bảng chính |
| `lifemate_asr_validation_outputs.zip` | 991 audio, 8 active runs; cùng các thành phần trên | Không phải test độc lập; 100 benchmark nằm trong tập này |
| `zalo-lifemate-test-model.ipynb` | 33 cell, cấu hình chọn cả 8 model, stage validation; đọc AST/output metadata, không execute | Không dùng notebook output làm bằng chứng duy nhất |
| Annotation trong hai ZIP | Các template giọng/entity/review | Chưa điền; review mới ở error_review.csv chỉ dựa văn bản, không là gold |
| Cache/split/waveform gốc | Hash cache có trong config và khớp audit_verification.json trước đây; ID/path của output thuộc train | Không có 3 CSV gốc trong ZIP để tự tái kiểm tra development-disjoint/hash waveform; chưa nghe audio |

Quyết định và bảng ngắn: [docs/asr_model_selection.md](../../docs/asr_model_selection.md).

## Các file kết quả

| File | Nội dung |
|---|---|
| [metrics_recomputed.csv](metrics_recomputed.csv) | Benchmark, validation và 891 ID còn lại; overall/final/history/độ dài; edit counts, mẫu số, success-only, failure, tốc độ/RAM/VRAM |
| [per_audio_recomputed.csv](per_audio_recomputed.csv) | Tính lại edit counts từng ID, không chép transcript vào file phân tích |
| [paired_comparisons.csv](paired_comparisons.csv) | 28 cặp mỗi stage, CI WER/CER theo hội thoại, số câu thắng/thua/hòa, lỗi chung và bổ sung |
| [audit.json](audit.json) | SHA256 đầu vào, checksum export, ID/reference/role/config/source/resume checks, annotation và giới hạn |
| [run_manifest.json](run_manifest.json) | Đúng checkpoint/revision, generation/model config và runtime của từng active run |
| [load_warmup_sessions.csv](load_warmup_sessions.csv) | Load time, warm-up và session; benchmark 4 model đầu tái dùng 8 smoke record cùng config, validation chạy 991 record/session riêng |
| [inference_errors.csv](inference_errors.csv) | Turbo 1/991, Whisper-small 4/991 lỗi token-limit/thiếu EOS; raw text lặp chỉ giữ để chẩn đoán trong ZIP gốc |
| [text_diagnostics_summary.csv](text_diagnostics_summary.csv), [cases](text_diagnostic_cases.csv) | Cờ ký tự ngoài Latin, rỗng, độ dài và lặp; đây là gợi ý review, không phải hallucination/critical error gold |
| [review_panel_ids.csv](review_panel_ids.csv), [error_review.csv](error_review.csv) | Panel metadata seed 2026 (10 final, 22 history), manual text review 32 câu trên medium/PhoASR/Qwen1.7 và 12 ca mining; không suy tỷ lệ toàn tập từ mining |
| [sources.json](sources.json) | Nguồn license chính thức và giới hạn xác minh |

## Quy ước và kiểm tra

- Normalization: `nfc_lower_punct_symbol_to_space_keep_accents_digits_v1`; NFC/lowercase, category P/S thành khoảng trắng, gộp whitespace, giữ dấu. Không number-normalize.
- WER corpus theo khoảng trắng; CER bỏ khoảng trắng. Error/missing hypothesis rỗng; n_ok=0 không có metric nhận dạng hợp lệ. Không lấy trung bình WER từng câu.
- All/final/history lần lượt có 14.376/6.766/7.610 token reference; validation 46.494 ký tự không khoảng trắng, 4.887,05725 giây audio. Reference chuẩn hóa không rỗng.
- Mọi run chính cùng 991 ID, reference, sample_uid, role, duration và config runtime T4/FP16/batch1/eager. Hai ZIP cùng cache hash, source snapshot khớp request; fingerprint run và model/revision của prediction khớp config. Metric tính lại bằng RapidFuzz khớp summary với rtol=1e-10, atol=1e-9.
- Chọn run từ comparison_status, không gom các thư mục cũ. Benchmark có leftover scoring_patch cũ, nhưng active config/source/per-audio/metric đều được đối chiếu độc lập. Không sửa raw output.
- 100 benchmark ID đều thuộc validation; raw prediction trên phần giao giống nhau 100/100 mỗi model, latency/session khác. Không cộng thành 1.091 audio. Kết quả 891 còn lại chỉ là robustness check, không là test mới độc lập.
- Bootstrap multinomial lấy mẫu có hoàn lại theo sample_uid hội thoại, mỗi lần tính corpus metric từ tổng edits/độ dài; 10.000 mẫu, seed 2026. Có 81 cụm benchmark, 309 cụm validation. CI 95% khám phá, không điều chỉnh 28 cặp, không chứng minh tương đương khi chứa 0.
- `oracle_wer_diagnostic_only` dùng min edits theo reference chỉ để hiểu tính bổ sung, không phải hệ thống có thể triển khai. Medium/PhoASR cùng có lỗi trên 833/991 câu; các lỗi chung không đồng nghĩa cùng đúng một lỗi hoặc cùng nguyên nhân.
- Cờ short: ref >=10 token và hyp <1/2 ref. Long: ref >=4 và hyp >max(30,2,5*ref). Lặp: hyp >=20 và cùng trigram lặp >=4 lần. Chưa xác minh mất đoạn/hallucination bằng audio. Non-Latin đếm ký tự alphabetic không có Unicode name bắt đầu LATIN, không phải bộ nhận dạng ngôn ngữ.
- Latency p50/p95 và weighted RTF chỉ trên ca thành công; gồm đọc file, preprocessing, chuyển GPU, generate/forward, decode và synchronize. Load/download/warm-up tách riêng. RSS là peak process lấy mẫu 10ms, có thể bỏ sót peak và chịu ảnh hưởng kernel/chạy tuần tự; VRAM là PyTorch allocated/reserved, không phải toàn máy.

## Tái lập

Từ gốc project, Python cần NumPy, Pandas, RapidFuzz. Không cần torch, GPU, weights hay dataset audio để phân tích:

```powershell
python -m lifemate.asr.tools.analyze_asr_outputs --benchmark D:\\private-asr\\lifemate_asr_benchmark_outputs.zip --validation D:\\private-asr\\lifemate_asr_validation_outputs.zip --notebook D:\\private-asr\\zalo-lifemate-test-model.ipynb --output src/lifemate/asr/analysis/asr_selection_20261029
python tools/write_asr_selection_report.py
python -m unittest discover -s src/lifemate/asr/tests -v
```

Script thứ nhất chỉ đọc các file được chỉ rõ, chặn environment/state trước đọc, không execute code notebook, không gửi dữ liệu ra ngoài. `error_review.csv` là review thủ công đã điền, không bị script phân tích ghi đè; tái chạy số học không thay thế review/annotation. Script thứ hai tạo memo từ CSV, không sửa predictions.

Thư mục này không chứa audio hoặc transcript đầy đủ, nhưng có ID nội bộ và ghi chú từ dữ liệu cuộc thi: giữ trong đội, không publish cùng dataset.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
decision=dict(model_id='vinai/PhoWhisper-medium',revision='55a7e3eb6c906de891f8f06a107754427dd3be79',
    confidence='moderate_engineering_choice_not_proven_accuracy_winner',
    scope='5 preselected models, with all 8 actual exported models compared',
    closest_challenger='Qualcomm-AI-Research/PhoASR-whisper-small',
    challenger_wer_advantage_pp=float(-pair.delta_wer_pp),
    recommendation='Integrate PhoWhisper-medium now; listen/adjudicate critical cases and confirm use terms before deciding whether PhoASR replaces it.',
    fine_tuning_now=False,automatic_asr_fallback_recommended=False)
(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Team memo whitespace word count:',len(memo.split()))
