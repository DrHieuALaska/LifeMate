"""Build the standalone Kaggle notebook from the reviewed Python module."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
from lifemate.asr import asr_benchmark as b

source=(ROOT/'asr_benchmark.py').read_text(encoding='utf-8')
cells=[]

def md(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))

def code(role,text,hidden=False):
    cells.append(dict(cell_type='code',execution_count=None,outputs=[],
        metadata=dict(lifemate_role=role,**({'jupyter':{'source_hidden':True}} if hidden else {})),
        source=text.splitlines(keepends=True)))

md('''# LifeMate — chọn model ASR phù hợp
**Đã điền ba input của bạn. Bật GPU + Internet (nếu luật cho phép), rồi Run All trong kernel mới.**
Mặc định: smoke 8 audio với 5 ứng viên chính, sau đó đổi `STAGE='benchmark'` để so cùng 100 audio. Chưa có kết quả chứng minh model nào thắng trên LifeMate.

| Bước | Chức năng |
|---|---|
| 1–3 | Cài thư viện, cấu hình, nạp mã dùng chung |
| 4–6 | Kiểm tra cache/audio, GPU, chuẩn bị lượt chạy |
| 7 | **PhoWhisper-medium**: ứng viên chuyên biệt tiếng Việt |
| 8 | **Whisper-large-v3-turbo**: chất lượng/tốc độ |
| 9–10 | **Qwen3-ASR-0.6B / 1.7B-hf**: so chất lượng và chi phí hai cỡ |
| 11 | **wav2vec2 tiếng Việt**: ứng viên CTC cỡ nhỏ |
| 12–14 | Whisper/PhoWhisper-small và PhoASR-small: tùy chọn, mặc định bỏ qua |
| 15–16 | Bảng chọn model và ZIP |

Giữ notebook/output **private** vì chứa transcript. Không đọc environment/tool labels, không gửi audio ra API. Reference chỉ dùng chấm điểm.
''')
md('## 1. Cài thư viện\nChạy trước khi import model. Giữ nguyên torch/CUDA của Kaggle; offline thì đặt `INSTALL=False` và chuẩn bị packages trước.\n')
code('install','''INSTALL = True
if INSTALL:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        '--upgrade-strategy','only-if-needed',
        'transformers==5.14.1', 'huggingface-hub==1.24.0', 'tokenizers==0.22.2',
        'safetensors==0.8.0', 'soundfile==0.13.1', 'psutil==7.1.0','librosa==0.11.0'])
import subprocess, sys
DEPENDENCY_CHECK=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True).stdout
print(DEPENDENCY_CHECK or 'pip check: không có nội dung báo cáo')
# Nếu đã import các thư viện này trước khi cài: restart kernel rồi Run All.
''')
md('## 2. Cấu hình\nBa đường dẫn đã điền. Chỉ sửa nếu tên mount thay đổi. `benchmark` = 100 audio; `smoke` = 8; `validation` = 991 nếu cache hợp lệ.\n')
code('config','''from pathlib import Path
DATASET_ROOT = Path('/kaggle/input/datasets/jamel27/zalo-aichallange-lifemate-dataset')
AUDIT_INPUT = Path('/kaggle/input/notebooks/jamel27/zalo-lifemate-audit')
FOLLOWUP_INPUT = Path('/kaggle/input/notebooks/jamel27/zalo-lifemate-audit-follow-up')
CACHE_DIR = None  # Tự lấy CSV trong FOLLOWUP_INPUT/lifemate_asr_followup hoặc ZIP follow-up.
DATASET_ZIP = None  # Chỉ đặt khi ZIP dataset có tên khác train+public_test.zip hoặc data.zip.
WORK_DIR = Path('/kaggle/working/lifemate_asr_inputs')  # Cache/audio chọn lọc; không đưa vào ZIP kết quả.
OUTPUT_DIR = Path('/kaggle/working/lifemate_asr_benchmark')
RESUME_DIR = None  # Output private cũ; lần đầu giữ None. Copy xong tự đặt lại None.
STAGE = 'smoke'  # Đạt smoke trước, rồi đổi benchmark (100) / validation (991).
DEVICE = 'cuda:0'
PRECISION = 'float16'  # CPU: DEVICE='cpu', PRECISION='float32'
OFFLINE = False
RUN_INFERENCE = True
RETRY_ERRORS = False
MAX_NEW_TOKENS = 400
MAX_LATENCY_P95_MS = None  # Điền SLA triển khai nếu đã biết; không tự đặt ngưỡng.
MAX_VRAM_GB = None  # Giới hạn PyTorch peak reserved trên máy đích, không phải tổng VRAM hệ thống.
CANDIDATES = [
    dict(name='phowhisper_medium',model_id='vinai/PhoWhisper-medium',family='seq2seq',revision='main',local_path=None),
    dict(name='whisper_turbo',model_id='openai/whisper-large-v3-turbo',family='seq2seq',revision='main',local_path=None),
    dict(name='qwen3_asr_06b',model_id='Qwen/Qwen3-ASR-0.6B-hf',family='qwen3_asr',revision='main',local_path=None,max_new_tokens=512),
    dict(name='qwen3_asr_17b',model_id='Qwen/Qwen3-ASR-1.7B-hf',family='qwen3_asr',revision='main',local_path=None,max_new_tokens=512),
    dict(name='whisper_small',model_id='openai/whisper-small',family='seq2seq',revision='main',local_path=None),
    dict(name='phowhisper_small',model_id='vinai/PhoWhisper-small',family='seq2seq',revision='main',local_path=None),
    dict(name='wav2vec2_vi',model_id='nguyenvulebinh/wav2vec2-base-vietnamese-250h',family='ctc',revision='main',local_path=None),
    dict(name='phoasr_small',model_id='Qualcomm-AI-Research/PhoASR-whisper-small',family='seq2seq',revision='main',local_path=None),
]
SELECTED_MODELS = ['phowhisper_medium','whisper_turbo','qwen3_asr_06b','qwen3_asr_17b','wav2vec2_vi']
# Bật model tùy chọn bằng cách thêm đúng name vào SELECTED_MODELS (đọc license trước khi chốt).
# Offline: mount đủ checkpoint/processor/tokenizer/config/weights, đặt local_path và OFFLINE=True.
''')
md('## 3. Nạp mã dùng chung\nChỉ định nghĩa xử lý audio, scoring, resume và ZIP; **chưa tải/chạy model**. Cell dài có thể thu gọn.\n')
code('implementation',f'BENCHMARK_SOURCE = {source!r}\nexec(compile(BENCHMARK_SOURCE,"asr_benchmark.py","exec"))\nBENCHMARK_SOURCE_SHA256 = {hashlib.sha256(source.encode()).hexdigest()!r}\nprint("Pipeline:", CODE_VERSION)\n',hidden=True)
md('## 4. Kiểm tra input và chọn audio\nƯu tiên cache follow-up đã sửa; không lấy metadata cũ từ audit. Nếu chỉ có ZIP, lấy đúng CSV/audio cần dùng, không giải nén environment hoặc toàn dataset.\n')
code('inputs','''from IPython.display import display
CACHE_DIR, cache_provenance = resolve_followup_cache(FOLLOWUP_INPUT,WORK_DIR,CACHE_DIR)
references, joined, benchmark, cache_hashes = load_cache(CACHE_DIR)
rows = select_rows(joined,benchmark,STAGE)
print('Cache đang dùng:',CACHE_DIR)
print('Audit cũ:',AUDIT_INPUT,'— mount:',AUDIT_INPUT.is_dir(),'(không dùng thay follow-up)')
print('Tập đã chọn:',STAGE,len(rows),rows.kind.value_counts().to_dict())
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
if RESUME_DIR is not None:
    if not Path(RESUME_DIR).is_dir(): raise FileNotFoundError(RESUME_DIR)
    if any(OUTPUT_DIR.iterdir()): raise RuntimeError('Resume cần OUTPUT_DIR trống để tránh ghi đè.')
    shutil.copytree(RESUME_DIR,OUTPUT_DIR,dirs_exist_ok=True)
    RESUME_DIR=None
if RUN_INFERENCE:
    AUDIO_ROOT, audio_provenance = prepare_selected_audio(rows,DATASET_ROOT,WORK_DIR,DATASET_ZIP)
else:
    AUDIO_ROOT=DATASET_ROOT
    audio_provenance=dict(mode='not_checked_scoring_only')
write_json(OUTPUT_DIR/'input_provenance.json',dict(cache=cache_provenance,audio=audio_provenance,
    original_audit_mount=str(AUDIT_INPUT),followup_mount=str(FOLLOWUP_INPUT)))
(OUTPUT_DIR/'dependency_check.txt').write_text(globals().get('DEPENDENCY_CHECK','Không chạy cell pip check.'),encoding='utf-8')
rows[['ref_uid','sample_uid','kind','duration']].to_csv(OUTPUT_DIR/('selection_'+STAGE+'.csv'),index=False)
annotation_templates(rows,OUTPUT_DIR)
print('Audio root:',AUDIO_ROOT,'—',audio_provenance['mode'])
print('Input hợp lệ. Giữ nguyên split và ID cho mọi model.')
''')
md('## 5. Kiểm tra GPU và thư viện\nDừng sớm nếu thiếu package/GPU; chưa tải checkpoint.\n')
code('runtime','''if RUN_INFERENCE:
    import torch
    import transformers
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC, Qwen3ASRForConditionalGeneration, AutoProcessor
    import soundfile, psutil, huggingface_hub
    if DEVICE.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('Bật GPU trong Settings của Kaggle rồi chạy lại.')
        torch.cuda.get_device_properties(torch.device(DEVICE))
    elif DEVICE!='cpu': raise ValueError('DEVICE phải là cuda:0 hoặc cpu.')
    if DEVICE=='cpu' and PRECISION!='float32': raise ValueError('CPU phải dùng PRECISION=float32.')
    if PRECISION not in ['float16','float32']: raise ValueError('Precision chưa được hỗ trợ.')
    if transformers.__version__!='5.14.1': raise RuntimeError('Bản này cần Transformers 5.14.1 để chạy Qwen native. Chạy cell cài trong kernel mới.')
    env=runtime_info()
    write_json(OUTPUT_DIR/'runtime.json',env)
    print('GPU:',env['gpu'],'| torch:',env['versions']['torch'],'| transformers:',env['versions']['transformers'])
else:
    print('Chỉ chấm lại: không cần GPU hoặc tải checkpoint.')
''')
md('## 6. Chuẩn bị lượt chạy\nLưu config, code và ID; mỗi model dùng cùng audio. Tự ghi lại bảng kết quả sau từng model để giữ tiến độ.\n')
code('plan','''cfg=dict(output_dir=str(OUTPUT_DIR),dataset_root=str(AUDIO_ROOT),stage=STAGE,
    device=DEVICE,precision=PRECISION,offline=OFFLINE,retry_errors=RETRY_ERRORS,
    max_new_tokens=MAX_NEW_TOKENS,source_sha256=BENCHMARK_SOURCE_SHA256)
selected=[c for c in CANDIDATES if c['name'] in SELECTED_MODELS]
if not selected or len(selected)!=len(SELECTED_MODELS): raise ValueError('SELECTED_MODELS có tên trùng/lạ hoặc rỗng.')
write_json(OUTPUT_DIR/'benchmark_request.json',dict(config=cfg,candidates=selected,
    selected_ref_uids=rows.ref_uid.tolist(),cache_hashes=cache_hashes,
    expected_audio_per_model=len(rows),normalization=NORMALIZATION,
    selection_constraints=dict(max_latency_p95_ms=MAX_LATENCY_P95_MS,max_vram_gb=MAX_VRAM_GB)))
(OUTPUT_DIR/'benchmark_source.py').write_text(BENCHMARK_SOURCE,encoding='utf-8')
run_dirs=[]
def refresh_results():
    global summary,scored,comparison_status
    summary,scored=evaluate(rows,OUTPUT_DIR,run_dirs=run_dirs)
    paired_bootstrap(scored,OUTPUT_DIR/'paired_bootstrap.csv')
    make_review(scored,OUTPUT_DIR)
    selection_table(summary,scored,OUTPUT_DIR/'model_selection.csv',MAX_LATENCY_P95_MS,MAX_VRAM_GB)
    status=[]
    for spec in selected:
        matches=[d for d in set(run_dirs) if json.loads((d/'config.json').read_text())['model_id']==spec['model_id']]
        if not matches:
            status.append(dict(model=spec['name'],model_id=spec['model_id'],run='',stage=STAGE,
                n_expected=len(rows),n_ok=0,n_error=0,n_missing=len(rows),state='not_run_or_checkpoint_error'))
        for d in matches:
            stats=aggregate(scored[d.name])
            state='ready' if stats['n_ok']==len(rows) else ('incomplete' if stats['n_missing'] else 'completed_with_errors')
            status.append(dict(model=spec['name'],model_id=spec['model_id'],run=d.name,stage=STAGE,
                n_expected=len(rows),n_ok=stats['n_ok'],n_error=stats['n_error'],n_missing=stats['n_missing'],state=state))
    comparison_status=pd.DataFrame(status)
    comparison_status.to_csv(OUTPUT_DIR/'comparison_status.csv',index=False)

def run_one(name):
    global run_dirs
    specs=[s for s in selected if s['name']==name]
    if not specs:
        print('Bỏ qua model không nằm trong SELECTED_MODELS:',name)
        return
    print('MODEL:',specs[0]['model_id'],'| stage:',STAGE,'| audio:',len(rows))
    if RUN_INFERENCE:
        current=run_benchmark(rows,specs,cfg,cache_hashes)
    else:
        current=[]
        for d in OUTPUT_DIR.iterdir():
            if not d.is_dir() or not (d/'config.json').is_file(): continue
            saved=json.loads((d/'config.json').read_text())
            if (saved.get('model_id')==specs[0]['model_id'] and saved.get('source_sha256')==cfg['source_sha256']
                and saved.get('cache_hashes')==cache_hashes): current.append(d)
    run_dirs=list(dict.fromkeys(run_dirs+current))
    refresh_results()
    display(comparison_status[comparison_status.model==name])
refresh_results()
''')
md('## 7. PhoWhisper-medium — ứng viên chính\n`vinai/PhoWhisper-medium`: chuyên biệt tiếng Việt, tăng năng lực so với small; greedy `vi/transcribe`.\n')
code('model_phowhisper_medium',"run_one('phowhisper_medium')\n")
md('## 8. Whisper-large-v3-turbo — ứng viên chính\n`openai/whisper-large-v3-turbo`: thử cân bằng chất lượng/tốc độ, không mặc định tốt nhất.\n')
code('model_turbo',"run_one('whisper_turbo')\n")
md('## 9. Qwen3-ASR-0.6B — ứng viên chính\n`Qwen/Qwen3-ASR-0.6B-hf`: runtime native Transformers, tiếng Việt cố định, không context/hotword; tối đa 512 token. Dùng FP16 trên T4.\n')
code('model_qwen06',"run_one('qwen3_asr_06b')\n")
md('## 10. Qwen3-ASR-1.7B — ứng viên chính\n`Qwen/Qwen3-ASR-1.7B-hf`: so lợi ích chất lượng với chi phí của bản 0.6B trên cùng dữ liệu. Chưa đo VRAM/T4; OOM phải ghi lỗi.\n')
code('model_qwen17',"run_one('qwen3_asr_17b')\n")
md('## 11. wav2vec2 tiếng Việt — ứng viên chính\n`nguyenvulebinh/wav2vec2-base-vietnamese-250h`: so CTC cỡ nhỏ, greedy không LM ngoài. Giữ đủ audio >10 s và ghi cảnh báo; kiểm tra license CC-BY-NC-4.0 trước khi chốt triển khai.\n')
code('model_ctc',"run_one('wav2vec2_vi')\n")
md('## 12. Whisper-small — tùy chọn\nMặc định bỏ qua; thêm `whisper_small` vào SELECTED_MODELS nếu muốn đối chiếu kết quả cũ.\n')
code('model_whisper',"run_one('whisper_small')\n")
md('## 13. PhoWhisper-small — tùy chọn\nMặc định bỏ qua; thêm `phowhisper_small` nếu muốn so chi phí của small/medium.\n')
code('model_phowhisper',"run_one('phowhisper_small')\n")
md('## 14. PhoASR-whisper-small — tùy chọn\n`Qualcomm-AI-Research/PhoASR-whisper-small`: model Việt mới, có hạn chế giọng miền Trung và license riêng. Thêm `phoasr_small` nếu quyền sử dụng phù hợp.\n')
code('model_phoasr',"run_one('phoasr_small')\n")
md('''## 15. So sánh để chọn model
`ready` = đủ prediction thành công, chưa phải kết luận model tốt. WER/CER càng thấp càng tốt; vẫn cần review tên/số/phủ định. Lỗi được chấm như câu rỗng; `n_ok=0` không có WER hợp lệ. CER bỏ khoảng trắng, giữ dấu; không đổi số chữ thành chữ số.
Bootstrap lấy mẫu theo hội thoại; CI chứa 0 chưa đủ kết luận thắng. Latency gồm đọc audio đến decode, tách load/warm-up; batch 1. RAM là RSS lấy mẫu, VRAM là PyTorch peak allocated/reserved, không phải toàn hệ thống.
`model_selection.csv`: loại run thiếu/lỗi hoặc vượt giới hạn bạn đặt, đánh dấu nhóm không bị model khác vượt cả WER lẫn latency. Chưa tự chọn người thắng: cần validation đầy đủ và review tên/số/phủ định. Model bị load lỗi không có dòng metric, xem comparison_status/model_errors.
''')
code('comparison','''refresh_results()
display(comparison_status)
if not summary.empty:
    columns=['run','n','n_ok','n_error','n_missing','wer','cer_no_spaces','failure_rate',
             'latency_p50_ms','latency_p95_ms','rtf_weighted','rss_peak_bytes','vram_peak_allocated_bytes','vram_peak_reserved_bytes']
    display(summary.loc[summary.subset=='all',columns])
else:
    print('Chưa có run để chấm. Xem model_errors.jsonl hoặc chạy các cell model.')
selection=selection_table(summary,scored,OUTPUT_DIR/'model_selection.csv',MAX_LATENCY_P95_MS,MAX_VRAM_GB)
display(selection)
print('Smoke chỉ kiểm tra pipeline. Chạy benchmark 100, rồi validation 991 cho ứng viên cạnh tranh để chọn model.')
print('Chưa đủ model vẫn tải ZIP để phân tích lỗi; n_missing không phải kết quả thành công.')
''')
md('''## 16. Tạo ZIP và tải về
Chạy riêng cell này được nếu output còn trên đĩa. ZIP gồm prediction/reference, metrics, config/runtime, code, lỗi, annotation và checksum; loại audio, environment và weights. **Giữ private và gửi lại ZIP để phân tích.**
Annotation là biểu mẫu, chưa phải nhãn thật. Nghe trước khi xem prediction, dùng `uncertain` nếu chưa chắc; không suy vùng miền từ transcript. ASR không tự suy người nhận/thời gian/tool call.
''')
code('export','''from pathlib import Path
from IPython.display import FileLink, display
import os
zip_source=Path(globals().get('OUTPUT_DIR','/kaggle/working/lifemate_asr_benchmark'))
'''+inspect.getsource(b.package_outputs)+'''
zip_path=package_outputs(zip_source,zip_source.parent/'lifemate_asr_benchmark_outputs')
print('Tải ZIP này, giữ private:',zip_path)
display(FileLink(os.path.relpath(zip_path,Path.cwd())))
''')
for i,c in enumerate(cells): c['id']=f'lifemate-{i:02d}'
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
(ROOT/'notebook/lifemate_asr_benchmark.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Built lifemate_asr_benchmark.ipynb:',len(cells),'cells')
