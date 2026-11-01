# Hướng dẫn ASR trước khi dọn folder

Lưu nguyên nội dung README trước ngày 01/11/2026 để bảo toàn các ghi chú/thay đổi có sẵn.
Các đường dẫn và hướng dẫn builder dưới đây thuộc bố cục cũ; xem [README hiện tại](../../README.md) để chạy.
Không diễn giải các số benchmark pretrained/991 như kết quả fine-tune trên valid 504.

# LifeMate ASR

ASR runtime, benchmark, notebook và tài liệu nằm cùng trong thư mục này.

```text
asr/
├── base.py, phowhisper.py      # Interface nhóm và adapter local
├── team_data.py               # Đọc dev manifest/preprocess của nhóm
├── asr_benchmark.py           # Benchmark model và scoring
├── notebook/                 # Các notebook so sánh và chạy full 991
├── docs/                     # Quyết định model và nghiên cứu
├── tools/                    # Build notebook, phân tích output
├── tests/                    # Tests tổng hợp, không cần GPU
└── analysis/                 # Hướng dẫn + kết quả riêng được ignore
```

## Dùng dữ liệu/preprocess của nhóm

Repo lớn đã có `lifemate.exploration`, bộ đọc manifest và pipeline evaluation.
Lấy dataset/preprocess private qua kênh nhóm; GitHub không chứa dữ liệu hoặc
các manifest đã chốt. Đặt bộ dữ liệu tại `train+public_test/` ở gốc repo,
giữ `reports/eda/validation_split/{train,dev,holdout}_ids.txt` của nhóm.
Không chạy lại EDA để thay split đã chốt.

```python
from pathlib import Path
from lifemate.asr import PhoWhisperASR
from lifemate.asr.team_data import load_dev_audio

root = Path("train+public_test")
rows = load_dev_audio(root, root / "reports/eda/validation_split/dev_ids.txt")
asr = PhoWhisperASR(Path("src/lifemate/asr/models/phowhisper-medium"))
for row in rows:
    transcript = asr.transcribe(row.audio_path)
    # Giao transcript cho agent; row.reference chỉ dùng chấm offline.
```

Loader tái dùng `read_ids` và path guard của repo chung, giữ thứ tự **193 dev
sample**, lấy audio hiện tại và history user/human. Không dùng tool labels,
không truy cập environment/public-test, không tự chia lại dữ liệu. Thiếu file
hoặc reference thì dừng để sửa nguồn. Reference không đi vào inference.

Cài runtime bằng `python -m pip install -e ".[asr]"`. Chuẩn bị checkpoint local
`vinai/PhoWhisper-medium` revision `55a7e3eb6c906de891f8f06a107754427dd3be79`;
người cấp checkpoint xác minh revision. Adapter không tự tải weights hoặc gửi
audio ra API. Load một lần, mặc định CPU FP32; GPU dùng `device="cuda:0"`.
Audio cần mono 16 kHz và tối đa 30 giây; chưa resample/chia đoạn tự động.
`transcribe` trả raw text; lỗi inference trả `ASRError`, lỗi file/load model
được caller xử lý. Không dùng text dở dang hoặc tự bật fallback.

## Benchmark cũ và tài liệu

- [Quyết định chọn model và ưu tiên fine-tune](docs/asr_model_selection.md).
- [Nghiên cứu và giao thức](docs/research_report.md).
- [Chạy thẳng 991 audio: PhoWhisper-large và Qwen](notebook/zalo_lifemate_full_test.ipynb).
- [Chạy thẳng 991 audio: omniASR](notebook/zalo_lifemate_omni_full_test.ipynb).
- [Chạy thẳng 991 audio: IchigoWhisper và Sherpa dưới 2B](notebook/zalo_lifemate_ichigo_sherpa_under2b.ipynb).
- [Chạy thẳng 991 audio: Zipformer tiếng Việt 153M và 291M](notebook/zalo_lifemate_zipformer_153m_291m_full_test.ipynb).
- [Chạy lại riêng Zipformer 291M sau lỗi vocabulary](notebook/zalo_lifemate_zipformer_291m_fixed_full_test.ipynb).
- [Chạy thẳng 991 audio: Cohere Transcribe, lượt thử riêng 2,066B](notebook/zalo_lifemate_cohere_full_test.ipynb).
- [Phân tích output private](analysis/README.md).

Notebook giữ thí nghiệm lịch sử **991 audio** với cache follow-up riêng gồm
`references_enriched.csv`, `split_manifest.csv`, `benchmark_manifest.csv`.
Import notebook lên Kaggle private, chỉnh input/cache rồi chạy smoke trước,
sau đó benchmark/validation nếu dùng notebook lịch sử. Các bản full 991 ở trên
chạy một lượt Run All, không cần chọn stage. Không dùng cache 991 audio thay cho **193 sample
dev của nhóm** hoặc so trực tiếp điểm giữa hai tập. Notebook là source-only;
output chạy thật phải giữ private. Đã bỏ các notebook audit cũ vì repo nhóm
có pipeline dữ liệu riêng.

Bản dưới 2B tách runtime tự động bằng uv: Ichigo dùng Python 3.11/PyTorch 2.5.1
và GPU, Sherpa dùng Python 3.11/ONNX trên CPU 2 threads. Chỉ so chất lượng trên
cùng ID/reference; không xem latency CPU/GPU là phép so cùng phần cứng.
Ichigo được đếm cả Whisper-medium nền và floating buffers; Sherpa đếm tensor
trong ONNX làm bound bảo thủ, không gọi là số trainable chính xác. Worker dừng
trước inference nếu count/bound >=2.000.000.000. Cohere Transcribe bị loại vì
[metadata chính thức](https://huggingface.co/api/models/CohereLabs/cohere-transcribe-03-2026)
ghi 2.065.804.048 tham số; quantization không làm giảm số tham số.
Kết quả Ichigo/Sherpa đã được cập nhật trong hai tài liệu ở trên;
không suy chất lượng mọi biến thể từ một cấu hình đã đo.

Bản Zipformer 153M/291M chạy lần lượt `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h`
và `hataphu/zipformer-k2-rnn-lm-vi`, FP32 ONNX, CPU 2 threads, greedy,
không RNNLM/hotword/punctuation model, giữ cùng 991 ID/scorer với bản 68M.
GPU không bắt buộc; bật Internet, thêm hai input cũ rồi Run All.
File `config.json` của hynt thực tế là bảng token symbol/id, worker đọc như text.
Hataphu cần chấp nhận quyền tải trên Hugging Face, bật Kaggle Secret `HF_TOKEN`;
token chỉ dùng ở model gated, không ghi vào ZIP. Nếu thiếu quyền, kết quả model
public trước đó vẫn được xuất. Quy mô 153M/291M theo card, worker ghi bound
tensor thực tế và dừng nếu >=2B. ZIP mới ghi **153M: 991/991 OK, WER 15,34%,
CER 10,65%**, thấp nhất trong 12 checkpoint/cấu hình có prediction đã đo.
Đã đối chiếu 4 ZIP của tập 991: cùng ID/reference/hội thoại/kind/duration;
đây vẫn là development validation cũ, không phải test độc lập.
153M có card CC-BY-NC-ND-4.0, chưa chốt quyền fine-tune/triển khai;
68M giữ vai trò tích hợp nhẹ và nhánh xác minh fine-tune, large + LoRA dự phòng.

291M đã tải checkpoint nhưng lỗi validator token trước inference (**0 OK/991 missing**),
chưa có WER/CER hợp lệ. Dùng notebook sửa riêng ở trên, phiên Kaggle mới,
hai input như cũ và secret `HF_TOKEN`; giữ `RESUME_ZIP = None`, Run All.
Bản sửa kiểm tra output ID theo metadata ONNX, giữ nguyên mapping token,
dùng `tokens.txt` ở root và output mới; không cần chạy lại 153M. Kiểm tra giả lập
đã qua, chưa xác nhận inference 291M thật. Chi tiết ở hai tài liệu được liên kết.

Bản Cohere chạy riêng theo yêu cầu bổ sung, không áp giới hạn dưới 2B của bản
Ichigo/Sherpa. Dùng đúng 991 ID/scorer, Python Kaggle trực tiếp, native Transformers
5.14.1, FP16, `language='vi'`, greedy, tối đa 512 token. Checkpoint yêu cầu quyền tải:
chấp nhận điều kiện tại trang model bằng tài khoản Hugging Face, rồi thêm token có
quyền đọc vào Kaggle Secrets tên `HF_TOKEN` và bật quyền secret cho notebook.
Token không đi vào source/request/config/ZIP; audio không gửi qua API ASR.
Cell quyền tải dừng sớm trước download weights nếu chưa có quyền. Chưa có kết quả
GPU Cohere trong repo để kết luận hơn/kém PhoWhisper-large.

## Nghe kiểm tra lỗi thủ công

Từ output validation private đã chạy, tạo trang nghe và danh sách folder lỗi:

```powershell
python -m lifemate.asr.tools.build_manual_review --outputs D:/private-asr/validation.zip --dataset src/lifemate/asr/data/train+public_test.zip --output src/lifemate/asr/analysis/manual_review_new
```

Mở `index.html` trong thư mục output để nghe, xem reference/model text, lọc
folder và ghi nhãn nguyên nhân/giọng sau khi nghe. Xuất CSV và sao lưu JSON
để giữ nhãn. `review_errors.csv`, `exact_matches.csv`, `folders_summary.csv`
và `folders_to_check.txt` tách ca lỗi/câu khớp; khớp là theo reference sau
chuẩn hóa, chưa phải gold đã nghe xác minh. Toàn bộ output/audio nằm trong
`analysis/` được ignore. Không suy vùng miền hoặc tỷ lệ lỗi theo giọng chỉ
từ nhóm câu sai; muốn so giọng cần một mẫu đại diện gồm cả đúng và sai.

## Kiểm tra

Tại gốc repo:

```powershell
python -m pip install -e ".[eda,asr-analysis]"
python -m lifemate.asr.tools.build_notebook
python -m unittest discover -s tests -v
python -m unittest discover -s src/lifemate/asr/tests -v
```

Tests dùng fixture/mock, không chứng minh model chạy thật hoặc đo lại WER.
`data/`, `models/`, `checkpoints/`, `artifacts/` và kết quả `analysis/` được
ignore. Các bản nghiên cứu/audit gốc vẫn giữ ở workspace ASR cũ trên máy.
