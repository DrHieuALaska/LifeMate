# LifeMate ASR

ASR runtime, benchmark, notebook và tài liệu nằm cùng trong thư mục này.

```text
asr/
├── base.py, phowhisper.py      # Interface nhóm và adapter local
├── team_data.py               # Đọc dev manifest/preprocess của nhóm
├── asr_benchmark.py           # Benchmark model và scoring
├── notebook/                 # Một notebook benchmark standalone
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

- [Quyết định chọn PhoWhisper-medium](docs/asr_model_selection.md).
- [Nghiên cứu và giao thức](docs/research_report.md).
- [Notebook benchmark](notebook/lifemate_asr_benchmark.ipynb).
- [Phân tích output private](analysis/README.md).

Notebook giữ thí nghiệm lịch sử **991 audio** với cache follow-up riêng gồm
`references_enriched.csv`, `split_manifest.csv`, `benchmark_manifest.csv`.
Import notebook lên Kaggle private, chỉnh input/cache rồi chạy smoke trước,
sau đó benchmark/validation. Không dùng cache 991 audio thay cho **193 sample
dev của nhóm** hoặc so trực tiếp điểm giữa hai tập. Notebook là source-only;
output chạy thật phải giữ private. Đã bỏ các notebook audit cũ vì repo nhóm
có pipeline dữ liệu riêng.

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
