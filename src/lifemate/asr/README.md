# ASR của LifeMate

Code inference, benchmark, tài liệu, notebook và tests ASR được giữ cùng trong
`src/lifemate/asr/`. `base.py` giữ nguyên giao thức chung của nhóm.

## Dùng với agent

Tại gốc repo, cài dependencies khi cần chạy model:

```powershell
python -m pip install -e ".[asr]"
```

Chuẩn bị checkpoint `vinai/PhoWhisper-medium` revision
`55a7e3eb6c906de891f8f06a107754427dd3be79` ở thư mục local được ignore.
Adapter không tải weights hoặc gửi audio tới API. Người chuẩn bị checkpoint
phải xác minh nguồn/revision; đường dẫn local tự nó không chứng minh revision.

```python
from pathlib import Path
from lifemate.asr import ASRError, PhoWhisperASR

asr = PhoWhisperASR(Path("src/lifemate/asr/models/phowhisper-medium"))
# GPU: thêm device="cuda:0" (mặc định FP16); CPU mặc định FP32.
try:
    transcript = asr.transcribe(Path("src/lifemate/asr/data/example.wav"))
except ASRError:
    # Controller quyết định yêu cầu người dùng nói lại.
    raise
```

Khởi tạo một lần rồi dùng lại cho audio hiện tại/history được phép. Trả chuỗi
`text_model` gốc cho `AgentInput.current_transcript`; normalization chỉ dùng
scoring. Không truyền gold transcript, tool labels hoặc trạng thái evaluator
vào ASR. Caller xử lý lỗi đọc file/load model và `ASRError`; không dùng text dở
dang làm transcript hợp lệ. Audio cần mono 16 kHz, tối đa 30 giây; bản này chưa
có resampling hoặc chia audio dài. Không sửa evaluator/agent hay tự bật fallback.

## Benchmark và tài liệu

- [Hướng dẫn benchmark Kaggle](docs/README.md).
- [Quyết định model và giới hạn bằng chứng](docs/asr_model_selection.md).
- [Nghiên cứu và giao thức](docs/research_report.md).
- [Kiểm tra local trước tích hợp](docs/local_validation.md).
- [Notebook benchmark standalone](notebook/lifemate_asr_benchmark.ipynb).
- [Audit dữ liệu](notebook/zalo-lifemate-audit.ipynb) và
  [audit follow-up](notebook/zalo-lifemate-audit-follow-up.ipynb).

Notebook trong Git chỉ chứa source, không có output. Chạy trên dữ liệu được
phép trong workspace private. Kết quả ngày 29/10 là bằng chứng lịch sử đã cung
cấp, không phải phép đo inference mới sau tích hợp.

## Kiểm tra và tạo lại notebook

Tại gốc repo:

```powershell
python -m pip install -e ".[asr-analysis]"
python -m lifemate.asr.tools.build_notebook
python -m unittest discover -s tests -v
python -m unittest discover -s src/lifemate/asr/tests -v
```

`asr-analysis` chỉ cài NumPy/Pandas/RapidFuzz, không cài torch/model.
Các tests dùng dữ liệu tổng hợp và mock, không cần GPU, weights hoặc dữ liệu
cuộc thi. Kiểm tra model thật vẫn cần thực hiện riêng trên máy đích.

## File riêng trên máy

`analysis/`, `data/`, `artifacts/`, `models/`, `checkpoints/` được ignore ngay
tại module. Chúng vẫn có thể nằm trong thư mục ASR local. Không commit audio,
transcript, manifests, predictions, benchmark output hoặc notebook có output.
Một clone mới không có những file riêng này.

Tạo lại phân tích từ hai ZIP output private (thay đường dẫn theo máy):

```powershell
python -m lifemate.asr.tools.analyze_asr_outputs --benchmark D:/private-asr/benchmark.zip --validation D:/private-asr/validation.zip --notebook D:/private-asr/run.ipynb --output src/lifemate/asr/analysis/asr_selection_20261029
python -m lifemate.asr.tools.write_asr_selection_report
```

Script memo ghi vào `analysis/asr_selection_20261029/`, không ghi đè tài liệu
đã rà soát để đưa lên Git. Giữ split/development IDs của nhóm, không tự chia
lại hoặc dùng public-test để chọn model.
