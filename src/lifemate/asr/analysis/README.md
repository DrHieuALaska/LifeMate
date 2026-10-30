# Phân tích ASR

Chỉ README này được track. Các lần phân tích, CSV/JSON, transcript, ID,
manifest và output benchmark giữ local theo [quy định nhóm](../../../../CONTRIBUTING.md).

Tính lại thí nghiệm lịch sử từ hai ZIP private, tại gốc repo:

```powershell
python -m pip install -e ".[asr-analysis]"
python -m lifemate.asr.tools.analyze_asr_outputs --benchmark D:/private-asr/benchmark.zip --validation D:/private-asr/validation.zip --notebook D:/private-asr/run.ipynb --output src/lifemate/asr/analysis/asr_selection_20261029
```

Script này phân tích các output cũ, không chạy inference. Muốn chạy trên tập
dev mới, dùng manifest/preprocess của nhóm theo [hướng dẫn ASR](../README.md).
Không coi điểm 991 audio cũ là điểm trên dev 193 sample của nhóm.
