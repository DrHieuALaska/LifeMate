# Phân tích kết quả ASR

Thư mục này giữ các lần phân tích local ngay trong module ASR. Chỉ file hướng
dẫn này được track. CSV/JSON, manifests, review có ID và báo cáo chi tiết được
ignore theo quy định [CONTRIBUTING](../../../../CONTRIBUTING.md).

Ví dụ output: `asr_selection_20261029/`. Người clone repo mới cần lấy các ZIP
private từ thành viên được phép hoặc tự chạy benchmark trên dữ liệu hợp lệ.

Từ gốc repo:

```powershell
python -m pip install -e ".[asr-analysis]"
python -m lifemate.asr.tools.analyze_asr_outputs --benchmark D:/private-asr/benchmark.zip --validation D:/private-asr/validation.zip --notebook D:/private-asr/run.ipynb --output src/lifemate/asr/analysis/asr_selection_20261029
```

Memo bổ sung cần bảng review private `error_review.csv` của lần phân tích:

```powershell
python -m lifemate.asr.tools.write_asr_selection_report
```

Script ghi memo, decision và README vào thư mục lần phân tích được ignore.
Không ghi đè tài liệu đã rà soát ở `../docs/`.
