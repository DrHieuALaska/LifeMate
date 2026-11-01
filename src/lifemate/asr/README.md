# LifeMate ASR

Luồng hiện tại: **fine-tune → chọn best → nghe review valid → tích hợp inference**.
Web chạy trên máy bằng Python, không cần GPU, Node, API key hay upload audio lên dịch vụ ngoài.

## Bắt đầu review hai model

1. Import [notebook xuất predictions valid](notebook/zalo_lifemate_export_valid_review.ipynb) lên Kaggle.
   Add Input hai output continuation mới nhất của hynt và Sherpa, chọn CPU.
   Đường dẫn đã điền theo các run đã báo cáo; có thể điền thư mục mount ngoài cùng.
2. Save Version → **Save & Run All**, tải `finetune_valid_review.zip`.
   Notebook chọn best theo EM rồi WER và tính lại điểm từ đủ 504 predictions.
3. Tại gốc repo chạy:

```powershell
.\src\lifemate\asr\Start-Review.ps1
```

4. Mở **http://127.0.0.1:8765**, bấm **Nạp ZIP review**, chọn ZIP vừa tải.

ZIP audio local mặc định: `src/lifemate/asr/data/train+public_test.zip`.
Nếu dữ liệu ở nơi khác:

```powershell
.\src\lifemate\asr\Start-Review.ps1 -Dataset "D:\private\train+public_test.zip" -Bundle "$env:USERPROFILE\Downloads\finetune_valid_review.zip"
```

Nếu PowerShell chặn script, dùng Python trực tiếp (không cần đổi execution policy):

```powershell
python src/lifemate/asr/review_web/server.py
```

Server chỉ bind `127.0.0.1`. Ctrl+C để dừng; `-Port 8766` / `--port 8766` để đổi cổng.
ZIP `lifemate_finetune_handoff.zip` chỉ có config/source, không có predictions nên không thay thế ZIP review.

## Trên web

- Nghe audio và xem reference / hynt / Sherpa cạnh nhau; chọn tốc độ phát.
- Lọc câu cần đối chiếu, model khác nhau, đã/chưa nghe, final/history, số và gợi ý từ tiếng Anh.
- Đánh giá riêng mỗi model: đúng, khác chữ nhưng giữ ý, sai đổi ý nghĩa hoặc không chắc.
- Ghi lỗi tên riêng, số tiền, giờ, phủ định, thiếu/thêm từ, reference hoặc âm thanh.
- Đề xuất sửa reference riêng; giữ nguyên dữ liệu và điểm tự động gốc.
- Ghi chú tự lưu trong `analysis/finetune_review_web/` (ignored). Xuất **JSON** để sao lưu/nhập lại,
  xuất **CSV** để tổng hợp. Backup chỉ nhập vào đúng bộ predictions.
- Mở lại server sẽ nạp bộ review gần nhất. Chỉ phát WAV có SHA256 khớp output.
- Phím ← / → chuyển câu khi không ở ô nhập liệu.

Đây là **validation đã dùng chọn checkpoint**, không phải test độc lập. Màu vàng chỉ thể hiện
khác biệt văn bản, không tự kết luận đúng nghĩa, giọng hay mức độ nguy hiểm của lỗi.

## Notebook đang dùng

| Việc | Notebook |
|---|---|
| Xuất predictions cho web | [export valid review](notebook/zalo_lifemate_export_valid_review.ipynb) |
| Bảng điểm từng epoch | [fine-tune review](notebook/zalo_lifemate_finetune_review.ipynb) |
| ZIP config/source/tokenizer nhỏ | [handoff ZIP](notebook/zalo_lifemate_finetune_handoff_zip.ipynb) |
| Train tiếp hynt | [hynt continuation](notebook/zalo_lifemate_hynt_continue_finetune.ipynb) |
| Train tiếp Sherpa | [Sherpa continuation](notebook/zalo_lifemate_sherpa_vi_continue_finetune.ipynb) |
| Train mới hynt 5 epoch | [hynt fresh](notebook/zalo_lifemate_hynt_zipformer_finetune_fresh_5epoch.ipynb) |
| Train mới nhánh Sherpa | [Sherpa fresh](notebook/zalo_lifemate_sherpa_vi_68m_main_finetune.ipynb) |

Notebook pilot, bản sửa/resume cũ, benchmark 991 và ZIP toàn checkpoint được giữ trong
[notebook/archive](notebook/archive/README.md). Không dùng chúng thay notebook review valid mới.

## Cấu trúc

```text
asr/
├── base.py, team_data.py                # Interface ASR và manifest nhóm
├── asr_benchmark.py                     # Benchmark/scoring lịch sử
├── Start-Review.ps1                     # Chạy web local
├── review_web/                         # Server + giao diện HTML/CSS/JS, không CDN
├── notebook/                           # Luồng fine-tune/review hiện tại
│   └── archive/                        # Notebook lịch sử, giữ để tái lập
├── tools/                              # Builder/trainer chỉ giữ local
├── tests/                              # Kiểm tra chỉ giữ local
├── docs/                               # Tài liệu model; archive/ giữ hướng dẫn cũ
├── data/                               # Private dataset ZIP (ignored)
└── analysis/                           # Private metrics/review/ghi chú (ignored)
```

## Tích hợp và bằng chứng

CSV tổng hợp cho đồng đội đối chiếu ở [docs/results](docs/results/README.md).
Không đưa các CSV có transcript/UID/audio vào Git.

`base.py` định nghĩa `ASR.transcribe(Path) -> str`; `team_data.py` dùng manifest dev
193 mẫu của nhóm. Tập này khác validation fine-tune 504 audio; cần audit overlap trước khi
diễn giải điểm end-to-end. Web chỉ đọc predictions có sẵn, chưa chạy inference/export ONNX.

Các nghiên cứu pretrained/991 trong [model selection](docs/asr_model_selection.md) và
[research report](docs/research_report.md) là bằng chứng lịch sử, không thay số sau fine-tune.
[Hướng dẫn benchmark cũ](docs/archive/legacy_benchmark_workflow.md) giữ nội dung trước khi dọn thư mục.

Không xóa hay thay source data, checkpoint, split hoặc kết quả cũ. Giữ output competition private;
không commit audio, predictions, manifests, labels, weights, logs hoặc secrets.

## Kiểm tra

```powershell
python -m pytest src/lifemate/asr/tests/test_finetune_review_web.py -q
node --check src/lifemate/asr/review_web/app.js
```

Tests dùng fixture, không thay bằng chứng nghe người thật hay inference checkpoint thực tế.
