# Notebook lịch sử

Giữ để tái lập thí nghiệm cũ; không xóa dữ liệu hoặc checkpoint.

| Notebook | Vai trò |
|---|---|
| `zalo_lifemate_hynt_zipformer_full_test.ipynb` | Benchmark pretrained ONNX trên 991 audio lịch sử |
| `zalo_lifemate_hynt_zipformer_finetune.ipynb` | Bản training/resume sửa lỗi trước các run fresh |
| `zalo_lifemate_hynt_zipformer_finetune_fresh_1epoch.ipynb` | Pilot 1 epoch |
| `zalo_lifemate_finetune_zip_outputs.ipynb` | Export toàn bộ output/checkpoint; không cần cho web review |

Luồng hiện tại ở [README ASR](../../README.md): xuất predictions valid từ hai output continuation,
nạp web local và ghi chú. Các builder liên quan đã trỏ vào thư mục archive này.
