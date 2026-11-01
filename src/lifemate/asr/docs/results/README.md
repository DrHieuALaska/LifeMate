# Kết quả fine-tune để đối chiếu

Snapshot ngày 01/11/2026. Chỉ chứa số liệu tổng hợp được người dùng yêu cầu chia sẻ;
không có transcript, predictions từng audio, UID, đường dẫn audio hay weights.

- [finetune_summary.csv](finetune_summary.csv): baseline và best của hai model, tham số, trạng thái.
- [finetune_epoch_metrics.csv](finetune_epoch_metrics.csv): từng epoch, tử số/mẫu số WER/CER để kiểm tra phép tính.
- [finetune_group_metrics.csv](finetune_group_metrics.csv): baseline và best theo nhóm; các nhóm có thể giao nhau.

Nguồn: `lifemate_finetune_handoff.zip` người dùng cung cấp, đã kiểm tra CRC và SHA256 trong manifest.
Metrics được kiểm tra lại từ các counters, không tính lại edit distance từ transcript vì ZIP không chứa predictions.
Đây là snapshot CSV tĩnh; chạy notebook/export lại khi có thí nghiệm mới, không ghi đè raw output.

Hai model dùng cùng 4.532 train / 504 valid, seed 2026, cùng fingerprint trong CSV.
Chọn best bằng **EM cao nhất, hòa EM thì WER thấp nhất**. `selected_by_em` đánh dấu lựa chọn đó.
`*_pct` có đơn vị phần trăm (ví dụ 8.269096 nghĩa là 8,269096%), không phải tỷ lệ 0–1.
WER = word_edits / n_words; CER = char_edits / n_chars; EM = exact_normalized / n.

| Best theo EM | Đúng / 504 | WER % | CER % |
|---|---:|---:|---:|
| hynt epoch 9 | 242 | 8,269 | 6,231 |
| Sherpa epoch 3 | 240 | 8,059 | 5,770 |

hynt epoch 5 có WER thấp hơn epoch 9 (8,157%) nhưng ít câu exact hơn (239 so với 242).
Không kết luận model nào thắng toàn diện hoặc khoảng cách có ý nghĩa thống kê từ bảng này.

Các số liệu là **validation đã dùng chọn checkpoint**, chưa phải test độc lập.
Split có tách hội thoại/PCM theo metadata nhưng chưa xác minh tách người nói; 991 audio lịch sử
đã nằm trong nguồn chia train/valid, không dùng lại làm test độc lập sau fine-tune.
Nhóm `english_term_hint` chỉ là lexicon gợi ý, không phải nhãn code-switch/giọng do người chấm.
Checkpoint không có trong ZIP nên `checkpoint_weights_verified=false`; epoch best được suy từ metrics và state.

Web nghe review cần ZIP predictions riêng từ
[notebook export](../../notebook/zalo_lifemate_export_valid_review.ipynb), cộng dataset audio private của nhóm.
