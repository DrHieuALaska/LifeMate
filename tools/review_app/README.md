# LifeMate Result Review

App local để nghe và gán nhãn các output ASR/reasoning đã chạy. Bốn khung 01–04
cùng xuất hiện trên một màn hình và cuộn độc lập:

1. **Audio & ASR**: yêu cầu hiện tại và audio lịch sử kèm bản chép ASR.
2. **Input & lịch sử text**: context, hồ sơ, phản hồi text, state đầu vào và JSON.
3. **Output reasoning**: tool call dự đoán, đánh giá và output JSON do người review sửa.
4. **Tool được phép**: danh sách theo context của sample, mô tả và schema.

Nút “Ghi chú” mở ghi chú/nhóm lỗi. Thanh dưới có ô hoàn tất và nút lưu rồi chuyển
hội thoại. Nút ☰ ẩn/hiện danh sách. Số thứ tự được giữ nguyên khi lọc; tìm `#12`
để mở hội thoại số 012. Mỗi lần chỉ phát một audio.

Đây là công cụ kiểm tra thủ công output có sẵn. App không thực thi tool call,
không gọi ASR/LLM và không thay đổi benchmark train-only của repo.

## Chạy sau khi clone

Cần **Python 3.11+**, dữ liệu gốc dạng ZIP hoặc thư mục đã giải nén từ kênh chia sẻ của nhóm và output Kaggle.
App chỉ dùng thư viện chuẩn; không cần cài package, GPU hoặc model weights.

### Mở bằng một file trên Windows

Đặt dữ liệu đã giải nén cạnh repo như sau:

```text
thư_mục_chung/
  train+public_test/
  LifeMate/
    open_review.bat
```

Sau khi pull, chỉ cần **nhấp đúp `LifeMate/open_review.bat`**.
File tự tìm `../train+public_test` tính từ repo; nếu chưa có thư mục thì thử
`../train+public_test.zip`. Nếu đã có `config.json`, đường dẫn trong đó được ưu tiên.
Lần đầu chưa có output Kaggle, launcher tự tải về rồi mở web. Các lần sau dùng
output đã tải. Chỉ lần tải cần Internet; vẫn cần Python 3.11+ trên máy.

Giữ cửa sổ server mở khi dùng app. Mở file lần nữa sẽ mở lại web của server đang
chạy. Nếu đổi dữ liệu/cấu hình, đóng cửa sổ server cũ rồi mở lại. Nhãn vẫn lưu
trên ổ đĩa. Không cần chạy riêng `server.py` hoặc tiện ích tải trong cách mở này.

Các lệnh sau chạy từ thư mục repo `LifeMate`.

### Người pull repo nhận được gì?

| Thành phần | Có trong Git? | Cần chuẩn bị trên máy |
|---|---|---|
| Code app, giao diện, config mẫu, test giả lập | Có | Pull/clone repo |
| Audio + context/state gốc | Không | Nhận ZIP hoặc thư mục đã giải nén qua kênh dữ liệu của nhóm |
| ASR và output reasoning từ Kaggle | Không | Chạy tiện ích tải ở bước 1 hoặc dùng Kaggle CLI |
| Snapshot so sánh valid504 | Không | Tùy chọn; không cần để gán nhãn public test |
| Nhãn người dùng đã nhập | Không | Nhận file nhãn riêng nếu muốn tiếp tục phần người khác đã label |

Trên máy đang dùng app, các file dữ liệu vẫn nằm cục bộ trong `data/`,
`imports/` và vị trí ZIP gốc. `.gitignore` chỉ ngăn chúng được đưa lên Git;
người pull repo không tự có các file này và không cần tải model weights.

Để chia sẻ và tiếp tục nhãn đã lưu, dùng file `data/manual_public_labels.json`
qua kênh nội bộ và truyền `--annotations <đường-dẫn-file.json>`. Hai máy phải
dùng cùng bộ output vì file nhãn có fingerprint. File xuất JSONL phục vụ xem/
phân tích; app hiện không nhập trực tiếp JSONL vào kho nhãn.

### 1. Tải output về máy

Với notebook công khai hiện tại, tiện ích tải có sẵn dùng API chính thức:

```powershell
python tools/review_app/fetch_kaggle_results.py
```

File được lưu trong thư mục local bị Git bỏ qua:
`tools/review_app/imports/kaggle_20_sample_public_test/`.
Tên notebook là `drhieualaska/20-sample-public-test`; số mẫu lấy từ output
thực tế, không cố định theo slug “20-sample”.

Nếu đã dùng Kaggle CLI, có thể tải vào `artifacts/` thay vì tiện ích trên:

```powershell
kaggle kernels output drhieualaska/20-sample-public-test -p artifacts/review_app/kaggle_output
```

Notebook riêng tư cần tài khoản có quyền truy cập. Thiết lập thông tin đăng
nhập trên máy; không đưa token vào config hay repo. File Python tải kèm output
không được app tự động thực thi. Khi review một run khác, dùng một thư mục output
riêng để tránh trộn các artifact cũ và mới.

### 2. Mở app

Nếu dùng tiện ích tải mặc định:

```powershell
python tools/review_app/server.py --dataset "../train+public_test.zip" --open
```

Thay đường dẫn ZIP cho đúng máy. Nếu dùng output ở `artifacts/`:

```powershell
python tools/review_app/server.py --dataset "../train+public_test.zip" --public-results artifacts/review_app/kaggle_output/lifemate_benchmark_634 --open
```

**Nếu đã giải nén, truyền thẳng thư mục, không cần nén lại:**

```powershell
python tools/review_app/server.py --dataset "../train+public_test" --open
```

Có thể chọn thư mục bọc bên ngoài, thư mục có `train/` và `public_test/`, hoặc
chọn trực tiếp `public_test/` khi chỉ review public test. Giữ nguyên tên/cấu trúc
file gốc, gồm `public_test.jsonl`, `public_test_tools_catalog.json`, `audio/` và
`public_test_environment/`. Để xem thêm valid ASR, chọn package có cả `train/`.
App tự tìm manifest trong thư mục đã chọn và đọc WAV/state tại chỗ.

`--public-results` trỏ vào thư mục có `selected_ids.json`,
`batch_asr_transcripts.jsonl`, `BENCHMARK_NOT_FOR_SUBMISSION.jsonl` và
`metrics/journal/`. Nguồn dữ liệu phải là cùng package đã dùng tạo prediction.
App kiểm tra ID giữa các file, ghép audio/lịch sử theo sample và đối chiếu
submission trong từng journal.

Địa chỉ mặc định: **http://127.0.0.1:8770/**.
Đổi cổng bằng `--port 8772`. App chỉ bind loopback. Các lệnh Python trên cũng
dùng được trên Linux/macOS. Trên Windows có thể mở `start_review.bat` sau khi
cấu hình đường dẫn local.

### 3. Cấu hình local tùy chọn

Sao chép `config.example.json` thành `config.json` trong thư mục app và chỉnh
đường dẫn. `config.json` đã bị Git bỏ qua.

- Đường dẫn trong config được tính tương đối với thư mục app.
- `dataset_path` nhận ZIP hoặc thư mục. Khóa cũ `dataset_zip` vẫn được hỗ trợ.
- Đường dẫn truyền bằng CLI được tính từ thư mục chạy lệnh.
- CLI ưu tiên hơn config.
- `--annotations <file.json>` chọn nơi lưu nhãn riêng.
- `--valid-snapshot <review.json>` thêm màn hình so sánh valid tùy chọn.

## Gán nhãn và xem đầy đủ input

Public test không có transcript/output chuẩn. Dòng ASR và tool call trên app
là **prediction**. Ô sửa transcript/output để trống cho đến khi người review nhập.

Có thể nhập bản chép đúng, chọn đánh giá ASR/reasoning, đánh dấu nhóm lỗi và ghi
chú. Nhãn tự lưu sau khoảng một giây; kiểm tra “Đã lưu trên máy”. Nội dung JSON
sai cú pháp sẽ giữ mẫu hiện tại để người dùng sửa trước khi chuyển.

Audio lịch sử nằm ở khung 01. Phản hồi text và lịch sử tool nằm ở khung 02,
kèm nút trỏ tới lượt audio tương ứng. JSON đầu vào gồm sample gốc, record sau
khi ghép ASR cache, catalog tool và state môi trường ban đầu. State là dữ liệu
để tool tra cứu, không đồng nghĩa toàn bộ state nằm trong prompt LLM. Nếu
output không lưu dump prompt hoặc tool response lúc chạy, app không tự dựng
thêm các dữ liệu đó.

Nút “Xuất nhãn” tải JSONL gồm prediction gốc, fingerprint nguồn và nhãn đã nhập,
bao gồm nhãn nháp lẫn mẫu hoàn tất. Mẫu số thứ tự và nội dung prediction không
bị thay đổi khi gán nhãn.

## Dữ liệu cục bộ và Git

Những file/thư mục sau **không nằm trong bản code để push**:

- `config.json`: đường dẫn cấu hình riêng của từng máy.
- `data/`: snapshot valid, `manual_public_labels.json` và bản sao `.json.bak`.
- `imports/`: output Kaggle, transcript, journal và metadata tải về.
- Log, ảnh preview, cache Python và model weights.

Giữ ZIP hoặc thư mục dữ liệu ở vị trí riêng, truyền qua `--dataset`; app đọc audio tại chỗ.
Không commit audio, transcript, state, prediction, nhãn thủ công hoặc credentials.
Nếu cần chia sẻ dữ liệu review, dùng kênh nội bộ được phép. File nhãn có
fingerprint để tránh ghép sang một tập prediction khác.

## Snapshot valid tùy chọn

App chạy public-review mà không cần `data/review.json`. Nếu có snapshot riêng
của nhóm, đặt trong `data/` hoặc truyền `--valid-snapshot`; trang `/valid`
sẽ xuất hiện. Trang này giữ nguyên WER gốc và bản so sánh ASR. Với audio lịch sử,
output chuẩn là phản hồi assistant ngay sau lượt user, không phải nhãn của
yêu cầu cuối.

`prepare_data.py --workspace <original-workspace>` là tiện ích tạo snapshot
từ các báo cáo ASR local có đúng layout cũ; nó không cần chạy cho public review.
Dữ liệu đầu ra của tiện ích vẫn thuộc `data/` bị bỏ qua bởi Git.

## Kiểm thử

Từ repo:

```powershell
python -m unittest discover -s tests -p test_result_review_app.py -v
python -m unittest discover -s tests -v
```

Test được commit chỉ dùng ZIP, WAV, ASR và state **giả lập** tạo trong thư mục
tạm. Kiểm tra được chạy không có dataset thật, network, secret hoặc GPU; gồm
public-only sau clone, đường dẫn tương đối/CLI, snapshot valid tùy chọn, input,
audio seek, lưu/đọc lại/xuất nhãn và chặn truy cập file ngoài route cho phép.
