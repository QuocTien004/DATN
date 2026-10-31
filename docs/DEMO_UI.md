# Giao diện trình diễn DATN

Chạy từ repo root bằng Python 3.10. Giao diện Gradio Blocks dùng tiếng Việt,
chỉ nạp model, suy luận, đánh giá và phát lại. Không có bước train tự động.

## Có GPU — mô phỏng và suy luận

Dùng môi trường train hiện có nếu các dependency đang tương thích, hoặc tạo
venv riêng. Ví dụ PowerShell trên Windows:

```powershell
py -3.10 -m venv .venv-ui
.\.venv-ui\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-ui.txt
python scripts/demo_ui.py --checkpoint-dir checkpoints --output-dir demo_runs
```

Chọn wheel PyTorch phù hợp driver của máy. Cài CPU nếu không có CUDA. Mở URL
được in ra, mặc định `http://127.0.0.1:7860`. Chọn `cuda` trong giao diện
khi CUDA khả dụng. Gradio được chốt ở **6.0.1**, các tùy chọn theme/CSS đặt tại
`launch()` theo API bản này.

Đặt checkpoint đáng tin cậy trong thư mục truyền cho `--checkpoint-dir`.
Script quét file `.pt`, không sửa hoặc chuyển đổi checkpoint gốc. Chọn
`critic_warmup/continuous_seed0_refit_warmup_c23.pt` nếu có đúng artifact đó.
Nếu checkpoint đã chứa `models`, để trống WM riêng. Nếu chỉ chứa Actor,
chọn WM đi kèm bằng đường dẫn tương đối trong cùng thư mục checkpoint.
Loader kiểm chiều và nạp trọng số strict, không dùng fallback state giả.

Nhấn **Nạp model → Khởi tạo episode → Chạy**. Map seed quyết định bản đồ,
RNG seed dùng để lặp lại ngẫu nhiên, hai trường được giữ riêng. Chọn shield
trước khi khởi tạo, muốn đổi cấu hình thì bắt đầu episode mới. Horizon tối
đa trong UI là 5.000 bước.

Camera giữ đúng ảnh RGB đầu vào Encoder, thường 64×64. Ảnh được phóng lớn
để xem, không thêm dữ liệu ảnh cho mạng. Top-down là render của MetaDrive,
không phải dự đoán ảnh của RSSM. Bản ghi lưu ảnh PNG lossless.

## Chỉ CPU — giao diện và Replay

Chế độ này không cần cài torch hoặc MetaDrive:

```bash
python -m venv .venv-ui
# Kích hoạt venv theo hệ điều hành
python -m pip install -r requirements-ui.txt
python scripts/demo_ui.py --replay-only
```

Trong chế độ mô phỏng, tạm dừng rồi **Xuất Replay ZIP**. Chuyển ZIP sang môi trường CPU,
chọn trong tab **Replay trên CPU** và nhấn **Nạp Replay**. Có thể phát, dừng,
tiến từng bước hoặc kéo timeline. Chỉ frame đang xem được đọc từ đĩa, không
nạp toàn bộ ảnh episode vào RAM. ZIP import giới hạn 512 MB dữ liệu giải nén.
Với bản ghi lớn hơn, chuyển nguyên thư mục và nhập đường dẫn `manifest.json`.

Không tạo sẵn video/cảm biến giả để lấp chỗ trống. Khi chưa có bản ghi,
giao diện hiển thị trạng thái chưa có dữ liệu. Replay vẫn hoạt động khi
không có checkpoint và không tải model vào RAM.

## Colab tùy chọn

Notebook [DATN_Gradio_Demo.ipynb](../notebooks/DATN_Gradio_Demo.ipynb) tạo môi
trường Python 3.10 riêng vì Python mặc định của Colab có thể khác. Source ZIP
phải thuộc commit có giao diện này. Khi chưa push, tạo từ nhánh Bao local:

```bash
git archive --format=zip --output=DATN-demo-source.zip Bao
```

Upload ZIP qua sidebar Files của Colab. Notebook dùng Xvfb và OpenGL software
để có ảnh thật, còn torch có thể suy luận bằng GPU. Cách này có thể chậm hơn
máy GPU chạy native. Không thay driver NVIDIA của session.
Nếu dùng EGL đã setup đúng, xem script [setup_colab_egl.py](../scripts/setup_colab_egl.py)
và hướng dẫn [train](TRAINING_GUIDE.md). Không tự bật EGL chỉ vì có CUDA.

`--share` tạo link Gradio tạm để trình duyệt truy cập Colab. Chỉ dùng trong
phiên trình diễn tin cậy. Notebook có cell dừng process UI và lưu output
trên Drive. Hoàn tất thì ngắt runtime GPU trong Colab.

## Hiểu đúng các khung AI

- Encoder nhận **RGB và vector state**. RSSM posterior nhận embedding hiện
  tại và action thực thi trước đó để cập nhật bộ nhớ h/z.
- c23 là vector quan sát 23 chiều đi thêm trực tiếp vào Actor. Checkpoint
  đã audit không có continuous c. Bốn tọa độ vật cản đọc từ simulator.
- Actor c23 của Tien có thêm quy tắc phanh/chống kẹt trong `Actor.forward`
  khi eval. Vì vậy “tắt shield” trong UI chỉ tắt lớp Safety Shield bên ngoài,
  chưa đồng nghĩa toàn bộ lệnh là đầu ra thuần của mạng neural.
- Nhánh c19 continuous nếu có sẽ được thể hiện riêng từ checkpoint. Không
  đổi metadata hoặc đệm state để biến checkpoint19 thành checkpoint23.
- Biểu đồ latent hiển thị tối đa 32 phần tử đầu và thống kê trên toàn vector.
  Đây là giá trị thật, không gán ý nghĩa vật lý cho từng neuron.
- Reward/Continue prior được tính từ **prior kế tiếp sau action thực thi**,
  trước khi nhìn ảnh tiếp theo. Posterior kế tiếp là phép kiểm sau khi đã
  nhận ảnh mới. Hai cột này được so với reward và terminal của cùng action.
- Reward head xuất symlog. UI đảo về thang reward với ceiling symlog 3,26
  như imagination, vẫn giữ output symlog gốc và cờ clip trong Replay.
  Continue là sigmoid(logit). Target bằng `1 - terminated`, không cắt
  continue chỉ vì timeout. Tổng reward là reward thật từ wrapper, gồm base
  và custom shaping được lưu riêng.

Giá trị head được hiển thị để kiểm chứng. Nó chưa đạt tất cả gate và không
được gọi là ground truth hoặc dùng để tuyên bố RL đã ổn.

## Thử ảnh và cảm biến

Đóng mô phỏng rồi chuyển sang **AI & thử đầu vào**. Chọn PNG RGB đúng kích
thước checkpoint và dán vector JSON đúng thứ tự từ MetaDrive. Có thể lấy
vector từ `steps.jsonl` của recording. Đầu episode bật reset và dùng action
trước `[0, 0]`. Ảnh tiếp theo bỏ reset, nhập đúng action đã thực thi ở bước
trước, kể cả action sau shield. State memory index 5/6 là `(action+1)/2`
trong observation, còn trường action trước của UI nhận giá trị vật lý `[-1,1]`.

Không tự nối hai ảnh không liên tiếp thành một trajectory. Ảnh riêng không
có reward/terminal thực tế, nên UI chỉ hiển thị suy luận và không chấm đúng/sai.

## Thống kê và so sánh

Tab thống kê tính lại từ từng episode, không chỉ tin aggregate. JSON trong
Tien `082c7fa` gồm **20 maps 10000–10019, horizon1400**, ghi 17/20 finish khi
tắt shield và 19/20 khi bật. Đây là kết quả lịch sử của Tien, JSON chưa có
đủ checkpoint SHA/runtime để kết luận giao diện hiện tại tái lập được mức đó.
Bộ 10 maps horizon1000 trong audit 17/10 là phiên bản khác. Không trộn chúng.

Có thể tải hai file eval mới để so sánh, UI kiểm danh sách map seed và horizon,
nhưng cùng hai trường này chưa chứng minh mọi nguồn ngẫu nhiên/runtime đều
paired. Log `metrics.jsonl` được vẽ theo tên trục người dùng chọn. `env_steps`
khác optimizer updates, không quy đổi bằng cách đổi nhãn.

## Ghi nhận, tái lập và xử lý lỗi

Mỗi episode có `manifest.json`, `steps.jsonl` và `frames/`. Manifest ghi hash
checkpoint/WM, config, seed, phiên bản torch/Python, source commit và trạng
thái hoàn tất. Mỗi bước lưu sensor, action Actor, action thực thi, shield,
head prior/posterior, reward thật và terminal. Checksum bảo vệ JSONL và ảnh.
Ghi theo bước, episode bị ngắt vẫn có thể đọc các dòng hoàn chỉnh.

Live dùng một process spawn cho Panda3D/model, mọi thao tác simulator diễn ra
trong process đó. Lịch sử riêng theo phiên trình duyệt. Đóng phiên/nạp lại model
sẽ giải phóng worker, State có TTL dọn tài nguyên khi người dùng rời trang.
Tua Live chỉ xem dữ liệu đã ghi, không đưa trạng thái simulator quay ngược.

Nếu không có RGB hoặc chiều state sai, UI báo lỗi và không suy luận trên ảnh
giả. Nếu top-down lỗi, camera/sensor vẫn hiển thị cùng thông báo thiếu top-down.
Nếu worker hết thời gian, nạp lại model sau khi kiểm log console/render backend.

Kiểm tra code:

```bash
python -m unittest discover -s tests -q
python scripts/demo_ui.py --help
```

Preflight trên máy GPU của thành viên (không train):

```bash
python scripts/check_demo_runtime.py --checkpoint-dir checkpoints \
  --checkpoint critic_warmup/continuous_seed0_refit_warmup_c23.pt \
  --device cuda --steps 3 --output-dir demo_runs/render_check
```

Nếu chưa có CUDA, dùng `--device cpu` để kiểm camera/model trước. Mở
`runtime_check.json`, kiểm ảnh RGB/top-down trong recording và hash checkpoint.
Lỗi import MetaDrive nghĩa là môi trường chưa đủ dependency, chưa phải
bằng chứng driver/render hỏng.

Chạy preflight tại môi trường trình diễn để xác nhận camera, checkpoint và
render backend. Lưu Replay cùng manifest cho mỗi lượt đánh giá. Một lượt
demo không thay thế benchmark nhiều map hoặc chứng minh tổng quát hóa.
