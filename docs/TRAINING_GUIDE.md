# Hướng dẫn train lại và lưu artifact

Cập nhật 2026-09-25, branch `Bao`. Đọc [review Br_Bao](PILOT_REVIEW_20260925.md):
v2 chạy đúng pipeline nhưng 50k vẫn success 0%. Chạy gate 2k ở mục 5 trước khi
đầu tư train dài. Chỉ checkpoint legacy **không có contract 2** mới bắt buộc
pretrain WM lại; WM 5k contract 2 của Br_Bao có thể dùng để khởi tạo pilot mới.

## 1. Môi trường Python 3.10 trên Colab

Mount Drive bằng giao diện Colab hoặc cell do người dùng chạy. Trong notebook,
chạy cell sau (CLI `uv` có sẵn trên runtime đã kiểm tra):

```bash
%%bash
set -e
uv venv --python 3.10 /content/datn_py310
uv pip install --python /content/datn_py310/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
git clone --branch Bao https://github.com/QuocTien004/DATN.git /content/DATN
uv pip install --python /content/datn_py310/bin/python -r /content/DATN/requirements.txt
/content/datn_py310/bin/python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```

Nếu runtime chưa có `uv`, cài `python -m pip install uv`. Nếu chưa có Xvfb/xauth,
chạy `apt-get update` rồi `apt-get install -y xvfb xauth`. Kernel notebook có thể
vẫn là 3.13; các command bên dưới gọi thẳng Python 3.10 trong venv.

Trên Windows: tạo venv bằng `py -3.10 -m venv .venv`, activate và cài Torch +
`requirements.txt` theo README. Không dùng Xvfb trên Windows.

### Camera trên GPU Colab (tùy chọn, tiết kiệm thời gian)

Có CUDA không có nghĩa camera đang render trên GPU. Runtime T4 ngày 25/09 dùng
`llvmpipe` khi chạy Xvfb: probe 30 steps khoảng 0,835 giây/step. Với NVIDIA EGL,
renderer báo `Tesla T4/PCIe/SSE2`, probe khoảng 0,0103 giây/step (không tính model
update và startup; không coi đây là tốc độ train tổng thể).

```bash
/content/datn_py310/bin/python /content/DATN/scripts/setup_colab_egl.py
```

Script tải và **giải nén** thư viện OpenGL đúng phiên bản driver hiện tại từ apt
repository đã cấu hình, dưới `/content/datn_nvidia_gl/`. Không cài/thay kernel
driver. Nếu không tìm được package khớp, dừng và dùng Xvfb, không tự chọn driver
khác. Copy ba dòng `export` được in vào **cùng cell shell** với lệnh train/eval.
Sau đó chạy Python trực tiếp, bỏ `xvfb-run -a`. `DATN_RENDER_BACKEND=egl` chỉ hỗ
trợ headless (`use_render: false`). Không dùng setup này trên Windows.

Runtime đã thử có driver 580.82.07, package `libnvidia-gl-580=580.82.07-0ubuntu1`.
Backend/default, GPU và phiên bản Torch được ghi trong run manifest. Chuyển
renderer có thể gây khác biệt pixel; ghi backend khi so sánh các experiment.

## 2. Đúng bộ dữ liệu và checkpoint

Drive hiện chia `DoAn/Br_Bao/` và `DoAn/Br_Tien/`. Bộ đã kiểm tra ngày 25/09:
`DoAn/Br_Bao/data/replay_buffer/bootstrap_v2.npz` (25k, 64×64),
`DoAn/Br_Bao/checkpoints/retrain_v2/world_model/latest.pt` (5k updates),
`DoAn/Br_Bao/checkpoints/retrain_v2/online/online/latest.pt` (50k env steps).
Thư mục `online/online` của bộ này là do truyền dư `/online` vào `--ckpt-dir`.
Không trộn bootstrap/checkpoint 256×256 legacy với bộ 64×64.

Copy file cần dùng từ Drive sang `/content` trước khi load. Replay giải nén và
tạo mmap cạnh `.npz`, nên đọc trực tiếp từ Drive vừa chậm vừa tạo I/O lớn.
Không cần copy tất cả numbered checkpoint để chạy một test.

Để train mới, khuyến nghị collect bootstrap bằng code mới: replay lưu riêng
`done` (reset episode) và `terminated` (terminal thật). Replay cũ vẫn đọc được,
nhưng không thể khôi phục chính xác timeout/terminal từ một cờ `done` duy nhất.
Nếu tái dùng bootstrap 64×64 cũ, ghi rõ hạn chế này trong report, giữ đúng reward
config lúc collect và vẫn pretrain World Model mới.

## 3. Kiểm tra và collect bootstrap

Trong mỗi cell shell, vào đúng repo và chọn lại Python:

```bash
%%bash
set -e
cd /content/DATN
PY=/content/datn_py310/bin/python
$PY -m unittest discover -s tests -v
xvfb-run -a $PY scripts/collect_bootstrap.py --dry-run
xvfb-run -a $PY scripts/collect_bootstrap.py --steps 25000 --policy mixed_expert --out data/replay_buffer/bootstrap_v2.npz
```

Expected: RGB `(64,64,3) uint8`, state `(19,)`, action `(2,)`.
`mixed_expert` chọn expert 80% số bước, random 20%; có noise steering nhỏ.
Đây là mixed bootstrap, không phải dataset expert thuần. Kiểm tra return/episode
và preview trước khi pretrain. Mỗi run dùng output mới, giữ lại bootstrap gốc.

## 4. Pretrain World Model mới

```bash
%%bash
set -e
cd /content/DATN
PY=/content/datn_py310/bin/python
$PY scripts/train_world_model.py --device cuda --updates 5000 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_wm
```

Không thêm `--resume` với checkpoint cũ. Reward/continue heads hiện học ở
successor posterior, khớp cách imagination sử dụng chúng; checkpoint mới lưu
`transition_contract: 2`. Vẫn dùng kiến trúc 512 deterministic + 32×32 categorical,
batch 8, sequence 32. Nếu thiếu VRAM, giảm batch trước.

Để nối chính pretrain v2 đã gián đoạn:

```bash
$PY scripts/train_world_model.py --device cuda --updates 1000 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --resume checkpoints/retrain_v2/world_model/latest.pt \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_wm
```

`--updates` của WM là số update thêm. Training loss thấp chưa đủ chứng minh WM
dự đoán đúng terminal/reward ở trajectory chưa thấy; cần xem rollout và eval.

## 5. Gate online 2k trước khi train dài

Không resume actor 50k để thử entropy mới; tạo actor mới từ WM pretrain contract 2.
Với bộ Br_Bao đã có, copy bootstrap và WM vào đường dẫn tương ứng dưới `/content/DATN`
rồi bỏ qua collect/pretrain. Không copy mmap; nó được tạo lại từ `.npz`.

```bash
%%bash
set -e
cd /content/DATN
PY=/content/datn_py310/bin/python
xvfb-run -a $PY scripts/train_online.py --config configs/train_pilot.yaml --device cuda --seed 0 \
  --total-steps 2000 --eval-episodes 2 --eval-horizon 200 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --wm-checkpoint checkpoints/retrain_v2/world_model/latest.pt \
  --ckpt-dir checkpoints/pilot_v3_s0 --log-dir logs/pilot_v3_s0
```

`--ckpt-dir` nhận **root của run**, không thêm `/online`. Trên Windows bỏ
`xvfb-run -a`; gọi Python của venv 3.10. Training horizon vẫn 1000; chỉ eval bị
giới hạn 200 để chẩn đoán nhanh. Không so success/route của eval ngắn với eval
1000 steps cũ như cùng một benchmark.

Mỗi 250 env steps có 25 WM và 5 Actor-Critic updates. Batch 8 lấy 2 sequences
từ bootstrap được giữ riêng, 6 từ replay online. `term_ratio: 0.25` là tỷ lệ
sequence bắt buộc chứa một episode boundary; phần uniform còn lại vẫn có thể
chứa terminal, không có nghĩa chính xác 25% transition là crash.

Actor pilot: std trong `[0.1,0.5]`, init 0.3, squashed entropy coefficient 0.01,
imagination 15, không cộng noise ngoài policy. Các số này cần đánh giá, chưa phải nghiệm
tối ưu đã xác minh. Reward coefficients được giữ nguyên để tránh đổi cả mục tiêu
học cùng lúc với các lỗi pipeline.

Theo dõi success/out-of-road và route completion qua nhiều mốc. Thêm các metrics:
`rollout/throttle_saturation`, `rollout/steering_saturation`,
`rollout/lateral_available_mean`, `rollout/reward_base_mean`,
`rollout/reward_shaping_mean`, `wm/terminal_fraction`, `wm/continue_on_terminal`.
`ac/continue_mean` giữ nghĩa lịch sử là discounted continuation; metric mới
`ac/continue_probability_mean` bỏ gamma. Không dùng ngưỡng 0.95 cho trung bình
continue để tự kết luận model nhận biết crash: cần xem riêng terminal samples.

Thêm `rollout/idle_fraction`, `rollout/collection_seconds`; `eval_step_*.json` ghi
từng seed, horizon và terminal/truncated. Nếu không có episode hoàn tất trong
một đoạn 250 steps, metrics episode của đoạn là 0, không có nghĩa xe thất bại 0%.

Gate chấp nhận kỹ thuật: loss/gradient hữu hạn, model/replay khớp step, resume và
eval hoạt động. Gate chất lượng khác: xe có tiến lên, không liên tục đứng yên hoặc
steering dồn một phía; eval đủ horizon và nhiều seed phải cải thiện. Chỉ pass gate
kỹ thuật **không** đủ để giao train 50k/500k. Nên thử seed 0,1,2 với output riêng;
không tăng Stage 2 khi Stage 1 vẫn success 0%.

Eval đầy đủ sau pilot (giữ mặc định horizon 1000 trong checkpoint):

```bash
xvfb-run -a $PY scripts/eval.py --device cuda --episodes 20 --start-seed 10000 \
  --checkpoint checkpoints/pilot_v3_s0/online/latest.pt \
  --output logs/pilot_v3_s0/eval_full.json
```

Không resume để chạy tiếp dài cho tới khi đã xem gate. Nếu cần nối thử đến 5k,
lệnh ở mục 6 giữ đúng setting pilot; `--eval-horizon 1000` khôi phục eval đầy đủ.

## 6. Lưu và resume đúng cặp

```text
checkpoints/pilot_v3_s0/
  online/
    latest.pt
    online_buffer.npz
    online_step_001000.pt
    ...
logs/pilot_v3_s0/
  metrics.jsonl
  resolved_config.json
  run_start_000000.json
  eval_step_001000.json
```

`latest.pt` chứa tất cả WM, Actor/Critic, optimizer, config và counters. Replay
cạnh nó ghi `checkpoint_step`; resume kiểm tra bằng `env_steps`. Không fallback
về bootstrap khi thiếu replay. Numbered model checkpoint dùng để eval lịch sử;
chỉ resume được nếu còn replay đúng mốc đó. Repo chỉ giữ một replay online mới nhất
để tránh nhân bản hàng trăm MB mỗi lần save.

```bash
xvfb-run -a $PY scripts/train_online.py --device cuda --total-steps 5000 \
  --resume checkpoints/pilot_v3_s0/online/latest.pt \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --eval-horizon 1000 \
  --ckpt-dir checkpoints/pilot_v3_s0 --log-dir logs/pilot_v3_s0
```

Không cần `--wm-checkpoint` khi resume online. Không truyền bootstrap qua `--buffer`
trong lệnh resume. `--total-steps` là mốc tổng. `--eval-every 0` và `--ckpt-every 0`
tắt tác vụ định kỳ tương ứng; cuối training vẫn save.

Lệnh trên chỉ là ví dụ nối pilot **sau khi đã xem kết quả**, không phải khuyến nghị
tự động train tiếp nếu xe vẫn đứng yên/lệch lái.

Không truyền `--config` thì resume lấy training config từ checkpoint, không rơi
về defaults của `train.yaml`. CLI overrides vẫn có hiệu lực. Actor config luôn
lấy từ checkpoint; muốn thử entropy/horizon của Actor mới cần run khởi tạo mới.
Manifest `resolved_config.json` phản ánh setting thực thi, `run_start_*.json` giữ
lịch sử từng lần chạy và phiên bản Python/Torch/GPU.

Để lưu lên Drive: sau khi tiến trình dừng ở checkpoint hoàn chỉnh, copy cả thư mục
`checkpoints/pilot_v3_s0/`, `logs/pilot_v3_s0/`, bootstrap và config vào thư
mục run mới trên Drive. Không copy `latest.pt` ở một thời điểm rồi replay sau khi
training đã ghi mốc mới. Mỗi file được ghi tạm rồi rename; cặp model/replay có kiểm
tra step để phát hiện gián đoạn giữa hai lần ghi. Không có snapshot trạng thái
simulator/RNG, nên resume bắt đầu episode mới, không tái hiện bit-for-bit quỹ đạo.

## 7. Eval độc lập và smoke test

```bash
xvfb-run -a $PY scripts/eval.py --device cuda --episodes 20 --start-seed 10000 \
  --checkpoint checkpoints/pilot_v3_s0/online/latest.pt \
  --output logs/pilot_v3_s0/eval.json
```

Eval dùng WM và Actor cùng checkpoint, env config trong checkpoint, seed cố định
10000–10019. Không truyền `--wm-checkpoint` cho checkpoint online. JSON lưu từng
episode để so sánh đúng seed. Nếu eval checkpoint offline Actor-only của workflow
cũ, mới cần truyền thêm đúng `--wm-checkpoint` tương ứng.

Test nhanh trước khi train dài: copy config riêng và đặt env horizon 10,
buffer capacity 1000, sequence 8, imagination horizon 3, max_start_states 4.
Dùng replay nhỏ và output riêng. Chạy 8 env steps, 1 WM + 1 AC update/iteration,
save rồi resume đến 12; thêm `--episodes 1 --horizon 10` cho eval smoke. Không dùng
chỉ số từ horizon rút ngắn để báo chất lượng policy.

MetaDrive RGB có thể render bằng CPU trên Colab dù neural network dùng T4. Nếu
eval chậm, giới hạn số episode để kiểm tra pipeline; báo đúng số đã hoàn tất.
Sau khi tải report/artifact cần giữ về Drive hoặc máy local, ngắt runtime để
không tiếp tục tiêu quota.
