# Hướng dẫn train lại và lưu artifact

Cập nhật 2026-09-24, branch `Bao`. Đọc [RETRAINING_NOTES.md](RETRAINING_NOTES.md)
để biết vì sao cần pretrain lại. Checkpoint cũ dùng để đối chiếu/eval; không resume
training trực tiếp sau khi đổi transition contract.

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

## 2. Đúng bộ dữ liệu và checkpoint

Ở Drive của nhóm, bộ mới đã được đặt dưới `DoAn/data/replay_buffer/` và
`DoAn/checkpoints/`. Hai file `DoAn/bootstrap.npz` và `DoAn/latest.pt` ở cấp ngoài
là bộ cũ 256×256; đừng trộn với bộ mới 64×64.

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

## 5. Pilot online 50k steps

```bash
%%bash
set -e
cd /content/DATN
PY=/content/datn_py310/bin/python
xvfb-run -a $PY scripts/train_online.py --device cuda --seed 0 \
  --total-steps 50000 --steps-per-iter 1000 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --wm-checkpoint checkpoints/retrain_v2/world_model/latest.pt \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_online
```

Mỗi 1000 env steps có 25 WM và 25 Actor-Critic updates. Batch 8 lấy 2 sequences
từ bootstrap được giữ riêng, 6 từ replay online. `term_ratio: 0.25` là tỷ lệ
sequence bắt buộc chứa một episode boundary; phần uniform còn lại vẫn có thể
chứa terminal, không có nghĩa chính xác 25% transition là crash.

Actor mới: std trong `[0.1,0.5]`, init 0.3, entropy coefficient 0.003, không cộng
noise ngoài policy. Các số này là cấu hình pilot cần đánh giá, chưa phải nghiệm
tối ưu đã xác minh. Reward coefficients được giữ nguyên để tránh đổi cả mục tiêu
học cùng lúc với các lỗi pipeline.

Theo dõi success/out-of-road và route completion qua nhiều mốc. Thêm các metrics:
`rollout/throttle_saturation`, `rollout/steering_saturation`,
`rollout/lateral_available_mean`, `rollout/reward_base_mean`,
`rollout/reward_shaping_mean`, `wm/terminal_fraction`, `wm/continue_on_terminal`.
`ac/continue_mean` giữ nghĩa lịch sử là discounted continuation; metric mới
`ac/continue_probability_mean` bỏ gamma. Không dùng ngưỡng 0.95 cho trung bình
continue để tự kết luận model nhận biết crash: cần xem riêng terminal samples.

Nên thử ít nhất seed 0, 1, 2 với output/log riêng. Chỉ nối lên 100k/500k khi route
completion cải thiện lặp lại, bắt đầu có success và out-of-road giảm. Không tăng
độ khó sang Stage 2 khi Stage 1 vẫn success 0%.

## 6. Lưu và resume đúng cặp

```text
checkpoints/retrain_v2/
  world_model/latest.pt
  online/
    latest.pt
    online_buffer.npz
    online_step_020000.pt
    ...
logs/retrain_v2_online/
  metrics.jsonl
  resolved_config.json
```

`latest.pt` chứa tất cả WM, Actor/Critic, optimizer, config và counters. Replay
cạnh nó ghi `checkpoint_step`; resume kiểm tra bằng `env_steps`. Không fallback
về bootstrap khi thiếu replay. Numbered model checkpoint dùng để eval lịch sử;
chỉ resume được nếu còn replay đúng mốc đó. Repo chỉ giữ một replay online mới nhất
để tránh nhân bản hàng trăm MB mỗi lần save.

```bash
xvfb-run -a $PY scripts/train_online.py --device cuda --total-steps 100000 \
  --resume checkpoints/retrain_v2/online/latest.pt \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_online
```

Không cần `--wm-checkpoint` khi resume online. Không truyền bootstrap qua `--buffer`
trong lệnh resume. `--total-steps` là mốc tổng. `--eval-every 0` và `--ckpt-every 0`
tắt tác vụ định kỳ tương ứng; cuối training vẫn save.

Để lưu lên Drive: sau khi tiến trình dừng ở checkpoint hoàn chỉnh, copy cả thư mục
`checkpoints/retrain_v2/`, `logs/retrain_v2_online/`, bootstrap và config vào thư
mục run mới trên Drive. Không copy `latest.pt` ở một thời điểm rồi replay sau khi
training đã ghi mốc mới. Mỗi file được ghi tạm rồi rename; cặp model/replay có kiểm
tra step để phát hiện gián đoạn giữa hai lần ghi. Không có snapshot trạng thái
simulator/RNG, nên resume bắt đầu episode mới, không tái hiện bit-for-bit quỹ đạo.

## 7. Eval độc lập và smoke test

```bash
xvfb-run -a $PY scripts/eval.py --device cuda --episodes 20 --start-seed 10000 \
  --checkpoint checkpoints/retrain_v2/online/latest.pt \
  --output logs/retrain_v2_online/eval.json
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
