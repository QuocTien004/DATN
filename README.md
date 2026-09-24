# World Model + MetaDrive

Xe tự lái dùng RGB trước **64×64×3**, vector state 19 chiều và action
`[steering, throttle/brake] ∈ [-1, 1]`. Encoder + categorical RSSM học World Model;
Actor-Critic học bằng imagined rollout. Đây là implementation Dreamer-style riêng,
không phải bản sao đầy đủ DreamerV3.

## Bản sửa để train lại — 2026-09-24

Đã sửa reset RSSM giữa episode, căn reward/continue với successor latent, lưu/nạp
đúng replay vòng và cặp checkpoint–replay. Eval online dùng World Model nằm trong
chính checkpoint. Actor mới dùng giới hạn trơn và exploration thấp hơn; giữ riêng
25% bootstrap trong batch online. Xem [chi tiết thay đổi](docs/RETRAINING_NOTES.md).

**Checkpoint cũ vẫn eval được, nhưng cần pretrain WM mới và khởi tạo Actor mới để
train với bản sửa.** Các checkpoint mới có `transition_contract: 2`. Chưa có bằng
chứng policy sau sửa đã học lái tốt; cần pilot 50k steps và đánh giá nhiều seed.

## Cài đặt

Python **3.10**; MetaDrive 0.4.3; NumPy 1.x. Đã kiểm tra trên Colab T4 với
Python 3.10.21, Torch 2.11.0+cu128. Cài Torch phù hợp máy trước:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# CPU: dùng --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Không dùng `requirements-cpu.txt` / `requirements-cuda.txt`: các file đó đã bị xóa.
Colab cần venv Python 3.10 riêng và Xvfb cho camera headless; xem
[hướng dẫn đầy đủ](docs/TRAINING_GUIDE.md).

## Train mới → resume → eval

Chạy từ root repo. Trên Colab thêm `xvfb-run -a` trước các command dùng MetaDrive.
Chọn thư mục output mới cho mỗi experiment để giữ nguyên artifact cũ.

```bash
python scripts/collect_bootstrap.py --steps 25000 --policy mixed_expert \
  --out data/replay_buffer/bootstrap_v2.npz

python scripts/train_world_model.py --device cuda --updates 5000 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_wm

python scripts/train_online.py --device cuda --total-steps 50000 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --wm-checkpoint checkpoints/retrain_v2/world_model/latest.pt \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_online

python scripts/train_online.py --device cuda --total-steps 100000 \
  --resume checkpoints/retrain_v2/online/latest.pt \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --ckpt-dir checkpoints/retrain_v2 --log-dir logs/retrain_v2_online

python scripts/eval.py --device cuda --episodes 20 --start-seed 10000 \
  --checkpoint checkpoints/retrain_v2/online/latest.pt \
  --output logs/retrain_v2_online/eval.json
```

Resume tự nạp `online_buffer.npz` cạnh checkpoint và kiểm tra cùng `env_steps`.
`--total-steps` là mốc tổng, không phải số bước chạy thêm. Online eval không cần
`--wm-checkpoint`. Giữ cả `latest.pt`, `online_buffer.npz`, bootstrap và log khi
chuyển máy/Colab. Chi tiết lưu Drive và smoke test ở [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

## Cấu trúc

```text
configs/       environment, World Model, Actor-Critic và train loop
envs/          MetaDrive RGB/state wrapper + reward shaping
models/        Encoder, RSSM, Decoder, reward/continue heads, Actor-Critic
training/      collect, World Model update, imagination, online orchestration
evaluation/    recurrent policy + aggregate/per-seed metrics
scripts/       collect_bootstrap.py, train_world_model.py, train_online.py, eval.py
tests/         unit và regression tests
docs/          hướng dẫn cộng tác và train lại
```

Ảnh/biểu đồ cũ trong `docs/` phản ánh lượt train trước bản sửa, không phải kết quả
chất lượng của bản mới. Lượt cũ ở 400k có success 0%; ưu tiên kiểm tra lane keeping
trước khi dùng [Stage 2](configs/env_stage2_dense_traffic.yaml).
