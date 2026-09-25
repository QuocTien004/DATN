# World Model + MetaDrive

Xe tự lái dùng RGB trước **64×64×3**, vector state 19 chiều và action
`[steering, throttle/brake] ∈ [-1, 1]`. Encoder + categorical RSSM học World Model;
Actor-Critic học bằng imagined rollout. Đây là implementation Dreamer-style riêng,
không phải bản sao đầy đủ DreamerV3.

## Trước khi train dài — 2026-09-25

Lượt `Br_Bao` đã pretrain 5k và online 50k đúng contract v2, nhưng cả 100 episode
eval đều chưa thành công; mốc 50k chỉ đạt route completion 1,81%. Không chạy tiếp
50k/500k chỉ vì loss giảm. Xem [review và bằng chứng Drive](docs/PILOT_REVIEW_20260925.md).

Đã thêm entropy đúng cho action sau tanh, sửa gradient symexp tại 0, lưu config
thực thi/resume và per-seed eval. `configs/train_pilot.yaml` là gate **2.000 bước**,
chưa phải cấu hình đã chứng minh nâng success. WM 5k contract 2 và bootstrap v2
hiện có dùng lại được; khởi tạo Actor mới để thử setting mới.

## Bản sửa pipeline — 2026-09-24

Đã sửa reset RSSM giữa episode, căn reward/continue với successor latent, lưu/nạp
đúng replay vòng và cặp checkpoint–replay. Eval online dùng World Model nằm trong
chính checkpoint. Actor mới dùng giới hạn trơn và exploration thấp hơn; giữ riêng
25% bootstrap trong batch online. Xem [chi tiết thay đổi](docs/RETRAINING_NOTES.md).

**Checkpoint legacy không có `transition_contract: 2` vẫn eval được, nhưng cần
pretrain WM mới để train với pipeline v2.** Không nhầm chúng với checkpoint v2
của Br_Bao, vốn đã dùng đúng transition contract.

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
Colab cần venv Python 3.10 riêng; camera dùng Xvfb hoặc NVIDIA EGL tăng tốc trên
GPU qua `scripts/setup_colab_egl.py` (không thay driver hệ thống); xem
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

python scripts/train_online.py --config configs/train_pilot.yaml --device cuda \
  --total-steps 2000 --eval-episodes 2 --eval-horizon 200 \
  --buffer data/replay_buffer/bootstrap_v2.npz \
  --demo-buffer data/replay_buffer/bootstrap_v2.npz \
  --wm-checkpoint checkpoints/retrain_v2/world_model/latest.pt \
  --ckpt-dir checkpoints/pilot_v3_s0 --log-dir logs/pilot_v3_s0

python scripts/eval.py --device cuda --episodes 20 --start-seed 10000 \
  --checkpoint checkpoints/pilot_v3_s0/online/latest.pt \
  --output logs/pilot_v3_s0/eval_full.json
```

Nếu đã có bootstrap v2 + WM 5k của Br_Bao, copy về đúng đường dẫn và bỏ qua hai
lệnh collect/pretrain. Eval 200 bước trong pilot chỉ để chẩn đoán; lệnh eval độc
lập bên trên dùng horizon 1000 trong checkpoint. Xem kết quả rồi mới quyết định
resume. `--ckpt-dir` là root, script tự thêm `/online`, đừng truyền dư suffix này.

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
