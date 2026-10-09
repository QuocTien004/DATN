# Setup và workflow có thể tái lập

Cập nhật 09/10/2026. **Không mở RL hiện tại.** Các lệnh train dưới đây là smoke
WM legacy hoặc BC có điều kiện, không phải recipe đã đạt chất lượng lái.
Chạy từ root repo; các đường dẫn output là ví dụ tương đối, chọn thư mục mới.

## 1. Python và dependency

Dùng Python 3.10, MetaDrive 0.4.3, NumPy 1.x. Bộ dependency đã dùng kiểm thử:
Python 3.10, Torch 2.11.0, NumPy 1.26.4. GPU không bắt buộc cho unit tests.
Không giả định Python của notebook chính là Python của venv.

Linux/macOS (cần có Python 3.10 trước):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài bản CPU trên nền tảng có wheel tương ứng:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import sys, torch, numpy; print(sys.version); print(torch.__version__, numpy.__version__); print(torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Nếu dùng CUDA, thay lệnh cài Torch bằng wheel tương thích nền tảng/driver,
không cài hai bản Torch vào cùng venv. Ví dụ wheel đã dùng cho kiểm thử:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Đối chiếu [PyTorch installation](https://pytorch.org/get-started/locally/) khi nền
tảng khác. `requirements.txt` không pin toàn bộ dependency bắc cầu; lưu
`python -m pip freeze` cùng run để tái lập. Comment CUDA trong requirements là
ví dụ cũ, không phải cam kết driver phù hợp. Không cần torchvision/torchaudio
cho các model trong repo.

Colab: tạo venv Python 3.10 riêng nếu kernel dùng phiên bản khác. Có thể dùng
`uv venv --python 3.10 .venv` rồi gọi trực tiếp `.venv/bin/python`;
kích hoạt venv trong một shell cell không đổi Python kernel ở cell khác.
Không để artifact duy nhất trên ổ tạm của runtime.

## 2. Camera/headless

Collect và eval cần MetaDrive render được ảnh, kể cả `use_render: false`.
Trên Linux headless, cài Xvfb/xauth theo môi trường rồi thêm `xvfb-run -a`
trước command dùng environment. Không dùng Xvfb trên Windows.

Linux/Colab có NVIDIA EGL có thể dùng helper:

```bash
python scripts/setup_colab_egl.py --output .tools/nvidia_gl
```

Script tải/giải nén userspace OpenGL **khớp driver**, không thay kernel driver.
Copy các export được in vào cùng shell với collect/eval, rồi bỏ xvfb-run.
Nếu không tìm thấy package khớp, dừng setup EGL và dùng backend có sẵn.
CUDA khả dụng không đồng nghĩa camera render trên GPU.
Giữ renderer cố định giữa đối chứng; khác pixel có thể gây khác trajectory.

## 3. Kiểm CLI, quan sát và collect

Các lệnh help không cần data/checkpoint:

```bash
python scripts/collect_bootstrap.py --help
python scripts/train_world_model.py --help
python scripts/train_imitation.py --help
python scripts/eval.py --help
```

Reset một lần để kiểm camera và shape:

```bash
python scripts/collect_bootstrap.py --config configs/train.yaml --dry-run
```

Expected image `(64,64,3) uint8`, state `(19,) float32`, action `(2,)`.
Nếu shape khác, kiểm env config trước khi collect.

Smoke pure expert 256 bước (đủ thử pipeline, **không đủ học lái**):

```bash
python scripts/collect_bootstrap.py --config configs/train.yaml --env-config configs/env_metadrive.yaml --policy pure_expert --steps 256 --out data/replay_buffer/expert_smoke.npz
python scripts/preview_data.py --buffer data/replay_buffer/expert_smoke.npz --num-samples 4 --out-dir data/raw_rollouts/expert_smoke
```

Pure expert deterministic, clip action về [-1,1], không noise/random fallback;
collector tạo `expert_smoke.manifest.json` chứa checksum và env config.
Mixed expert dùng 80% expert +20% random và steering noise; chỉ dùng làm WM
bootstrap, **không dùng làm nhãn BC**. Không đổi nhãn manifest để vượt validation.

Để collect dataset nghiên cứu, chốt maps/seeds, traffic, reward, số bước trước;
sao chép các YAML vào cùng một thư mục cấu hình run hoặc truyền rõ
`--env-config`/`--wm-config`. CLI collect không có `--seed`: sửa seed trong
bản train config riêng. Data/training map phải tách eval maps.

## 4. Smoke WM legacy, không phải continuous c

```bash
python scripts/train_world_model.py --config configs/train.yaml --env-config configs/env_metadrive.yaml --wm-config configs/world_model.yaml --device cpu --updates 1 --batch-size 1 --seq-len 8 --buffer data/replay_buffer/expert_smoke.npz --ckpt-dir checkpoints/wm_smoke --log-dir logs/wm_smoke
```

Output: `checkpoints/wm_smoke/world_model/latest.pt`. Script tự thêm
`world_model/`; không truyền dư suffix. Checkpoint ghi contract 2, model
weights, optimizer, wm_cfg/env_cfg và dimensions. Một update chỉ kiểm tích hợp.

Resume đúng WM legacy, đúng config và replay:

```bash
python scripts/train_world_model.py --config configs/train.yaml --env-config configs/env_metadrive.yaml --wm-config configs/world_model.yaml --device cpu --updates 1 --batch-size 1 --seq-len 8 --buffer data/replay_buffer/expert_smoke.npz --resume checkpoints/wm_smoke/world_model/latest.pt --ckpt-dir checkpoints/wm_smoke --log-dir logs/wm_smoke
```

WM `--updates` là **số update thêm**. Không resume optimizer khi đổi kiến trúc.
CLI này không có loss c và sẽ từ chối RSSM continuous.
Các helper c trong source chưa có public CLI hoàn chỉnh train-from-scratch;
cần checkpoint/recipe đã xác minh, không tự ghép các flag tưởng tượng.

## 5. BC: latent-only smoke và continuous có điều kiện

Chạy BC 1 update từ bộ smoke ở trên:

```bash
python scripts/train_imitation.py --config configs/train_pilot.yaml --expert-buffer data/replay_buffer/expert_smoke.npz --wm-checkpoint checkpoints/wm_smoke/world_model/latest.pt --out-dir checkpoints/bc_smoke --updates 1 --batch-size 1 --save-every 1 --posterior deterministic --device cpu --seed 0
```

`--out-dir` phải rỗng. BC lưu `latest.pt`, numbered checkpoint, manifest,
metrics và source provenance. BC CLI hiện **không có --resume**.
Không dùng smoke checkpoint để kết luận route/finish.

Nếu đã được cấp đúng WM continuous và expert replay cùng env config, lệnh
sau minh họa **BC-only 1 update**, không tái tạo baseline lịch sử:

```bash
python scripts/train_imitation.py --config configs/train_pilot.yaml --expert-buffer data/replay_buffer/expert.npz --wm-checkpoint checkpoints/continuous_wm.pt --control-state --control-state-source continuous --posterior alternating --updates 1 --batch-size 1 --save-every 1 --out-dir checkpoints/continuous_bc_smoke --device cpu --seed 0
```

Hai input ở lệnh này phải có sẵn; repo không tự sinh `continuous_wm.pt`.
`observed` là GT-state đối chứng, `predicted` đòi head learned control_state,
`continuous` đòi RSSM c. Không thay control source để làm checkpoint load được.
BC chỉ học Actor; Critic và output variance chưa được chứng minh tốt.

## 6. Eval đúng checkpoint

Smoke eval 1 episode, 10 bước:

```bash
python scripts/eval.py --device cpu --checkpoint checkpoints/bc_smoke/latest.pt --episodes 1 --start-seed 10000 --horizon 10 --output logs/bc_smoke/eval_smoke.json
```

Eval checkpoint continuous baseline được cấp, ví dụ:

```bash
python scripts/eval.py --device cpu --checkpoint checkpoints/bc_baseline.pt --episodes 4 --start-seed 44200 --output logs/bc_baseline/eval_44200_44203.json
```

Đây là clean evaluation maps 44200–44203, **không phải** ablation dataset 72
episode hoặc bộ noise/swerve. CLI này không có flag off-BC perturbation.
Bỏ `--horizon` dùng horizon env trong checkpoint; kiểm metadata trước.
Checkpoint có `models` (BC/online self-contained) không truyền
`--wm-checkpoint`. Chỉ offline Actor-only mới cần đúng WM riêng.

Eval không có `--checkpoint` chạy random policy. Mặc định deterministic lấy
argmax posterior và deterministic Actor. JSON ghi aggregate + từng episode,
không tự lưu toàn bộ raw trajectory. Crash metric tổng gồm out-of-road;
đọc cả các trường riêng theo [evaluation](../evaluation/evaluate.py).

## 7. RL và resume lịch sử: chỉ tra cứu

`scripts/train.py` là offline Actor-Critic; `scripts/train_online.py` luân phiên
collect, WM update và AC update. **Không chạy hai workflow này hiện tại.**
Không đưa BC checkpoint vào online `--resume`: nó không có đủ online
counters/optimizer/replay. Fresh online từ WM cũng không tự nạp Actor BC.

Có thể tra parser an toàn:

```bash
python scripts/train.py --help
python scripts/train_online.py --help
```

Online `--total-steps` là mốc tổng env steps; `--ckpt-dir` tự thêm `online/`.
Resume online dùng training config trong checkpoint nếu không truyền config,
Actor config luôn từ checkpoint; cần replay đúng mốc. Cách lưu/check ở
[ARTIFACTS](ARTIFACTS.md). Có flag chạy được không phải cho phép vượt gate.

## 8. Thành viên mới nên tiếp tục ở đâu?

Ngày 09/10/2026 đã chạy kiểm tra 9 CLI help, dry-run camera, collect256,
preview, WM1update →resume thêm1, BC1update và eval1episode/horizon10 theo
chuỗi smoke ở trên. Đây là kiểm chứng tích hợp, không phải benchmark chất lượng.
Lệnh BC continuous có điều kiện cần artifact riêng, không được coi là tái lập
baseline chỉ từ các YAML mặc định.

Chạy tests → kiểm artifact/config → eval baseline đúng protocol → đọc
[gate](PILOT_REVIEW_20260925.md) → đăng ký ablation một thay đổi.
Giữ baseline/prior/Actor/heads/Critic nếu chỉ sửa posterior. Tách
train/validation/test/confirmation; không tuning test. Lưu evidence và kiểm
completed/gate trước khi mở eval vòng kín. Không tự mở RL hoặc lateral mapping.
