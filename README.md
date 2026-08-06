# World Model + MetaDrive (Đồ án tốt nghiệp)

Hệ thống lái xe tự động dùng **World Model (RSSM theo chuẩn DreamerV3)** + **Actor-Critic** (imagination), huấn luyện trên [MetaDrive](https://github.com/metadriverse/metadrive).

**Observation:** 1 camera trước RGB `(256×256×3)` + state vector 19 chiều → Encoder (CNN + MLP).

Sơ đồ cấu trúc dữ liệu: [`docs/data_schema.png`](docs/data_schema.png)

## Cấu trúc project

```text
configs/          YAML: env, world model, train
envs/             MetaDrive wrapper + reward
models/           Encoder, RSSM, Decoder, predictors, Actor-Critic (stubs Phase B/C)
utils/            Seed, logger, checkpoint, ReplayBuffer
training/         Collect + train loops
evaluation/       Metrics + evaluate
scripts/          CLI: collect_bootstrap, preview_data, train, eval
docs/             Sơ đồ / tài liệu
data/             Rollouts & buffer (không commit dữ liệu lớn)
checkpoints/      Weights (gitignored)
logs/             Logs (gitignored)
```

## Setup

```bash
git clone https://github.com/QuocTien004/DATN.git
cd DATN
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
# MetaDrive từ PyPI (không cần clone source vào repo)
pip install "metadrive-simulator>=0.4.3"
```

Nếu bạn giữ bản clone local `./metadrive`, có thể: `pip install -e ./metadrive`.

## Chạy Phase A (thu thập / kiểm tra obs)

```bash
# Kiểm tra shape ảnh + state
python scripts/collect_bootstrap.py --config configs/train.yaml --dry-run

# Thu thập bootstrap (random policy)
python scripts/collect_bootstrap.py --config configs/train.yaml

# Xem mẫu dữ liệu
python scripts/preview_data.py --config configs/train.yaml --steps 80 --num-samples 6
```

Kỳ vọng dry-run: `image=(256,256,3) uint8`, `state=(19,) float32`, `action_dim=2`.

## Pipeline huấn luyện

1. Collect → ReplayBuffer  
2. Train World Model (RSSM)  
3. Train Actor-Critic bằng imagination  
4. Evaluate trên map seed hold-out  

```bash
python scripts/train.py --config configs/train.yaml   # Phase B/C (đang skeleton)
python scripts/eval.py --config configs/train.yaml
```

## Phân công (nhóm 2)

Nhánh Git gợi ý: Người A → `Tien` · Người B → `Bao` · ghép chung → `main`.

### Người A — World Model

**Công việc hiện tại (xây + tự test riêng, không cần Actor):**

| # | Công việc | File | Xong khi nào (tự test) |
|---|-----------|------|------------------------|
| A1 | Collect bootstrap 20k–50k step (1 lần) | `scripts/collect_bootstrap.py` | Có `data/replay_buffer/bootstrap.npz` |
| A2 | Encoder: CNN(ảnh) + MLP(state) → `e_t` | `models/encoder.py` | Shape `e_t` đúng, VD `(B, 512)` |
| A3 | RSSM: `h_t`, posterior `z_t`, prior `ẑ_t` | `models/rssm.py` | `z`/`ẑ` shape `(B, 32, 32)` |
| A4 | Decoder + Reward/Continue predictors | `models/decoder.py`, `models/predictors.py` | Forward ra được ảnh/reward/continue |
| A5 | Vòng train WM trên buffer | `training/train_world_model.py` | Loss recon/KL chạy, không NaN; loss có giảm |
| A6 | Lưu checkpoint WM | `utils/checkpoint.py`, `checkpoints/world_model/` | Có file `.pt` load lại được |

Config liên quan: `configs/world_model.yaml`, `configs/train.yaml` (batch=16, seq=64).

### Người B — Actor-Critic

**Công việc hiện tại (xây + tự test riêng, chưa cần WM thật — dùng mock `(h,z)`):**

| # | Công việc | File | Xong khi nào (tự test) |
|---|-----------|------|------------------------|
| B1 | Actor: `(h,z)` → action `(steering, throttle)` | `models/actor_critic.py` | Action shape `(B, 2)`, trong [-1, 1] |
| B2 | Critic: `(h,z)` → value | `models/actor_critic.py` | Value shape `(B, 1)` hoặc `(B,)` |
| B3 | Imagination loop (horizon 15) với **mock** `(h,z)` | `training/train_agent.py` | Chạy H bước không lỗi shape |
| B4 | Loss / update Actor-Critic (trên rollout giả) | `training/train_agent.py` | `backward()` chạy được |
| B5 | Eval script + metrics trên MetaDrive | `evaluation/evaluate.py`, `evaluation/metrics.py`, `scripts/eval.py` | Chạy được random/policy giả, in success/collision |

Config liên quan: `configs/train.yaml` → `imagination_horizon` (trong `world_model.yaml`).

### File chung
`envs/metadrive_wrapper.py`, `utils/replay_buffer.py`, `training/collect.py`, `training/trainer.py`, `scripts/train.py`, `configs/env_metadrive.yaml`

**Lưu ý data:** Người A lấy bootstrap (policy random) lúc đầu để train WM.

## Trạng thái hiện tại

- [x] Phase A: cấu trúc repo, env wrapper (1 cam trước), ReplayBuffer, scripts collect/preview  
- [ ] Phase B: implement & train World Model  
- [ ] Phase C: Actor-Critic imagination  
- [ ] Phase D: eval, baseline, báo cáo  
