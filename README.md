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

| Vai trò | Module |
|---------|--------|
| World Model | `models/encoder.py`, `rssm.py`, `decoder.py`, `predictors.py`, `training/train_world_model.py` |
| Actor-Critic & tích hợp | `models/actor_critic.py`, `training/train_agent.py`, `evaluation/`, baseline PPO/SAC |

## Trạng thái hiện tại

- [x] Phase A: cấu trúc repo, env wrapper (1 cam trước), ReplayBuffer, scripts collect/preview  
- [ ] Phase B: implement & train World Model  
- [ ] Phase C: Actor-Critic imagination  
- [ ] Phase D: eval, baseline, báo cáo  
