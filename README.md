# World Model + MetaDrive

Autonomous driving with a **DreamerV3-style World Model (RSSM)** and **Actor-Critic** (imagination), trained in [MetaDrive](https://github.com/metadriverse/metadrive).

**Observation:** front RGB `(256×256×3)` + 19-D state · **Action:** `(steering, throttle)` ∈ [-1, 1]

## Setup

```bash
git clone https://github.com/QuocTien004/DATN.git
cd DATN
python -m venv venv
# Windows: venv\Scripts\activate

pip install -r requirements-cuda.txt   # or requirements-cpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Layout

```text
configs/       env_metadrive.yaml, world_model.yaml, train.yaml
envs/          MetaDrive wrapper
models/        Encoder, RSSM, Decoder, predictors, Actor-Critic
training/      collect, world-model / agent updates
evaluation/    metrics, evaluate
scripts/       CLI entrypoints
data/          replay buffer (gitignored)
checkpoints/   weights (gitignored)
```

## Hyperparameters

| | Train (laptop) | Reference |
|--|----------------|-----------|
| Image | 256×256 RGB | — |
| Latent | categorical 32×32 | DreamerV3-style |
| `batch_size` | 8 | 16 |
| `sequence_length` | 32 | 64 |
| WM `updates` | 5000 | — |
| Bootstrap steps | 20000 | — |
| Device | `cuda` | — |

See `configs/train.yaml`, `configs/world_model.yaml`, `configs/env_metadrive.yaml`.

## Data collection

```bash
python scripts/collect_bootstrap.py --config configs/train.yaml --dry-run
python scripts/collect_bootstrap.py --config configs/train.yaml
python scripts/preview_data.py --config configs/train.yaml --steps 80 --num-samples 6
```

Expect: `image=(256,256,3) uint8`, `state=(19,) float32`, `action_dim=2`.  
Output: `data/replay_buffer/bootstrap.npz`.

## Train World Model

```bash
python scripts/train_world_model.py --config configs/train.yaml --device cuda
python scripts/train_world_model.py --config configs/train.yaml --device cuda --resume checkpoints/world_model/latest.pt
```

Checkpoints: `checkpoints/world_model/`.

## Train agent / eval (skeleton)

```bash
python scripts/train.py --config configs/train.yaml
python scripts/eval.py --config configs/train.yaml
```

## Status

- [x] Env wrapper, replay buffer, bootstrap collect  
- [x] World Model (Encoder / RSSM / Decoder / predictors) + train loop  
- [x] Converged WM checkpoint (`checkpoints/world_model/latest.pt`)  
- [ ] Actor-Critic imagination training  
- [ ] Hold-out evaluation & report  
