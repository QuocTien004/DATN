# Phase C — Actor-Critic imagination

> Cập nhật: workflow online và transition contract v2 xem
> [TRAINING_GUIDE.md](TRAINING_GUIDE.md) và [RETRAINING_NOTES.md](RETRAINING_NOTES.md).
> Phần kết quả ngày 2026-09-04 bên dưới là kiểm tra lịch sử, không phải bản train v2.

Tài liệu này là contract ngắn để hai phần World Model và Actor-Critic tích hợp với
nhau. Implementation là **DreamerV3-style**, không phải bản sao đầy đủ DreamerV3.

## Artifact cần có

Không commit replay/checkpoint lớn vào Git. Đặt local theo mặc định:

```text
data/replay_buffer/bootstrap.npz
checkpoints/world_model/latest.pt
```

Replay sequence phải có:

- `images`: `(N, 64, 64, 3)`, `uint8` (bộ lịch sử 256×256 cần config riêng);
- `states`: `(N, 19)`, `float32`;
- `actions`: `(N, 2)`, `float32`, miền `[-1, 1]`;
- `rewards`, `dones`, `terminated`, `episode_start`: `(N,)`;
- metadata của `ReplayBuffer` hiện tại.

World Model checkpoint cần mapping `models` chứa ít nhất `encoder`, `rssm`,
`reward`, `continue`. Checkpoint chuẩn còn có `decoder`, `wm_cfg`, `image_shape`,
`state_dim`, `action_dim` và `step`. Phase C kiểm tra metadata trước khi train và
báo lỗi nếu latent/data dimensions không tương thích.

Interface hiện tại:

```text
RSSM state:
  h: (B, 512)
  z: (B, 32, 32) categorical one-hot/ST

RewardPredictor(h, z):   (B, 1) reward
ContinuePredictor(h, z): (B, 1) logits
MetaDrive action:        (B, 2) trong [-1, 1]
```

## Luồng Phase C

Replay RGB/state được Encoder và `RSSM.observe_step()` biến thành posterior start
states. Actor tạo tanh-squashed Normal action, sau đó `RSSM.imagine_step()` rollout
trong latent space. Reward/continue/value dự đoán được dùng để tính lambda-return.

World Model parameters bị freeze trong Actor-Critic update nhưng imagination không
được bọc toàn bộ bằng `torch.no_grad()`: gradient vẫn cần chạy qua dynamics từ
imagined return về action/Actor. Actor và Critic dùng hai optimizer riêng; Critic
target và imagined state trong Critic update đều detach.

## Test nhanh

```bash
python -m unittest discover -s tests -v
```

## Train, resume và evaluate

Train từ replay + pretrained World Model:

```bash
python scripts/train.py --config configs/train.yaml --device cuda \
  --buffer data/replay_buffer/bootstrap.npz \
  --wm-checkpoint checkpoints/world_model/latest.pt
```

Resume:

```bash
python scripts/train.py --config configs/train.yaml --device cuda \
  --buffer data/replay_buffer/bootstrap.npz \
  --wm-checkpoint checkpoints/world_model/latest.pt \
  --resume checkpoints/actor_critic/latest.pt
```

Evaluate recurrent latent policy:

```bash
python scripts/eval.py --config configs/train.yaml --device cuda \
  --checkpoint checkpoints/actor_critic/latest.pt \
  --wm-checkpoint checkpoints/world_model/latest.pt
```

`LatentActorPolicy` reset RSSM state và previous action ở đầu mỗi episode, rồi cập
nhật posterior từ observation mới trước khi lấy deterministic action.

## Colab

MetaDrive 0.4.3 đã được xác minh với Python 3.10.12. Colab không có `DISPLAY` cần
chạy RGB offscreen evaluation qua Xvfb:

```bash
xvfb-run -a python scripts/eval.py --config configs/train.yaml --device cuda \
  --checkpoint checkpoints/actor_critic/latest.pt \
  --wm-checkpoint checkpoints/world_model/latest.pt
```

Nên copy `bootstrap.npz` từ Drive sang `/content` trước khi train. `ReplayBuffer`
tạo file image mmap cạnh `.npz`; làm trực tiếp trên Drive vừa chậm vừa tạo file lớn.

## Trạng thái xác minh ngày 2026-09-04

Đã test trên Colab Tesla T4, Python 3.10.12, Torch 2.11.0 CUDA và MetaDrive 0.4.3:

- 8/8 unit/smoke tests pass;
- buffer thật 20.000 transition cho 19.317 sequence hợp lệ ở length 32;
- World Model checkpoint step 5000 load đủ các module;
- 2 Actor-Critic update thật trên GPU thành công, loss/gradient finite;
- checkpoint agent lưu đúng step 2 và latent/action dimensions;
- evaluation end-to-end trên hold-out seed 10000, horizon smoke 20 thành công qua
  Xvfb và trả đủ success/crash/return/length metrics.

Hai update và episode 20 bước chỉ xác minh integration, **không chứng minh policy đã
học lái tốt**. Training dài, tuning và report 20 hold-out episode vẫn là TODO.
