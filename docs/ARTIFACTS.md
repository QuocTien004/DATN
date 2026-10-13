# Data, checkpoint và cấu hình

## 1. Transition contract

`obs_t + executed_action_t → obs_(t+1), reward_t, terminated_t`.

Replay hiện hành lưu images (uint8 HWC), states (float32,19), actions
(float32,2, [-1,1]), rewards, dones, terminated, episode_start, cùng metadata
ring buffer và next observations. `ReplayBuffer.sample(..., include_next=True)`
tạo batch transition-aligned cho WM. Không đọc trực tiếp archive bằng cách
giả định mọi next obs luôn là row kế tiếp: có terminal boundary và tail.

`done = terminated hoặc truncated` quyết định reset. Timeout truncation không
tự là terminal thật. Archive cũ không có terminated fallback về done là hạn chế
dữ liệu, không phải khôi phục đúng timeout.

State theo observation MetaDrive đang dùng, không đổi interface tùy tiện:
heading index2, speed index3, previous executed action index5/6 ở thang [0,1],
lateral index8. Không coi các index này là đơn vị góc/metre thô. Lateral đang
phụ thuộc lane reference nên có thể nhảy khi reference đổi; navigation-reference
mapping chưa được chấp nhận. c và Actor đều có normalization lưu trong weights.

Pure expert replay có `<stem>.manifest.json`: schema_version1, policy pure_expert,
expert_deterministic true, action_transform clip[-1,1], replay_sha256 và env_cfg.
[Validator](../utils/demonstrations.py) kiểm metadata/hash trước BC.
Off-BC diagnostic NPZ không mặc nhiên là ReplayBuffer hay expert dataset;
phân biệt bc_action với executed_action.

## 2. Phân loại checkpoint

| Loại | Nội dung/đường dùng |
|---|---|
| WM legacy contract2 | models, optimizer, wm_cfg, env_cfg, dimensions, step; CLI WM resume |
| Offline Actor-Critic | actor_critic, optimizer/counters, liên kết WM; eval cần WM tương ứng |
| Online | WM + Actor/Critic + optimizers + config + env_steps + replay_path; replay đúng mốc |
| BC | WM + Actor/Critic config/weights + bc_updates + bc_optimizer + imitation metadata |
| Continuous BC diagnostic | BC có control_input continuous và wm_cfg chứa c; eval bằng model-state |
| Posterior/prior patch | Chỉ module được fit; không phải checkpoint eval self-contained |

BC save giữ Critic nhưng không có nghĩa Critic đã được train. Patch không được
đưa thẳng vào eval.py; chỉ áp vào bản sao đúng baseline sau kiểm hash/module
scope và xuất đủ metadata. Không sửa control_input thành observed để “chạy được”.

Nạp bằng [load_world_model_state](../models/world_model.py) kiểm module/shape.
Warm-start thêm control embedding/c có API tường minh, không load strict=False
đại trà. Sau thay kiến trúc không dùng optimizer cũ với parameter list mới.

## 3. Xem metadata an toàn

Chỉ nạp artifact tin cậy. Helper hiện gọi torch.load mà không truyền weights_only;
chế độ an toàn phụ thuộc phiên bản PyTorch (bản 2.11 mặc định weights_only=True).
Không tự tắt bảo vệ khi file lạ bị từ chối. Ví dụ chạy từ root:

```python
from utils.checkpoint import load_checkpoint
ck = load_checkpoint("checkpoints/bc_baseline.pt", map_location="cpu")
for key in ("checkpoint_kind", "transition_contract", "control_input",
            "image_shape", "state_dim", "action_dim", "bc_updates", "env_steps"):
    print(key, ck.get(key))
print(ck.get("wm_cfg", {}))
print(ck.get("actor_critic_cfg", {}))
```

Không commit output metadata chưa rà soát: manifest có thể chứa đường dẫn run.

## 4. Config thực tế và độ ưu tiên

| File | Mặc định/ý nghĩa |
|---|---|
| configs/env_metadrive.yaml | 64×64, horizon1000, traffic0.1, custom reward bật |
| configs/world_model.yaml | Encoder512, h512, z32×32; không c, không control head |
| configs/train.yaml | seed0, CPU, seq32, WM batch8; config legacy có RL dài, không phải kế hoạch được duyệt |
| configs/train_pilot.yaml | Config pilot lịch sử 2k env steps, imagination15; không có chứng nhận gate |
| configs/env_stage2_dense_traffic.yaml | Stage2 riêng; không tự nâng độ khó khi Stage1 chưa đạt |

`load_experiment_configs` tìm env/WM YAML **cùng thư mục với train config** nếu
không truyền override. Đây không phải deep-merge train_pilot với train.yaml.
BC chỉ đọc train config đã chỉ định, WM/env lấy từ checkpoint và expert manifest.
Eval ưu tiên env/WM/Actor config trong checkpoint khi có; CLI env/horizon có thể
override nên phải ghi lại để so sánh đúng. WM legacy resume dựng model từ config
được truyền, không tự lấy mọi setting từ checkpoint.

Các khóa opt-in: encoder.control_embedding, rssm.categorical_embed_dim,
rssm.continuous_state, control_state_predictor, predictors.continuous_state_dim;
`rssm.continuous_state.exact_action_memory` là ablation P2 mặc định tắt;
phải lưu flag cùng checkpoint config, không thể suy ra từ weights.
Actor control_state_dim/source/mean/std, Critic critic_control_state_dim.
Xem source tương ứng và [kiến trúc](phase_c.md); không có một YAML mặc định
bao trùm mọi nhánh experimental.

Reward environment ở [envs/reward.py](../envs/reward.py) cộng shaping vào reward
gốc: progress có lateral factor, lateral penalty, idle tùy config,
success/crash bổ sung. Reward head WM là dự đoán reward này, không phải hàm
shaping. Không đổi reward config âm thầm khi tái dùng replay/checkpoint.

## 5. Lưu/resume không mất đối chứng

Checkpoint experimental RL30k đã audit là evidence suy giảm, không phải
baseline mới hoặc đầu vào được duyệt để resume. Wrapper reward/continue
chưa tương thích loader WM chuẩn. Giữ BC gốc và không đổi weights/config
để ép load; xem [audit và kế hoạch bàn giao](P1_P4_AUDIT.md).

Online output root tự thêm `online/`: latest.pt + online_buffer.npz.
Checkpoint env_steps phải bằng replay checkpoint_step. Numbered checkpoint chỉ
resume được nếu giữ replay đúng mốc; repo không giữ mọi phiên bản replay.
Không thay replay thiếu bằng bootstrap.

Save dùng file tạm/rename cho từng file, không phải transaction cả cặp.
Chụp bản sao sau khi dừng ở mốc hoàn chỉnh, kiểm step/hash trước khi dùng.
Online resume bắt đầu episode mới, không lưu toàn bộ simulator/RNG để tái hiện
bit-for-bit. WM updates là số thêm, online total-steps là mốc tổng, BC CLI chưa resume.

Replay load có thể tạo image mmap cạnh NPZ. Làm việc trên ổ local đủ dung lượng,
giữ NPZ gốc, không phân phối mmap như dữ liệu gốc. Luôn giữ cùng source/config,
dependency freeze, checksum, metrics, gate và trajectory cần thiết. Không chỉ
giữ latest.pt rồi bỏ mất baseline hoặc provenance.
