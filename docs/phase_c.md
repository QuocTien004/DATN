# Kiến trúc World Model và Actor-Critic

## 1. Pipeline

World Model (WM) học biểu diễn observation và hậu quả action.
Behavior Cloning (BC) học bắt chước expert. Reinforcement Learning (RL) tối ưu
Actor bằng reward dự đoán trong imagined rollout. Các bước có loss/optimizer riêng.

```text
obs_t + executed_action_t → obs_(t+1), reward_t, terminal_t
            ↓
Encoder → posterior RSSM (h,z,[c]) → Actor BC → action thật để eval
            ↓
   prior + action của Actor (không đọc observation tương lai)
            ↓ lặp
 imagined state → reward/continue + Critic → mục tiêu RL [đang khóa]
```

Vector state 19 chiều do simulator cung cấp. Posterior c học từ embedding của
vector này khi quan sát thật; không gọi hệ thống là điều khiển chỉ bằng camera.

Pipeline trên đã triển khai và kiểm thử, chưa chứng nhận chất lượng RL.
Audit P1–P4 được tích hợp vào Bao, giữ guard và BC. P2 exact action-memory
chỉ opt-in, mặc định false; P1 GT-start là oracle. Xem
[kết quả và Next Steps](P1_P4_AUDIT.md) trước khi refit transition.

## 2. RSSM h/z/c

Recurrent State-Space Model là mô hình trạng thái có bộ nhớ.
Source: [Encoder](../models/encoder.py), [RSSM](../models/rssm.py),
[ContinuousState](../models/continuous_state.py).

| Biến | Cấu trúc | Vai trò |
|---|---|---|
| h | GRU, mặc định 512 chiều | Bộ nhớ từ h trước, z trước, action trước |
| z | 32 nhóm ×32 lớp categorical one-hot với straight-through gradient | Biểu diễn ngẫu nhiên; prior đọc h, posterior đọc h + embedding |
| c (opt-in) | Vector liên tục, baseline dùng 19 chiều | State điều khiển có posterior và transition riêng |

Encoder mặc định ghép CNN ảnh + MLP state thành embedding 512.
Opt-in `encoder.control_embedding` bổ sung Linear + ELU từ state chuẩn hóa;
baseline dùng 64 chiều riêng. `rssm.categorical_embed_dim` giữ prefix 512 cho
h/z cũ. Posterior c đọc slice riêng qua `embed_start/embed_stop`;
`posterior_hidden` quy định số lớp. Các thiết lập opt-in không nằm trong YAML mặc định.

**Posterior** cập nhật khi có observation mới: h qua GRU, z đọc embedding,
c dự đoán từ embedding điều khiển. **Prior** chạy khi không có observation:
h/z tiến theo action, `c_next = c + transition(c, action)`.
Transition là MLP, không nhận GT tương lai, không gọi encoder trong imagined step.

Posterior c hiện không nhận h/c trước trực tiếp. Memory action 5/6 học từ
observation, không nối tắt action thật vào Actor. c ở tọa độ chuẩn hóa;
`decode(c) = c * state_std + state_mean` là đổi thang cố định, không phải
decoder học thêm. Actor có normalization control riêng, phải giữ cả hai bộ.
`RSSM.get_feature()` vẫn chỉ trả h/z; module dùng c phải yêu cầu tường minh.

## 3. World Model

[Builder](../models/world_model.py) tạo encoder, RSSM, decoder (có thể bỏ khi
eval), reward/continue heads; tùy config có control-state predictor.

- Decoder tái tạo ảnh/state từ h/z. Reconstruction loss thấp không chứng minh
  latent đủ để điều khiển hoặc rollout dài.
- Reward head học reward symlog tại **successor state** sau action.
  Imagination dùng symexp đổi lại, có giới hạn đầu ra trong implementation.
- Continue head trả logit. Sigmoid là xác suất tiếp tục, discount là
  `gamma * sigmoid(logit)`. Target là `1 - terminated`, không phải
  `1 - done` khi timeout chỉ là cắt episode.
- Heads mặc định h/z; opt-in `predictors.continuous_state_dim` nối c chuẩn hóa.
  Có code không đồng nghĩa head đã qua terminal gate.

Loss WM legacy gồm reconstruction, KL categorical, reward và continue.
KL balance phân gradient cho prior/dynamics và posterior/representation;
free-nats đặt sàn KL. Trainer legacy **từ chối RSSM có c** vì thiếu supervision
nhánh đó. Không thêm c vào YAML rồi chạy CLI WM cũ để “train đủ”.

[Helpers nhánh c](../training/continuous_state.py) tách current-state supervision
và rollout H1/H5/H15: dự đoán liên tiếp 1/5/15 bước, không sửa lại bằng GT.
Future targets chỉ tính loss; mask padding và không đi qua reset episode.

## 4. Actor, Critic và BC

[Actor](../models/actor_critic.py) nhận h + z flatten, tùy chọn control 19 chiều.
MLP trả mean/std của Normal; action qua tanh. Eval deterministic dùng action từ
mean; training có thể lấy mẫu. Transform/entropy lấy từ checkpoint.

| Nguồn control | Quan sát thật | Imagination |
|---|---|---|
| latent-only | Chỉ h/z | Đường RL legacy có code |
| observed | GT state hiện tại, đối chứng | Không có GT tương lai, bị chặn |
| predicted | Head dự đoán state từ h/z | Helper experimental, chưa đạt gate |
| continuous | Posterior c → decode | Prior c → decode, diagnostic opt-in |

[BC loss](../training/imitation.py) khớp deterministic action với executed expert
action. Chỉ optimizer Actor đổi; WM/Critic không học bằng BC. Không trực tiếp
huấn luyện output std. Replay phải pure expert có manifest/checksum, không
đổi nhãn mixed/random thành expert.

Critic dự đoán value symlog; slow Critic tạo bootstrap cho lambda-return.
Mặc định Critic đọc h/z; `critic_control_state_dim` cho phép nhận model-state.
BC tốt không chứng minh Critic đã tốt.

## 5. Imagination và khóa RL

[training/train_agent.py](../training/train_agent.py) lấy posterior làm điểm đầu
rồi lặp Actor → prior → heads. Lambda-return kết hợp reward, discount, value.
Actor nhận gradient qua dynamics đóng băng; Critic học target detach.
Freeze WM không đồng nghĩa bọc toàn bộ imagination bằng no_grad.

`ActorCritic.require_imagination_ready()` chặn Actor có control bridge trong
đường gọi mặc định. `diagnostic_control=True` là ngoại lệ nghiên cứu tường minh,
không phải readiness. Diagnostic RL continuous yêu cầu Critic cùng control dim.

**Không có công tắc khóa RL toàn repo**: legacy latent-only vẫn chạy được.
Khóa hiện tại gồm guard control bridge và quy trình không mở RL khi gate fail.
Chưa có CLI public hoàn chỉnh tái tạo mọi thí nghiệm c từ đầu; helper đã test
không thay thế recipe và artifact đúng.

## 6. Các nhánh cần phân biệt

- Head/decoder trên h/z cũ khác c có transition riêng.
- Tăng reconstruction weight khác sửa posterior c.
- Joint WM + imitation là opt-in; WM loss chưa hỗ trợ predicted/continuous control.
- Posterior refit chỉ đổi control embedding + posterior; transition refit chỉ đổi prior.
- Lateral navigation-reference chưa triển khai, không đổi nghĩa state index 8.

Xem [trạng thái gate](PILOT_REVIEW_20260925.md) trước khi chọn workflow.
