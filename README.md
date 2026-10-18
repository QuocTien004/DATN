# World Model + MetaDrive

Dự án nghiên cứu điều khiển xe trong MetaDrive bằng World Model và Actor-Critic
theo hướng Dreamer. Đây là implementation riêng, không phải bản tái lập đầy đủ
DreamerV3. Input gồm RGB trước 64×64×3 và state 19 chiều; action là
`[steering, throttle/brake]` trong `[-1,1]`.

## Trạng thái ngày 17/10/2026

Baseline nghiên cứu đang giữ là Actor học Behavior Cloning (BC), nhận latent
h/z và state do nhánh continuous c tạo ra. Không thay baseline bằng các refit
chưa đạt gate. **RL vẫn khóa theo quy trình nghiên cứu; không train dài.**

Prior c, tương thích Actor sau sửa posterior và reward/continue chưa đủ gate.
Code chạy được không có nghĩa model đã hoàn thiện. YAML mặc định vẫn là
latent-only lịch sử, không tự tạo baseline c.
Xem [trạng thái và giới hạn](docs/PILOT_REVIEW_20260925.md).
Pipeline World Model → BC → Actor–Critic và hệ thống đánh giá đã được triển
khai, kiểm thử. Benchmark BC lịch sử đạt **82,10% route, 14/24 finish** trên
maps 42000–42011 với hai seed train Actor 0/1, không phải kết quả của YAML
mặc định hoặc của riêng checkpoint seed0.

Bao đã tích hợp audit P1–P4: P1/P2/P3 có cải thiện cục bộ nhưng chưa qua gate,
P4 chưa xác minh Critic. P2 mặc định tắt, P1/P3 chỉ diagnostic, readiness guard
giữ nguyên. Checkpoint RL30k có **30.000 env steps / 750 updates cho mỗi mạng
Actor và Critic**. Paired cùng map/seed: BC **57,24%, 2/8 finish** → RL30k
**18,86%, 0/8**. Checkpoint này là evidence suy giảm, không thay BC baseline.
Các bộ benchmark và eval30k 28,94% được tách ở
[audit và cách tái lập](docs/P1_P4_AUDIT.md). Log kiểm thử tích hợp gần nhất:
**137/137 pass**; tests pass không chứng minh policy đã đạt chất lượng.

Audit mới đã đọc checkpoint `continuous_seed0_refit_warmup_c23` và tái tính
H1/H5/H15 trên 150 episode có checksum. Refit exact-memory hai seed cải thiện
lateral H15 nhưng **vẫn fail gate**. `c23` là BC với observation/control 23 chiều
trực tiếp, RSSM chỉ có h/z, không phải continuous prior đã được sửa.
JSON của Tien ghi 84,46% route, 5/10 finish khi chạy thuần và 98,68%, 10/10
khi có shield trên maps 10000–10009. Đây là kết quả lưu sẵn của hệ kết hợp,
chưa tái lập vòng kín hoặc xác nhận tổng quát hóa. Xem
[audit tuần 12–17/10](docs/P1_P4_AUDIT.md#cập-nhật-17102026--refit-và-checkpoint-c23).

## Hướng tiếp theo (Next Steps)

Refit exact-memory đã có kết quả, chưa cần lặp lại cùng lượt train. Ưu tiên
giải quyết gate heading/lateral và kiểm lại protocol map/RNG. Hướng dự phòng
được đề xuất là WM + BC, đánh giá riêng model-state, observed-state và shield.
Chuyển cách học sang offline không tự khắc phục sai số tưởng tượng. Xem
[kế hoạch bàn giao cập nhật](docs/P1_P4_AUDIT.md#hướng-tiếp-theo-sau-audit-1710).

## Bắt đầu

Dùng Python 3.10, tạo và kích hoạt venv, cài PyTorch phù hợp nền tảng rồi cài
`requirements.txt`. Lệnh cụ thể ở [TRAINING_GUIDE](docs/TRAINING_GUIDE.md).
Unit tests không cần checkpoint:

```bash
python -m unittest discover -s tests -v
```

Data/checkpoint không nằm trong Git. Phải có đúng artifact và metadata,
không đổi tên hoặc sửa metadata checkpoint cũ để giả làm baseline mới.

## Tài liệu

- [Kiến trúc và pipeline](docs/phase_c.md): RSSM h/z/c, posterior/prior, BC, imagination.
- [Setup, collect, train BC, eval](docs/TRAINING_GUIDE.md): các lệnh có trong source.
- [Data, checkpoint, cấu hình](docs/ARTIFACTS.md): contract v2 và load/resume đúng loại.
- [Trạng thái và gate](docs/PILOT_REVIEW_20260925.md): baseline, experimental, chưa đạt.
- [Tests và tái lập](docs/notes.md).
- [Lịch sử sửa lỗi](docs/RETRAINING_NOTES.md).
- [Mục lục và hình lịch sử](docs/README.md).

## Source chính

| Thư mục | Vai trò |
|---|---|
| `envs/` | Observation, reset/step, reward shaping |
| `models/` | Encoder, RSSM, decoder, heads, Actor/Critic |
| `training/` | WM loss, BC, posterior/prior ablation, imagination |
| `evaluation/` | Policy có bộ nhớ và metrics |
| `scripts/` | CLI collect/train/eval |
| `configs/` | Cấu hình khởi tạo, không thay metadata checkpoint |

Không commit data, checkpoint, credential hoặc log cá nhân. Giữ output riêng
cho từng experiment, không ghi đè artifact đối chứng.
