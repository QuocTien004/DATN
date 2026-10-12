# World Model + MetaDrive

Dự án nghiên cứu điều khiển xe trong MetaDrive bằng World Model và Actor-Critic
theo hướng Dreamer. Đây là implementation riêng, không phải bản tái lập đầy đủ
DreamerV3. Input gồm RGB trước 64×64×3 và state 19 chiều; action là
`[steering, throttle/brake]` trong `[-1,1]`.

## Trạng thái ngày 12/10/2026

Baseline nghiên cứu đang giữ là Actor học Behavior Cloning (BC), nhận latent
h/z và state do nhánh continuous c tạo ra. Không thay baseline bằng các refit
chưa đạt gate. **RL vẫn khóa theo quy trình nghiên cứu; không train dài.**

Prior c, tương thích Actor sau sửa posterior và reward/continue chưa đủ gate.
Code chạy được không có nghĩa model đã hoàn thiện. YAML mặc định vẫn là
latent-only lịch sử, không tự tạo baseline c.
Xem [trạng thái và giới hạn](docs/PILOT_REVIEW_20260925.md).
Nhánh tích hợp P1–P4 giữ baseline và readiness guard; P2 chỉ opt-in,
P1/P3 chỉ diagnostic. Xem [audit và cách tái lập](docs/P1_P4_AUDIT.md).

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
