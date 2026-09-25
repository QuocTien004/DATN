# Thay đổi phục vụ train lại — 2026-09-24

Đây là lịch sử bản v2. Kết quả train v2 50k và thay đổi tiếp theo ngày 25/09 ở
[PILOT_REVIEW_20260925.md](PILOT_REVIEW_20260925.md). Thư mục Drive cũ bên dưới
đã được nhóm chuyển vào `DoAn/Br_Bao/smoke/`; link vẫn giữ nguyên.

## Vì sao cần sửa trước khi tăng số bước

Log lượt cũ dừng ở 403k; checkpoint mới được chia sẻ là 400k. Trong log, cả 40 lần
eval có success 0%. Ở 400k: route completion 15.20%, episode length 60.35 và
out-of-road 100%. Không dùng loss giảm để khẳng định đã học lái.

Kiểm tra trực tiếp checkpoint trên T4 hoàn tất 4 seed 10000–10003: cả 4 out-of-road,
không có success. Đây là subset ngắn để chẩn đoán, không thay thế eval 20 episode.
Ga trên các seed này hầu như luôn >0.9; mean khoảng 0.964, đúng mức tanh(2) của
mean bị clamp cứng. Lateral factor có mặt ở toàn bộ các bước đã ghi, nên giả thuyết
lane gate bị tắt không được xác nhận trong lần test này.

## Các sửa đổi

| Vấn đề | Hành vi sau sửa | Code |
|---|---|---|
| Batch bỏ `is_first` | Truyền reset flag đến WM và posterior start states | `training/batches.py` |
| Reward/continue target đặt tại obs_t nhưng imagination đọc next latent | Train heads từ obs_(t+1), action_t, kể cả terminal next obs | `training/train_world_model.py` |
| KL balance 0.8 đặt lên posterior/representation | Đặt 0.8 lên prior/dynamics, 0.2 lên representation; học cả current và successor prior | `training/train_world_model.py` |
| `done` trộn terminal và timeout | Replay mới có `terminated` riêng; timeout reset RSSM nhưng không học như terminal thật | `utils/replay_buffer.py`, collectors |
| Load bỏ write pointer và dùng lại mmap chỉ theo size | Giữ idx/full, unwrap khi mở rộng ring, rebuild mmap từ archive hiện tại | `utils/replay_buffer.py` |
| Tail transition không có successor khi sample | Lưu next obs mới nhất cùng terminal next obs; bỏ sequence thiếu successor trong archive cũ | `utils/replay_buffer.py` |
| Resume mặc định quay về bootstrap | Link replay trong checkpoint, kiểm tra replay step và checkpoint step | `utils/online_state.py`, `scripts/train_online.py` |
| Eval riêng dùng WM offline cho Actor online | Dùng WM trong chính online checkpoint; configure seed range trước khi tạo env | `scripts/eval.py` |
| Eval cắt episode nhưng counters còn tồn | Reset return/length/recurrent state; đánh dấu boundary khi collect lại | `training/online_trainer.py` |
| Hard clamp làm gradient bằng 0 ngoài miền | Actor mới dùng mean tanh trơn, std sigmoid; giữ legacy transforms khi eval checkpoint cũ | `models/actor_critic.py` |
| Bootstrap bị replay vòng ghi đè | Giữ bản bootstrap riêng và mix 25% sequences trong batch | `training/online_trainer.py` |
| Hai nguồn exploration và throttle std lớn | Giảm std/entropy, tắt Gaussian noise cộng ngoài | `configs/train.yaml` |
| Stage 2 khác resolution và dùng reward keys không được đọc | Đồng bộ 64×64 và schema reward đang dùng | `configs/env_stage2_dense_traffic.yaml` |
| Khó truy nguồn experiment | Lưu config/CLI/git revision + dirty status; per-seed eval JSON | CLI train/eval |

Các lỗi code là điều đã xác minh; chúng chưa chứng minh từng lỗi đóng góp bao nhiêu
vào failure của lượt cũ. Không khẳng định bạn đã resume sai chỉ từ đường biểu diễn.
Muốn biết cần có lịch sử lệnh chạy/phiên Colab.

## Contract v2 và migration

Checkpoint mới có `transition_contract: 2`. Online training/resume và WM resume
từ chối checkpoint cũ để tránh tiếp tục dùng reward/continue heads học sai căn
thời gian. Pretrain World Model từ đầu, tạo Actor mới, giữ checkpoint cũ để eval.
Replay bootstrap 64×64 cũ có thể đọc nhưng thiếu `terminated`; code fallback về
`done`, không thể phân biệt timeout cũ. Recollect bằng code mới là đường chuẩn.

Reward coefficients vẫn giữ nguyên; thêm log base/shaping/lateral để tune dựa trên
bằng chứng. Smooth bounds, entropy 0.003 và demo ratio 0.25 là cấu hình pilot, chưa
được chứng minh nâng success rate. Chạy pilot nhiều seed theo
[TRAINING_GUIDE.md](TRAINING_GUIDE.md) trước khi đầu tư train dài.

## Xác minh

21 tests đã pass trên CPU local và Python 3.10/Colab T4. Regression tests kiểm tra reset flag, RSSM reset isolation, successor head alignment,
timeout target, circular replay round-trip, stale mmap, latest next observation,
demo mixing, gradient qua smooth bounds và cặp model/replay không khớp.
Các test nhỏ chỉ xác minh tính đúng của pipeline. Báo cáo raw kết quả GPU và log
chẩn đoán được lưu trong [Drive: DoAn/smoke/2026-09-24_retrain_v2](https://drive.google.com/drive/folders/16-wucShQmW0T-jys5jn2J4P3M5Y2USGV);
không đưa data/checkpoint lớn vào Git.

GPU smoke dùng kiến trúc đầy đủ với dữ liệu thật: 2 WM updates; online train 8
steps có periodic eval; resume tới 12 rồi 16 steps; eval độc lập từ online
checkpoint. WM CLI với các flags trong guide cũng chạy được 1 update. Horizon
smoke được rút còn 4 bước để kiểm tra tích hợp, không dùng đo success rate.
