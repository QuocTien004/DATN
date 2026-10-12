# Trạng thái nghiên cứu và gate

Cập nhật 09/10/2026. Giữ tên file để bảo toàn liên kết lịch sử; đây không còn
là hướng dẫn mở pilot RL tháng 9.

## Baseline và mức sẵn sàng

Baseline đang giữ là checkpoint BC continuous-state, Actor dùng h/z +
model-generated c. Refit posterior/prior là candidate, không ghi đè baseline.
YAML mặc định latent-only không phải recipe tái tạo checkpoint baseline này.
Model/data không nằm trong Git; phải có metadata/checksum đúng để so sánh.

| Nhánh | Trạng thái |
|---|---|
| Latent-only WM/Actor-Critic | Có train/eval và tests, không phải policy đạt chất lượng |
| Latent + observed GT | Đối chứng thiếu thông tin điều khiển, không dùng GT tương lai |
| Predicted-state head h/z | Experimental, chưa đủ gate |
| Continuous c + BC | Baseline chẩn đoán, chưa phải hệ thống hoàn thiện |
| Refit prior c off-BC | Chưa qua regression cũ, dù confirmation có cải thiện |
| Refit posterior state-only | State error giảm nhưng có BC regression, chưa chấp nhận |
| Posterior + action-anchor tổng | Diagnostic fail, không promote/eval vòng kín candidate |
| Posterior + budget riêng action/memory | Đã thử có đối chứng, vẫn fail diagnostic; giữ baseline |
| Reward/continue refit h/z hoặc h/z+c | Chưa đạt, tạm dừng để giải quyết c |
| Critic/control và diagnostic RL | Có helper opt-in, không được chạy tiếp hiện tại |

Loss action tổng có thể giảm chủ yếu ở ga/phanh trong khi steering/memory xấu
đi. Phải báo riêng tọa độ, không dùng trung bình che regression.
Phép thử budget riêng trên clean train đã giảm lệch steering nhưng làm state
regression. Khả thi khi chỉnh tự do từng đầu ra không có nghĩa posterior với
trọng số chung học được nghiệm đó. Không tăng update hay nới gate để nhận model.

## Thứ tự gate

1. Posterior và tương thích Actor: state/heading/memory 5/6 đúng hơn, action
   clean trong giới hạn đã chốt; sau đó mới eval BC vòng kín.
2. Prior c: H1/H5/H15 so hold-baseline, clean và noise/swerve/fast, timeout,
   chuyển lane reference và near-failure nếu dữ liệu đủ.
3. Reward/continue: terminal thật, out-of-road/crash, timeout; cả train và maps
   chưa thấy. Continue trung bình cao không chứng minh nhận diện terminal.
4. Critic/action ranking: không đánh giá tốt giả action gây hỏng.
5. BC không regression, rồi mới xin mở pilot RL ngắn có early-stop.

Hiện chưa đủ các gate: **RL khóa, không train 5k/50k/100k**.
Confirmation đẹp không xóa regression cũ. Không sửa posterior, prior và
representation cùng một ablation.

## Ngưỡng diagnostic gần nhất

Ablation action-anchor dùng training seed 0/1; train maps 44000–44011,
validation 44100–44105, test 44200–44211, confirmation 44300–44311.
Mỗi split có clean/noise/swerve/fast/timeout. Đây là tập chẩn đoán có can thiệp
action, không phải benchmark lái thuần expert.

Gate từng seed trên cả validation/test/confirmation:

- Clean action RMSE mỗi trục ≤80% state-only cùng seed.
- Clean và moving-offBC: RMSE tổng state, heading, memory 5/6 ≤110% đối chứng.
- Heading moving-offBC ≤50% baseline posterior gốc.

Đây là quy ước admission bảo thủ của thí nghiệm, không phải chuẩn học thuật
phổ quát hoặc gate mở RL. Test/confirmation không chọn trọng số/checkpoint.
Action-anchor tổng và budget riêng đều không đạt; chưa có route/finish mới
của các candidate. Đối chứng cùng seed/config được giữ để phân biệt thay đổi
objective với thay đổi runtime.

## Đọc kết quả đúng phạm vi

Route là phần lộ trình đi được, finish là sự kiện đến đích. Báo out-of-road,
crash, timeout, số episode hoàn tất và horizon. Đứng yên không crash chưa là an toàn.

Benchmark cần checkpoint/data hash, maps, training/reset/noise seed,
traffic/reward config, posterior sampling, renderer và kết quả từng episode.
Không so trung bình khác maps/horizon. Lặp A/A cùng baseline để đo biến động.
Tài liệu này không phải kho raw evidence hoặc bảng xếp hạng.

Review tháng 9 cho thấy pipeline chạy được chưa đồng nghĩa học lái được.
Hướng dẫn chạy tiếp RL cũ đã gỡ khỏi onboarding; lịch sử code còn ở
[RETRAINING_NOTES](RETRAINING_NOTES.md). Không dùng loss graph cũ để chứng minh
baseline c hiện tại.
