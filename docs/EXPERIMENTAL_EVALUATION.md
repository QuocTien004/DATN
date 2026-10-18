# Eval thử nghiệm observation23 và traffic shield

Các tùy chọn này phục vụ kiểm chứng nhánh Tien. Baseline vẫn là BC model-state19.
RL còn khóa, `exact_action_memory=false` mặc định. Không có checkpoint trong Git.

## Chọn đúng observation contract

Mặc định môi trường giữ vector19. `env_cfg.state_dim: 23` mới nối thêm4 đặc
trưng vật cản, giữ nguyên19 tọa độ đầu. Script eval chọn `state_dim` theo
checkpoint và từ chối config ghi chiều khác. Không đổi metadata của checkpoint
c19 để dùng state23. Nguồn vật cản là thông tin đối tượng simulator, không phải
vật thể được mạng nhận diện từ ảnh.

Checkpoint c23 đã audit dùng Actor `control_state_source=observed`, không có
continuous RSSM. Dùng được cho eval BC có observation. Việc load thành công
không cung cấp state23 tương lai trong imagination và không mở quyền RL.

## Lệnh kiểm chứng

Chạy từ root với môi trường đã setup theo [TRAINING_GUIDE](TRAINING_GUIDE.md).
`ARTIFACTS/policy.pt` là placeholder cho checkpoint có đủ metadata và weights.

```bash
python scripts/eval.py --checkpoint ARTIFACTS/policy.pt --device cpu \
  --episodes 10 --start-seed 10000 --output OUTPUT/pure.json

python scripts/eval.py --checkpoint ARTIFACTS/policy.pt --device cpu \
  --episodes 10 --start-seed 10000 --safety-shield --output OUTPUT/shield.json
```

Đây là lệnh có trong source, chưa phải lượt benchmark mới đã chạy. Dùng output
mới, ghi checkpoint/source hash, runtime và toàn bộ config. Cùng map seed chưa
đủ đảm bảo paired nếu global RNG, traffic hoặc runtime khác. Khi có protocol
riêng, giữ map seed tách khỏi seed dùng lặp lại ngẫu nhiên. Chọn thêm maps mới
trước khi gọi kết quả là confirmation.

## Giới hạn shield

Shield là bộ điều khiển quy tắc có thể sửa ga/phanh và cả steering ở tốc độ
thấp. Nó đọc vị trí/vận tốc xe xung quanh, chưa có bảo đảm an toàn. Logic nhận
diện giao lộ theo tên lane và các ngưỡng hiện tại chưa được xác nhận tổng quát.

Wrapper ghi lại action đã can thiệp cho posterior ở bước kế tiếp và reset
trạng thái mỗi episode. JSON đánh dấu `safety_shield` và lưu số bước can thiệp
trong `shield_interventions`. Return luôn cộng reward thật từ môi trường.
Sửa action memory có thể đổi trajectory, nên phải đánh giá lại bản tích hợp.
Không dùng kết quả có shield để mô tả riêng chất lượng Actor hoặc prior.

Chưa nhập đường `--safe-collect`, regularizer RL hoặc script pilot của Tien.
Kết quả và giới hạn bằng chứng xem [audit](P1_P4_AUDIT.md).
