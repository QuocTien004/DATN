# HƯỚNG DẪN HUẤN LUYỆN ONLINE RL (DREAMERV3 + METADRIVE)

Tài liệu này hướng dẫn toàn bộ quy trình huấn luyện Online Reinforcement Learning cho xe tự hành trong môi trường MetaDrive sử dụng thuật toán DreamerV3.

---

## 1. Yêu Cầu Môi Trường & Kiểm Tra

1. Mở terminal PowerShell tại thư mục gốc của project:
   ```powershell
   cd d:\DoAnTotNghiep\worldmodel-metadrive
   ```
2. Kích hoạt môi trường ảo Python:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Kiểm tra kết nối GPU CUDA:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
   ```
   *Yêu cầu: In ra `CUDA Available: True`.*

---

## 2. Lệnh Khởi Chạy Huấn Luyện (Chính Thức)

Chạy vòng lặp Online RL kết hợp (Thu thập dữ liệu $\to$ Cập nhật World Model $\to$ Huấn luyện Actor-Critic trong không gian tưởng tượng $\to$ Đánh giá định kỳ):

```powershell
python scripts/train_online.py --device cuda --total-steps 500000 --steps-per-iter 1000 --ckpt-dir checkpoints/exp_official_01
```

### Ý nghĩa các tham số:
- `--device cuda`: Chạy huấn luyện trên GPU NVIDIA.
- `--total-steps 500000`: Tổng số bước tương tác với môi trường MetaDrive (500.000 steps).
- `--steps-per-iter 1000`: Mỗi vòng lặp thu thập 1.000 bước thực tế rồi cập nhật mạng (25 bước WM, 25 bước Actor-Critic).
- `--ckpt-dir checkpoints/exp_official_01`: Thư mục lưu checkpoint định kỳ (tự động lưu `online_step_020000.pt`, `online_step_040000.pt`,... và `latest.pt`).

---

## 3. Khôi Phục (Resume) Khi Bị Gián Đoạn

Nếu quá trình huấn luyện bị ngắt quãng (mất điện, tắt máy), bạn tiếp tục huấn luyện từ checkpoint gần nhất bằng cách thêm cờ `--resume`:

```powershell
python scripts/train_online.py --device cuda --resume checkpoints/exp_official_01/online/latest.pt --ckpt-dir checkpoints/exp_official_01
```

Script sẽ tự động khôi phục toàn bộ trọng số mạng (WM, Actor, Critic, Optimizer) và tiếp tục đếm bước từ thời điểm dừng lại.

---

## 4. Theo Dõi & Vẽ Biểu Đồ Trực Quan

Trong lúc terminal 1 đang train, bạn có thể mở một terminal PowerShell thứ 2 (đã kích hoạt venv) để vẽ đồ thị:

```powershell
python scripts/plot_metrics.py
```

Lệnh này sẽ tự động đọc `logs/metrics.jsonl` và xuất ra file ảnh:
- **`docs/training_curves.png`**: Biểu đồ phân tích tổng hợp 6 nhóm chỉ số quan trọng.

### Các chỉ số cần chú ý:
1. **`eval/mean_route_completion`**: Tỷ lệ hoàn thành lộ trình (kỳ vọng tăng dần từ $5\% \to 30\% \to 50\%+$).
2. **`eval/out_of_road_rate`**: Tỷ lệ xe văng khỏi đường (kỳ vọng giảm từ $1.0 \to 0.0$).
3. **`ac/policy_std`**: Độ biến thiên khám phá (kỳ vọng duy trì khỏe mạnh ở mức $0.30 \sim 0.45$, không bị sụp đổ).
4. **`ac/continue_mean`**: Khả năng nhận biết rủi ro của World Model trong tưởng tượng (kỳ vọng $< 0.95$).

---

## 5. Đánh Giá Độc Lập (Evaluation)

Sau khi huấn luyện xong, chạy script đánh giá chính thức trên 20 bản đồ kiểm thử độc lập (seeds 10000–10019):

```powershell
python evaluation/evaluate.py --checkpoint checkpoints/exp_official_01/online/latest.pt --wm-checkpoint checkpoints/world_model/latest.pt --num-episodes 20 --device cuda
```
