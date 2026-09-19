# HƯỚNG DẪN HUẤN LUYỆN TOÀN DIỆN (PRE-TRAIN & ONLINE RL)

Tài liệu này hướng dẫn toàn bộ quy trình từ đầu (end-to-end) gồm 3 giai đoạn:
1. **Giai đoạn 1 (Phase A1)**: Thu thập dữ liệu khởi tạo (Bootstrap Collection)
2. **Giai đoạn 2 (Phase A2)**: Tiền huấn luyện World Model (Offline Pre-training)
3. **Giai đoạn 3 (Phase B/C)**: Huấn luyện tăng cường trực tuyến (Online RL với DreamerV3)

---

## 1. Yêu Cầu Môi Trường & Thiết Bị

1. Mở PowerShell tại thư mục gốc của dự án:
   ```powershell
   cd d:\DoAnTotNghiep\worldmodel-metadrive
   ```
2. Kích hoạt Virtual Environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Kiểm tra GPU CUDA:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
   ```
   *Yêu cầu: In ra `CUDA Available: True` và tên card đồ họa (ví dụ: `NVIDIA GeForce GTX 1650`).*

---

## 2. Giai Đoạn 1: Thu Thập Dữ Liệu Khởi Tạo (Bootstrap Data Collection)

Mục đích: Thu thập 25.000 bước lái xe mẫu (chuyên gia kết hợp) để tạo tập dữ liệu ban đầu cho World Model học nhận diện đường sá, xe cộ và vật lý môi trường.

```powershell
# Chạy thu thập 25.000 bước bằng policy mixed_expert
python scripts/collect_bootstrap.py --config configs/train.yaml --policy mixed_expert --steps 25000
```

### Kết quả đầu ra:
- File mảng numpy: `data/replay_buffer/bootstrap.npz`
- File bộ nhớ ảnh: `data/replay_buffer/bootstrap_images.mmap`
- Tập dữ liệu gồm ~73 episode lái xe hoàn chỉnh (độ dài trung bình ~340 bước/episode), có đủ dữ liệu rẽ trái, rẽ phải, tăng tốc và phanh.

---

## 3. Giai Đoạn 2: Tiền Huấn Luyện World Model (Offline Pre-training)

Mục đích: Huấn luyện mạng Encoder (CNN + MLP), RSSM (Recurrent State-Space Model), Decoder, Reward Predictor và Continue Predictor trên dữ liệu bootstrap vừa thu thập.

```powershell
# Huấn luyện 5.000 gradient updates trên GPU
python scripts/train_world_model.py --config configs/train.yaml --device cuda --updates 5000 --batch-size 8 --log-every 20 --ckpt-every 500
```

### Nếu bị gián đoạn và muốn Resume:
```powershell
python scripts/train_world_model.py --config configs/train.yaml --device cuda --resume checkpoints/world_model/latest.pt
```

### Kết quả đầu ra:
- Checkpoints lưu tại: `checkpoints/world_model/wm_step_005000.pt` và `checkpoints/world_model/latest.pt`.
- Loss tổng thể sẽ hội tụ từ `~1.2` xuống mức `~0.24` (tái tạo ảnh sắc nét, sai số chuyển trạng thái thấp).
- Vẽ biểu đồ tiền huấn luyện:
  ```powershell
  python scripts/plot_metrics.py
  ```
  *(Biểu đồ xuất ra tại `docs/pretrain_curves.png`)*

---

## 4. Giai Đoạn 3: Huấn Luyện Online RL (DreamerV3 Loop)

Mục đích: Xe tự lái tương tác trực tiếp với MetaDrive. Dữ liệu mới liên tục được nạp vào buffer để cập nhật World Model, đồng thời Actor-Critic được huấn luyện trong không gian tưởng tượng (latent imagination 30 bước).

```powershell
# Huấn luyện Online RL chính thức (500k steps)
python scripts/train_online.py --device cuda --total-steps 500000 --steps-per-iter 1000 --ckpt-dir checkpoints/exp_official_01
```

### Ý nghĩa các tham số:
- `--device cuda`: Huấn luyện mạng nơ-ron trên GPU.
- `--total-steps 500000`: Tổng số bước tương tác với MetaDrive (500k steps).
- `--steps-per-iter 1000`: Mỗi vòng lặp thu thập 1.000 bước thực tế rồi cập nhật 25 bước WM và 25 bước Actor-Critic.
- `--ckpt-dir checkpoints/exp_official_01`: Lưu checkpoint định kỳ vào `checkpoints/exp_official_01/online/`.

### Khôi phục (Resume) nếu mất điện / tắt máy:
```powershell
python scripts/train_online.py --device cuda --resume checkpoints/exp_official_01/online/latest.pt --ckpt-dir checkpoints/exp_official_01
```

---

## 5. Theo Dõi Tiến Trình & Vẽ Biểu Đồ

Mở một cửa sổ PowerShell thứ 2 (đã kích hoạt `venv`) để theo dõi:

```powershell
python scripts/plot_metrics.py
```
- Tự động xuất biểu đồ trực quan ra file **`docs/training_curves.png`**.
- Xem nhanh dòng log mới nhất:
  ```powershell
  Get-Content logs/metrics.jsonl -Tail 1
  ```

### Các chỉ số cần quan sát:
| Chỉ số | Kỳ vọng khi xe học tốt | Ý nghĩa |
| :--- | :--- | :--- |
| `eval/mean_route_completion` | Tăng từ $5\% \to 30\% \to 50\%+$ | Xe vượt qua khúc cua đầu tiên và đi xa hơn. |
| `eval/out_of_road_rate` | Giảm từ $1.0 \to 0.0$ | Tỷ lệ văng lề giảm dần. |
| `ac/policy_std` | Giữ ổn định $0.30 \sim 0.45$ | Duy trì khám phá, không bị liệt góc lái. |
| `ac/continue_mean` | $< 0.95$ | World Model nhận biết được nguy hiểm khi xe chệch làn. |

---

## 6. Đánh Giá Độc Lập Sau Khi Huấn Luyện (Evaluation)

Chạy kiểm thử trên 20 map độc lập chưa từng gặp khi train (seeds 10000–10019):

```powershell
python evaluation/evaluate.py --checkpoint checkpoints/exp_official_01/online/latest.pt --wm-checkpoint checkpoints/world_model/latest.pt --num-episodes 20 --device cuda
```
