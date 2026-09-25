# Kiểm tra lượt train Br_Bao — 2026-09-25

Nguồn: `MyDrive/DoAn/Br_Bao`, log WM/online, `resolved_config.json`, checkpoint
và replay. Bằng chứng chi tiết, script chẩn đoán và kết quả Colab lưu tại
[Drive: smoke/2026-09-25_pilot_v3](https://drive.google.com/drive/folders/1vYmI09TFyxVuCzLuv6bwRfCCXcYTV8Oe).
Không ghi đè artifact gốc của người train.

## Người train có làm đúng không?

Phần lớn là đúng: manifest ghi commit `6b25044`, bootstrap mới 25.000 transitions,
WM pretrain 5.000 updates, online 50.000 steps; RGB 64×64, contract 2, demo ratio
0.25, không thêm action noise. Replay cuối có 75.000 transitions và
`checkpoint_step=50000`, khớp model. Trọng số WM đều hữu hạn. `git_dirty=true`
nên không thể khẳng định source máy bạn hoàn toàn giống commit nếu thiếu git diff.

`--ckpt-dir checkpoints/retrain_v2/online` tạo `online/online/` vì script tự thêm
`online/`. Đây là nhầm vị trí lưu, không phải bằng chứng train sai. Truyền root
`checkpoints/retrain_v2` ở các lượt sau; không cần di chuyển/xóa bộ cũ.

## Chất lượng chưa đạt

Mỗi mốc eval đủ 20 episodes, seeds 10000–10019, horizon 1000 theo manifest:

| Env steps | Success | Out-of-road | Route completion | Episode length |
|---:|---:|---:|---:|---:|
| 10.000 | 0% | 100% | 1,84% | 19,65 |
| 20.000 | 0% | 100% | 1,80% | 25,00 |
| 30.000 | 0% | 0% | 1,37% | 1000,00 |
| 40.000 | 0% | 55% | 1,87% | 510,85 |
| 50.000 | 0% | 65% | 1,81% | 389,15 |

Mốc 30k không crash nhưng gần như không tiến lên và chạm time limit; không phải
một policy an toàn đã học lái. Không tăng lên 500k hoặc Stage 2 với kết quả này.

Audit T4 lấy 2.048 transitions/cặp WM–replay, có ưu tiên sequence chứa boundary.
Đây là **chẩn đoán trên replay đã thấy**, không phải test set độc lập:

- WM pretrain trên bootstrap: prior nhận ra 91,2% terminal tại ngưỡng continue
  0,5. Trên replay online chứa hành vi mới, chỉ 24,1% trong mẫu tương ứng. Đây là
  dấu hiệu thay đổi phân bố; loss pretrain thấp không bảo đảm dự đoán tốt nơi
  actor mới chạy tới.
- WM sau online trên online replay: prior terminal recall 89,6%; vẫn dự đoán
  reward âm nhẹ hơn thực tế (trung bình −0,71 so với −1,69 trong nhóm reward âm).
  Không suy ra accuracy toàn tập từ mẫu ưu tiên terminal này.
- Actor 50k trên posterior của online replay: steering mean −0,903, 72,5% action
  mean có |steering| >0,9. Trên bootstrap steering mean −0,931. Actor bị lệch lái
  rõ rệt, không chỉ vấn đề định dạng file hay máy chạy.

## Code và setting thay đổi

1. `models/actor_critic.py`: `entropy_mode: squashed` tính entropy của action sau
   tanh + affine. Cách cũ `H(Normal)` không phụ thuộc mean trực tiếp, nên thưởng
   entropy vẫn có thể cao khi steering đã dồn sát biên. Dùng log Jacobian ổn định,
   đối chiếu `torch.distributions.TransformedDistribution` trong regression test.
   Công thức dựa trên [PyTorch distributions](https://docs.pytorch.org/docs/2.11/distributions.html).
   Đây là sửa mục tiêu regularization; chưa chứng minh nó là nguyên nhân duy nhất.
2. `training/train_agent.py`: sửa đạo hàm `symexp` tại 0 từ 0 thành đúng 1. Giữ
   nguyên giá trị hàm và giới hạn đầu ra. Không quy toàn bộ failure cho lỗi này.
3. `configs/train_pilot.yaml`: gate 2.000 bước, collect mỗi 250, 25 WM/5 AC updates,
   imagination 15, entropy 0,01. So với run cũ: WM cập nhật 4× trên mỗi env step,
   AC 20 thay vì 25 updates/1000 steps. Giữ reward, data, kiến trúc, batch 8,
   sequence 32 và 25% bootstrap. Đây là ứng viên thử nghiệm, không phải cấu hình
   tối ưu đã xác minh; thay nhiều biến nên không phải ablation đơn biến.
4. `scripts/train_online.py`: resume mặc định lấy training config đã lưu; manifest
   ghi giá trị CLI thực sự dùng, Python/Torch/GPU và thêm file `run_start_*.json`
   để không mất lịch sử khi resume. Actor config vẫn lấy từ checkpoint nhằm tránh
   đổi distribution ngầm. Checkpoint cũ thiếu `entropy_mode` giữ mode `normal`.
5. `training/online_trainer.py`: thêm idle fraction, thời gian collect, per-seed
   JSON mỗi mốc eval; `--eval-horizon` cho phép chẩn đoán ngắn nhưng không rút
   horizon train. Log loss/return không thay thế đánh giá hành vi.
6. `scripts/setup_colab_egl.py`, `envs/metadrive_wrapper.py`: tùy chọn camera GPU
   headless. Chỉ giải nén userspace OpenGL khớp driver vào thư mục riêng, không
   thay kernel driver; chọn bằng `DATN_RENDER_BACKEND=egl`. Probe đã xác nhận T4
   renderer nhanh hơn llvmpipe; xem cách dùng trong guide.

Vị trí code để đối chiếu: `models/actor_critic.py:L83-L155`,
`training/train_agent.py:L120-L125`, `scripts/train_online.py:L88-L113`,
`scripts/train_online.py:L294-L326`, `training/online_trainer.py:L390-L397`,
`scripts/setup_colab_egl.py:L15-L49`, `envs/metadrive_wrapper.py:L53-L61`.

## Bước tiếp theo cho người train

### Kết quả gate Colab, không chỉ smoke vài bước

Python 3.10.21, Torch 2.11.0+cu128, T4; dùng bootstrap v2 và WM 5k của bạn,
Actor mới. Chạy 2.000 env steps, save, resume tới 3.000. Mỗi mốc eval dưới đây có
4 seed 10000–10003, **horizon đầy đủ 1000**, không dùng horizon 4/200 của smoke cũ.

| Mốc | Success | Out-of-road | Route completion | Episode length |
|---:|---:|---:|---:|---:|
| 1.000 | 0/4 | 4/4 | 12,28% | 205,0 |
| 2.000 | 0/4 | 4/4 | 6,25% | 76,0 |
| 3.000 sau resume | 0/4 | 4/4 | 4,16% | 40,5 |

**Gate kỹ thuật đã chạy, gate chất lượng không đạt. Không giao cấu hình này train
50k/500k ngay.** Route completion đang giảm trên subset này. Không so trực tiếp
trung bình 4 seed với trung bình 20 seed của máy thật để tuyên bố cải thiện.
Sửa entropy/symexp có regression test và lý do kỹ thuật, nhưng thử nghiệm này
chưa chứng minh các sửa đổi đủ để học lái. Các thông số pilot vẫn là thí nghiệm.

Lượt đầu dùng CPU render bị chủ động dừng sau mốc log 250 để chuyển EGL; giữ log
ở `pilot_logs/`, `pilot.log`, không tính vào 3k của run chính. Với policy inference,
250 bước collect đầu tiên giảm từ 247,84 giây xuống 6,50 giây trên GPU renderer.
Đây là thời gian collect, không gồm toàn bộ update/checkpoint/eval. Mỗi backend
có thể tạo pixel hơi khác nên không coi hai lượt là tái hiện bit-for-bit.

Kiểm tra bổ sung đã hoàn tất:

- 26/26 unit/regression tests pass cả local và trên runtime Colab Python 3.10.
  Training WM/Actor-Critic của pilot chạy CUDA thật, không chỉ test tensor nhỏ.
- Resume 2k→3k giữ `steps_per_iter=250`, 25 WM/5 AC updates, imagination 15,
  `entropy_mode=squashed`. Model/replay đều step 3000, replay size 28.000.
- Eval độc lập checkpoint Br_Bao 50k trên cùng 4 seed: success 0/4, route 1,99%,
  3 episode chạm horizon 1000, 1 out-of-road. File `legacy_eval.json` trong bộ
  evidence là baseline **v2 50k này**, không phải checkpoint 400k contract cũ.
- Thử RSSM posterior sampled thay vì argmax, giữ actor deterministic tại 3k:
  success 0/4, route 4,19%, cả 4 out-of-road; gần kết quả 4,16% của mode cũ.
  Phép thử nhỏ này không ủng hộ giả thuyết chỉ đổi mode latent là giải quyết được.
  Không thay mặc định eval trong code sau phép thử.

Evidence gồm `audit.json`, original run logs, `gpu_pilot_logs/` (các JSON per-seed,
manifest từng lần chạy), log unit/train/resume/eval, `input_manifest.json` và
`artifact_manifest.json` có SHA-256, source snapshot và cặp
`checkpoints/online/latest.pt` + `online_buffer.npz` ở mốc 3k. Không copy mmap
hoặc package driver lên Drive. Checkpoint này dùng để chẩn đoán/reproduce, **không
phải policy đã đạt chất lượng lái**.

### Cách dùng kết quả

Dùng **actor mới**, có thể tái dùng `bootstrap_v2.npz` và WM pretrain 5k contract 2
của lượt này. Không bắt buộc collect/pretrain lại chỉ vì thay entropy. Không resume
actor 50k bị lệch lái để thử cấu hình mới, vì resume giữ config/optimizer cũ.
Làm theo gate 2k trong [TRAINING_GUIDE.md](TRAINING_GUIDE.md), không chạy ngay 50k.

Chỉ pass số học/pipeline chưa đủ để gửi train dài. Cần xem hành động, idle,
route completion và eval đủ horizon; sau đó chạy seed khác với output riêng.
Nếu tiếp tục đứng yên hoặc lệch lái, giữ artifact rồi điều tra WM/action-gradient
và thử ablation riêng, không chỉ tăng số bước hay tăng crash penalty.
