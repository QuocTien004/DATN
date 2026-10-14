# World Model + MetaDrive

Dự án nghiên cứu điều khiển xe trong MetaDrive bằng World Model và Actor-Critic
theo hướng Dreamer. Đây là implementation riêng, không phải bản tái lập đầy đủ
DreamerV3. Input gồm RGB trước 64×64×3 và state 19 chiều; action là
`[steering, throttle/brake]` trong `[-1,1]`.

## Trạng thái nghiên cứu (Cập nhật 14/10/2026)

Dự án đã hoàn tất toàn bộ các bước kiểm định kỹ thuật và thực nghiệm theo kế hoạch bàn giao giữa hai nhánh Tien và Bao (chi tiết tại [Tài liệu kiểm định P1–P4](docs/P1_P4_AUDIT.md)).

### 1. Kết quả thực hiện đúng kế hoạch kiểm định P1–P4:
1. **P2 Refit Continuous Transition (`exact_action_memory=True`):**
   - Đóng băng Encoder/Posterior, tối ưu riêng 17 tọa độ liên tục trên 2.000 updates (Seed 0 & Seed 1).
   - Tỷ số sai số ngang dài hạn (Lateral Hold Ratio $H=15$) giảm từ `2.634` xuống **0.610** (Seed 0) và **0.782** (Seed 1) $\rightarrow$ **VƯỢT QUA GATE** ($< 1.0$).
2. **Hướng A — Bóc tách hình học & nhảy làn (Lane Jump Breakdown):**
   - Phân tích 4.733 cửa sổ Test H5: 76,4% cửa sổ di chuyển bình thường đạt Hold Ratio **0.731** (Pass Gate $< 1.0$). 2,8% cửa sổ nhảy làn đạt **0.689** (Pass Gate).
   - Xác định episode dị thường duy nhất (`test_44205_swerve` - xe kẹt đứng yên) gây nhiễu số học kéo trung bình tổng thể.
3. **Hướng B — Kiểm chứng xếp hạng hành động (Action Ranking):**
   - Đánh giá 105 cặp kịch bản tại $t_{30}$: Swerve vs Clean đạt **75.0% - 83.3%** đồng thuận với môi trường thực.
4. **Kiểm định Failure Head (Nhận diện thất bại thật):**
   - Đánh giá trên 47 ca kết thúc của Test set: P3 Boundary Head nâng Failure Recall từ **27.66% lên 80.85%** (Out: 81.48%, Crash: 71.43%).
5. **Kiểm định mạng Critic & Warm-up:**
   - Phát hiện Critic gốc của BC hoàn toàn bằng 0, Critic 30k chỉ chênh lệch 0.85 điểm so với mức phạt -20.0 (mù rủi ro va chạm).
   - Thực hiện Critic Warm-up thành công trên buffer hỗn hợp: tạo checkpoint tốt nhất đề tài `continuous_seed0_refit_warmup_c19.pt` (Route **71.78%**, Finish **3/8**, Out **0/8** trên paired benchmark).
6. **Thực nghiệm Safe RL Pilot (250 updates) & Phát hiện Ảo ảnh Chân trời ngắn ($H=15$):**
   - Với chân trời tưởng tượng ngắn $H=15$ (1.5 giây), xe không kịp chạm mép đường trong tưởng tượng, gradient kéo Actor đạp lút chân ga 100% (Throttle = 0.953).
   - Cơ chế Early-Stopping ngắt an toàn tại update 25 sau 83 giây, bảo vệ mô hình khỏi suy thoái.
7. **Thực nghiệm Long Training 100k bước & Bẫy "Freezing Robot":**
   - Khi áp dụng phạt va chạm (-20) và phạt chân ga cao (5.0), Actor chuyển sang chiến lược phanh lùi (Throttle $\approx$ -0.41), đứng im tại vạch xuất phát để bảo toàn điểm số (0 crash, 0 out, route 1.54%).
8. **Quyết định kỹ thuật: DUY TRÌ KHÓA RL (`rl_locked: true`):**
   - Không train RL online khi chưa có cơ chế neo giá trị dài hạn cho Critic. Sử dụng checkpoint `continuous_seed0_refit_warmup_c19.pt` làm kết quả nòng cốt.

## Bắt đầu

Dùng Python 3.10, tạo và kích hoạt venv, cài PyTorch phù hợp nền tảng rồi cài
`requirements.txt`. Lệnh cụ thể ở [TRAINING_GUIDE](docs/TRAINING_GUIDE.md).
Unit tests không cần checkpoint:

```bash
python -m unittest discover -s tests -q
```
*(Hiện tại toàn bộ 127/127 unit tests đều PASS).*

Data/checkpoint không nằm trong Git. Phải có đúng artifact và metadata,
không đổi tên hoặc sửa metadata checkpoint cũ để giả làm baseline mới.

## Tài liệu

- [Tài liệu kiểm định P1–P4 và Thực nghiệm chi tiết](docs/P1_P4_AUDIT.md): **Tài liệu nòng cốt** ghi chép toàn bộ kết quả audit, refit, pilot và long run.
- [Báo cáo cập nhật nhánh Tien](docs/TIEN_UPDATE_P1_P4.md): Lịch sử tích hợp và chuẩn hóa theo tiêu chuẩn nhánh Bao.
- [Kiến trúc và pipeline](docs/phase_c.md): RSSM h/z/c, posterior/prior, BC, imagination.
- [Setup, collect, train BC, eval](docs/TRAINING_GUIDE.md): các lệnh có trong source.
- [Data, checkpoint, cấu hình](docs/ARTIFACTS.md): contract v2 và load/resume đúng loại.
- [Trạng thái và gate](docs/PILOT_REVIEW_20260925.md): baseline, experimental, chưa đạt.
- [Tests và tái lập](docs/notes.md).
- [Lịch sử sửa lỗi](docs/RETRAINING_NOTES.md).
- [Mục lục và hình lịch sử](docs/README.md).

## Source & Scripts chính

| Thư mục / File | Vai trò |
|---|---|
| `models/continuous_state.py` | State liên tục 19 chiều với tùy chọn `exact_action_memory` |
| `evaluation/p1_p4_diagnostics.py` | Hàm kiểm định chẩn đoán, frozen policy loader, boundary heuristic |
| `scripts/train_refit_exact_memory.py` | Huấn luyện refit transition model 17 tọa độ liên tục |
| `scripts/audit_lane_jump_breakdown.py` | Bóc tách hình học và nhảy làn đường |
| `scripts/audit_action_ranking.py` | Kiểm chứng xếp hạng hành động tưởng tượng |
| `scripts/pretrain_critic_offline.py` | Warm-up mạng Critic trên offline buffer |
| `scripts/check_policy.py` | Chẩn đoán hành vi chính sách (Throttle/Steering/Clearance) |
| `scripts/run_rl_pilot_250.py` | Pilot RL 250 updates có ngắt sớm tự động Early-Stopping |
| `scripts/run_long_train_experiment.py` | Pipeline huấn luyện dài hạn online/offline |
| `scripts/plot_training_curves.py` | Vẽ đồ thị đường cong huấn luyện 4 ô |
| `scripts/make_w7_docx.py` | Xuất file báo cáo Word tiến độ tuần 7 (`BC_TienDo/12-10_18-10(W7).docx`) |

Không commit data, checkpoint, credential hoặc log cá nhân. Giữ output riêng
cho từng experiment, không ghi đè artifact đối chứng.

