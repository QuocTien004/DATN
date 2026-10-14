# Cập nhật nhánh Tien: Hoàn tất tích hợp P1, P2, P3, P4 (World Model Fixes & RL Pilot Setup)

Tài liệu này ghi chú lại các thay đổi quan trọng đã thực hiện trên nhánh `Tien` nhằm khắc phục các lỗi vật lý và chuẩn bị cho đợt tinh chỉnh Actor-Critic (RL).

## 1. Tích hợp P1 (Skip-connection GT Continuous State)
- **File sửa đổi:** `training/train_agent.py`
- **Nội dung:** Thay đổi hàm `posterior_start_states`, ép buộc ghi đè trạng thái liên tục `c` của posterior bằng toạ độ vật lý thực tế (`gt_c`) lấy từ dataset thay vì dùng bản dự đoán của mô hình. Điều này triệt tiêu sai số ban đầu của World Model trước khi bắt đầu chuỗi tưởng tượng.

## 2. Tích hợp P2 (Hard-assign Action Memory cho Prior)
- **File sửa đổi:** `models/continuous_state.py`
- **Nội dung:** Sửa đổi phương thức `ContinuousState.imagine()`. Thay vì để MLP tự dự đoán toàn bộ trạng thái tiếp theo, hệ thống giờ đây thực hiện gán cứng (hard-assign) hành động vật lý chính xác `(action + 1.0) * 0.5` vào các chỉ số bộ nhớ số 5 và 6 của vector trạng thái liên tục. Giúp loại bỏ hoàn toàn hiện tượng trôi dạt (drift) tín hiệu lái khi tưởng tượng dài hạn (H15).

## 3. Tích hợp P3 (Modular Boundary Head)
- **File sửa đổi:** `models/predictors.py`
- **Nội dung:** Bổ sung module mới `ModularBoundaryHead`. Lớp này đóng vai trò màng lọc độc lập bọc ngoài các hàm `reward` và `continue` cũ. Thuật toán tự động trích xuất toạ độ `c`, tính toán khoảng cách an toàn với lề đường (`clearance`). Nếu xe chạm ngưỡng `0.0768`, module sẽ can thiệp và chèn phạt trực tiếp: ép `continue_logit` về `-10.0` (cắt cầu quỹ đạo) và ghim điểm thưởng tối đa về `-0.6`. Khắc phục triệt để hội chứng "mù mép đường".

## 4. Đạt mốc P4 & Triển khai kịch bản RL Pilot
- **File thêm mới:** `scripts/run_rl_pilot.py`, `logs/p4_eval/run_p4_actor_critic_test.py` (chạy script độc lập)
- **Nội dung:** 
  - Kịch bản chẩn đoán `run_p4_actor_critic_test.py` chứng minh World Model đã được sửa (P1+P2+P3) chấm điểm (Discounted Return) cho hành vi lạng lách (1.058) thấp hơn hẳn hành vi lái an toàn (1.372).
  - Kịch bản huấn luyện mới `run_rl_pilot.py` chuyên dụng cho đợt tinh chỉnh RL. Kịch bản có khả năng đóng băng (Freeze) World Model, tự động áp dụng P2, P3 và tải trọng số Actor của BC. 
  - Đã tiến hành chạy thử nghiệm vòng Pilot đầu tiên và xác nhận hiện tượng *Catastrophic Forgetting* khi không có ràng buộc BC, mở đường cho bước phát triển hàm phạt KL-Constraint sắp tới.

---

## 5. Chuẩn hóa & Hoàn thiện theo Kế hoạch Kiểm định của nhánh Bao (14/10/2026)

Sau đợt bàn giao và kiểm định chéo với nhánh `Bao` (tài liệu [P1_P4_AUDIT.md](P1_P4_AUDIT.md)), toàn bộ mã nguồn trên nhánh `Tien` đã được chuẩn hóa lại theo nguyên tắc fail-closed và kiểm chứng thực nghiệm nghiêm ngặt:

1. **Chuẩn hóa P2:** Chuyển đổi từ can thiệp gán cứng sang tùy chọn cấu hình `exact_action_memory: bool` có kiểm tra tính hợp lệ của action, bảo toàn khả năng tương thích ngược của checkpoint cũ.
2. **Loại bỏ bypass, duy trì khóa RL:** Giữ nguyên hàm kiểm tra an toàn `require_imagination_ready()` trong `models/actor_critic.py` để ngăn chặn việc train RL khi các gate chưa đạt chuẩn.
3. **Thực hiện đầy đủ kế hoạch kiểm định thực nghiệm:**
   - **Refit Transition $H=15$:** Đạt Hold Ratio **0.610** (Vượt qua Gate $< 1.0$).
   - **Bóc tách lỗi hình học (Lane Jump Breakdown):** 76,4% cửa sổ thông thường đạt Hold Ratio **0.731** (Pass Gate).
   - **Action Ranking:** Tỷ lệ đồng thuận nhận biết hành vi đánh võng nguy hiểm đạt **75.0% - 83.3%**.
   - **Critic Warm-up & Checkpoint tốt nhất:** Tạo ra checkpoint `continuous_seed0_refit_warmup_c19.pt` đạt thành tích cao nhất đề tài (**71.78% Route, 3/8 Finish, 0/8 Out** trên paired benchmark).
   - **Phát hiện Ảo ảnh Chân trời ngắn ($H=15$) & Hiện tượng Freezing Robot (100k steps):** Giải thích tường tận cơ chế thất bại của RL trong môi trường xe tự lái có World Model và chứng minh hiệu quả ngắt bảo vệ của cơ chế Early-Stopping.
4. **Trạng thái kiểm thử:** 127/127 unit tests đều đạt chuẩn.
