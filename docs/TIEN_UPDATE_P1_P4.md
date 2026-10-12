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
