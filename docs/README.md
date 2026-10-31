# Tài liệu dự án

Bắt đầu từ [README](../README.md). Đối chiếu source ngày 17/10/2026;
giữ tên một số file để bảo toàn liên kết lịch sử.

1. [Kiến trúc h/z/c và pipeline](phase_c.md).
2. [Setup và các lệnh chạy](TRAINING_GUIDE.md).
3. [Data, checkpoint, cấu hình](ARTIFACTS.md).
4. [Trạng thái, gate, nhánh experimental](PILOT_REVIEW_20260925.md).
5. [Tests và tái lập](notes.md).
6. [Lịch sử sửa lỗi và migration](RETRAINING_NOTES.md).
7. [Audit P1–P4, các bộ benchmark và Next Steps](P1_P4_AUDIT.md):
   gồm kiểm chứng refit exact-memory và checkpoint c23, RL vẫn khóa (17/10/2026).
8. [Eval thử nghiệm state23 và traffic shield](EXPERIMENTAL_EVALUATION.md):
   opt-in, giữ state19 mặc định, chưa phải benchmark đã tái lập trên Bao.
9. [Ghi chép nhánh Tien](TIEN_UPDATE_P1_P4.md): đọc cùng đính chính trong audit.
10. [Hướng dẫn demo 3D của Tien](SAFETY_SHIELD_AND_DEMO_GUIDE.md): tài liệu lịch sử, kết quả từng bộ map cần đối chiếu JSON.

## Hình lịch sử

Giao diện mới: [DEMO_UI](DEMO_UI.md) — Live, AI, từng bước, Replay CPU,
thống kê và notebook Colab. Đọc phần timing và giới hạn bằng chứng trước
khi trình bày kết quả.

Giữ để tham khảo, không dùng làm benchmark baseline hiện tại:

- [data_schema.png](data_schema.png): sơ đồ cũ 256×256, thiếu phân biệt
  terminated/timeout. Contract hiện hành ở [ARTIFACTS](ARTIFACTS.md).
- [schema_sample.png](schema_sample.png): ảnh mẫu cũ 256×256, không phải input
  mặc định 64×64.
- [pretrain_curves.png](pretrain_curves.png), [training_curves.png](training_curves.png):
  đồ thị lịch sử không kèm đủ dataset/seed/checkpoint hash để tái lập độc lập.
  Không suy ra chất lượng phiên bản hiện tại từ các hình này.

Hướng dẫn mở RL cũ đã thay bằng gate hiện hành. Lịch sử sửa code vẫn giữ trong
Git; tài liệu public không chứa liên kết artifact riêng.
