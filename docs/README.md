# Tài liệu dự án

Bắt đầu từ [README](../README.md). Đối chiếu source ngày 09/10/2026;
giữ tên một số file để bảo toàn liên kết lịch sử.

1. [Kiến trúc h/z/c và pipeline](phase_c.md).
2. [Setup và các lệnh chạy](TRAINING_GUIDE.md).
3. [Data, checkpoint, cấu hình](ARTIFACTS.md).
4. [Trạng thái, gate, nhánh experimental](PILOT_REVIEW_20260925.md).
5. [Tests và tái lập](notes.md).
6. [Lịch sử sửa lỗi và migration](RETRAINING_NOTES.md).
7. [Tài liệu kiểm định P1–P4 và kết quả thực nghiệm toàn diện](P1_P4_AUDIT.md): **Tài liệu chính** ghi nhận đầy đủ refit H15, lane jump breakdown, action ranking, failure recall, critic audit & warm-up, RL pilot, ảo ảnh chân trời ngắn $H=15$, và hiện tượng Freezing Robot 100k bước.
8. [Cập nhật nhánh Tien](TIEN_UPDATE_P1_P4.md): Quá trình chuẩn hóa kỹ thuật và tích hợp theo chuẩn nhánh Bao.

## Hình lịch sử

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
