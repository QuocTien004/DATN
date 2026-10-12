# Lịch sử sửa lỗi và migration

Ghi chú còn giá trị kỹ thuật, không phải hướng dẫn resume RL.
Trạng thái hiện tại xem [gate](PILOT_REVIEW_20260925.md).

## Transition contract v2

| Vấn đề lịch sử | Contract/code hiện tại |
|---|---|
| Mất reset RSSM | Truyền is_first; reset h/z/c và previous action |
| Reward/continue lệch thời điểm | Heads học successor posterior sau executed action |
| KL balance sai nhánh | Tách dynamics/prior và representation/posterior |
| Timeout gộp terminal | done để reset, terminated cho target continue |
| Replay vòng mất write pointer/mmap cũ | Giữ idx/full, rebuild mmap từ archive |
| Tail thiếu next obs | Lưu latest next obs và terminal next obs |
| Resume lệch model/replay | Kiểm env_steps = checkpoint_step, không fallback bootstrap |
| Actor online eval với WM khác | Checkpoint có models dùng chính WM bên trong |
| Eval làm lệch collector | Reset recurrent/episode counters khi collect lại |
| Bootstrap bị ghi đè | Demo replay riêng, mix theo demo_ratio |

Source: [batches](../training/batches.py), [WM loss](../training/train_world_model.py),
[replay](../utils/replay_buffer.py), [resume validation](../utils/online_state.py),
[online trainer](../training/online_trainer.py), [eval](../scripts/eval.py).

Checkpoint thiếu `transition_contract: 2` có thể eval lịch sử nhưng trainer
WM/online hiện từ chối resume theo contract mới. Replay thiếu terminated
fallback về done không thể khôi phục timeout đã mất. Không sửa metadata
để vượt kiểm tra.

## Actor và số học

Actor mới hỗ trợ mean tanh trơn, std sigmoid và entropy sau tanh
(`entropy_mode: squashed`). Checkpoint legacy thiếu config được eval với
transform legacy. Không áp setting mới âm thầm lên weights cũ.
Đạo hàm symexp tại 0 đã sửa. Regression tests kiểm giá trị/gradient, không
chứng minh các sửa đổi đủ để RL học lái.

## Nhánh experimental sau đó

- BC tách RL; kiểm provenance pure expert và bounded executed actions.
- Bridge observed/predicted/continuous có metadata và readiness guard.
- Continuous c có posterior/residual prior riêng, giữ đường h/z cũ.
- Posterior replay chỉ học control embedding + posterior; clean action-anchor
  truyền gradient qua Actor đóng băng, không neo action off-BC.
- Heads h/z+c và Critic nhận c là opt-in, chưa qua gate toàn hệ thống.

Xem [kiến trúc](phase_c.md), [artifact](ARTIFACTS.md).
Không resume optimizer legacy sau khi thêm module/đổi parameter list.
Giữ baseline, đối chứng cùng input/seed/config; thay từng giả thuyết.
