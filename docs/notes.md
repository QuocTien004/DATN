# Kiểm thử, tái lập và bàn giao

## Chạy tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_continuous_posterior.py' -v
```

Chạy từ root trong môi trường đủ dependency. Suite hiện nhiều hơn bộ 8 smoke
tests ban đầu; không lấy số test lịch sử làm số test của HEAD.

Kiểm lại sau tích hợp P1–P4 vào Bao ngày 13/10/2026, Python 3.10 CPU:
**127/127 regression tests pass**. Bao giữ
readiness guard, mặc định P2 tắt và checkpoint BC bất biến. Các test oracle,
memory timing/normalization/gradient, strict checkpoint load và guard kiểm
correctness, không cấp quyền chạy RL. Hướng tiếp quản ở
[Next Steps](P1_P4_AUDIT.md#next-steps--kế-hoạch-chưa-thực-hiện).

| Nhóm test | Ý nghĩa |
|---|---|
| test_actor_critic, test_trainer | Range/action, lambda-return, freeze, checkpoint |
| test_evaluation | Reset recurrent và aggregate/per-episode |
| test_training_regressions | Timing, reset, terminal, replay/resume, số học |
| test_world_model_arch, test_world_model_loss_scales | Shape và loss WM |
| test_imitation, test_joint_world_model | BC optimizer scope, provenance, joint opt-in |
| test_control_bridge, test_control_prediction | Control source, GT guard, predicted state |
| test_continuous_state, test_continuous_transition | c, không GT tương lai, rollout mask |
| test_continuous_posterior | Chỉ posterior/embedding đổi, clean-only action gradient |
| test_separate_replay | Budget riêng từng action/memory, không bù trừ giữa tọa độ, frozen scope |
| test_control_predictors, test_reward_diagnostics | Heads h/z+c, metric terminal/reward |
| test_p1_p4_audit | Oracle tường minh, P2 opt-in/gradient/checkpoint, P3 shape/scale, return và guard |

Tên trên tương ứng file `tests/<tên>.py`. Test tensor nhỏ pass không chứng minh
policy lái tốt. Smoke environment kiểm camera/step/load bằng horizon ngắn;
benchmark cần horizon đầy đủ.

## Tái lập

Trước run ghi commit + source diff/hash; SHA-256 checkpoint/data; config resolve;
dependency versions; seed model/sampler/environment/noise; renderer và posterior
sampled/deterministic; updates, batch, horizon. Giữ train/validation/test/
confirmation riêng. Chuẩn hóa và trọng số chỉ lấy từ train; không chọn model
tốt nhất trên test rồi gọi đó là generalization.

Chốt giả thuyết, đối chứng, metric, gate, early-stop trước ablation. Chỉ thay một
thành phần. Kiểm optimizer scope và so weights đóng băng bằng giá trị.

Sau run kiểm completed/results/gate/checksum; console “done” không đủ.
Lưu trajectory khi cần; phân biệt bc_action với executed_action.
Không chạy lại job đã hoàn tất chỉ vì mất console.

## Metric và độ biến động

- State MSE/RMSE: nêu tọa độ observation hay normalized c; tách heading/memory.
- Action: tách steering/ga-phanh; nêu so expert hay Actor baseline.
- Prior: H1/H5/H15, hold, drift; không teacher-force.
- BC: route, finish, out/crash/timeout từng map/mode, không chỉ mean.
- A/A: lặp cùng baseline/runtime/seed. Sai khác pixel nhỏ có thể lớn dần theo
  vòng kín; không quy hết cho model, cũng không lấy noise để bỏ gate.

## Bàn giao

Không nạp checkpoint pickle không tin cậy. Không ghi đè baseline. Model/replay
cùng mốc; lưu source/config, metric, checkpoint, trajectory, manifest trước
khi tắt runtime tạm thời. Chỉ quảng bá candidate sau gate.
Public docs không chứa credential, định danh cá nhân, liên kết artifact riêng
hoặc log chưa rà soát.
