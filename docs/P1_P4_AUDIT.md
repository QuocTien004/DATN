# Tích hợp có kiểm chứng P1–P4

Trạng thái 12/10/2026: giữ baseline BC của Bao, **RL còn khóa**. Đây là
nhánh tích hợp chọn lọc, không phải xác nhận toàn bộ thay đổi của Tien.
Không dùng kết quả GT-start để tuyên bố posterior đã dự đoán tốt.

## Những gì được giữ và giới hạn

| Ý tưởng | Quyết định | Lý do |
|---|---|---|
| P1: khởi đầu từ state thật | Chỉ giữ hàm oracle trong diagnostic | Đo giới hạn khi bỏ sai số đầu vào; BC và policy thật vẫn dùng posterior học được |
| P2: memory action chính xác | Giữ opt-in, mặc định tắt | State sau bước đi có index 5/6 bằng `(executed_action + 1)/2`; không đồng nghĩa heading/lateral hết drift |
| P3: heuristic sát lề | Chỉ giữ phép tính diagnostic, chưa thay head production | Threshold chưa chứng minh được recall/FPR trên thất bại thật; không phải collision detector đầy đủ |
| P4: so action | Kiểm tra reward thật và paired rollout riêng | Tổng reward tưởng tượng thấp hơn không chứng minh Critic đúng |
| Bỏ guard / monkeypatch pilot | Không tích hợp | Chưa qua gate, checkpoint wrapper chưa tương thích loader chuẩn |

Không nhập script pilot có bypass. `ActorCritic.require_imagination_ready()`
vẫn chặn Actor dùng control-state trong đường RL thông thường. Diagnostic
không tự cấp quyền chạy RL, dù một vài metric tốt hơn.

## P2: timing, normalization, checkpoint

Trong `wm_cfg.rssm.continuous_state`, tùy chọn mới:

```yaml
exact_action_memory: false
```

Chỉ bật `true` cho ablation MetaDrive state19/action2. Nhánh prior nhận
`c[t], action[t]` và tạo `c[t+1]`. Hai memory coordinate mới được đổi về
không gian chuẩn hóa bằng `state_mean[5:7]`, `state_std[5:7]` của checkpoint.
Action phải là hành động thực thi hữu hạn trong `[-1,1]`, không phải action
thô trước clipping. Không lấy state thật ở tương lai.

Key mặc định thiếu hoặc `false` giữ đúng hành vi cũ. Trọng số có cùng keys;
**config là một phần bắt buộc của artifact**, không được chỉ gửi `state_dict`
rồi mong loader tự biết prior đã bật P2. Lưu một checkpoint ablation mới kèm
`wm_cfg`, không sửa checkpoint baseline. Có tests về timing, gradient,
normalization và save/load config cùng weights.

## P3: không nhầm đơn vị và kết luận

Diagnostic tái hiện công thức của source Tien: clearance xấp xỉ bằng
`min(state[0], state[1]) - 0.16 * abs(state[2] - 0.5)`, threshold `0.0725`,
sharpness `120`, chỉ kích hoạt boundary khi `state[3] > 0.05`.
Đây là coordinate observation đã scale, không phải khoảng cách mét.

Penalty `-0.6` nằm trong **symlog**, tương đương khoảng `-0.8221` reward gốc;
không phải penalty thất bại `-20`. Continue là minimum giữa head cũ và
boundary logit; code không luôn ép thành `-10`. Xe chậm/dừng vẫn có thể đã
ở ngoài đường hoặc va chạm, nên speed mask cần kiểm chứng riêng.

Giữ trọng số head cũ không đảm bảo nó nhận ra va chạm trong prior rollout.
Khi nghiên cứu gradient, đóng băng trọng số nhưng vẫn giữ graph theo input;
`no_grad` quanh head sẽ cắt đường gradient đó. Diagnostic mới giữ shape
scalar `(B,1)` cả khi batch chỉ có một phần tử và không thay schema checkpoint.

## Tái lập audit offline

Cần checkpoint baseline được phép sử dụng, SHA-256 và dataset đã thu từ đúng
checkpoint đó. Mỗi collection có `collection_complete.json` với `episodes`
(`split`, `map`, `mode`, `n`, `sha256`), và thư mục `dataset/` cùng cấp.
File `{split}_{map}_{mode}.npz` phải chứa N action/reward/terminal/out/crash/
finish/truncated và N+1 node `state`, `c`, `hard`. `hard` ghép h và z đã flatten.
Timeout thuần túy không đổi thành terminal. Không nối qua reset giữa hai episode.

```bash
python scripts/audit_p1_p4.py --help
python scripts/audit_p1_p4.py \
  --checkpoint ARTIFACTS/baseline.pt --baseline-sha256 SHA256_CUA_BASELINE \
  --manifest DATA/collection_complete.json \
  --manifest CONFIRMATION/collection_complete.json \
  --output OUTPUT/audit_moi --device cpu --seeds 19 20 21
```

Các đường dẫn viết hoa là placeholder cần thay bằng artifact hợp lệ.
Có thể dùng `--device cuda` khi môi trường có GPU. Output phải mới/rỗng.
`--prior-checkpoint` nhận refit có key `branch`; script yêu cầu mọi trọng số
ngoài `transition.*` bằng baseline để tránh thay nhiều nhánh cùng lúc.

P0/P2 dùng posterior học được; P1/P1P2 là oracle-start. H0 dùng posterior tại
successor; H1/H5/H15 dùng action thực thi liên tiếp, không teacher forcing.
Ba seed điều khiển categorical sampling, **không phải ba seed train độc lập**;
transition c vốn deterministic với cùng start/action. Sampling prior dùng
stride 8, thêm t30 và 15 transition cuối; posterior đánh giá mọi transition.
Do đó không trộn số đếm prior với benchmark full-transition.

Evidence gồm config/hash, metric theo split/map/mode, dự đoán từng episode,
data checks, gate và completion. Hãy đọc `completed.json` cùng `gate.json`,
không suy ra hoàn tất từ dòng console. File gate mặc định đóng RL;
review c, reward/terminal, ranking thật và Critic vẫn cần trước mọi pilot.

Ranking riêng dùng cùng checkpoint/manifest/hash/output mới, thay entrypoint
bằng `scripts/audit_action_ranking.py` (xem `--help`). So các action plan đã
ghi từ t30, chỉ nhận clean/intervention có history khớp trong 1e-5. Hai ứng
viên dùng cùng seed và horizon chung tối đa15; đối chiếu tổng reward thật
discount0.997, bỏ tie thật có độ chênh ≤0.1 khi tính agreement. Lưu từng bước
reward/continue dự đoán. Đây chưa phải kiểm chứng Critic hay policy RL mới.

## Điều kiện mở pilot

Audit offline ngày11/10, baseline BC seed0/update2000, posterior deterministic
đã cache: trên test maps44200–44211 (9.429transitions,47terminal), P3 tăng
terminal recall từ51,06% lên70,21%, nhưng chưa đạt90%; reward MSE/mean-train
tăng từ1,513 lên1,550 (cần <1). Trên confirmation44300–44311
(11.447transitions,48terminal), recall43,75%→66,67%. Đây là dữ liệu frozen-BC
với5mode clean/noise/swerve/fast/timeout, không phải benchmark policy RL.
Ngay cả P3 nhận GT tại successor trong đối chứng H0 cũng chỉ đạt80,85%/75%
recall test/confirmation. Không dùng oracle này trong vận hành.

Không có xác nhận train dài thành công chỉ từ một đồ thị hoặc imagined return.
Cần log/config/checkpoint cùng source revision và tách rõ env steps với
Actor-Critic updates. Khi đủ gate mới đề xuất pilot tối đa 250 updates,
giữ baseline, eval paired cùng map/seed và early-stop trước khi chạy.
Không áp dụng KL/BC regularization để che một World Model chưa đủ tin cậy.

## Kiểm chứng checkpoint online, không nhầm với resume

Artifact online được kiểm ngày12/10 có mốc10.000/20.000/30.000 **env steps**,
tương ứng250/500/750bước optimizer Actor và Critic. Trọng số Actor/Critic
có đổi; encoder/posterior giữ nguyên. Prior c trùng refit seed0/update2000
đã audit, các base head trong wrapper vẫn là head baseline. Vì vậy có bằng
chứng đã train, nhưng không có bằng chứng các gate đã qua. Không suy từ
tên file “30k” thành30.000RLupdates hoặc policy tốt hơn.

Checkpoint experimental bọc reward/continue bằng `head.*` nhưng config vẫn
mô tả head chuẩn `net.*`, nên **loader World Model chuẩn không resume được**.
Không bỏ strict load hoặc guard để che lỗi này. Hàm
`evaluation.p1_p4_diagnostics.frozen_policy_for_audit` chỉ tải strict các
thành phần Encoder/RSSM/Actor–Critic cần cho đánh giá xe thật, yêu cầu
transition contract2, state19/action2 và learned-continuous control.
Mọi weights đóng băng; reward/continue/decoder và optimizers không được tải.
Đây không phải checkpoint migration hay đường train RL thay thế.

Bản tích hợp có127tests pass, gồm policy-only load, từ chối GT fallback,
strict keys và readiness guard vẫn khóa. Khi tiếp quản, chạy toàn bộ:

```bash
python -m unittest discover -s tests -q
```

Giữ checkpoint gốc bất biến, lưu SHA-256/config/source cùng kết quả. Chỉ
so chất lượng policy bằng eval paired cùng runtime/map/seed và reward thật;
không dùng fixed-input action probe hoặc imagined return thay cho finish,
route completion và out/crash trong môi trường.

Đánh giá frozen-policy ngày12/10: baseline BC seed0/update2000 vs artifact
online seed0/env-step30.000, MetaDrive0.4.3, horizon1000, traffic0,1, learned
posterior và action deterministic. Maps44200/44205/44500/44501, mỗi map hai
global RNG seed `map+1234` và `map+2234`:16episode tổng,8mỗi policy.
Route trung bình57,24%→18,86%, finish2/8→0/8; out4/8→4/8,
crash2/8→4/8. Riêng hai map mới44500/44501: route69,23%→17,40%.
30k thấp hơn BC ở cả8cặp. Đây là benchmark nhỏ trên4map, **không phải tám
seed train độc lập hoặc tỷ lệ thành công toàn bộ MetaDrive**. BC vẫn thất
bại ở3/4map; kết quả chỉ cho thấy bản30k không bảo toàn baseline trong phép
so này. Giữ BC và khóa RL, không lấy imagined return để phủ nhận regression.
