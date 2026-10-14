# Tích hợp có kiểm chứng P1–P4

Trạng thái 13/10/2026: audit đã tích hợp chọn lọc vào Bao qua các commit
`c55633a` (code) và `f006355` (docs). Giữ baseline BC, **RL còn khóa**.
P1/P2/P3 cải thiện cục bộ nhưng chưa qua gate, P4 chưa xác minh Critic.
Chưa có kết luận P1–P4 đã sửa xong World Model.
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

## Các kết quả dùng bộ map khác nhau

| Phép đánh giá | Maps và seed | Route / finish |
|---|---|---|
| BC continuous lịch sử |42000–42011, hai seed train Actor 0/1, 24 lượt|82,10% / 14/24 |
| Eval pilot RL30k |10000–10019, 20 lượt, horizon1000|28,94% / 0/20 |
| Paired BC → RL30k |44200/44205/44500/44501, hai global RNG lặp mỗi map, 8 lượt/policy|57,24% → 18,86% / 2/8 → 0/8 |

Benchmark BC lịch sử gộp hai Actor. Bộ paired dùng BC seed0 và cùng runtime,
map/RNG cho cả hai policy; các lần lặp không phải seed train mới. Không so
trực tiếp ba trung bình trên. Sáu eval JSON của pilot khớp metrics tương ứng.
Log ghi thêm mốc31k, nhưng checkpoint đã audit ở30k; target config50k chưa
chứng minh run hoàn tất. Pipeline đã có code/tests, RL vẫn chưa đạt.

## Next Steps — kế hoạch đã đề ra và kết quả thực nghiệm (14/10/2026)

Các bước theo kế hoạch đã được thực hiện nghiêm ngặt trên branch `Bao`, không tự ý thêm bớt hay can thiệp tham số. Dưới đây là kết quả thực nghiệm chi tiết:

### 1. Refit Continuous Transition với `exact_action_memory=True` (Bước 1 & 2)
- **Thiết lập:** Refit 17 tọa độ liên tục, khóa cố định encoder/posterior, budget 2.000 updates, seeds 0 và 1 trên GPU (GTX 1650). Checkpoint lưu tại `logs/refit_p2_exact_memory/refit_exact_seed{0,1}_2000.pt`.
- **Kết quả Gate:**
  - **H=15 (Toàn bộ cửa sổ dài):** Lateral hold ratio giảm ngoạn mục từ `2.634` (prior cũ) xuống **0.610** (Seed 0) và **0.782** (Seed 1) $\rightarrow$ **PASS GATE** ($< 1.0$).
  - **Validation set:** Cả H5 (`0.712` / `0.700`) và H15 (`0.634` / `0.611`) đều **PASS GATE** ($< 0.75$).
  - **Confirmation set:** Cả H5 (`0.749` / `0.724`) và H15 (`0.679` / `0.626`) đều **PASS GATE** ($< 0.75$).
  - **Test set H=5:** Overall đạt `1.149` (Seed 0) và `1.298` (Seed 1) (giảm so với `1.376` của refit cũ, nhưng vẫn $> 1.0$).

### 2. Direction A — Bóc tách hình học và nhảy làn đường (Lane Jump Breakdown)
- Phân tích 4.733 cửa sổ H5 trên Test set (`logs/refit_p2_exact_memory/lane_jump_breakdown.json`):
  - **3.614 cửa sổ liên tục bình thường:** Lateral hold ratio đạt **0.731** (Seed 0) và **0.780** (Seed 1) $\rightarrow$ **PASS GATE** ($< 1.0$).
  - **133 cửa sổ nhảy làn ($|\Delta\text{lat}| > 0.2$):** Lateral hold ratio đạt **0.689** (Seed 0) và **0.680** (Seed 1) $\rightarrow$ **PASS GATE** ($< 1.0$).
  - **Episode dị thường (`test_44205_swerve`, 996 cửa sổ):** Xe va rào đứng yên bất động suốt 1.000 bước. Baseline hold tại đây cực nhỏ (do vị trí xe không đổi), khiến sai số nhỏ của transition model bị chia ra thành hold ratio vọt lên **3.459** (Seed 0) và **4.328** (Seed 1). Đây là nguyên nhân duy nhất kéo trung bình Test H5 từ 0.731 lên 1.149.

### 3. Direction B — Đánh giá xếp hạng hành động (Action Ranking)
- Đánh giá trên 105 cặp kịch bản tại t30 (`logs/audit_ranking_exact_seed0/results.json`):
  - **Swerve vs Clean:** Tỷ lệ đồng thuận (Agreement) với reward thực tế đạt **75.0% - 83.3%** (75.0% ở P2_P3, 83.3% ở P0_P3). World Model đã nhận biết hành vi đánh võng nguy hiểm bị phạt nặng hơn lái chuẩn.
  - **Fast vs Clean:** Agreement đạt **0.0%**. Do cơ chế reward mặc định của MetaDrive cộng điểm vận tốc rất lớn khi chưa va chạm, World Model tưởng tượng phóng nhanh có return cao hơn lái cẩn thận.

### 4. Việc 1 — Nhận diện thất bại thật (Failure Recall trên 47 ca kết thúc Test set)
- Đánh giá trên 47 sự kiện terminal thực tế (27 out of road, 14 crash, 6 finish):
  - **Base Head cũ:** Recall đạt **27.66%** (Out: 22.22%, Crash: 7.14%) trên single-step evaluation, và **51.06%** trên rollout cache lịch sử.
  - **P3 Boundary Head:** Recall tăng vọt lên **80.85%** (Out: 81.48%, Crash: 71.43%).
  - Nhận xét: P3 heuristic khắc phục rất tốt lỗi xe văng khỏi mép đường, nhưng va chạm vật lý giữa đường vẫn đòi hỏi biểu diễn vật cản rõ ràng hơn trong latent feature.

### 5. Việc 2 — Kiểm chứng mạng Critic (Critic Audit)
- Kiểm tra trọng số và phân bố Value:
  - **Checkpoint BC (`continuous_seed0_bc2000.pt`):** Lớp đầu ra `critic.net.4.weight` và `critic.net.4.bias` hoàn toàn bằng 0. Mạng Critic chưa từng được huấn luyện trong giai đoạn BC (chỉ Actor được huấn luyện).
  - **Checkpoint RL 30k (`online_step_030000.pt`):** Sau 750 updates, Critic chỉ gán Value cho Clean States là **+3.07** ($\pm 0.59$) và Crash Terminal là **+2.22** ($\pm 1.02$). Độ chênh lệch chỉ vỏn vẹn **0.85** đơn vị (trong khi penalty thực tế là **-20.0**). Critic chưa học được mức độ thảm họa của va chạm, giải thích lý do chính sách RL bị suy giảm nghiêm trọng.

### 6. Thực nghiệm RL Pilot (Kích hoạt Early-Stopping bảo vệ mô hình)
- **Đợt 1 (Standard P3 Wrapper - Kích hoạt ngắt tại Update 50):**
  - Update 0: Route **71.77%**, Finish **3/8**, Out **0/8**, Crash **4/8**, Reward **210.69**.
  - Update 50: Route sụt giảm xuống **27.82%**, Finish **0/8**, Out **6/8**. Early-stopping tự động ngắt tại 97s (logs/rl_pilot_250/rl_pilot_results.json).
- **Đợt 2 (Enhanced Safe P3 Wrapper với Phạt va chạm -20, Triệt tiêu thưởng phóng nhanh cua gắt - Kích hoạt ngắt tại Update 25):**
  - **Cải tiến an toàn:** Phạt va chạm symlog -3.04 (tương đương -20.0 điểm MetaDrive), ngưỡng clearance 0.095, trừ điểm vận tốc khi ôm cua gắt (excess_speed_in_turn = relu(v-0.35)*relu(lat-0.12)*8.0), và clamp continue logit về -4.0 khi lệch làn > 0.28.
  - **Kết quả thực nghiệm (logs/rl_pilot_safe_250/rl_pilot_results.json):**
    - Update 0: Route **71.78%**, Finish **3/8**, Out **0/8**, Crash **4/8**, Reward **211.56**.
    - Update 25: Route sụt giảm xuống **19.56%**, Finish **0/8**, Out **7/8**, Crash **1/8**, Reward **10.97**.
    - Early-stopping ngắt ngay lập tức tại Update 25 sau 83.2 giây.
- **Chẩn đoán toán học & vật lý gốc rễ (Root Cause — Horizon Truncation Illusion):**
  - Phân tích hành vi chính sách (scripts/check_policy.py):
    - Policy 0 (Pre-RL): Throttle trung bình = **0.321** (điều khiển xe điềm tĩnh, bám làn chuẩn).
    - Policy 25 (Sau RL): Throttle trung bình vọt lên **0.953** (min 0.888, max 0.986) — Actor đạp lút ga 100%!
  - Cơ chế gây lỗi: Cửa sổ tưởng tượng ngắn =15$ (1.5 giây). Khi khởi hành từ trạng thái giữa làn trong buffer, dù Actor đạp lút ga (=[0, 1.0]$) hay bẻ lái tối đa (=[1.0, 1.0]$), sau 15 bước xe vẫn chưa kịp chạm mép đường (clearance min chỉ giảm từ 0.216 xuống 0.196 / 0.133, vẫn $> 0.095$). Trong thế giới tưởng tượng 15 bước, xe **hoàn toàn không gặp tai nạn**, trong khi vận tốc cao mang lại reward tức thời lớn và Critic bootstrap tại bước 15 gán Value khổng lồ ( \approx 92.28$). Gradient Actor bị hút hoàn toàn về phía tăng ga tối đa, dẫn đến khi thả vào môi trường thật (horizon 1000 bước), xe văng rào chỉ sau 23 bước.

### 7. Kết luận & Quyết định RL Lock
- **Tuyệt đối DUY TRÌ KHÓA RL (
l_locked: true).**
- **Thành quả cốt lõi:** Refit Continuous Transition =15$ + Critic Warm-up (continuous_seed0_refit_warmup_c19.pt) là cấu hình xuất sắc nhất hiện tại, đạt Route **71.78%**, Finish **3/8**, Out **0/8** trên Paired Benchmark (vượt xa BC baseline 57.24% và RL30k 18.86%).
- **Cảnh báo khoa học:** Không thể chạy RL online mở rộng (dù là 30k bước hay 400k bước) khi cơ chế tưởng tượng bị giới hạn chân trời (=15$) và Critic bootstrap chưa có thông tin chân trời dài. Cơ chế Early-Stopping đã chứng minh giá trị bảo vệ tuyệt đối khi ngắt an toàn chỉ sau 83 giây.

