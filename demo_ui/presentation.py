"""Small Vietnamese views derived only from real records/model output."""
from __future__ import annotations
import html
import numpy as np

# StateObservation (MetaDrive 0.4.3, no side/lane/vehicle lidar beams):
# ego 0..8, two navigation waypoints (five normalized fields each).
SENSORS = [
    'Mép trái / tổng bề rộng chuẩn', 'Mép phải / tổng bề rộng chuẩn',
    'Hướng xe so với làn (chuẩn hóa)', '(Tốc độ + 1) / (max_speed + 1)', 'Steering chuẩn hóa [0,1]',
    'Action trước: steering → (a+1)/2', 'Action trước: ga/phanh → (a+1)/2', 'Yaw rate (chuẩn hóa)',
    'Lệch ngang / MAX_LANE_WIDTH + 0,5 (clip [0,1])',
    'Waypoint 1: chiếu dọc', 'Waypoint 1: chiếu ngang', 'Waypoint 1: bán kính đường cong',
    'Waypoint 1: hướng đường cong', 'Waypoint 1: góc đường cong',
    'Waypoint 2: chiếu dọc', 'Waypoint 2: chiếu ngang', 'Waypoint 2: bán kính đường cong',
    'Waypoint 2: hướng đường cong', 'Waypoint 2: góc đường cong',
    'Khoảng cách xe trước / 50 m', 'Vận tốc tương đối / 20 m/s',
    'Lệch ngang xe trước / 3,5 m', 'Nguy cơ giao cắt',
]


def sensor_rows(state):
    return [[i, SENSORS[i] if i < len(SENSORS) else f'State {i}', round(float(v),6)]
            for i,v in enumerate(state)]


def _escape(value):
    return html.escape(str(value))


def number(value, digits=3):
    return '—' if value is None else f'{float(value):.{digits}f}'


def metrics_html(row=None):
    row = row or {}
    flags = row.get('flags',{})
    status = ('Về đích' if flags.get('arrive_dest') else 'Ra đường' if flags.get('out_of_road')
              else 'Va chạm' if any(flags.get(k) for k in ['crash','crash_vehicle','crash_object','crash_sidewalk'])
              else 'Hết thời gian' if row.get('truncated') else 'Đang chạy' if row else 'Chưa có episode')
    action = row.get('action_executed',[None,None])
    data = [('QUÃNG ĐƯỜNG', number(100*row['route_completion'],1)+'%' if 'route_completion' in row else '—'),
            ('GÓC LÁI', number(action[0])), ('GA / PHANH', number(action[1])),
            ('TỔNG REWARD', number(row.get('episode_return'),1)), ('KẾT QUẢ',status)]
    return '<div class="telemetry">'+''.join(f'<div><span>{label}</span><strong>{_escape(value)}</strong></div>' for label,value in data)+'</div>'


def pipeline_html(features=None, state_dim=None):
    features = features or {}
    c = features.get('c')
    parts = [('RGB + cảm biến', str(state_dim or '19 / 23')+' tọa độ'),
             ('Encoder', str(features.get('embedding',{}).get('dim','—'))+' features'),
             ('RSSM · h / z', f"{features.get('h',{}).get('dim','—')} / {features.get('z',{}).get('dim','—')}"),
             ('Actor', 'steering · ga/phanh')]
    content = '<div class="signal-chain">'+''.join(f'<div><small>{i+1:02d}</small><b>{label}</b><span>{_escape(value)}</span></div>' for i,(label,value) in enumerate(parts))+'</div>'
    if state_dim == 23:
        content += '<p class="fine-print">c23: state quan sát đi thêm trực tiếp vào Actor. Actor có quy tắc phanh/chống kẹt của Tien. Safety Shield là lớp can thiệp bên ngoài.</p>'
    elif c:
        content += '<p class="fine-print">Nhánh c do RSSM tạo: '+str(c['dim'])+' chiều.</p>'
    return content


def prediction_rows(row):
    prior = row.get('prior',{})
    post = row.get('posterior_next',{})
    rows = [
        ['Reward bước kế (đơn vị môi trường)', prior.get('reward'), post.get('reward'), row.get('reward')],
        ['Continue bước kế (xác suất)', prior.get('continue_probability'), post.get('continue_probability'), row.get('continue_target')],
        ['Reward symlog (output gốc)',prior.get('reward_symlog'),post.get('reward_symlog'),
         float(np.sign(row['reward'])*np.log1p(abs(row['reward']))) if 'reward' in row else None],
    ]
    return [[r[0],*[None if v is None else round(float(v),4) for v in r[1:]]] for r in rows]


def latent_html(features):
    blocks = []
    for name, summary in features.items():
        values = summary.get('preview', [])
        if not values:
            continue
        maximum = max(max(abs(float(x)) for x in values),1e-6)
        bars = ''.join(f'<span title="{i}: {float(x):.4f}" style="height:{8+44*abs(float(x))/maximum:.1f}px;opacity:{.35+.65*abs(float(x))/maximum:.3f}"></span>' for i,x in enumerate(values))
        blocks.append(f'<section class="latent-block"><b>{_escape(name)}</b><small>{summary["dim"]} chiều · μ {number(summary["mean"])} · σ {number(summary["std"])}</small><div class="latent-bars">{bars}</div></section>')
    return '<div class="latent-grid">'+''.join(blocks)+'</div>' if blocks else '<p class="fine-print">Nạp checkpoint và chạy một bước để xem embedding, h/z và c nếu có.</p>'
