"""Read source evaluation JSON and training logs without requiring a model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def read_evaluation(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    episodes = data.get('episodes', data.get('results'))
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('File cần danh sách episodes/results để tính lại thống kê.')
    def mean(key):
        values = [float(row[key]) for row in episodes if key in row]
        if len(values) != len(episodes) or not np.isfinite(values).all():
            raise ValueError('Thiếu hoặc sai chỉ số ' + key)
        return float(np.mean(values))
    summary = dict(episodes=len(episodes), route=mean('route_completion'),
                   finish=mean('success'), crash=mean('collision'), out=mean('out_of_road'))
    seeds = [row.get('seed') for row in episodes]
    return dict(summary=summary, seeds=seeds, horizon=data.get('horizon'),
                episodes=episodes, source=Path(path).name,
                provenance=data.get('provenance'))


def compare_evaluations(left, right):
    a, b = read_evaluation(left), read_evaluation(right)
    paired = (a['seeds'] == b['seeds'] and None not in a['seeds'] and a['horizon'] == b['horizon'])
    rows = []
    for name, result in [('Offline — tắt shield', a), ('Offline + Safety Shield', b)]:
        s = result['summary']
        rows.append([name, s['episodes'], round(100*s['route'], 2),
                     int(round(s['finish']*s['episodes'])),
                     int(round(s['crash']*s['episodes'])), int(round(s['out']*s['episodes'])),
                     result['horizon'], result['source']])
    note = ('Danh sách map seed và horizon khớp.' if paired else 'Hai file khác seed/horizon. Không diễn giải chênh lệch là đối chứng paired.')
    if not a['provenance'] or not b['provenance']:
        note += ' JSON lịch sử thiếu checkpoint hash/runtime đầy đủ. Đây là kết quả lưu sẵn, chưa tái lập bởi giao diện.'
    return rows, note


def read_training_log(path, limit=20000):
    """Keep the x-axis field name. Never relabel env steps as updates."""
    rows = []
    with Path(path).open(encoding='utf-8') as file:
        for index, line in enumerate(file):
            if index >= limit:
                raise ValueError('Log trên 20.000 dòng. Chọn file hoặc trích đoạn nhỏ hơn.')
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError('Log chưa có dữ liệu.')
    keys = sorted({key for row in rows for key, value in row.items()
                   if isinstance(value, (float, int)) and not isinstance(value, bool)})
    if not keys:
        raise ValueError('Log chưa có chỉ số số để vẽ.')
    return rows, keys


def plot_comparison(rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    if len(rows)!=2:
        raise ValueError('Cần hai policy để vẽ đối chứng.')
    figure=Figure(figsize=(10,2.8),facecolor='#111b24')
    axes=figure.subplots(1,2)
    labels=['Tắt shield','Bật shield']
    colors=['#7e96a5','#4cd4b0']
    values=[[row[2] for row in rows],[100*row[3]/row[1] for row in rows]]
    for ax,series,title in zip(axes,values,['Route trung bình (%)','Về đích (%)']):
        ax.set_facecolor('#111b24')
        ax.barh(labels,series,color=colors,height=.45)
        ax.invert_yaxis();ax.set_xlim(0,108)
        ax.set_title(title,color='#e9f0f4',fontsize=11,loc='left')
        ax.tick_params(colors='#b9c8d0',labelsize=9)
        for i,value in enumerate(series):ax.text(value+1,i,f'{value:.1f}',va='center',color='#e9f0f4',fontsize=9)
        for spine in ax.spines.values():spine.set_visible(False)
    figure.tight_layout()
    return figure


def plot_training(rows, x, y):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    points = [(row[x], row[y]) for row in rows if x in row and y in row
              and np.isfinite([row[x], row[y]]).all()]
    if not points:
        raise ValueError('Không có cặp chỉ số hữu hạn để vẽ.')
    figure = Figure(figsize=(10, 3.4), facecolor='#111b24')
    ax = figure.subplots()
    ax.set_facecolor('#111b24')
    ax.plot(*zip(*points), color='#4cd4b0', linewidth=1.6)
    ax.set_xlabel(x, color='#b9c8d0'); ax.set_ylabel(y, color='#b9c8d0')
    ax.tick_params(colors='#b9c8d0'); ax.grid(alpha=.12)
    for spine in ax.spines.values():
        spine.set_color('#31404b')
    figure.tight_layout()
    return figure
