"""Gradio Blocks presentation, with lazy model/simulator imports."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import sys

import gradio as gr
import numpy as np

from demo_ui.analytics import compare_evaluations, plot_comparison, plot_training, read_training_log
from demo_ui.controller import Controller
from demo_ui.presentation import (latent_html, metrics_html, pipeline_html,
                                  prediction_rows, sensor_rows)

ROOT = Path(__file__).resolve().parents[1]


def build_theme():
    theme=gr.themes.Base(primary_hue='emerald',neutral_hue='slate',font=['Segoe UI','Arial','sans-serif'])
    # Keep tables, labels and controls readable even if the browser prefers light.
    values=theme.to_dict()['theme']
    theme=theme.set(**{key[:-5]:value for key,value in values.items()
                       if key.endswith('_dark') and key[:-5] in values})
    palette=dict(background_fill_primary='#0b1219',background_fill_secondary='#111b24',
                 body_background_fill='#0b1219',block_background_fill='#111b24',
                 input_background_fill='#0d161e',body_text_color='#e9f0f4',
                 border_color_primary='rgba(179,204,218,.13)',
                 block_label_background_fill='#111b24',block_label_text_color='#b9c8d0',
                 block_title_text_color='#b9c8d0',table_even_background_fill='#111b24',
                 table_odd_background_fill='#14212c',table_border_color='rgba(179,204,218,.13)',
                 button_primary_background_fill='#4cd4b0',button_primary_text_color='#0b1219',
                 button_primary_background_fill_hover='#6ee3c3',
                 button_secondary_background_fill='#1b2c38',button_secondary_text_color='#e9f0f4')
    both=dict(palette)
    both.update({k+'_dark':v for k,v in palette.items() if k+'_dark' in values})
    return theme.set(**both)


def build_app(controller=None):
    ctx = controller or Controller()

    def guarded(fn):
        def call(*args):
            try:
                return fn(*args)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
        call.__name__ = fn.__name__
        return call

    def live_view(result):
        row = result.get('record', {})
        prepared = result.get('prepared',{})
        state = row.get('state',prepared.get('state',[]))
        features = row.get('features', prepared.get('features',{}))
        index = row.get('step',0)
        warning = row.get('topdown_warning') or result.get('topdown_warning')
        status = ('Episode đã kết thúc. Có thể xuất Replay.' if result.get('done')
                  else 'Đã ghi bước '+str(index)+'.' if row else 'Đã khởi tạo. Sẵn sàng chạy hoặc tiến một bước.')
        if warning:
            status += ' ' + warning
        return (result.get('rgb'), result.get('topdown'), metrics_html(row),
                prediction_rows(row), sensor_rows(state), latent_html(features),
                pipeline_html(features,len(state) if state else None),
                gr.Slider(maximum=max(1,result.get('length',index+1)-1), value=index, interactive=bool(row)),
                status, result.get('folder',''))

    def replay_view(result):
        row = result['record']
        meta = result['metadata']
        status = (f"Bước {row['step']+1}/{result['length']} · map {meta.get('map_seed','—')} · "
                  f"shield {'bật' if meta.get('safety_shield') else 'tắt'} · "
                  f"checkpoint {meta.get('checkpoint_name','—')} · "
                  f"{'episode hoàn tất' if meta.get('complete') else 'bản ghi dừng giữa chừng'}")
        return (result['rgb'],result['topdown'],metrics_html(row),prediction_rows(row),
                sensor_rows(row['state']),latent_html(row.get('features',{})),
                gr.Slider(maximum=max(1,result['length']-1),value=row['step']),status,
                meta)

    with gr.Blocks(title='DATN · Mô phỏng xe tự hành', analytics_enabled=False) as demo:
        token = gr.State(value=None, time_to_live=1800, delete_callback=ctx.close)
        gr.HTML('<header class="lab-header"><div><p>DATN / AUTONOMOUS DRIVING</p><h1>Mô phỏng xe tự hành</h1></div><div class="caption">Camera, tín hiệu và hành trình trong cùng một không gian.<br>MetaDrive · World Model · Actor–Critic</div></header>')
        with gr.Tabs(selected='replay' if ctx.replay_only else 'live'):
            with gr.Tab('Mô phỏng & từng bước',id='live'):
                with gr.Accordion('Thiết lập mô phỏng',open=True,elem_id='session-settings',elem_classes=['settings-panel']) as settings:
                    with gr.Row(elem_classes=['settings-row']):
                        checkpoints=gr.Dropdown(ctx.checkpoints(),label='Checkpoint',scale=4,min_width=260,interactive=not ctx.replay_only)
                        refresh=gr.Button('Quét lại',scale=0,min_width=92)
                        device=gr.Radio(['cpu','cuda'],value='cpu',label='Thiết bị suy luận',scale=1,min_width=180,elem_classes=['device-selector'],interactive=not ctx.replay_only)
                        load=gr.Button('Nạp model',scale=0,min_width=126,interactive=not ctx.replay_only)
                    with gr.Row(elem_classes=['settings-row']):
                        seed=gr.Number(value=10000,label='Map seed',precision=0,min_width=128,interactive=not ctx.replay_only)
                        rng=gr.Number(value=0,label='RNG seed',precision=0,min_width=128,interactive=not ctx.replay_only)
                        horizon=gr.Number(value=1400,label='Giới hạn bước',precision=0,min_width=128,interactive=not ctx.replay_only)
                        shield=gr.Checkbox(value=False,label='Safety Shield bên ngoài',min_width=208,interactive=not ctx.replay_only)
                        create=gr.Button('Khởi tạo episode',variant='primary',min_width=176,interactive=not ctx.replay_only)
                    with gr.Accordion('Tùy chọn World Model',open=False,elem_classes=['compact-details']):
                        wm=gr.Textbox(label='WM riêng (để trống nếu đã nhúng)',placeholder='world_model/latest.pt',interactive=not ctx.replay_only)
                    model_status=gr.Markdown('Chế độ Replay CPU. Chuyển sang tab Replay để mở bản ghi.' if ctx.replay_only else 'Chọn checkpoint và nạp model để bắt đầu.',elem_id='model-status')
                with gr.Row(elem_id='live-controls',elem_classes=['transport-bar']):
                    run=gr.Button('▶ Chạy',variant='primary',min_width=100,interactive=not ctx.replay_only)
                    pause=gr.Button('Ⅱ Tạm dừng',min_width=116)
                    step=gr.Button('Tiến 1 bước →',min_width=128,interactive=not ctx.replay_only)
                    end=gr.Button('Kết thúc phiên chạy',min_width=156,interactive=not ctx.replay_only)
                    speed=gr.Radio([1,2,5,10],value=5,label='Nhịp hiển thị (bước/giây)',min_width=230)
                with gr.Row(elem_id='live-workspace',elem_classes=['observation-workspace']):
                    with gr.Column(scale=4,min_width=460,elem_classes=['camera-desk']):
                        with gr.Row(elem_classes=['camera-pair']):
                            rgb=gr.Image(label='Camera RGB · đầu vào Encoder',type='numpy',interactive=False,height=320,min_width=200,elem_id='camera-rgb',elem_classes=['camera-view','rgb-view'])
                            top=gr.Image(label='Top-down · MetaDrive',type='numpy',interactive=False,height=320,min_width=200,elem_id='camera-top',elem_classes=['camera-view'])
                        timeline=gr.Slider(0,1,value=0,step=1,label='Xem lại bước đã ghi (tạm dừng trước khi tua)',interactive=False)
                        status=gr.Markdown('Khởi tạo episode để bắt đầu.',elem_id='main-status')
                    with gr.Column(scale=1,min_width=256,elem_classes=['signal-rail']):
                        telemetry=gr.HTML(metrics_html(),elem_id='live-signals')
                with gr.Accordion('Phân tích mô hình',open=False,elem_id='live-analysis',elem_classes=['analysis-panel']):
                    pipeline=gr.HTML(pipeline_html())
                    predictions=gr.Dataframe(headers=['Đại lượng','Prior trước ảnh kế','Posterior sau ảnh kế','Thực tế'],datatype=['str','number','number','number'],label='Dự đoán và kết quả sau cùng một action',interactive=False)
                    gr.Markdown('Reward hiển thị ở đơn vị môi trường. Continue thật = 1 − terminated, timeout riêng vẫn bằng 1. Ảnh và cảm biến ở bước t đi cùng action t và kết quả sau action đó.',elem_classes=['helper-note'])
                    with gr.Row():
                        with gr.Column():
                            sensors=gr.Dataframe(headers=['Index','Cảm biến / chuẩn hóa','Giá trị'],datatype=['number','str','number'],label='Cảm biến',max_height=360,interactive=False)
                        with gr.Column():
                            latent=gr.HTML(latent_html({}))
                with gr.Column(elem_id='live-results',elem_classes=['results-panel']):
                    gr.Markdown('### Bản ghi episode',elem_classes=['section-title'])
                    with gr.Row():
                        export=gr.Button('Xuất Replay ZIP',scale=0,min_width=176,elem_id='export-replay')
                        download=gr.File(label='Replay ZIP',height=80,interactive=False,elem_id='replay-download')
                        folder=gr.Textbox(label='Thư mục lưu',interactive=False)
                live_timer=gr.Timer(.2,active=False)

            with gr.Tab('AI & thử đầu vào',id='inference'):
                gr.Markdown('### Theo dõi một lịch sử RSSM\nNạp model ở tab Mô phỏng. Ảnh phải là camera MetaDrive RGB đúng kích thước checkpoint. Vector cảm biến cần đúng thứ tự và chuẩn hóa, không nhập giá trị vật lý thay vector chuẩn hóa. Cần đóng mô phỏng trước khi thử ảnh riêng.')
                with gr.Row():
                    input_image=gr.Image(label='Ảnh camera MetaDrive RGB',type='numpy',image_mode='RGB',sources=['upload'],height=280)
                    with gr.Column():
                        sensor_json=gr.Textbox(label='Vector 19 hoặc 23 cảm biến (JSON array)',lines=5,placeholder='Dán vector từ bản ghi MetaDrive, giữ nguyên thứ tự.')
                        first=gr.Checkbox(value=True,label='Bắt đầu episode mới / reset RSSM')
                        previous=gr.Textbox(label='Action thực thi trước ảnh này',value='[0, 0]')
                        infer=gr.Button('Suy luận quan sát này',variant='primary',interactive=not ctx.replay_only)
                with gr.Column(elem_classes=['results-panel']):
                    gr.Markdown('### Kết quả suy luận',elem_classes=['section-title'])
                    infer_status=gr.Markdown('Bước đầu dùng latent khởi tạo và action trước bằng 0. Nối chuỗi cần bỏ reset và cung cấp đúng action đã thực thi. Không trộn ảnh từ hai episode.')
                    infer_chain=gr.HTML(pipeline_html())
                    with gr.Row():
                        infer_sensors=gr.Dataframe(headers=['Index','Cảm biến / chuẩn hóa','Giá trị'],label='Cảm biến đầu vào',max_height=360,interactive=False)
                        infer_latent=gr.HTML(latent_html({}))
                    infer_result=gr.JSON(label='Action, heads và lịch sử (ảnh rời không có reward thực tế)')

            with gr.Tab('Replay trên CPU',id='replay'):
                gr.Markdown('### Phát lại đồng bộ\nReplay dùng frame RGB đã ghi, top-down nếu có, sensor/action và kết quả đúng cùng bước. Không cần PyTorch, GPU hoặc MetaDrive. Nạp ZIP đã xuất, hoặc nhập thư mục recording trên máy này.')
                with gr.Accordion('Mở bản ghi',open=True,elem_id='replay-settings',elem_classes=['settings-panel']) as replay_settings:
                    with gr.Row(elem_classes=['settings-row']):
                        replay_zip=gr.File(label='Replay ZIP',file_types=['.zip'],type='filepath',height=88,min_width=220)
                        replay_path=gr.Textbox(label='Thư mục episode hoặc manifest.json trên máy',placeholder='demo_runs/episode/manifest.json',scale=2,min_width=260)
                        replay_load=gr.Button('Nạp Replay',variant='primary',scale=0,min_width=144)
                with gr.Row(elem_classes=['transport-bar']):
                    replay_play=gr.Button('▶ Phát',variant='primary',min_width=100)
                    replay_pause=gr.Button('Ⅱ Dừng',min_width=100)
                    replay_next=gr.Button('Bước kế →',min_width=128)
                    replay_rate=gr.Radio([1,2,5,10],value=5,label='Tốc độ phát (bước/giây)',min_width=230)
                with gr.Row(elem_id='replay-workspace',elem_classes=['observation-workspace']):
                    with gr.Column(scale=4,min_width=460,elem_classes=['camera-desk']):
                        with gr.Row(elem_classes=['camera-pair']):
                            replay_rgb=gr.Image(label='Camera RGB đã ghi',interactive=False,height=320,min_width=200,elem_id='replay-rgb',elem_classes=['camera-view','rgb-view'])
                            replay_top=gr.Image(label='Top-down đã ghi',interactive=False,height=320,min_width=200,elem_id='replay-top',elem_classes=['camera-view'])
                        replay_slider=gr.Slider(0,1,value=0,step=1,label='Bước đang xem')
                        replay_status=gr.Markdown('Chưa nạp Replay.',elem_id='replay-status')
                    with gr.Column(scale=1,min_width=256,elem_classes=['signal-rail']):
                        replay_metrics=gr.HTML(metrics_html(),elem_id='replay-signals')
                with gr.Accordion('Phân tích bản ghi',open=False,elem_classes=['analysis-panel']):
                    replay_predictions=gr.Dataframe(headers=['Đại lượng','Prior bước kế','Posterior bước kế','Thực tế'],label='Reward / Continue',interactive=False)
                    with gr.Row():
                        with gr.Column():
                            replay_sensors=gr.Dataframe(headers=['Index','Cảm biến / chuẩn hóa','Giá trị'],label='Cảm biến',max_height=360,interactive=False)
                        with gr.Column():
                            replay_latent=gr.HTML(latent_html({}))
                with gr.Accordion('Thông tin bản ghi',open=False,elem_classes=['results-panel']):
                    replay_meta=gr.JSON(label='Manifest: checkpoint, cấu hình, seed và trạng thái hoàn tất')
                replay_timer=gr.Timer(.2,active=False)

            with gr.Tab('Thống kê & đối chứng',id='statistics'):
                gr.Markdown('### Offline và Safety Shield\nThống kê tính từ từng episode trong JSON. Tắt shield chỉ tắt lớp bên ngoài, Actor c23 vẫn có quy tắc điều khiển tích hợp.')
                default_a=ROOT/'evaluation/baseline_c23_pure.json'
                default_b=ROOT/'evaluation/baseline_c23_shield.json'
                comparison=[]; comparison_note='Chọn hai file eval để tính.'
                if default_a.is_file() and default_b.is_file():
                    comparison,comparison_note=compare_evaluations(default_a,default_b)
                with gr.Row():
                    eval_a=gr.File(label='JSON Offline — tắt shield',file_types=['.json'],type='filepath')
                    eval_b=gr.File(label='JSON Offline + shield',file_types=['.json'],type='filepath')
                    compare=gr.Button('Tính lại so sánh')
                comparison_table=gr.Dataframe(value=comparison,headers=['Policy','Lượt','Route %','Finish','Crash','Out','Horizon','Nguồn'],interactive=False)
                comparison_status=gr.Markdown(comparison_note)
                comparison_plot=gr.Plot(value=plot_comparison(comparison) if comparison else None,label='Hai policy trên đúng bộ eval đã chọn')
                gr.Markdown('### Đường cong huấn luyện từ log thật\nGiữ nguyên tên trục trong log. `env_steps`, `step` và optimizer updates không tự quy đổi cho nhau.')
                with gr.Row():
                    log_file=gr.File(label='metrics.jsonl',file_types=['.jsonl'],type='filepath')
                    x_key=gr.Dropdown([],label='Trục X',interactive=True)
                    y_key=gr.Dropdown([],label='Trục Y',interactive=True)
                    plot_button=gr.Button('Vẽ biểu đồ')
                plot=gr.Plot(label='Chỉ số huấn luyện')
                log_state=gr.State(value=[])

            with gr.Tab('Hướng dẫn',id='guide'):
                gr.Markdown('''### Bắt đầu sử dụng

**Mô phỏng:** chọn checkpoint và thiết bị, nạp model rồi khởi tạo episode.
Chạy liên tục hoặc tiến từng bước. Camera và cột tín hiệu hiển thị dữ liệu
cùng bước. Đổi map, seed hoặc shield thì bắt đầu episode mới.

**Replay:** mở ZIP hoặc thư mục episode đã lưu. Phát, dừng hoặc chọn bước
trên thanh thời gian. Chế độ này dùng CPU và không cần nạp model.

**Thử đầu vào:** đóng mô phỏng, nhập ảnh RGB và state cùng quan sát. Reset
RSSM ở đầu episode. Khi nối lịch sử, nhập action đã thực thi ở bước trước.

### Đọc tín hiệu

Steering và ga/phanh là action chuẩn hóa trong [-1,1]. Tiến độ tuyến lấy
từ MetaDrive. Reward/Continue dự đoán được so với kết quả sau cùng action.
Mở phần phân tích để xem prior, posterior, cảm biến và bộ nhớ RSSM.

Thanh thời gian chỉ xem dữ liệu đã ghi, không quay ngược simulator.
Ảnh rời không có reward thực tế. Thanh latent thể hiện giá trị đặc trưng,
không phải độ tin cậy của mô hình.

### Model và bản ghi

c23 dùng h/z và state hiện tại trực tiếp. Bốn cảm biến vật cản lấy từ
simulator. Actor có quy tắc phanh/chống kẹt, Safety Shield là lớp can thiệp
bên ngoài. Reward/Continue còn có sai số. Giao diện phục vụ suy luận và
đánh giá, RL tiếp tục khóa.

Tạm dừng trước khi xuất Replay ZIP. Bản ghi lưu ảnh, cảm biến, action,
checkpoint hash và cấu hình để đọc lại trên CPU.
''')

        live_outputs=[rgb,top,telemetry,predictions,sensors,latent,pipeline,timeline,status,folder]
        replay_outputs=[replay_rgb,replay_top,replay_metrics,replay_predictions,replay_sensors,replay_latent,replay_slider,replay_status,replay_meta]

        def on_load_model(token, checkpoint, wm_checkpoint, device):
            meta=ctx.load(token,checkpoint,wm_checkpoint,device)
            cleared=list(live_view({}))
            cleared[-2]='Đã nạp model. Khởi tạo episode mới để chạy.'
            return (f"**{html.escape(meta['checkpoint_name'])}** · {meta['state_dim']}D · RGB {meta['image_shape'][0]}×{meta['image_shape'][1]} · {meta['control_source']} · {device}\n\nSHA256 `{meta['checkpoint_sha256']}` · RL khóa.",
                    gr.Timer(active=False),*cleared,None,'Model vừa đổi. Bắt đầu lịch sử ảnh mới bằng reset RSSM.',
                    pipeline_html(),latent_html({}),[])

        def on_create(token,seed,rng,horizon,shield):
            return (*live_view(ctx.open(token,seed,rng,horizon,shield)),gr.Timer(active=False),gr.Accordion(open=False))

        def on_step(token):
            return live_view(ctx.step(token))

        def on_run(token):
            s=ctx.get(token)
            with s.lock:
                if not s.folder or s.done:
                    raise ValueError('Cần khởi tạo episode chưa kết thúc trước khi chạy.')
                s.running=True
            return gr.Timer(active=True)

        def on_pause(token):
            s=ctx.get(token)
            with s.lock:s.running=False
            return gr.Timer(active=False)

        def on_tick(token):
            s=ctx.get(token)
            with s.lock:
                if not s.running:return tuple(gr.skip() for _ in live_outputs)+(gr.Timer(active=False),)
                try:
                    result=ctx.step(token)
                except Exception:
                    s.running=False
                    raise
                return (*live_view(result),gr.Timer(active=s.running))

        def on_end(token):
            s=ctx.get(token)
            with s.lock:
                s.running=False
                if s.worker:s.worker.call('end')
                s.done=True
            return gr.Timer(active=False),'Đã đóng mô phỏng. Bản ghi vẫn được giữ, có thể xuất Replay hoặc thử ảnh.'

        def on_infer(token,image,vector,action,reset):
            if image is None:raise ValueError('Cần ảnh RGB MetaDrive thật.')
            sensors=json.loads(vector); previous=json.loads(action)
            result=ctx.infer(token,image,sensors,previous,reset)
            return (result,pipeline_html(result['features'],len(result['state'])),
                    latent_html(result['features']),sensor_rows(result['state']),
                    f"Đã xử lý {result['history_steps']} quan sát trong lịch sử hiện tại. Action: {result['action']}. Heads posterior thuộc ảnh đang nhập, không có ground truth reward để đối chiếu.")

        def on_replay_load(token,path,archive):
            return (*replay_view(ctx.load_replay(token,path,archive)),gr.Timer(active=False),gr.Accordion(open=False))

        def on_replay_seek(token,index):
            s=ctx.get(token)
            with s.lock:s.replaying=False
            return (*replay_view(ctx.replay_frame(token,index)),gr.Timer(active=False))

        def on_replay_play(token):
            s=ctx.get(token)
            with s.lock:
                if s.replay is None:raise ValueError('Nạp Replay trước.')
                if s.cursor>=len(s.replay)-1:s.cursor=-1
                s.replaying=True
            return gr.Timer(active=True)

        def on_replay_pause(token):
            s=ctx.get(token)
            with s.lock:s.replaying=False
            return gr.Timer(active=False)

        def on_replay_tick(token):
            s=ctx.get(token)
            with s.lock:
                if not s.replaying:return tuple(gr.skip() for _ in replay_outputs)+(gr.Timer(active=False),)
                result=ctx.replay_frame(token,s.cursor+1)
                s.replaying=s.cursor<len(s.replay)-1
                return (*replay_view(result),gr.Timer(active=s.replaying))

        def on_log(path):
            rows,keys=read_training_log(path)
            x=next((k for k in ['env_steps','step','steps','update','updates'] if k in keys),keys[0])
            y=next((k for k in ['actor_loss','wm_loss','loss','critic_loss','mean_return'] if k in keys and k!=x),
                   next((k for k in keys if k!=x),x))
            return rows,gr.Dropdown(choices=keys,value=x),gr.Dropdown(choices=keys,value=y)

        def on_compare(a,b):
            rows,note=compare_evaluations(a,b)
            return rows,note,plot_comparison(rows)

        def start_session(request:gr.Request):
            ctx.get(request.session_hash)
            return request.session_hash

        def unload(request:gr.Request):
            ctx.close(request.session_hash)

        demo.load(start_session,outputs=token,api_name='start_session')
        demo.unload(unload)
        opts=dict(concurrency_id='model-actions',concurrency_limit=1)
        refresh.click(lambda:gr.Dropdown(choices=ctx.checkpoints()),outputs=checkpoints,api_name='scan_checkpoints')
        load.click(guarded(on_load_model),[token,checkpoints,wm,device],
                   [model_status,live_timer]+live_outputs+[infer_result,infer_status,infer_chain,infer_latent,infer_sensors],api_name='load_model',**opts)
        create.click(guarded(on_create),[token,seed,rng,horizon,shield],live_outputs+[live_timer,settings],api_name='create_episode',**opts)
        step.click(guarded(on_step),token,live_outputs,api_name='next_step',**opts)
        run.click(guarded(on_run),token,live_timer,api_name='run',**opts)
        pause.click(guarded(on_pause),token,live_timer,queue=False,api_name='pause')
        live_timer.tick(guarded(on_tick),token,live_outputs+[live_timer],show_progress='hidden',trigger_mode='always_last',api_visibility='private',**opts)
        speed.change(lambda rate:gr.Timer(value=1/float(rate)),speed,live_timer,queue=False,api_visibility='private')
        timeline.release(guarded(lambda t,i:live_view(ctx.seek_live(t,i))),[token,timeline],live_outputs,api_name='seek_live',**opts)
        end.click(guarded(on_end),token,[live_timer,status],api_name='close_episode',**opts)
        export.click(guarded(ctx.export),token,download,api_name='export_replay',**opts)
        infer.click(guarded(on_infer),[token,input_image,sensor_json,previous,first],[infer_result,infer_chain,infer_latent,infer_sensors,infer_status],api_name='infer',**opts)
        replay_load.click(guarded(on_replay_load),[token,replay_path,replay_zip],replay_outputs+[replay_timer,replay_settings],api_name='replay_load')
        replay_slider.release(guarded(on_replay_seek),[token,replay_slider],replay_outputs+[replay_timer],api_name='replay_seek')
        replay_play.click(guarded(on_replay_play),token,replay_timer,api_name='replay_play')
        replay_pause.click(guarded(on_replay_pause),token,replay_timer,queue=False,api_name='replay_pause')
        replay_next.click(guarded(lambda t:replay_view(ctx.replay_frame(t,ctx.get(t).cursor+1))),token,replay_outputs,api_name='replay_next')
        replay_timer.tick(guarded(on_replay_tick),token,replay_outputs+[replay_timer],show_progress='hidden',trigger_mode='always_last',api_visibility='private')
        replay_rate.change(lambda rate:gr.Timer(value=1/float(rate)),replay_rate,replay_timer,queue=False,api_visibility='private')
        compare.click(guarded(on_compare),[eval_a,eval_b],[comparison_table,comparison_status,comparison_plot],api_name='compare_evaluations')
        log_file.change(guarded(on_log),log_file,[log_state,x_key,y_key],api_name='load_log')
        plot_button.click(guarded(plot_training),[log_state,x_key,y_key],plot,api_name='plot_log')
    demo.queue(max_size=16)
    demo.datn_controller=ctx
    return demo


def main():
    parser=argparse.ArgumentParser(description='DATN Vietnamese Gradio demonstration')
    parser.add_argument('--checkpoint-dir',default='checkpoints')
    parser.add_argument('--output-dir',default='demo_runs')
    parser.add_argument('--replay-only',action='store_true')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--share',action='store_true',help='Create temporary Gradio link (optional, e.g. Colab)')
    args=parser.parse_args()
    ctx=Controller(args.checkpoint_dir,args.output_dir,replay_only=args.replay_only)
    demo=build_app(ctx)
    theme=build_theme()
    try:
        demo.launch(server_name=args.host,server_port=args.port,share=args.share,
                    css=(Path(__file__).with_name('style.css')).read_text(),theme=theme,
                    allowed_paths=[str(ctx.output_root)])
    finally:ctx.close_all()


if __name__=='__main__':main()
