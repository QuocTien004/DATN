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

    with gr.Blocks(title='DATN · Phòng kiểm chứng xe tự hành', analytics_enabled=False) as demo:
        token = gr.State(value=None, time_to_live=1800, delete_callback=ctx.close)
        gr.HTML('<header class="lab-header"><div><p>DATN / WORLD MODEL LAB</p><h1>Phòng kiểm chứng xe tự hành</h1></div><div class="caption">Quan sát xe. Dừng từng bước. Đối chiếu quyết định.<br>Offline c23 · MetaDrive · RSSM / Actor–Critic</div></header>')
        with gr.Tabs(selected='replay' if ctx.replay_only else 'live'):
            with gr.Tab('Mô phỏng & từng bước',id='live'):
                with gr.Row():
                    with gr.Column(scale=1,min_width=260,elem_id='model-panel'):
                        gr.Markdown('### Thiết lập phiên chạy')
                        checkpoints=gr.Dropdown(ctx.checkpoints(),label='Checkpoint Actor / Offline / Online',interactive=not ctx.replay_only)
                        refresh=gr.Button('Quét lại checkpoint',size='sm')
                        wm=gr.Textbox(label='WM riêng (để trống nếu đã nhúng)',placeholder='world_model/latest.pt',interactive=not ctx.replay_only)
                        device=gr.Radio(['cpu','cuda'],value='cpu',label='Thiết bị suy luận',interactive=not ctx.replay_only)
                        load=gr.Button('Nạp model',variant='primary',interactive=not ctx.replay_only)
                        model_status=gr.Markdown('Chế độ Replay CPU — không nạp model/MetaDrive.' if ctx.replay_only else 'Chưa nạp checkpoint. Chỉ chọn checkpoint đáng tin cậy của dự án.')
                        seed=gr.Number(value=10000,label='Map seed',precision=0)
                        rng=gr.Number(value=0,label='RNG seed (tách khỏi map)',precision=0)
                        horizon=gr.Number(value=1400,label='Giới hạn bước episode',precision=0)
                        shield=gr.Checkbox(value=False,label='Bật Safety Shield bên ngoài')
                        create=gr.Button('Khởi tạo episode',interactive=not ctx.replay_only)
                        gr.Markdown('Thay seed/checkpoint/shield cần khởi tạo episode mới. RL không được chạy từ giao diện.')
                    with gr.Column(scale=4,min_width=480):
                        telemetry=gr.HTML(metrics_html())
                        with gr.Row():
                            rgb=gr.Image(label='Camera RGB — đầu vào thật của Encoder',type='numpy',interactive=False,height=310,elem_id='camera-rgb',elem_classes=['camera-view','rgb-view'])
                            top=gr.Image(label='Top-down — MetaDrive render',type='numpy',interactive=False,height=310,elem_id='camera-top',elem_classes=['camera-view'])
                        with gr.Row():
                            run=gr.Button('▶ Chạy',variant='primary',interactive=not ctx.replay_only)
                            pause=gr.Button('Ⅱ Tạm dừng')
                            step=gr.Button('Tiến 1 bước →',interactive=not ctx.replay_only)
                            end=gr.Button('Kết thúc phiên chạy',interactive=not ctx.replay_only)
                        with gr.Row():
                            timeline=gr.Slider(0,1,value=0,step=1,label='Xem lại bước đã ghi (tạm dừng trước khi tua)',interactive=False)
                            speed=gr.Radio([1,2,5,10],value=5,label='Nhịp hiển thị (bước/giây)')
                        status=gr.Markdown('Khởi tạo episode để bắt đầu.',elem_id='main-status')
                        pipeline=gr.HTML(pipeline_html())
                        predictions=gr.Dataframe(headers=['Đại lượng','Prior trước khi thấy ảnh kế','Posterior sau ảnh kế','Thực tế'],datatype=['str','number','number','number'],label='Dự đoán bước kế và kết quả sau action',interactive=False)
                        gr.Markdown('Reward đã đổi từ symlog về đơn vị môi trường, giới hạn hiển thị như code imagination. Continue thật = 1 − terminated. Timeout riêng vẫn có Continue = 1. Ảnh/sensor của bước t đi cùng action t và kết quả sau action đó.')
                        with gr.Accordion('Cảm biến và bộ nhớ ở bước đang chọn',open=False):
                            sensors=gr.Dataframe(headers=['Index','Cảm biến / chuẩn hóa','Giá trị'],datatype=['number','str','number'],interactive=False)
                            latent=gr.HTML(latent_html({}))
                        with gr.Row():
                            export=gr.Button('Xuất Replay ZIP')
                            download=gr.File(label='Bản ghi có thể phát lại trên CPU',interactive=False)
                        folder=gr.Textbox(label='Thư mục bản ghi trên máy chạy',interactive=False)
                live_timer=gr.Timer(.2,active=False)

            with gr.Tab('AI & thử đầu vào'):
                gr.Markdown('### Theo dõi một lịch sử RSSM\nNạp model ở tab Mô phỏng. Ảnh phải là camera MetaDrive RGB đúng kích thước checkpoint. Vector cảm biến cần đúng thứ tự và chuẩn hóa, không nhập giá trị vật lý thay vector chuẩn hóa. Cần đóng mô phỏng trước khi thử ảnh riêng.')
                with gr.Row():
                    input_image=gr.Image(label='Ảnh camera MetaDrive RGB',type='numpy',image_mode='RGB',sources=['upload'],height=280)
                    with gr.Column():
                        sensor_json=gr.Textbox(label='Vector 19 hoặc 23 cảm biến (JSON array)',lines=5,placeholder='Dán vector từ bản ghi MetaDrive, giữ nguyên thứ tự.')
                        first=gr.Checkbox(value=True,label='Bắt đầu episode mới / reset RSSM')
                        previous=gr.Textbox(label='Action thực thi trước ảnh này',value='[0, 0]')
                        infer=gr.Button('Suy luận quan sát này',variant='primary',interactive=not ctx.replay_only)
                infer_status=gr.Markdown('Bước đầu dùng latent khởi tạo và action trước bằng 0. Nối chuỗi cần bỏ reset và cung cấp đúng action đã thực thi. Không trộn ảnh từ hai episode.')
                infer_chain=gr.HTML(pipeline_html())
                infer_latent=gr.HTML(latent_html({}))
                infer_sensors=gr.Dataframe(headers=['Index','Cảm biến / chuẩn hóa','Giá trị'],interactive=False)
                infer_result=gr.JSON(label='Action, heads và lịch sử thực tế (không có reward thật khi chỉ upload ảnh)')

            with gr.Tab('Replay trên CPU',id='replay'):
                gr.Markdown('### Phát lại đồng bộ\nReplay dùng frame RGB đã ghi, top-down nếu có, sensor/action và kết quả đúng cùng bước. Không cần PyTorch, GPU hoặc MetaDrive. Nạp ZIP đã xuất, hoặc nhập thư mục recording trên máy này.')
                with gr.Row():
                    replay_zip=gr.File(label='Replay ZIP',file_types=['.zip'],type='filepath')
                    replay_path=gr.Textbox(label='Thư mục episode hoặc manifest.json trên máy',placeholder='demo_runs/20261031_…/manifest.json')
                    replay_load=gr.Button('Nạp Replay',variant='primary')
                replay_metrics=gr.HTML(metrics_html())
                with gr.Row():
                    replay_rgb=gr.Image(label='Camera RGB đã ghi',interactive=False,height=310,elem_classes=['camera-view','rgb-view'])
                    replay_top=gr.Image(label='Top-down đã ghi',interactive=False,height=310,elem_classes=['camera-view'])
                with gr.Row():
                    replay_play=gr.Button('▶ Phát',variant='primary')
                    replay_pause=gr.Button('Ⅱ Dừng')
                    replay_next=gr.Button('Bước kế →')
                    replay_rate=gr.Radio([1,2,5,10],value=5,label='Tốc độ phát (bước/giây)')
                replay_slider=gr.Slider(0,1,value=0,step=1,label='Bước đang xem')
                replay_status=gr.Markdown('Chưa nạp Replay.')
                replay_predictions=gr.Dataframe(headers=['Đại lượng','Prior bước kế','Posterior bước kế','Thực tế'],interactive=False)
                with gr.Accordion('Sensor, latent và nguồn bản ghi',open=False):
                    replay_sensors=gr.Dataframe(headers=['Index','Cảm biến / chuẩn hóa','Giá trị'],interactive=False)
                    replay_latent=gr.HTML(latent_html({}))
                    replay_meta=gr.JSON(label='Manifest: hash, config, seed, runtime và trạng thái hoàn tất')
                replay_timer=gr.Timer(.2,active=False)

            with gr.Tab('Thống kê & đối chứng'):
                gr.Markdown('### Offline thuần và Safety Shield\nCác số dưới được tính từ episode trong JSON. “Tắt shield” chỉ tắt lớp bên ngoài, Actor c23 của Tien hiện vẫn có quy tắc điều khiển tích hợp.')
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

            with gr.Tab('Chuẩn bị bảo vệ'):
                gr.Markdown('''### Kiểm tra trước buổi trình diễn

1. Trên máy GPU, nạp đúng checkpoint và kiểm hash/config hiển thị. Khởi tạo map đã kiểm tra, chạy ngắn rồi xuất Replay dự phòng.
2. Cho hội đồng chọn map seed mới. Báo route, finish, crash, out và timeout kể cả khi xe thất bại. Giữ RNG seed riêng khi so có/không shield.
3. Tạm dừng, tiến một bước và xem sensor → embedding → h/z → Actor. Thanh bộ nhớ chỉ minh họa các giá trị thực, không chứng minh từng neuron có ý nghĩa vật lý.
4. Tua về bước đã ghi chỉ đổi phần đang xem. Tiếp tục Live chạy từ trạng thái simulator mới nhất, không khôi phục thế giới về quá khứ.
5. Với chế độ chỉ CPU hoặc render không khả dụng, dùng tab Replay. Giao diện hiển thị rõ khi bản ghi thiếu top-down.

### Phạm vi kết quả

c23 dùng state hiện tại trực tiếp ngoài latent h/z. Bốn cảm biến vật cản đọc từ simulator. Không trình bày chúng như vật thể được mạng tự phát hiện từ ảnh. Reward/Continue là dự đoán chưa đạt toàn bộ gate. RL vẫn khóa.

Thí nghiệm Online, checkpoint và tài liệu cũ được giữ để đối chiếu. Giao diện chỉ suy luận/đánh giá, không có nút huấn luyện.
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
            return (*live_view(ctx.open(token,seed,rng,horizon,shield)),gr.Timer(active=False))

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
            return (*replay_view(ctx.load_replay(token,path,archive)),gr.Timer(active=False))

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
        create.click(guarded(on_create),[token,seed,rng,horizon,shield],live_outputs+[live_timer],api_name='create_episode',**opts)
        step.click(guarded(on_step),token,live_outputs,api_name='next_step',**opts)
        run.click(guarded(on_run),token,live_timer,api_name='run',**opts)
        pause.click(guarded(on_pause),token,live_timer,queue=False,api_name='pause')
        live_timer.tick(guarded(on_tick),token,live_outputs+[live_timer],show_progress='hidden',trigger_mode='always_last',api_visibility='private',**opts)
        speed.change(lambda rate:gr.Timer(value=1/float(rate)),speed,live_timer,queue=False,api_visibility='private')
        timeline.release(guarded(lambda t,i:live_view(ctx.seek_live(t,i))),[token,timeline],live_outputs,api_name='seek_live',**opts)
        end.click(guarded(on_end),token,[live_timer,status],api_name='close_episode',**opts)
        export.click(guarded(ctx.export),token,download,api_name='export_replay',**opts)
        infer.click(guarded(on_infer),[token,input_image,sensor_json,previous,first],[infer_result,infer_chain,infer_latent,infer_sensors,infer_status],api_name='infer',**opts)
        replay_load.click(guarded(on_replay_load),[token,replay_path,replay_zip],replay_outputs+[replay_timer],api_name='replay_load')
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
