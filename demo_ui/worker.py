"""One spawned process owns Panda3D, CUDA, and the recurrent policy."""
from __future__ import annotations
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import traceback

import numpy as np

from demo_ui.recording import EpisodeWriter, json_safe


def worker_main(connection):
    engine = env = writer = None
    observation = info = prepared = shield = None
    episode_return = 0.
    try:
        while True:
            command, args = connection.recv()
            try:
                if command == 'close':
                    connection.send(('ok', None))
                    break
                if command == 'load':
                    from demo_ui.inference import InferenceEngine
                    if env is not None:
                        env.close(); env = None
                    if writer is not None:
                        writer.close(); writer = None
                    engine = InferenceEngine(**args)
                    connection.send(('ok', engine.metadata))
                elif command == 'infer':
                    if engine is None:
                        raise ValueError('Chọn và nạp checkpoint trước.')
                    if env is not None:
                        raise ValueError('Đóng mô phỏng trước khi đổi lịch sử RSSM để thử ảnh.')
                    result = engine.observe(**args)
                    result['prior_next'] = engine.prior(result['action'])
                    connection.send(('ok', result))
                elif command == 'open':
                    if engine is None:
                        raise ValueError('Nạp checkpoint trước khi chạy MetaDrive.')
                    if env is not None:
                        env.close(); env = None
                    if writer is not None:
                        writer.close()
                    from envs.metadrive_wrapper import make_env
                    from evaluation.safety_shield import AdaptiveTrafficAvoidanceShield
                    from utils.seed import set_seed
                    rng_seed = int(args.get('rng_seed', 0))
                    set_seed(rng_seed)
                    cfg = dict(engine.env_cfg, start_seed=int(args['seed']), num_scenarios=1,
                               horizon=int(args['horizon']), use_render=False)
                    if not 1 <= cfg['horizon'] <= 5000:
                        raise ValueError('Horizon cần trong 1..5000.')
                    env = make_env(cfg)
                    observation, info = env.reset(seed=int(args['seed']))
                    engine.reset()
                    prepared = engine.observe(observation['image'], observation['state'])
                    shield = AdaptiveTrafficAvoidanceShield() if args.get('shield') else None
                    episode_return = 0.
                    try:
                        commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
                    except (OSError, subprocess.SubprocessError):
                        commit = os.getenv('DATN_SOURCE_COMMIT', 'unknown')
                    writer = EpisodeWriter(args['output_root'], dict(engine.metadata, env_cfg=cfg,
                                           map_seed=int(args['seed']), rng_seed=rng_seed,
                                           safety_shield=shield is not None, source_commit=commit,
                                           python=platform.python_version(), runtime_platform=platform.platform()))
                    writer.metadata['source_archive_sha256'] = os.getenv('DATN_SOURCE_ARCHIVE_SHA256')
                    import importlib.metadata
                    writer.metadata['metadrive_version'] = importlib.metadata.version('metadrive-simulator')
                    writer.metadata['numpy_version'] = np.__version__
                    top, warning = render_topdown(env)
                    connection.send(('ok', dict(rgb=observation['image'], topdown=top,
                                     prepared=prepared, metadata=writer.metadata,
                                     folder=str(writer.folder), topdown_warning=warning)))
                elif command == 'step':
                    if env is None or writer is None or writer.finished:
                        raise ValueError('Episode chưa khởi tạo hoặc đã kết thúc.')
                    old_obs = observation
                    top, warning = render_topdown(env)
                    actor_action = np.asarray(prepared['action'], dtype=np.float32)
                    action = actor_action.copy()
                    intervention = {}
                    if shield is not None:
                        raw = env._env
                        vehicle = getattr(raw,'agent',None) or getattr(raw,'current_track_vehicle',None) or getattr(raw,'vehicle',None)
                        action, intervention = shield.modulate_action(action, vehicle)
                    prior = engine.prior(action)
                    engine.policy.record_executed_action(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    following = engine.observe(observation['image'], observation['state'], previous_action=action)
                    episode_return += float(reward)
                    flags = {name: bool(info.get(name, False)) for name in
                             ['arrive_dest','out_of_road','crash','crash_vehicle','crash_object','crash_sidewalk']}
                    row = dict(state=old_obs['state'], action_actor=actor_action, action_executed=action,
                               prior=prior, posterior_next=following['posterior'], features=prepared['features'],
                               reward=float(reward), reward_base=info.get('reward_base'),
                               reward_shaping=info.get('reward_shaping'), episode_return=episode_return,
                               continue_target=1-int(terminated), terminated=bool(terminated), truncated=bool(truncated),
                               route_completion=float(info.get('route_completion',0) or 0),
                               flags=flags, shield=intervention, topdown_warning=warning,
                               simulation_time_s=writer.count*float(env._env.config['physics_world_step_size'])*int(env._env.config['decision_repeat']))
                    row = writer.append(row, old_obs['image'], top)
                    prepared = following
                    if terminated or truncated:
                        writer.close(complete=True, reason='terminated' if terminated else 'timeout')
                    connection.send(('ok', dict(record=row, rgb=old_obs['image'], topdown=top,
                                     folder=str(writer.folder), length=writer.count,
                                     done=bool(terminated or truncated))))
                elif command == 'end':
                    if writer is not None:
                        writer.close(reason='user_stop')
                    if env is not None:
                        env.close(); env = None
                    engine.reset() if engine else None
                    connection.send(('ok', None))
                else:
                    raise ValueError('Unknown worker command')
            except Exception as exc:
                traceback.print_exc()
                connection.send(('error', f'{type(exc).__name__}: {exc}'))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if writer is not None:
            writer.close(reason='worker_closed')
        if env is not None:
            env.close()
        connection.close()


def render_topdown(env):
    try:
        frame = env.render(mode='top_down', film_size=(640, 420), screen_size=(640, 420))
        if frame is None:
            return None, 'MetaDrive chưa trả về ảnh top-down.'
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError('Top-down không phải ảnh RGB.')
        return np.asarray(frame[:,:,:3], dtype=np.uint8), None
    except Exception as exc:
        return None, 'Top-down không khả dụng: ' + str(exc)


class ModelWorker:
    def __init__(self, timeout=120):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child,), daemon=True)
        self.process.start()
        child.close()
        self.timeout = timeout
        self.lock = threading.RLock()

    def call(self, command, **args):
        with self.lock:
            if not self.process.is_alive():
                raise RuntimeError('Tiến trình model đã đóng. Nạp lại checkpoint.')
            self.connection.send((command, args))
            if not self.connection.poll(self.timeout):
                self.close(force=True)
                raise TimeoutError('Model/render quá thời gian. Kiểm tra console và render backend rồi nạp lại.')
            status, value = self.connection.recv()
            if status != 'ok':
                raise ValueError(value)
            return value

    def close(self, force=False):
        with self.lock:
            if self.process.is_alive() and not force:
                try:
                    self.connection.send(('close', {}))
                    self.process.join(3)
                except (OSError, EOFError):
                    pass
            if self.process.is_alive():
                self.process.terminate(); self.process.join(3)
            self.connection.close()
