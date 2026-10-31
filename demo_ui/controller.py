"""Per-browser state and transport controls, independent of Gradio."""
from __future__ import annotations
import atexit
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

from demo_ui.recording import EpisodeReader, export_episode, import_episode


@dataclass
class Session:
    worker: object = None
    metadata: dict = field(default_factory=dict)
    folder: str | None = None
    frame: dict | None = None
    replay: EpisodeReader | None = None
    cursor: int = 0
    running: bool = False
    replaying: bool = False
    done: bool = False
    touched: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.RLock)

    def close(self):
        with self.lock:
            self.running = self.replaying = False
            if self.worker:
                self.worker.close()
                self.worker = None


class Controller:
    def __init__(self, checkpoint_root='checkpoints', output_root='demo_runs', *, replay_only=False):
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.replay_only = replay_only
        self.sessions = {}
        self.lock = threading.Lock()
        atexit.register(self.close_all)

    def get(self, token):
        if not token:
            raise ValueError('Phiên trình duyệt chưa sẵn sàng. Tải lại trang.')
        with self.lock:
            session = self.sessions.setdefault(token, Session())
            session.touched = time.monotonic()
            return session

    def close(self, token):
        with self.lock:
            session = self.sessions.pop(token, None)
        if session:
            session.close()

    def close_all(self):
        for token in list(self.sessions):
            self.close(token)

    def checkpoints(self):
        if not self.checkpoint_root.is_dir():
            return []
        return [str(p.relative_to(self.checkpoint_root))
                for p in sorted(self.checkpoint_root.rglob('*.pt'))][:500]

    def load(self, token, checkpoint, wm_checkpoint, device):
        if self.replay_only:
            raise ValueError('Đang ở chế độ Replay CPU. Khởi động lại không có --replay-only để suy luận.')
        if not checkpoint:
            raise ValueError('Chưa chọn checkpoint. Đặt file vào thư mục checkpoint rồi bấm Quét lại.')
        target = (self.checkpoint_root / checkpoint).resolve()
        if not target.is_relative_to(self.checkpoint_root) or not target.is_file():
            raise ValueError('Checkpoint không nằm trong thư mục đã cấu hình.')
        wm_target = None
        if wm_checkpoint:
            wm_target = (self.checkpoint_root / wm_checkpoint).resolve()
            if not wm_target.is_relative_to(self.checkpoint_root) or not wm_target.is_file():
                raise ValueError('Checkpoint WM không nằm trong thư mục đã cấu hình.')
        s = self.get(token)
        with s.lock:
            s.close()
            from demo_ui.worker import ModelWorker
            s.worker = ModelWorker()
            try:
                s.metadata = s.worker.call('load', checkpoint=str(target),
                                          wm_checkpoint=str(wm_target) if wm_target else None, device=device)
            except Exception:
                s.close()
                raise
            s.folder = None; s.frame = None; s.done = False
            return s.metadata

    def open(self, token, seed, rng_seed, horizon, shield):
        s = self.get(token)
        with s.lock:
            if s.worker is None:
                raise ValueError('Nạp checkpoint trước.')
            for value in [seed, rng_seed, horizon]:
                if int(value) != value:
                    raise ValueError('Seed và horizon phải là số nguyên.')
            result = s.worker.call('open', seed=int(seed), rng_seed=int(rng_seed),
                                   horizon=int(horizon), shield=bool(shield), output_root=str(self.output_root))
            s.folder = result['folder']; s.frame = result
            s.done = s.running = False; s.cursor = 0
            return result

    def step(self, token):
        s = self.get(token)
        with s.lock:
            if not s.folder or s.worker is None:
                raise ValueError('Khởi tạo episode trước.')
            if s.done:
                s.running = False
                return s.frame
            s.frame = s.worker.call('step')
            s.done = s.frame['done']; s.running = s.running and not s.done
            s.cursor = s.frame['record']['step']
            return s.frame

    def seek_live(self, token, index):
        s = self.get(token)
        with s.lock:
            if s.running:
                raise ValueError('Tạm dừng trước khi xem lại timeline.')
            if not s.folder:
                raise ValueError('Chưa có episode.')
            reader = EpisodeReader(s.folder)
            row, rgb, top = reader.frame(max(0,min(int(index),len(reader)-1)))
            return dict(record=row, rgb=rgb, topdown=top, folder=s.folder,
                        length=len(reader), done=s.done)

    def load_replay(self, token, path=None, archive=None):
        s = self.get(token)
        with s.lock:
            s.replaying = False
            if archive:
                s.replay = import_episode(archive, self.output_root)
            elif path:
                s.replay = EpisodeReader(path)
            else:
                raise ValueError('Chọn ZIP hoặc nhập thư mục/manifest Replay.')
            if not len(s.replay):
                raise ValueError('Episode chưa có frame đã ghi.')
            s.cursor = 0
            return self.replay_frame(token, 0)

    def replay_frame(self, token, index):
        s = self.get(token)
        with s.lock:
            if s.replay is None:
                raise ValueError('Chọn và nạp Replay trước.')
            s.cursor = max(0, min(int(index), len(s.replay)-1))
            row, rgb, top = s.replay.frame(s.cursor)
            return dict(record=row, rgb=rgb, topdown=top, metadata=s.replay.metadata,
                        length=len(s.replay), folder=str(s.replay.folder))

    def infer(self, token, image, sensors, previous_action, reset):
        s = self.get(token)
        with s.lock:
            if s.worker is None:
                raise ValueError('Nạp checkpoint ở tab Mô phỏng trước.')
            return s.worker.call('infer', image=image, state=sensors,
                                 previous_action=previous_action, reset=bool(reset))

    def export(self, token):
        s = self.get(token)
        with s.lock:
            if s.running:
                raise ValueError('Tạm dừng trước khi xuất Replay.')
            if not s.folder:
                raise ValueError('Chưa có episode để xuất.')
            return export_episode(s.folder)
