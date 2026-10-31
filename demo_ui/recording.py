"""Incremental, portable episode recordings with lossless model input frames."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np
from PIL import Image

SCHEMA = 'datn-demo-v1'
MAX_STEPS = 10000
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Đường dẫn frame nằm ngoài episode.')
    return path


class EpisodeWriter:
    def __init__(self, output_root, metadata):
        self.folder = Path(output_root).resolve() / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
        self.folder.mkdir(parents=True)
        (self.folder / 'frames').mkdir()
        self.count = 0
        self.finished = False
        self.metadata = dict(metadata, schema=SCHEMA, steps=0, complete=False,
                             timing='obs_t -> action_t -> reward_t, obs_t+1. Prior heads predict the successor before seeing obs_t+1.',
                             continue_target='1 - terminated; timeout/truncated alone retains continue=1')
        self.stream = (self.folder / 'steps.jsonl').open('x', encoding='utf-8', buffering=1)
        write_json(self.folder / 'manifest.json', self.metadata)

    def append(self, record, rgb, topdown=None):
        if self.finished or self.count >= MAX_STEPS:
            raise ValueError('Episode đã đóng hoặc vượt giới hạn ghi.')
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Frame RGB phải là uint8 HWC 3 kênh.')
        name = f'frames/{self.count:06d}_rgb.png'
        Image.fromarray(rgb).save(self.folder / name)
        row = dict(json_safe(record), step=self.count, rgb=name,
                   rgb_sha256=sha256(self.folder / name))
        if topdown is not None:
            td = np.asarray(topdown, dtype=np.uint8)
            name_td = f'frames/{self.count:06d}_top.png'
            Image.fromarray(td).save(self.folder / name_td)
            row['topdown'] = name_td
            row['topdown_sha256'] = sha256(self.folder / name_td)
        else:
            row['topdown'] = None
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        self.count += 1
        # Recoverable after every step, including interrupted Colab sessions.
        self.metadata['steps'] = self.count
        write_json(self.folder / 'manifest.json', self.metadata)
        return row

    def close(self, *, complete=False, reason='user_stop'):
        if self.finished:
            return
        self.stream.close()
        self.metadata.update(steps=self.count, complete=bool(complete), stop_reason=reason,
                             steps_sha256=sha256(self.folder / 'steps.jsonl'))
        write_json(self.folder / 'manifest.json', self.metadata)
        self.finished = True


class EpisodeReader:
    """Seek by line offsets and load only the selected pair of images."""
    def __init__(self, path):
        path = Path(path).resolve()
        self.folder = path if path.is_dir() else path.parent
        self.metadata = json.loads((self.folder / 'manifest.json').read_text(encoding='utf-8'))
        if self.metadata.get('schema') != SCHEMA:
            raise ValueError('Replay chưa đúng schema datn-demo-v1. Cần recording từ giao diện hoặc CLI demo.')
        steps = self.folder / 'steps.jsonl'
        if steps.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('Metadata episode quá lớn.')
        self.offsets = []
        with steps.open('rb') as file:
            while True:
                offset = file.tell()
                line = file.readline(512 * 1024)
                if not line:
                    break
                if not line.endswith(b'\n'):
                    break  # Ignore a partial last line after an interruption.
                if len(self.offsets) >= MAX_STEPS:
                    raise ValueError('Episode vượt 10.000 bước.')
                self.offsets.append(offset)
        expected = self.metadata.get('steps_sha256')
        if expected and sha256(steps) != expected:
            raise ValueError('Checksum steps.jsonl không khớp manifest.')
        if expected and len(self.offsets) != self.metadata.get('steps'):
            raise ValueError('Số bước không khớp manifest.')

    def __len__(self):
        return len(self.offsets)

    def row(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError('Bước Replay ngoài phạm vi.')
        with (self.folder / 'steps.jsonl').open('rb') as file:
            file.seek(self.offsets[index])
            return json.loads(file.readline())

    def frame(self, index):
        row = self.row(index)
        def load(key):
            if not row.get(key):
                return None
            path = contained(self.folder, row[key])
            if row.get(key + '_sha256') and sha256(path) != row[key + '_sha256']:
                raise ValueError('Checksum ảnh không khớp bản ghi: ' + key)
            with Image.open(path) as image:
                if image.width * image.height > 4096 * 4096:
                    raise ValueError('Frame quá lớn.')
                return np.asarray(image.convert('RGB'))
        return row, load('rgb'), load('topdown')


def export_episode(folder):
    reader = EpisodeReader(folder)
    if not len(reader):
        raise ValueError('Chưa có bước để xuất.')
    output = reader.folder.with_suffix('.zip')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(reader.folder.rglob('*')):
            if file.is_file() and file.suffix in {'.json', '.jsonl', '.png'}:
                archive.write(file, file.relative_to(reader.folder).as_posix())
    return str(output)


def import_episode(archive, output_root):
    folder = Path(output_root).resolve() / ('import_' + uuid.uuid4().hex)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 2 * MAX_STEPS + 10 or sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES:
            raise ValueError('Replay ZIP vượt giới hạn 512 MB hoặc quá nhiều file.')
        for member in members:
            target = contained(folder, member.filename)
            if target.suffix not in {'.json', '.jsonl', '.png'} and not member.is_dir():
                raise ValueError('ZIP Replay chỉ nhận JSON/JSONL/PNG.')
        folder.mkdir(parents=True)
        for member in members:
            if member.is_dir():
                continue
            target = contained(folder, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as file, bundle.open(member) as source:
                while data := source.read(1024 * 1024):
                    file.write(data)
    return EpisodeReader(folder)
