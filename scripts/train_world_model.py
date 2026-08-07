#!/usr/bin/env python
"""
A5–A6: Train World Model on bootstrap replay buffer + save checkpoints.

Example
-------
python scripts/train_world_model.py --config configs/train.yaml --updates 500
python scripts/train_world_model.py --config configs/train.yaml --updates 2000 --device cpu
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

import numpy as np
import torch

from models.decoder import Decoder
from models.encoder import Encoder
from models.predictors import ContinuePredictor, RewardPredictor
from models.rssm import RSSM
from training.train_world_model import train_world_model_step
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_experiment_configs
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train World Model (A5-A6)")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument(
        "--buffer",
        type=str,
        default=None,
        help="Path to bootstrap.npz (default: data/replay_buffer/bootstrap.npz)",
    )
    p.add_argument("--updates", type=int, default=None, help="Gradient updates")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None, help="Override sequence_length")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint .pt to resume",
    )
    return p.parse_args()


def _resolve_device(name: str | None) -> torch.device:
    if name is None or name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA not available -> using CPU")
        return torch.device("cpu")
    return torch.device(name)


def batch_to_torch(
    batch: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert numpy batch (image HWC uint8) -> torch (B,T,C,H,W) float."""
    img = batch["image"]  # (B, T, H, W, C)
    if img.ndim != 5:
        raise ValueError(f"Expected image (B,T,H,W,C), got {img.shape}")
    image = torch.from_numpy(img).to(device=device, dtype=torch.float32) / 255.0
    image = image.permute(0, 1, 4, 2, 3).contiguous()  # B,T,C,H,W

    return {
        "image": image,
        "state": torch.from_numpy(batch["state"]).to(device=device, dtype=torch.float32),
        "action": torch.from_numpy(batch["action"]).to(device=device, dtype=torch.float32),
        "reward": torch.from_numpy(batch["reward"]).to(device=device, dtype=torch.float32),
        "done": torch.from_numpy(batch["done"].astype(np.bool_)).to(device=device),
    }


def build_models(wm_cfg: dict, image_shape: tuple, state_dim: int, action_dim: int, device: torch.device):
    enc = Encoder(wm_cfg, state_dim=state_dim, image_shape=image_shape).to(device)
    rssm = RSSM(wm_cfg, embed_dim=enc.embed_dim, action_dim=action_dim).to(device)
    dec = Decoder(
        wm_cfg,
        deter_dim=rssm.deter_dim,
        stoch_size=rssm.stoch_size,
        image_shape=image_shape,
        state_dim=state_dim,
    ).to(device)
    reward = RewardPredictor(wm_cfg, rssm.deter_dim, rssm.stoch_size).to(device)
    cont = ContinuePredictor(wm_cfg, rssm.deter_dim, rssm.stoch_size).to(device)
    models = {
        "encoder": enc,
        "rssm": rssm,
        "decoder": dec,
        "reward": reward,
        "continue": cont,
    }
    return models


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    train_cfg = configs["train"]
    wm_cfg = configs["world_model"]
    env_cfg = configs["env"]

    set_seed(int(train_cfg.get("seed", 0)))
    device = _resolve_device(args.device or train_cfg.get("device", "cpu"))
    if device.type == "cuda":
        # benchmark=True makes the *first* step very slow (algo search)
        torch.backends.cudnn.benchmark = False

    buffer_path = Path(
        args.buffer
        or str(Path(train_cfg.get("paths", {}).get("buffer_dir", "data/replay_buffer")) / "bootstrap.npz")
    )
    if not buffer_path.exists():
        raise FileNotFoundError(
            f"Buffer not found: {buffer_path}\n"
            "Run A1 first: python scripts/collect_bootstrap.py --config configs/train.yaml --steps 20000"
        )

    print(
        f"Loading buffer: {buffer_path} (~{buffer_path.stat().st_size / 1e9:.1f} GB compressed)...",
        flush=True,
    )
    buffer = ReplayBuffer.load(buffer_path)
    # Prefer CLI, then train.yaml (npz meta inside npz may still be old seq=64)
    cfg_seq = int(train_cfg.get("buffer", {}).get("sequence_length", buffer.sequence_length))
    buffer.sequence_length = int(args.seq_len) if args.seq_len is not None else cfg_seq
    buffer._valid_starts_cache = None
    print("Buffer:", buffer.summary(), flush=True)

    image_shape = tuple(buffer.image_shape)  # H,W,C
    state_dim = buffer.state_dim
    action_dim = buffer.action_dim

    print("Building models on", device, "...", flush=True)
    models = build_models(wm_cfg, image_shape, state_dim, action_dim, device)
    params = [p for m in models.values() for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=float(wm_cfg.get("training", {}).get("lr", 1e-4)))

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        for k, m in models.items():
            m.load_state_dict(ckpt["models"][k])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from {args.resume} @ step={start_step}", flush=True)

    wm_train = train_cfg.get("wm_train", {})
    updates = int(args.updates if args.updates is not None else wm_train.get("updates", 1000))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else wm_train.get("batch_size", 8)
    )
    log_every = int(args.log_every if args.log_every is not None else wm_train.get("log_every", 20))
    ckpt_every = int(args.ckpt_every if args.ckpt_every is not None else wm_train.get("checkpoint_every", 200))

    ckpt_dir = Path(train_cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")) / "world_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        experiment_name=f"{train_cfg.get('experiment_name', 'wm')}_wm",
        log_dir=train_cfg.get("paths", {}).get("log_dir", "logs"),
        wandb_cfg=train_cfg.get("wandb", {}),
    )

    print(
        f"Train WM: updates={updates}, batch_size={batch_size}, "
        f"seq_len={buffer.sequence_length}, device={device}",
        flush=True,
    )
    print("Caching valid sequence starts...", flush=True)
    n_starts = len(buffer._valid_start_indices())
    print(f"Valid starts: {n_starts}", flush=True)
    print("Running first update...", flush=True)

    for step in range(start_step + 1, start_step + updates + 1):
        try:
            raw = buffer.sample(batch_size, include_next=False)
        except MemoryError:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raw = buffer.sample(batch_size, include_next=False)
        batch = batch_to_torch(raw, device)
        del raw
        metrics = train_world_model_step(batch, models, optimizer, wm_cfg)
        del batch

        if step % log_every == 0 or step == start_step + 1:
            logger.log(metrics, step=step)

        if step % ckpt_every == 0 or step == start_step + updates:
            path = ckpt_dir / f"wm_step_{step:06d}.pt"
            save_checkpoint(
                path,
                {
                    "step": step,
                    "models": {k: m.state_dict() for k, m in models.items()},
                    "optimizer": optimizer.state_dict(),
                    "wm_cfg": wm_cfg,
                    "env_cfg": env_cfg,
                    "image_shape": image_shape,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                },
            )
            # also refresh "latest"
            latest = ckpt_dir / "latest.pt"
            save_checkpoint(
                latest,
                {
                    "step": step,
                    "models": {k: m.state_dict() for k, m in models.items()},
                    "optimizer": optimizer.state_dict(),
                    "wm_cfg": wm_cfg,
                    "env_cfg": env_cfg,
                    "image_shape": image_shape,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                },
            )
            print(f"Saved checkpoint -> {path}")

    logger.finish()
    print("Done. Latest checkpoint:", ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()
