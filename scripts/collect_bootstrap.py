#!/usr/bin/env python
"""
Phase A entrypoint: collect bootstrap transitions into the replay buffer.

Examples
--------
# Print observation shapes only (1 reset):
python scripts/collect_bootstrap.py --config configs/train.yaml --dry-run

# Collect random-policy data:
python scripts/collect_bootstrap.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# Nested MetaDrive package must precede repo root (folder also named metadrive/).
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

from envs.metadrive_wrapper import make_env
from training.collect import collect_steps, random_action
from utils.config import load_experiment_configs
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap data collection for World Model")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument("--steps", type=int, default=None, help="Override bootstrap.num_steps")
    p.add_argument("--dry-run", action="store_true", help="Reset once and print obs shapes")
    p.add_argument("--out", type=str, default=None, help="Output .npz path for buffer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    train_cfg = configs["train"]
    env_cfg = configs["env"]

    seed = int(train_cfg.get("seed", 0))
    set_seed(seed)

    logger = Logger(
        experiment_name=f"{train_cfg.get('experiment_name', 'wm')}_bootstrap",
        log_dir=train_cfg.get("paths", {}).get("log_dir", "logs"),
        wandb_cfg={"enabled": False},
    )

    env = make_env(env_cfg)
    try:
        obs, info = env.reset(seed=seed)
        image = obs["image"]
        state = obs["state"]
        action_dim = int(env.action_space.shape[0])

        print("=== Observation check ===")
        print(f"image.shape = {image.shape}, dtype = {image.dtype}")
        print(f"state.shape = {state.shape}, dtype = {state.dtype}")
        print(f"action_dim  = {action_dim}")
        print(f"info keys   = {list(info.keys())[:20]}...")

        if args.dry_run:
            logger.finish()
            return

        buf_cfg = train_cfg.get("buffer", {})
        boot_cfg = train_cfg.get("bootstrap", {})
        num_steps = int(args.steps if args.steps is not None else boot_cfg.get("num_steps", 20000))

        buffer = ReplayBuffer(
            capacity=int(buf_cfg.get("capacity", 100_000)),
            image_shape=tuple(image.shape),
            state_dim=int(state.shape[0]),
            action_dim=action_dim,
            sequence_length=int(buf_cfg.get("sequence_length", 64)),
        )

        stats = collect_steps(
            env,
            buffer,
            num_steps=num_steps,
            policy_fn=lambda o: random_action(env.action_space),
        )
        logger.log(stats)

        out = args.out or str(
            Path(train_cfg.get("paths", {}).get("buffer_dir", "data/replay_buffer"))
            / "bootstrap.npz"
        )
        saved = buffer.save(out)
        print(f"Saved buffer ({len(buffer)} transitions) → {saved}")
        print(f"Buffer summary: {buffer.summary()}")
    finally:
        env.close()
        logger.finish()


if __name__ == "__main__":
    main()
