#!/usr/bin/env python
"""
Preview sample transitions from a live short collect or from a saved buffer.

Examples
--------
# Collect a few steps and visualize (no need for full bootstrap first):
python scripts/preview_data.py --config configs/train.yaml --steps 100 --num-samples 8

# Preview from an existing buffer:
python scripts/preview_data.py --buffer data/replay_buffer/bootstrap.npz --num-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

import matplotlib.pyplot as plt
import numpy as np

from envs.metadrive_wrapper import make_env
from training.collect import collect_steps, random_action
from utils.config import load_experiment_configs
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview collected MetaDrive samples")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--buffer", type=str, default=None, help="Path to .npz buffer")
    p.add_argument("--steps", type=int, default=120, help="Steps to collect if no --buffer")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/raw_rollouts/preview",
        help="Where to save preview images / summary",
    )
    return p.parse_args()


def _build_buffer_from_env(configs: dict, num_steps: int) -> ReplayBuffer:
    train_cfg = configs["train"]
    env_cfg = configs["env"]
    set_seed(int(train_cfg.get("seed", 0)))

    env = make_env(env_cfg)
    try:
        obs, _ = env.reset(seed=int(train_cfg.get("seed", 0)))
        buf_cfg = train_cfg.get("buffer", {})
        buffer = ReplayBuffer(
            capacity=max(num_steps + 10, 256),
            image_shape=tuple(obs["image"].shape),
            state_dim=int(obs["state"].shape[0]),
            action_dim=int(env.action_space.shape[0]),
            sequence_length=min(16, max(4, num_steps // 4)),
        )
        stats = collect_steps(
            env,
            buffer,
            num_steps=num_steps,
            policy_fn=lambda o: random_action(env.action_space),
        )
        print("Collect stats:", stats)
        return buffer
    finally:
        env.close()


def _pick_indices(n: int, k: int) -> np.ndarray:
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=np.int64)
    return np.linspace(0, n - 1, num=k, dtype=np.int64)


def save_preview(buffer: ReplayBuffer, out_dir: Path, num_samples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(buffer)
    if n == 0:
        raise RuntimeError("Buffer is empty — nothing to preview.")

    idxs = _pick_indices(n, num_samples)
    rows = int(np.ceil(len(idxs) / 4))
    cols = min(4, len(idxs))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    samples = []
    for ax_i, t in enumerate(idxs):
        img = buffer.images[t]
        state = buffer.states[t]
        action = buffer.actions[t]
        reward = float(buffer.rewards[t])
        done = bool(buffer.dones[t])

        # Save individual frame
        frame_path = out_dir / f"sample_{ax_i:02d}_t{t}.png"
        plt.imsave(frame_path, img)

        ax = axes[ax_i]
        ax.imshow(img)
        ax.set_title(f"t={t} r={reward:.2f} done={int(done)}", fontsize=9)
        ax.axis("off")

        sample = {
            "index": int(t),
            "image_file": frame_path.name,
            "image_shape": list(img.shape),
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": reward,
            "done": done,
        }
        samples.append(sample)

        print(f"\n--- sample {ax_i} (t={t}) ---")
        print(f"image: {img.shape} uint8, min={img.min()}, max={img.max()}")
        print(f"action: {np.round(action, 3)}")
        print(f"reward: {reward:.4f} | done: {done}")
        print(f"state[{len(state)}]: {np.round(state, 3)}")

    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")

    grid_path = out_dir / "preview_grid.png"
    fig.suptitle(f"MetaDrive bootstrap preview ({len(idxs)} / {n} transitions)", fontsize=12)
    fig.tight_layout()
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    summary = {
        "buffer_size": n,
        "num_shown": len(idxs),
        "image_shape": list(buffer.image_shape),
        "state_dim": buffer.state_dim,
        "action_dim": buffer.action_dim,
        "reward_mean": float(np.mean(buffer.rewards[:n])),
        "reward_std": float(np.std(buffer.rewards[:n])),
        "done_rate": float(np.mean(buffer.dones[:n])),
        "samples": samples,
    }
    summary_path = out_dir / "preview_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Saved ===")
    print(f"grid   : {grid_path}")
    print(f"frames : {out_dir / 'sample_*.png'}")
    print(f"summary: {summary_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.buffer:
        buffer = ReplayBuffer.load(args.buffer)
        print(f"Loaded buffer: {args.buffer} | {buffer.summary()}")
    else:
        configs = load_experiment_configs(args.config, args.env_config, None)
        print(f"Collecting {args.steps} steps for preview...")
        buffer = _build_buffer_from_env(configs, args.steps)
        # Also dump a tiny buffer next to preview for reuse
        tiny_path = out_dir / "preview_buffer.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        buffer.save(tiny_path)
        print(f"Saved tiny buffer -> {tiny_path}")

    save_preview(buffer, out_dir, args.num_samples)


if __name__ == "__main__":
    main()
