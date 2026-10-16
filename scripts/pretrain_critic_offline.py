#!/usr/bin/env python
"""Offline pretraining / warm-up for Critic on real environment returns.

Fixes the Critic blindness problem where:
- Baseline BC Critic has zero final weights (output 0 everywhere).
- Online RL Critic after 750 updates only had a 0.85 delta between clean and crash states.
This script fits the Critic to discounted Monte Carlo returns computed from
real offline trajectories (including the -20.0 penalty on crashes and out-of-road).
Supports matching Critic control_state_dim=19 when Actor uses continuous control.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor_critic import Critic


def compute_discounted_returns(rewards: np.ndarray, gamma: float = 0.997) -> np.ndarray:
    """Compute true discounted return G_t = sum_{k=0} gamma^k * r_{t+k}."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        returns[t] = acc
    return returns


def symlog(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Symmetric logarithm: sign(x) * log1p(|x|)."""
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    return torch.sign(x) * torch.log1p(torch.abs(x))


def load_dataset(data_dir: str, splits: list[str], gamma: float = 0.997, target_c_dim: int = 0):
    """Load offline transitions and compute discounted symlog return targets."""
    all_hard = []
    all_c = []
    all_targets = []
    all_clean_mask = []
    all_crash_mask = []
    all_out_mask = []

    files = []
    for split in splits:
        files.extend(sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npz"))))

    print(f"Loading {len(files)} files from {data_dir} for splits: {splits}")
    for f in files:
        d = np.load(f)
        r = d["reward"]
        h = d["hard"]  # shape (T+1, 1536)
        c = d["c"]     # shape (T+1, 19) or (T+1, 23)
        n = len(r)
        h = h[:n]  # match T steps
        c = c[:n]
        if target_c_dim == 23 and c.shape[-1] == 19:
            pad = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(c), 1))
            c = np.concatenate([c, pad], axis=-1)
        g = compute_discounted_returns(r, gamma=gamma)
        sym_g = symlog(g)

        is_clean = "_clean" in f
        is_crash = d["crash"][:n].astype(bool)
        is_out = d["out"][:n].astype(bool)

        all_hard.append(h)
        all_c.append(c)
        all_targets.append(sym_g)
        all_clean_mask.append(np.full(n, is_clean, dtype=bool))
        all_crash_mask.append(is_crash)
        all_out_mask.append(is_out)

    hard_tensor = torch.from_numpy(np.concatenate(all_hard)).float()
    c_tensor = torch.from_numpy(np.concatenate(all_c)).float()
    target_tensor = torch.from_numpy(np.concatenate(all_targets)).float().unsqueeze(-1)
    clean_mask = np.concatenate(all_clean_mask)
    crash_mask = np.concatenate(all_crash_mask)
    out_mask = np.concatenate(all_out_mask)

    print(f"Dataset total transitions: {len(hard_tensor)}")
    print(f"  Clean steps: {clean_mask.sum()}")
    print(f"  Crash terminal steps: {crash_mask.sum()}")
    print(f"  Out terminal steps: {out_mask.sum()}")
    print(f"  Target symlog min={target_tensor.min():.3f}, max={target_tensor.max():.3f}, mean={target_tensor.mean():.3f}")

    return hard_tensor, c_tensor, target_tensor, clean_mask, crash_mask, out_mask


def evaluate_critic(critic: Critic, test_hard: torch.Tensor, test_c: torch.Tensor, test_clean: np.ndarray, test_crash: np.ndarray, test_out: np.ndarray, device: torch.device):
    """Evaluate Critic value distribution across clean, crash, and out states."""
    critic.eval()
    with torch.no_grad():
        x = test_hard.to(device)
        kwargs = {}
        if critic.control_state_dim > 0:
            kwargs["control_state"] = test_c.to(device)
        v = critic(x[:, :512], x[:, 512:], **kwargs).cpu().numpy()

    clean_v = v[test_clean]
    crash_v = v[test_crash]
    out_v = v[test_out]

    metrics = {
        "clean_mean": float(clean_v.mean()) if len(clean_v) > 0 else 0.0,
        "clean_std": float(clean_v.std()) if len(clean_v) > 0 else 0.0,
        "crash_mean": float(crash_v.mean()) if len(crash_v) > 0 else 0.0,
        "crash_min": float(crash_v.min()) if len(crash_v) > 0 else 0.0,
        "crash_max": float(crash_v.max()) if len(crash_v) > 0 else 0.0,
        "out_mean": float(out_v.mean()) if len(out_v) > 0 else 0.0,
        "delta_clean_crash": float(clean_v.mean() - crash_v.mean()) if len(clean_v) > 0 and len(crash_v) > 0 else 0.0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", required=True, help="Baseline BC checkpoint")
    parser.add_argument("--data-dir", default="d:/DoAnTotNghiep/worldmodel-metadrive/data/offbc_episodes")
    parser.add_argument("--out-checkpoint", required=True, help="Where to save checkpoint with warm-up Critic")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--term-ratio", type=float, default=0.25, help="Ratio of terminal/negative return samples per batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-state", action="store_true", default=True, help="Condition Critic on model control state (dim=19)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load baseline checkpoint
    print(f"Loading baseline checkpoint: {args.baseline_checkpoint}")
    ckpt = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    ac_cfg = ckpt["actor_critic_cfg"].copy()

    # Configure Critic
    critic_cfg = ac_cfg.copy()
    if args.control_state and ac_cfg.get("control_state_dim", 0) > 0:
        critic_cfg["critic_control_state_dim"] = ac_cfg["control_state_dim"]
        critic_cfg["control_state_mean"] = ac_cfg["control_state_mean"]
        critic_cfg["control_state_std"] = ac_cfg["control_state_std"]
        print(f"Configuring Critic with matching control_state_dim={critic_cfg['critic_control_state_dim']}")
    else:
        critic_cfg["critic_control_state_dim"] = 0

    critic = Critic(critic_cfg, deter_dim=512, stoch_size=1024).to(device)

    # Load Train and Test datasets
    train_hard, train_c, train_targets, _, _, _ = load_dataset(args.data_dir, splits=["train"], gamma=args.gamma, target_c_dim=critic_cfg["critic_control_state_dim"])
    test_hard, test_c, _, test_clean, test_crash, test_out = load_dataset(args.data_dir, splits=["test"], gamma=args.gamma, target_c_dim=critic_cfg["critic_control_state_dim"])

    # Pre-train evaluation
    print("\n--- Pre-training Critic Evaluation on Test Set ---")
    pre_metrics = evaluate_critic(critic, test_hard, test_c, test_clean, test_crash, test_out, device)
    print(f"Clean Mean: {pre_metrics['clean_mean']:.4f} (std={pre_metrics['clean_std']:.4f})")
    print(f"Crash Mean: {pre_metrics['crash_mean']:.4f} (range [{pre_metrics['crash_min']:.4f}, {pre_metrics['crash_max']:.4f}])")
    print(f"Out Mean:   {pre_metrics['out_mean']:.4f}")
    print(f"Delta (Clean - Crash): {pre_metrics['delta_clean_crash']:.4f}")

    # Training loop
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    n_samples = len(train_hard)
    batch_size = args.batch_size

    print(f"\nStarting {args.updates} updates with batch size {batch_size}...")
    start_time = time.time()
    critic.train()

    train_hard_dev = train_hard.to(device)
    train_c_dev = train_c.to(device)
    train_targets_dev = train_targets.to(device)

    # Terminal oversampling matching train_pilot.yaml buffer.term_ratio: 0.25
    term_idx = torch.where((train_targets_dev.squeeze(-1) < 0))[0]
    non_term_idx = torch.where((train_targets_dev.squeeze(-1) >= 0))[0]
    n_term_batch = int(batch_size * args.term_ratio) if args.term_ratio > 0 and len(term_idx) > 0 else 0
    n_non_term_batch = batch_size - n_term_batch

    for step in range(1, args.updates + 1):
        if n_term_batch > 0:
            idx_term = term_idx[torch.randint(0, len(term_idx), (n_term_batch,), device=device)]
            idx_non = non_term_idx[torch.randint(0, len(non_term_idx), (n_non_term_batch,), device=device)]
            idx = torch.cat([idx_term, idx_non])
        else:
            idx = torch.randint(0, n_samples, (batch_size,), device=device)
        
        batch_h = train_hard_dev[idx, :512]
        batch_z = train_hard_dev[idx, 512:]
        batch_y = train_targets_dev[idx].squeeze(-1)

        kwargs = {}
        if critic.control_state_dim > 0:
            kwargs["control_state"] = train_c_dev[idx]

        pred = critic(batch_h, batch_z, **kwargs)
        loss = F.mse_loss(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optimizer.step()

        if step % 200 == 0 or step == args.updates:
            print(f"Step {step:4d}/{args.updates} | Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")

    # Post-training evaluation
    print("\n--- Post-training Critic Evaluation on Test Set ---")
    post_metrics = evaluate_critic(critic, test_hard, test_c, test_clean, test_crash, test_out, device)
    print(f"Clean Mean: {post_metrics['clean_mean']:.4f} (std={post_metrics['clean_std']:.4f})")
    print(f"Crash Mean: {post_metrics['crash_mean']:.4f} (range [{post_metrics['crash_min']:.4f}, {post_metrics['crash_max']:.4f}])")
    print(f"Out Mean:   {post_metrics['out_mean']:.4f}")
    print(f"Delta (Clean - Crash): {post_metrics['delta_clean_crash']:.4f}")

    # Save new checkpoint
    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_ckpt = ckpt.copy()
    new_ac = ckpt["actor_critic"].copy()
    for k, v in critic.state_dict().items():
        new_ac[f"critic.{k}"] = v.cpu()
        new_ac[f"slow_critic.{k}"] = v.cpu()

    new_ckpt["actor_critic"] = new_ac
    new_ckpt["actor_critic_cfg"] = critic_cfg
    new_ckpt["critic_warmup_metrics"] = {
        "pre": pre_metrics,
        "post": post_metrics,
        "updates": args.updates,
        "train_seconds": elapsed,
        "critic_control_state_dim": critic.control_state_dim,
    }

    torch.save(new_ckpt, out_path)
    print(f"\nSaved updated checkpoint with matching control-state Critic to: {out_path}")


if __name__ == "__main__":
    main()
