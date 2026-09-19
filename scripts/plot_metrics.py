#!/usr/bin/env python
"""
Plot training and evaluation metrics from local metrics.jsonl log files.

Usage:
    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --log-file logs/metrics.jsonl --output docs/training_curves.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training and evaluation curves.")
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/metrics.jsonl",
        help="Path to metrics.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/training_curves.png",
        help="Path to save the generated figure",
    )
    return parser.parse_args()


def plot_online_dashboard(records: list[dict], output_path: Path) -> bool:
    """Plot the 9-panel comprehensive dashboard for Online RL training."""
    online_records = [r for r in records if "rollout/total_env_steps" in r]
    eval_records = [
        r for r in records if "eval/mean_route_completion" in r or "eval/mean_episode_return" in r
    ]

    if not online_records:
        print("[Info] No online training records found yet in log file.")
        return False

    steps = np.array([r["rollout/total_env_steps"] for r in online_records])
    eval_steps = np.array(
        [r.get("step", r.get("rollout/total_env_steps", 0)) for r in eval_records]
    )

    current_step = int(steps[-1]) if len(steps) > 0 else 0
    total_episodes = int(online_records[-1].get("rollout/total_episodes", 0))

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle(
        f"World Model + Actor-Critic Online RL Dashboard (Step {current_step:,} | Episodes: {total_episodes})",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.96, hspace=0.36, wspace=0.26)

    # 1. Critic Value & Lambda Return Mean
    ax = axes[0, 0]
    val_mean = np.array([r.get("ac/value_mean", 0.0) for r in online_records])
    ret_mean = np.array([r.get("ac/lambda_return_mean", 0.0) for r in online_records])
    ax.plot(steps, val_mean, "r-o", markersize=3.0, lw=1.8, label=f"Critic Value (curr: {val_mean[-1]:.2f})")
    ax.plot(steps, ret_mean, "b--", alpha=0.85, lw=1.5, label=f"Lambda Return (curr: {ret_mean[-1]:.2f})")
    ax.axhline(0, color="gray", linestyle=":", lw=1.0)
    ax.set_title("1. Critic Value & Lambda Return", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Value / Return", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 2. Route Completion (%)
    ax = axes[0, 1]
    route_rollout = np.array([r.get("rollout/route_completion", 0.0) * 100.0 for r in online_records])
    ax.plot(steps, route_rollout, color="#00acc1", lw=1.6, alpha=0.8, label=f"Rollout Route % (curr: {route_rollout[-1]:.1f}%)")
    if len(eval_records) > 0:
        route_eval = np.array([r.get("eval/mean_route_completion", 0.0) * 100.0 for r in eval_records])
        ax.plot(eval_steps, route_eval, "mo-", linewidth=2.2, markersize=6, label=f"Eval Route % (curr: {route_eval[-1]:.1f}%)")
    ax.set_title("2. Route Completion (%)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Completion (%)", fontsize=10)
    ax.set_ylim(-2, max(105.0, float(np.max(route_rollout)) * 1.1 if len(route_rollout) else 105.0))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 3. Action Exploration (Std Steering & Throttle)
    ax = axes[0, 2]
    std_steer = np.array([r.get("ac/std_steering", 0.0) for r in online_records])
    std_throt = np.array([r.get("ac/std_throttle", 0.0) for r in online_records])
    ax.plot(steps, std_steer, color="#2e7d32", linewidth=1.8, label=f"Std Steering (curr: {std_steer[-1]:.3f})")
    ax.plot(steps, std_throt, color="#d32f2f", linewidth=1.8, label=f"Std Throttle (curr: {std_throt[-1]:.3f})")
    ax.set_title("3. Action Exploration (Std)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Standard Deviation", fontsize=10)
    max_std = max(float(np.max(std_steer)) if len(std_steer) else 1.0, float(np.max(std_throt)) if len(std_throt) else 1.0)
    ax.set_ylim(0.0, max(1.1, max_std * 1.15))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 4. Policy Entropy
    ax = axes[1, 0]
    entropy = np.array([r.get("ac/entropy", 0.0) for r in online_records])
    ax.plot(steps, entropy, color="#7b1fa2", linewidth=2.0, label=f"Entropy (curr: {entropy[-1]:.3f})")
    ax.axhline(0, color="gray", linestyle=":", lw=1.0)
    ax.set_title("4. Policy Entropy", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Entropy", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    # 5. Episode Length
    ax = axes[1, 1]
    len_rollout = np.array([r.get("rollout/mean_length", 0.0) for r in online_records])
    ax.plot(steps, len_rollout, color="#1976d2", lw=1.8, alpha=0.85, label=f"Rollout Length (curr: {len_rollout[-1]:.0f})")
    if len(eval_records) > 0:
        len_eval = np.array([r.get("eval/mean_episode_length", 0.0) for r in eval_records])
        ax.plot(eval_steps, len_eval, "ro-", linewidth=2.0, markersize=6, label=f"Eval Length (curr: {len_eval[-1]:.0f})")
    ax.set_title("5. Episode Survival Length (Steps)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Steps survived", fontsize=10)
    max_len = max(float(np.max(len_rollout)) if len(len_rollout) else 50.0, 50.0)
    ax.set_ylim(0, max(1000.0, max_len * 1.1))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 6. Mean Returns
    ax = axes[1, 2]
    ret_rollout = np.array([r.get("rollout/mean_return", 0.0) for r in online_records])
    ax.plot(steps, ret_rollout, color="#388e3c", lw=1.6, alpha=0.8, label=f"Rollout Return (curr: {ret_rollout[-1]:.1f})")
    if len(eval_records) > 0:
        ret_eval = np.array([r.get("eval/mean_episode_return", 0.0) for r in eval_records])
        ax.plot(eval_steps, ret_eval, "s-", color="#1b5e20", linewidth=2.2, markersize=6, label=f"Eval Return (curr: {ret_eval[-1]:.1f})")
    ax.axhline(0, color="gray", linestyle=":", lw=1.0)
    ax.set_title("6. Episode Returns (MetaDrive)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Return", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 7. Actor & Critic Loss
    ax = axes[2, 0]
    actor_loss = np.array([r.get("ac/actor_loss", 0.0) for r in online_records])
    critic_loss = np.array([r.get("ac/critic_loss", 0.0) for r in online_records])
    ax.plot(steps, actor_loss, color="#f57c00", linewidth=1.8, label=f"Actor Loss (curr: {actor_loss[-1]:.3f})")
    ax.plot(steps, critic_loss, color="#5d4037", linewidth=1.8, label=f"Critic Loss (curr: {critic_loss[-1]:.3f})")
    ax.set_title("7. Actor & Critic Loss", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 8. Gradient Norms
    ax = axes[2, 1]
    a_grad = np.array([r.get("ac/actor_grad_norm", 0.0) for r in online_records])
    c_grad = np.array([r.get("ac/critic_grad_norm", 0.0) for r in online_records])
    ax.plot(steps, a_grad, color="#ab47bc", lw=1.6, alpha=0.85, label=f"Actor Grad (curr: {a_grad[-1]:.2f})")
    ax.plot(steps, c_grad, color="#d32f2f", linewidth=1.8, label=f"Critic Grad (curr: {c_grad[-1]:.2f})")
    ax.axhline(10.0, color="red", linestyle="--", lw=1.2, alpha=0.7, label="Clip Threshold (10.0)")
    ax.set_title("8. Gradient Norms", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Grad Norm", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # 9. World Model Losses
    ax = axes[2, 2]
    wm_loss = np.array([r.get("wm/loss", 0.0) for r in online_records])
    wm_img = np.array([r.get("wm/loss_recon_img", 0.0) for r in online_records])
    wm_state = np.array([r.get("wm/loss_recon_state", 0.0) for r in online_records])
    wm_rew = np.array([r.get("wm/loss_reward", 0.0) for r in online_records])
    wm_cont = np.array([r.get("wm/loss_continue", 0.0) for r in online_records])
    wm_kl = np.array([r.get("wm/loss_kl", 0.0) for r in online_records])

    ax.plot(steps, wm_loss, "k-", linewidth=2.0, label=f"Total Loss ({wm_loss[-1]:.3f})")
    ax.plot(steps, wm_rew, "r--", lw=1.4, label=f"Reward ({wm_rew[-1]:.3f})")
    ax.plot(steps, wm_cont, "g--", lw=1.4, label=f"Continue ({wm_cont[-1]:.3f})")
    ax.plot(steps, wm_kl, "m--", lw=1.4, label=f"KL ({wm_kl[-1]:.3f})")
    ax.plot(steps, wm_img, "b--", lw=1.4, label=f"Image ({wm_img[-1]:.3f})")
    ax.plot(steps, wm_state, "c--", lw=1.4, label=f"State ({wm_state[-1]:.3f})")
    ax.set_title("9. World Model Losses", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Env Steps", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Success] Saved online metrics plot to: {output_path.resolve()}")
    return True


def plot_pretrain_curves(records: list[dict], output_path: Path) -> None:
    """Plot Phase A World Model pre-training loss curves."""
    pt_loss = [r.get("loss") for r in records if "loss" in r and "loss_recon_img" in r]
    pt_img = [r.get("loss_recon_img") for r in records if "loss_recon_img" in r]
    pt_state = [r.get("loss_recon_state") for r in records if "loss_recon_state" in r]
    pt_cont = [r.get("loss_continue") for r in records if "loss_continue" in r]
    pt_reward = [r.get("loss_reward") for r in records if "loss_reward" in r]
    pt_kl = [r.get("loss_kl") for r in records if "loss_kl" in r]
    pt_steps = [
        r.get("step", i) for i, r in enumerate(records) if "loss" in r and "loss_recon_img" in r
    ]

    if not pt_loss:
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("World Model Phase A (Pre-train) Metrics", fontsize=14, fontweight="bold")

    axs[0, 0].plot(pt_steps, pt_loss, color="black", linewidth=2, label="Total Loss")
    axs[0, 0].plot(pt_steps, pt_kl, color="purple", linewidth=1.5, label="KL Loss")
    axs[0, 0].set_title("Total & KL Loss")
    axs[0, 0].set_xlabel("Updates")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend()

    axs[0, 1].plot(pt_steps, pt_img, color="#1f77b4", label="Image Loss")
    axs[0, 1].plot(pt_steps, pt_state, color="#2ca02c", label="State Loss")
    axs[0, 1].set_title("Reconstruction Losses")
    axs[0, 1].set_xlabel("Updates")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].legend()

    axs[1, 0].plot(pt_steps, pt_reward, color="red", label="Reward Loss")
    axs[1, 0].set_title("Reward Loss (Huber)")
    axs[1, 0].set_xlabel("Updates")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend()

    axs[1, 1].plot(pt_steps, pt_cont, color="orange", label="Continue Loss")
    axs[1, 1].set_title("Continue (Terminal) Loss")
    axs[1, 1].set_xlabel("Updates")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout()
    pt_out = output_path.parent / "pretrain_curves.png"
    fig.savefig(pt_out, dpi=200)
    plt.close(fig)
    print(f"[Success] Saved pretrain plot to: {pt_out.resolve()}")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file)

    if not log_path.exists():
        candidates = list(Path("logs").glob("**/metrics.jsonl"))
        if candidates:
            log_path = candidates[0]
            print(f"[Info] Found log file at: {log_path}")
        else:
            print(f"[Error] Log file not found: {args.log_file}")
            return

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("[Warn] Log file is empty.")
        return

    output_path = Path(args.output)
    has_online = plot_online_dashboard(records, output_path)
    plot_pretrain_curves(records, output_path)
    if not has_online:
        print(
            f"[Note] Only pre-training curves plotted because no online training steps were recorded yet in {log_path}."
        )


if __name__ == "__main__":
    main()
