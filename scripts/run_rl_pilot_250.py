#!/usr/bin/env python
"""RL Pilot: 250 updates with Early-Stopping on Paired Benchmark.

Adheres strictly to the guidelines in docs/P1_P4_AUDIT.md:
- Uses the verified checkpoint (Refit Transition + C19 Critic Warm-up).
- Integrates P3 Boundary head (edge clearance heuristic) during imagination.
- Runs at most 250 Actor-Critic imagination updates.
- Evaluates periodically against the exact paired benchmark (maps 44200, 44205, 44500, 44501; 8 episodes).
- Early-stops immediately if policy performance degrades compared to BC baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
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

from envs.metadrive_wrapper import MetaDriveImageEnv, make_env
from evaluation.evaluate import LatentActorPolicy
from evaluation.p1_p4_diagnostics import boundary_predictions
from models.actor_critic import ActorCritic
from models.world_model import build_world_model, load_world_model_state
from training.batches import replay_batch_to_torch
from training.train_agent import posterior_start_states, train_actor_critic_step
from utils.replay_buffer import ReplayBuffer

PAIRED_BENCHMARK = [
    {"map": 44200, "seed": 44200 + 1234},
    {"map": 44200, "seed": 44200 + 2234},
    {"map": 44205, "seed": 44205 + 1234},
    {"map": 44205, "seed": 44205 + 2234},
    {"map": 44500, "seed": 44500 + 1234},
    {"map": 44500, "seed": 44500 + 2234},
    {"map": 44501, "seed": 44501 + 1234},
    {"map": 44501, "seed": 44501 + 2234},
]

# Baseline metrics from docs/P1_P4_AUDIT.md:
BC_BASELINE_METRICS = {
    "route_mean": 57.24,
    "finish": 2,
    "out": 4,
    "crash": 2,
}


class P3SafeWrappedPredictor(nn.Module):
    """Enhanced P3 Boundary & Speed-Curtailed Predictor for Imagined Rollouts."""

    def __init__(self, base_reward: nn.Module, base_continue: nn.Module, rssm_continuous: nn.Module, target_key: str):
        super().__init__()
        self.base_reward = base_reward
        self.base_continue = base_continue
        self.rssm_continuous = rssm_continuous
        self.target_key = target_key
        self.continuous_state_dim = 19

    def forward(self, h: torch.Tensor, z: torch.Tensor, *, c: torch.Tensor | None = None) -> torch.Tensor:
        r_sym = self.base_reward(h, z)
        c_logit = self.base_continue(h, z)
        if c is not None and self.rssm_continuous is not None:
            # Decode c to physical state19 space (continuous * state_std + state_mean)
            phys_s = self.rssm_continuous.decode(c)
            # 1. Realistic -20 penalty (symlog -3.04) and safety threshold 0.095
            p3 = boundary_predictions(
                phys_s, r_sym, c_logit,
                threshold=0.095,
                alpha=0.18,
                sharpness=120.0,
                speed_threshold=0.05,
                penalty_symlog=-3.04,
            )
            r_out = p3["reward"]
            c_out = p3["continue"]

            # 2. Curate speed bonus in curves / lateral deviations
            lat_dev = (phys_s[:, 2:3] - 0.5).abs()
            speed = phys_s[:, 3:4]
            excess_speed_in_turn = torch.relu(speed - 0.35) * torch.relu(lat_dev - 0.12) * 8.0
            r_out = r_out - excess_speed_in_turn

            # 3. Clamp continue logit if vehicle drifts dangerously wide (> 0.28)
            extreme_dev = lat_dev > 0.28
            c_out = torch.where(extreme_dev, torch.minimum(c_out, torch.full_like(c_out, -4.0)), c_out)

            return r_out if self.target_key == "reward" else c_out
        return r_sym if self.target_key == "reward" else c_logit


def evaluate_paired_benchmark(policy: LatentActorPolicy, env_cfg_base: dict, device: torch.device) -> dict:
    """Evaluate policy on the exact 8 paired episodes and compute route, finish, out, crash."""
    routes = []
    finishes = 0
    outs = 0
    crashes = 0
    rewards = []

    for item in PAIRED_BENCHMARK:
        cfg = dict(env_cfg_base)
        cfg["start_seed"] = item["seed"]
        cfg["num_scenarios"] = 1
        cfg["horizon"] = 1000

        env = make_env(cfg)
        obs, _ = env.reset()
        policy.reset()

        ep_r = 0.0
        done = False
        final_info = {}

        while not done:
            action = policy(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc
            final_info = info

        env.close()

        route_comp = final_info.get("route_completion", 0.0) * 100.0
        routes.append(route_comp)
        rewards.append(ep_r)
        if final_info.get("arrive_dest", False):
            finishes += 1
        elif final_info.get("crash", False):
            crashes += 1
        elif final_info.get("out_of_road", False):
            outs += 1

    return {
        "route_mean": float(np.mean(routes)),
        "route_std": float(np.std(routes)),
        "finish": finishes,
        "out": outs,
        "crash": crashes,
        "reward_mean": float(np.mean(rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/critic_warmup/continuous_seed0_refit_warmup_c19.pt")
    parser.add_argument("--buffer", default="data/replay_buffer/bootstrap_v2.npz")
    parser.add_argument("--total-updates", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--early-stop-drop", type=float, default=15.0, help="Max allowed drop in route completion below BC (57.24%)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", default="logs/rl_pilot_250")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== RL PILOT 250 UPDATES (WITH EARLY-STOPPING) ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Buffer: {args.buffer}")
    print(f"BC Baseline Reference: Route={BC_BASELINE_METRICS['route_mean']}%, Finish={BC_BASELINE_METRICS['finish']}/8")

    # 1. Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wm_cfg = ckpt["wm_cfg"]
    ac_cfg = ckpt["actor_critic_cfg"]

    world_model = build_world_model(
        wm_cfg,
        tuple(ckpt["image_shape"]),
        ckpt["state_dim"],
        ckpt["action_dim"],
        device,
        include_decoder=False,
    )
    load_world_model_state(world_model, ckpt)

    # Wrap reward and continue heads with P3 Boundary Heuristics
    rssm = world_model["rssm"]
    orig_reward = world_model["reward"]
    orig_continue = world_model["continue"]
    world_model["reward"] = P3SafeWrappedPredictor(orig_reward, orig_continue, rssm.continuous, "reward").to(device)
    world_model["continue"] = P3SafeWrappedPredictor(orig_reward, orig_continue, rssm.continuous, "continue").to(device)

    actor_critic = ActorCritic(
        ac_cfg,
        deter_dim=rssm.deter_dim,
        stoch_size=rssm.stoch_size,
        action_dim=ckpt["action_dim"],
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])

    # Freeze world model
    for m in world_model.values():
        m.eval().requires_grad_(False)

    # Optimizers
    actor_opt = torch.optim.Adam(actor_critic.actor.parameters(), lr=float(ac_cfg.get("actor_lr", 3e-5)))
    critic_opt = torch.optim.Adam(actor_critic.critic.parameters(), lr=float(ac_cfg.get("critic_lr", 3e-5)))

    # 2. Load Replay Buffer for imagination starting states
    print(f"\nLoading replay buffer: {args.buffer}")
    buffer = ReplayBuffer.load(Path(args.buffer))
    print(f"Buffer size: {len(buffer)}")

    # Policy for evaluation
    eval_policy = LatentActorPolicy(world_model["encoder"], world_model["rssm"], actor_critic, device=device, deterministic=True)
    env_cfg_base = dict(ckpt["env_cfg"])

    # 3. Initial baseline evaluation at update 0
    print("\n--- Running Evaluation at Update 0 (Pre-RL Baseline) ---")
    metrics_0 = evaluate_paired_benchmark(eval_policy, env_cfg_base, device)
    print(f"Update 0 Results: Route={metrics_0['route_mean']:.2f}% (std={metrics_0['route_std']:.2f}%), Finish={metrics_0['finish']}/8, Out={metrics_0['out']}/8, Crash={metrics_0['crash']}/8")

    history = [
        {"update": 0, "metrics": metrics_0, "loss": {}}
    ]

    stop_early = False
    stop_reason = ""
    min_allowed_route = BC_BASELINE_METRICS["route_mean"] - args.early_stop_drop

    print(f"\nStarting {args.total_updates} imagination updates with Early-Stop threshold route < {min_allowed_route:.1f}%...")
    start_time = time.time()

    for update_idx in range(1, args.total_updates + 1):
        # Sample batch of 8 sequences of length 32
        buffer.sequence_length = 32
        batch_np = buffer.sample(batch_size=8, term_ratio=0.25)
        batch = replay_batch_to_torch(batch_np, device=device)

        start_states = posterior_start_states(
            batch,
            world_model,
            max_states=64,
            deterministic=False,
            include_control=True,
        )

        loss_info = train_actor_critic_step(
            start_states,
            world_model,
            actor_critic,
            actor_opt,
            critic_opt,
            ac_cfg,
            diagnostic_control=True,
        )

        if update_idx % 10 == 0:
            print(f"Update {update_idx:3d}/{args.total_updates} | Actor Loss: {loss_info['actor_loss']:.4f} | Critic Loss: {loss_info['critic_loss']:.4f} | Img Rew: {loss_info['imagined_reward_mean']:.4f} | Val: {loss_info['value_mean']:.4f}")

        # Periodic Evaluation
        if update_idx % args.eval_every == 0 or update_idx == args.total_updates:
            print(f"\n--- Running Evaluation at Update {update_idx} ---")
            eval_metrics = evaluate_paired_benchmark(eval_policy, env_cfg_base, device)
            print(f"Update {update_idx} Metrics: Route={eval_metrics['route_mean']:.2f}%, Finish={eval_metrics['finish']}/8, Out={eval_metrics['out']}/8, Crash={eval_metrics['crash']}/8")

            record = {
                "update": update_idx,
                "metrics": eval_metrics,
                "loss": loss_info,
            }
            history.append(record)

            # Check Early-Stopping Condition
            if eval_metrics["route_mean"] < min_allowed_route:
                stop_early = True
                stop_reason = f"Route completion dropped to {eval_metrics['route_mean']:.2f}% (below threshold {min_allowed_route:.2f}%). Early-stopping to prevent policy regression."
                print(f"\n[ALERT - EARLY STOPPING TRIGGERED]: {stop_reason}")
                break

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s. Stopped early: {stop_early}")

    # Save summary and checkpoint
    out_summary = {
        "total_updates_run": update_idx,
        "stopped_early": stop_early,
        "stop_reason": stop_reason,
        "history": history,
        "elapsed_seconds": elapsed,
        "bc_baseline_reference": BC_BASELINE_METRICS,
    }
    summary_path = log_dir / "rl_pilot_results.json"
    summary_path.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(f"Results saved to {summary_path}")

    # Save final pilot checkpoint
    ckpt_out = {
        "models": {k: m.state_dict() for k, m in world_model.items() if k not in ["reward", "continue"]},
        "actor_critic": actor_critic.state_dict(),
        "actor_critic_cfg": ac_cfg,
        "wm_cfg": wm_cfg,
        "env_cfg": ckpt["env_cfg"],
        "pilot_summary": out_summary,
    }
    ckpt_save_path = log_dir / f"pilot_checkpoint_update_{update_idx}.pt"
    torch.save(ckpt_out, ckpt_save_path)
    print(f"Checkpoint saved to {ckpt_save_path}")


if __name__ == "__main__":
    main()
