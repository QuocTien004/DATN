#!/usr/bin/env python
"""Live 3D Visual Demo: Watch the autonomous World Model policy drive in MetaDrive.

Usage
-----
# Demo with 23D Policy + Safety Shield (Recommended):
python scripts/demo_drive.py --safety-shield --seed 10000

# Demo pure neural policy without shield:
python scripts/demo_drive.py --seed 10000

# Test specific intersection / challenging seeds:
python scripts/demo_drive.py --safety-shield --seed 44204
python scripts/demo_drive.py --safety-shield --seed 44205

# Demo from a trained Online RL checkpoint:
python scripts/demo_drive.py --checkpoint checkpoints/exp_23d_online/online/latest.pt --safety-shield
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

import numpy as np
import torch

from envs.metadrive_wrapper import make_env
from evaluation.evaluate import LatentActorPolicy
from evaluation.safety_shield import AdaptiveTrafficAvoidanceShield, SafeLatentActorPolicy
from models.actor_critic import ActorCritic
from models.world_model import build_world_model, load_world_model_state
from utils.checkpoint import load_checkpoint
from utils.config import load_experiment_configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live 3D MetaDrive Autonomous Driving Demo")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default="configs/env_metadrive.yaml")
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/critic_warmup/continuous_seed0_refit_warmup_c23.pt",
        help="Path to Actor-Critic checkpoint (BC, warmup, or online RL)",
    )
    p.add_argument(
        "--wm-checkpoint",
        type=str,
        default="checkpoints/wm_23d/world_model/latest.pt",
        help="Path to World Model checkpoint",
    )
    p.add_argument("--seed", type=int, default=10000, help="Scenario seed to run (set -1 or use --random for random map)")
    p.add_argument("--random", action="store_true", help="Pick a random unseen map seed in [10000, 90000]")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to demonstrate")
    p.add_argument("--until-fail", action="store_true", help="Run continuously across maps until a failure (CRASH, OUT OF ROAD, or TIMEOUT) occurs")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument("--safety-shield", action="store_true", help="Enable Adaptive Traffic Avoidance Shield (ACC/AEB)")
    p.add_argument("--fps", type=float, default=25.0, help="Target display FPS for human viewing (default: 25)")
    p.add_argument("--horizon", type=int, default=1000, help="Max steps per episode")
    p.add_argument(
        "--map",
        type=str,
        default=None,
        help="Map size (number of blocks, e.g. 7, 10, 15; default is 3) or block string (e.g. 'SCXOCrR')",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.random or args.seed < 0:
        args.seed = int(np.random.randint(10000, 90000))
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    env_cfg = dict(configs["env"])

    if args.map is not None:
        try:
            map_val: int | str = int(args.map)
            num_blocks = map_val
        except ValueError:
            map_val = str(args.map)
            num_blocks = len(map_val)
        env_cfg["map"] = map_val
        if args.horizon == 1000 and num_blocks > 3:
            args.horizon = num_blocks * 450

    # Force 3D rendering for live visual demo
    env_cfg["use_render"] = True
    env_cfg["horizon"] = args.horizon
    env_cfg["start_seed"] = args.seed
    env_cfg["num_scenarios"] = 10000 if args.until_fail else max(args.episodes, 100)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    mode_str = "UNTIL FAIL (Infinite)" if args.until_fail else f"{args.episodes} Episodes"
    map_str = f"{env_cfg.get('map', 3)} blocks"
    print("=" * 70)
    print("METADRIVE WORLD MODEL LIVE 3D DEMONSTRATION")
    print(f"Device: {device} | Start Seed: {args.seed} | Map: {map_str} | Mode: {mode_str} | Shield: {'ENABLED (ACC/AEB)' if args.safety_shield else 'DISABLED (Pure Neural)'}")
    print("=" * 70)

    # 1. Load Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")
    print(f"[Demo] Loading policy checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, map_location=device)

    # 2. Load World Model
    if "models" in ckpt:
        print("[Demo] Using World Model embedded in policy checkpoint.")
        wm_cfg = ckpt.get("wm_cfg", configs["world_model"])
        state_dim = int(ckpt.get("state_dim", env_cfg.get("state_dim", 23)))
        action_dim = int(ckpt.get("action_dim", 2))
        image_shape = tuple(ckpt.get("image_shape", (64, 64, 3)))
        world_model = build_world_model(wm_cfg, image_shape, state_dim, action_dim, device, include_decoder=False)
        load_world_model_state(world_model, ckpt)
    else:
        wm_path = Path(args.wm_checkpoint)
        if not wm_path.exists():
            raise FileNotFoundError(f"World Model checkpoint not found: {wm_path}")
        print(f"[Demo] Loading World Model checkpoint: {wm_path}")
        wm_ckpt = load_checkpoint(wm_path, map_location=device)
        wm_cfg = wm_ckpt.get("wm_cfg", configs["world_model"])
        state_dim = int(wm_ckpt.get("state_dim", env_cfg.get("state_dim", 23)))
        action_dim = int(wm_ckpt.get("action_dim", 2))
        image_shape = tuple(wm_ckpt.get("image_shape", (64, 64, 3)))
        world_model = build_world_model(wm_cfg, image_shape, state_dim, action_dim, device, include_decoder=False)
        load_world_model_state(world_model, wm_ckpt)

    # 3. Build Actor-Critic
    ac_cfg = ckpt.get("actor_critic_cfg", configs["train"].get("actor_critic", {}))
    if ckpt.get("control_input") in {"predicted", "continuous"}:
        ac_cfg = dict(ac_cfg, control_state_source=ckpt["control_input"])
    actor_critic = ActorCritic(
        ac_cfg,
        deter_dim=world_model["rssm"].deter_dim,
        stoch_size=world_model["rssm"].stoch_size,
        action_dim=action_dim,
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval()

    # 4. Create Policy
    raw_policy = LatentActorPolicy(
        world_model["encoder"],
        world_model["rssm"],
        actor_critic,
        device=device,
        deterministic=True,
    )

    # 5. Initialize MetaDrive 3D Environment
    print("[Demo] Launching MetaDrive 3D Window...")
    env = make_env(env_cfg)

    policy = raw_policy
    shield = None
    if args.safety_shield:
        shield = AdaptiveTrafficAvoidanceShield(
            corridor_width=2.4,
            threat_distance=28.0,
            min_gap=5.0,
            desired_time_gap=1.2,
        )
        policy = SafeLatentActorPolicy(raw_policy, shield=shield, env=env)
    elif hasattr(policy, "set_env"):
        policy.set_env(env)

    step_delay = 1.0 / max(args.fps, 1.0)

    try:
        ep = 0
        streak = 0
        while args.until_fail or ep < args.episodes:
            if args.random and ep > 0:
                current_seed = args.seed + int(np.random.randint(1, 9999))
            else:
                current_seed = args.seed + ep
            obs, info = env.reset(seed=current_seed)
            policy.reset()

            done = False
            step = 0
            ep_return = 0.0

            ep_label = f"{ep + 1}/INF (Streak: {streak})" if args.until_fail else f"{ep + 1}/{args.episodes}"
            print(f"\n--- Starting Demonstration Episode {ep_label} (Seed: {current_seed}) ---")
            print(f"{'Step':>5} | {'Speed (km/h)':>12} | {'Steer':>7} | {'Throttle':>8} | {'Route %':>8} | {'Lead Dist':>9} | {'Threat':>6} | {'Shield Status'}")
            print("-" * 85)

            while not done:
                start_time = time.monotonic()

                # Policy action
                action = policy(obs)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_return += float(reward)
                step += 1

                # Render 3D window
                env.render()

                # Extract telemetry from 23D state and vehicle
                state_vec = obs["state"]
                lead_dist_m = float(state_vec[19] * 50.0) if len(state_vec) >= 23 else 0.0
                threat_pct = float(state_vec[22] * 100.0) if len(state_vec) >= 23 else 0.0
                raw_env = getattr(env, "_env", env)
                vehicle = getattr(raw_env, "agent", None)
                speed_kmh = float(getattr(vehicle, "speed_km_h", info.get("velocity", 0.0) * 3.6))
                route_pct = float(info.get("route_completion", 0.0)) * 100.0

                shield_str = "OFF"
                if shield is not None:
                    diag = getattr(policy, "last_shield_diag", {})
                    if diag.get("min_lead_dist", 999.0) < 900.0:
                        lead_dist_m = float(diag["min_lead_dist"])
                    if diag.get("throttle_override", 0.0) > 0.0:
                        shield_str = "[ACTIVE] (ACC/AEB)"
                    else:
                        shield_str = "PASS-THROUGH"

                if step % 5 == 0 or done:
                    print(
                        f"{step:5d} | {speed_kmh:10.1f}   | {action[0]:7.2f} | {action[1]:8.2f} | {route_pct:7.1f}% | {lead_dist_m:7.1f} m | {threat_pct:5.0f}% | {shield_str}"
                    )

                # Maintain realistic viewing FPS
                elapsed = time.monotonic() - start_time
                if elapsed < step_delay:
                    time.sleep(step_delay - elapsed)

            # Episode summary
            outcome = "COMPLETED" if info.get("arrive_dest", False) else (
                "CRASH" if (info.get("crash", False) or info.get("crash_vehicle", False)) else (
                    "OUT OF ROAD" if info.get("out_of_road", False) else "TIMEOUT"
                )
            )
            print("-" * 85)
            print(f"Episode {ep + 1} (Seed {current_seed}) Result: {outcome}")
            print(f"Final Route Completion: {route_pct:.2f}% | Steps: {step} | Total Reward: {ep_return:.2f}")
            print("-" * 85)

            if outcome == "COMPLETED":
                streak += 1
            elif args.until_fail:
                print(f"\n[STOPPED ON FAILURE] Failure detected on Seed {current_seed}: {outcome} after {streak} consecutive successes!")
                print(f"To replay this exact failing scenario, run:\n  .\\venv\\Scripts\\python.exe scripts/demo_drive.py --seed {current_seed} --safety-shield")
                break

            ep += 1

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted by user.")
    finally:
        env.close()
        print("[Demo] Environment closed cleanly.")


if __name__ == "__main__":
    main()
