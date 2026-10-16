#!/usr/bin/env python
"""Evaluate final integrated AdaptiveTrafficAvoidanceShield on 8 Paired Benchmark Seeds.

Compares:
1. Baseline Pure World Model Policy
2. Enhanced Safe Policy (Adaptive Traffic Avoidance Shield with Curve Speed Management and Firm AEB)
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import time
import numpy as np
import torch

from envs.metadrive_wrapper import make_env
from evaluation import LatentActorPolicy, SafeLatentActorPolicy, AdaptiveTrafficAvoidanceShield
from models.actor_critic import ActorCritic
from models.world_model import build_world_model, load_world_model_state

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


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/critic_warmup/continuous_seed0_refit_warmup_c19.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    wm = build_world_model(
        ckpt["wm_cfg"],
        tuple(ckpt["image_shape"]),
        ckpt["state_dim"],
        ckpt["action_dim"],
        device,
        include_decoder=False,
    )
    load_world_model_state(wm, ckpt)

    ac = ActorCritic(
        ckpt["actor_critic_cfg"],
        wm["rssm"].deter_dim,
        wm["rssm"].stoch_size,
        ckpt["action_dim"],
    ).to(device)
    ac.load_state_dict(ckpt["actor_critic"])

    base_policy = LatentActorPolicy(wm["encoder"], wm["rssm"], ac, device=device, deterministic=True)
    shield = AdaptiveTrafficAvoidanceShield()
    safe_policy = SafeLatentActorPolicy(base_policy, shield=shield)

    env_cfg_base = dict(ckpt["env_cfg"])

    print("=================================================================")
    print("      FINAL INTEGRATED SHIELD BENCHMARK (8 EPISODES)             ")
    print("=================================================================")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}\n")

    results = []
    finishes = 0
    crashes = 0
    outs = 0

    for item in PAIRED_BENCHMARK:
        cfg = dict(env_cfg_base)
        cfg["start_seed"] = item["seed"]
        cfg["num_scenarios"] = 1
        cfg["horizon"] = 1000

        env = make_env(cfg)
        safe_policy.set_env(env)
        obs, _ = env.reset()
        safe_policy.reset()

        t0 = time.time()
        step = 0
        done = False
        final_info = {}
        overrides = 0

        while not done:
            action = safe_policy(obs)
            if safe_policy.last_shield_diag.get("throttle_override", 0) > 0:
                overrides += 1
            obs, r, term, trunc, info = env.step(action)
            step += 1
            done = term or trunc
            final_info = info

        env.close()
        dt = time.time() - t0

        route = float(final_info.get("route_completion", 0.0) or 0.0) * 100.0
        is_finish = bool(final_info.get("arrive_dest", False))
        is_crash = bool(final_info.get("crash", False) or final_info.get("crash_vehicle", False))
        is_out = bool(final_info.get("out_of_road", False))

        if is_finish:
            status = "FINISH"
            finishes += 1
        elif is_crash:
            status = "CRASH"
            crashes += 1
        elif is_out:
            status = "OUT"
            outs += 1
        else:
            status = "TRUNC"

        res = {
            "map": item["map"],
            "seed": item["seed"],
            "status": status,
            "step": step,
            "route": route,
            "overrides": overrides,
            "crash_vehicle": bool(final_info.get("crash_vehicle", False)),
            "out_of_road": is_out,
            "time_sec": dt,
        }
        results.append(res)
        print(f"Seed {res['seed']} (Map {res['map']}): {res['status']:6s} | Step: {res['step']:4d} | Route: {res['route']:5.1f}% | Overrides: {res['overrides']:3d} | Time: {dt:.1f}s")

    routes = [r["route"] for r in results]
    summary = {
        "count": len(results),
        "route_mean": float(np.mean(routes)),
        "route_std": float(np.std(routes)),
        "finish": finishes,
        "crash": crashes,
        "out": outs,
        "episodes": results,
    }

    out_file = Path("logs/final_shield_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    print(f"Total Episodes : {len(results)}")
    print(f"Route Mean     : {summary['route_mean']:.2f}% (std: {summary['route_std']:.2f})")
    print(f"Finishes       : {finishes}/{len(results)} ({finishes/len(results)*100:.1f}%)")
    print(f"Crashes        : {crashes}/{len(results)} ({crashes/len(results)*100:.1f}%)")
    print(f"Out of Road    : {outs}/{len(results)}")
    print(f"Saved to       : {out_file}")


if __name__ == "__main__":
    run_benchmark()
