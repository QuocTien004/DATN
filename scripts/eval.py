#!/usr/bin/env python
"""
Evaluation entrypoint.

Phase A: evaluates a random policy (sanity check for metrics + env).
Phase C: load Actor-Critic checkpoint and evaluate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

import numpy as np

from envs.metadrive_wrapper import make_env
from evaluation.evaluate import evaluate_policy
from training.collect import random_action
from utils.config import load_experiment_configs
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate policy on MetaDrive")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="Actor-Critic ckpt (Phase C)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, None)
    train_cfg = configs["train"]
    env_cfg = configs["env"]
    eval_cfg = train_cfg.get("eval", {})

    set_seed(int(train_cfg.get("seed", 0)))
    env = make_env(env_cfg)

    if args.checkpoint:
        raise NotImplementedError(
            "Loading Actor-Critic from checkpoint is Phase C. "
            "For now omit --checkpoint to eval random policy."
        )

    def policy_fn(obs: dict) -> np.ndarray:
        return random_action(env.action_space)

    try:
        metrics = evaluate_policy(
            env,
            policy_fn,
            num_episodes=int(args.episodes or eval_cfg.get("num_episodes", 5)),
            start_seed=int(eval_cfg.get("start_seed", 10000)),
        )
        print("=== Eval metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
