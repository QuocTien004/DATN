#!/usr/bin/env python
"""Evaluate a random policy or a recurrent latent Actor on hold-out maps."""

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
import torch

from envs.metadrive_wrapper import make_env
from evaluation.evaluate import LatentActorPolicy, evaluate_policy
from models.actor_critic import ActorCritic
from models.world_model import build_world_model, load_world_model_state, validate_world_model_metadata
from training.collect import random_action
from utils.checkpoint import load_checkpoint
from utils.config import load_experiment_configs
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate policy on MetaDrive")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="Actor-Critic checkpoint")
    p.add_argument("--wm-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--start-seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None, help="Save aggregate and per-seed JSON")
    p.add_argument("--horizon", type=int, default=None, help="Short smoke tests only; omit for full eval")
    return p.parse_args()


def _resolve_device(name: str | None) -> torch.device:
    requested = name or "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -> using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    train_cfg = configs["train"]
    env_cfg = configs["env"]
    eval_cfg = train_cfg.get("eval", {})

    set_seed(int(train_cfg.get("seed", 0)))
    num_episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("num_episodes", 20))
    start_seed = int(args.start_seed if args.start_seed is not None else eval_cfg.get("start_seed", 10000))
    if num_episodes <= 0:
        raise ValueError("--episodes must be positive")

    if args.checkpoint:
        device = _resolve_device(args.device or train_cfg.get("device", "cpu"))
        agent_checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        if args.env_config is None and "env_cfg" in agent_checkpoint:
            env_cfg = dict(agent_checkpoint["env_cfg"])
        checkpoint_root = Path(
            train_cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
        )
        wm_path = Path(
            args.wm_checkpoint
            or agent_checkpoint.get("world_model_checkpoint")
            or checkpoint_root / "world_model" / "latest.pt"
        )
        if "models" in agent_checkpoint:
            if args.wm_checkpoint:
                raise ValueError("Online checkpoint already contains its WM; omit --wm-checkpoint")
            wm_checkpoint = agent_checkpoint
        elif not wm_path.exists():
            raise FileNotFoundError(
                f"World Model checkpoint not found: {wm_path}. "
                "Pass --wm-checkpoint PATH."
            )
        else:
            wm_checkpoint = load_checkpoint(wm_path, map_location="cpu")
        image_shape = tuple(
            wm_checkpoint.get(
                "image_shape",
                agent_checkpoint.get(
                    "image_shape",
                    (
                        int(env_cfg.get("image_height", 256)),
                        int(env_cfg.get("image_width", 256)),
                        3,
                    ),
                ),
            )
        )
        state_dim = int(
            wm_checkpoint.get("state_dim", agent_checkpoint.get("state_dim", 19))
        )
        action_dim = int(
            wm_checkpoint.get(
                "action_dim",
                agent_checkpoint.get("action_dim", 2),
            )
        )
        validate_world_model_metadata(wm_checkpoint, image_shape=(int(env_cfg.get("image_height", 64)), int(env_cfg.get("image_width", 64)), 3))
        wm_cfg = wm_checkpoint.get("wm_cfg", configs["world_model"])
        world_model = build_world_model(
            wm_cfg,
            image_shape,
            state_dim,
            action_dim,
            device,
            include_decoder=False,
        )
        load_world_model_state(world_model, wm_checkpoint)
        rssm = world_model["rssm"]
        ac_cfg = agent_checkpoint.get(
            "actor_critic_cfg", train_cfg.get("actor_critic", {})
        )
        if "actor_critic_cfg" not in agent_checkpoint:
            # Historical online checkpoints predate smooth parameterization.
            ac_cfg = dict(ac_cfg, mean_transform="clamp", std_transform="log_clamp", min_std=0.30, max_std=1.0)
            print("[warn] Legacy online checkpoint has no Actor config; using original clamp transforms and std bounds 0.30..1.0")
        actor_critic = ActorCritic(
            ac_cfg,
            rssm.deter_dim,
            rssm.stoch_size,
            action_dim,
        ).to(device)
        try:
            actor_critic.load_state_dict(agent_checkpoint["actor_critic"])
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                f"Actor-Critic checkpoint is incomplete or incompatible: {exc}"
            ) from exc
        policy_fn = LatentActorPolicy(
            world_model["encoder"],
            rssm,
            actor_critic,
            device=device,
            deterministic=bool(eval_cfg.get("deterministic", True)),
        )
    else:
        def policy_fn(_obs: dict) -> np.ndarray:
            return random_action(env.action_space)

    env_cfg = dict(env_cfg, start_seed=start_seed, num_scenarios=num_episodes)
    if args.horizon is not None:
        env_cfg["horizon"] = args.horizon
    env = make_env(env_cfg)
    try:
        metrics = evaluate_policy(
            env,
            policy_fn,
            num_episodes=num_episodes,
            start_seed=start_seed,
            output_path=args.output,
        )
        print("=== Eval metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
