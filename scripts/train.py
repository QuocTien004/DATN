#!/usr/bin/env python
"""Train Actor-Critic offline from replay data and a pretrained World Model."""

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

import torch

from models.actor_critic import ActorCritic
from models.world_model import (
    build_world_model,
    load_world_model_state,
    validate_world_model_metadata,
)
from training.trainer import Trainer
from utils.checkpoint import load_checkpoint
from utils.config import load_experiment_configs
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed
from utils.online_state import require_current_contract


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Actor-Critic via latent imagination")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument("--buffer", type=str, default=None, help="Replay buffer .npz")
    p.add_argument("--wm-checkpoint", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Actor-Critic checkpoint")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    return p.parse_args()


def _resolve_device(name: str | None) -> torch.device:
    requested = name or "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -> using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def _check_agent_metadata(checkpoint: dict, actor_critic: ActorCritic) -> None:
    expected = {
        "deter_dim": actor_critic.deter_dim,
        "stoch_size": actor_critic.stoch_size,
        "action_dim": actor_critic.action_dim,
    }
    for key, value in expected.items():
        if key in checkpoint and int(checkpoint[key]) != value:
            raise ValueError(
                f"Actor-Critic checkpoint {key}={checkpoint[key]}, expected {value}"
            )


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    train_cfg = configs["train"]
    set_seed(int(train_cfg.get("seed", 0)))

    device = _resolve_device(args.device or train_cfg.get("device", "cpu"))
    paths_cfg = train_cfg.get("paths", {})
    buffer_path = Path(
        args.buffer
        or Path(paths_cfg.get("buffer_dir", "data/replay_buffer")) / "bootstrap.npz"
    )
    wm_checkpoint_path = Path(
        args.wm_checkpoint
        or Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
        / "world_model"
        / "latest.pt"
    )
    if not buffer_path.exists():
        raise FileNotFoundError(
            f"Replay buffer not found: {buffer_path}\n"
            "Collect or copy bootstrap.npz before Actor-Critic training."
        )
    if not wm_checkpoint_path.exists():
        raise FileNotFoundError(
            f"World Model checkpoint not found: {wm_checkpoint_path}\n"
            "Train the World Model first or pass --wm-checkpoint PATH."
        )

    print(f"Loading replay buffer: {buffer_path}", flush=True)
    buffer = ReplayBuffer.load(buffer_path)
    buffer.sequence_length = int(
        train_cfg.get("buffer", {}).get("sequence_length", buffer.sequence_length)
    )
    buffer._valid_starts_cache = None
    if not len(buffer._valid_start_indices()):
        raise RuntimeError(
            f"Replay buffer has no valid sequences of length {buffer.sequence_length}"
        )

    print(f"Loading World Model: {wm_checkpoint_path}", flush=True)
    wm_checkpoint = load_checkpoint(wm_checkpoint_path, map_location=device)
    require_current_contract(wm_checkpoint)
    validate_world_model_metadata(
        wm_checkpoint,
        image_shape=buffer.image_shape,
        state_dim=buffer.state_dim,
        action_dim=buffer.action_dim,
    )
    wm_cfg = wm_checkpoint.get("wm_cfg", configs["world_model"])
    world_model = build_world_model(
        wm_cfg,
        tuple(buffer.image_shape),
        buffer.state_dim,
        buffer.action_dim,
        device,
        include_decoder=False,
    )
    load_world_model_state(world_model, wm_checkpoint)
    for module in world_model.values():
        module.eval()
        module.requires_grad_(False)

    rssm = world_model["rssm"]
    agent_checkpoint = load_checkpoint(args.resume, map_location=device) if args.resume else None
    if agent_checkpoint is not None:
        require_current_contract(agent_checkpoint)
        train_cfg["actor_critic"] = agent_checkpoint["actor_critic_cfg"]
    ac_cfg = train_cfg.get("actor_critic", {})
    actor_critic = ActorCritic(
        ac_cfg,
        deter_dim=rssm.deter_dim,
        stoch_size=rssm.stoch_size,
        action_dim=buffer.action_dim,
    ).to(device)
    actor_optimizer = torch.optim.Adam(
        actor_critic.actor.parameters(), lr=float(ac_cfg.get("actor_lr", 8e-5))
    )
    critic_optimizer = torch.optim.Adam(
        actor_critic.critic.parameters(), lr=float(ac_cfg.get("critic_lr", 8e-5))
    )

    start_step = 0
    if args.resume:
        _check_agent_metadata(agent_checkpoint, actor_critic)
        try:
            actor_critic.load_state_dict(agent_checkpoint["actor_critic"])
            actor_optimizer.load_state_dict(agent_checkpoint["actor_optimizer"])
            critic_optimizer.load_state_dict(agent_checkpoint["critic_optimizer"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Actor-Critic checkpoint is incomplete or incompatible: {args.resume}: {exc}"
            ) from exc
        start_step = int(agent_checkpoint.get("step", 0))
        print(f"Resumed Actor-Critic @ update={start_step}", flush=True)

    logger = Logger(
        experiment_name=train_cfg.get("experiment_name", "wm_metadrive"),
        log_dir=train_cfg.get("paths", {}).get("log_dir", "logs"),
        wandb_cfg=train_cfg.get("wandb", {}),
    )
    trainer = Trainer(
        configs,
        buffer,
        logger,
        str(device),
        world_model=world_model,
        actor_critic=actor_critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        global_step=start_step,
        checkpoint_metadata={
            "transition_contract": 2,
            "env_cfg": wm_checkpoint.get("env_cfg", configs["env"]),
            "world_model_checkpoint": str(wm_checkpoint_path),
            "image_shape": tuple(buffer.image_shape),
            "state_dim": buffer.state_dim,
        },
    )
    print(
        f"Train Actor-Critic: device={device}, replay={buffer.summary()}, "
        f"start_update={start_step}",
        flush=True,
    )
    try:
        trainer.fit(
            updates=args.updates,
            batch_size=args.batch_size,
            log_every=args.log_every,
            checkpoint_every=args.ckpt_every,
        )
    finally:
        logger.finish()
    print(f"Done. Latest checkpoint: {trainer._checkpoint_dir() / 'latest.pt'}")


if __name__ == "__main__":
    main()
