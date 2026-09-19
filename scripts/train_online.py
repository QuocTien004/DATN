#!/usr/bin/env python
"""Online RL training: Collect -> Train WM -> Train AC -> Eval.

Usage
-----
  python scripts/train_online.py --device cuda
  python scripts/train_online.py --total-steps 10 --steps-per-iter 5 \
      --eval-every 10 --eval-episodes 1 --ckpt-every 10 --device cpu
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

import torch

from envs.metadrive_wrapper import make_env
from models.actor_critic import ActorCritic
from models.world_model import (
    build_world_model,
    load_world_model_state,
    validate_world_model_metadata,
)
from training.online_trainer import OnlineTrainer
from utils.checkpoint import load_checkpoint
from utils.config import load_experiment_configs
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online RL training loop")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument("--buffer", type=str, default=None,
                   help="Replay buffer .npz (default: data/replay_buffer/bootstrap.npz)")
    p.add_argument("--wm-checkpoint", type=str, default=None,
                   help="World Model checkpoint (default: checkpoints/world_model/latest.pt)")
    p.add_argument("--resume", type=str, default=None,
                   help="Online checkpoint to resume from")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="Custom checkpoint output directory (e.g., checkpoints/exp_01)")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Custom experiment name for logging")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--steps-per-iter", type=int, default=None)
    p.add_argument("--wm-updates", type=int, default=None)
    p.add_argument("--ac-updates", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    return p.parse_args()


def _resolve_device(name: str | None) -> torch.device:
    requested = name or "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -> using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    train_cfg = configs["train"]
    set_seed(int(train_cfg.get("seed", 0)))
    device = _resolve_device(args.device or train_cfg.get("device", "cpu"))
    paths_cfg = train_cfg.setdefault("paths", {})
    if args.ckpt_dir:
        paths_cfg["checkpoint_dir"] = args.ckpt_dir
    if args.exp_name:
        train_cfg["experiment_name"] = args.exp_name

    # ---- Load replay buffer ------------------------------------------------
    buffer_path = Path(
        args.buffer
        or Path(paths_cfg.get("buffer_dir", "data/replay_buffer")) / "bootstrap.npz"
    )
    if not buffer_path.exists():
        raise FileNotFoundError(
            f"Replay buffer not found: {buffer_path}\n"
            "Run scripts/collect_bootstrap.py first."
        )
    print(f"[Online] Loading initial replay buffer: {buffer_path}", flush=True)
    buffer = ReplayBuffer.load(buffer_path)

    # Expand buffer for online data and make writable
    online_cap = int(train_cfg.get("buffer", {}).get("capacity", 25000))
    buffer.make_writable(max(online_cap, len(buffer)))
    buffer.sequence_length = int(
        train_cfg.get("buffer", {}).get("sequence_length", buffer.sequence_length)
    )
    buffer._valid_starts_cache = None
    print(f"[Online] Buffer ready: {buffer.summary()}", flush=True)

    # ---- Load World Model checkpoint ---------------------------------------
    if args.wm_checkpoint:
        wm_ckpt_path = Path(args.wm_checkpoint)
    else:
        custom_wm_path = Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "world_model" / "latest.pt"
        base_wm_path = Path("checkpoints/world_model/latest.pt")
        wm_ckpt_path = custom_wm_path if custom_wm_path.exists() else base_wm_path
    if not wm_ckpt_path.exists():
        raise FileNotFoundError(
            f"World Model checkpoint not found: {wm_ckpt_path}\n"
            "Run scripts/train_world_model.py first."
        )
    print(f"[Online] Loading World Model: {wm_ckpt_path}", flush=True)
    wm_ckpt = load_checkpoint(wm_ckpt_path, map_location=device)
    validate_world_model_metadata(
        wm_ckpt,
        image_shape=buffer.image_shape,
        state_dim=buffer.state_dim,
        action_dim=buffer.action_dim,
    )
    wm_cfg = wm_ckpt.get("wm_cfg", configs["world_model"])
    world_model = build_world_model(
        wm_cfg,
        tuple(buffer.image_shape),
        buffer.state_dim,
        buffer.action_dim,
        device,
        include_decoder=True,  # decoder needed for online WM training
    )
    load_world_model_state(world_model, wm_ckpt)

    # ---- Build Actor-Critic ------------------------------------------------
    rssm = world_model["rssm"]
    ac_cfg = train_cfg.get("actor_critic", {})
    actor_critic = ActorCritic(
        ac_cfg,
        deter_dim=rssm.deter_dim,
        stoch_size=rssm.stoch_size,
        action_dim=buffer.action_dim,
    ).to(device)

    # ---- Optimizers --------------------------------------------------------
    wm_lr = float(wm_cfg.get("training", {}).get("lr", 1e-4))
    wm_params = [p for m in world_model.values() for p in m.parameters()]
    wm_optimizer = torch.optim.Adam(wm_params, lr=wm_lr)

    # Restore WM optimizer state if available
    if "optimizer" in wm_ckpt:
        try:
            wm_optimizer.load_state_dict(wm_ckpt["optimizer"])
            print("[Online] Resumed WM optimizer state.", flush=True)
        except (ValueError, RuntimeError) as exc:
            print(f"[Online] Could not restore WM optimizer: {exc}", flush=True)

    actor_optimizer = torch.optim.Adam(
        actor_critic.actor.parameters(),
        lr=float(ac_cfg.get("actor_lr", 8e-5)),
    )
    critic_optimizer = torch.optim.Adam(
        actor_critic.critic.parameters(),
        lr=float(ac_cfg.get("critic_lr", 8e-5)),
    )

    # ---- Resume from online checkpoint -------------------------------------
    start_env_steps = 0
    start_episodes = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            # Check if user omitted the /online/ subdirectory
            alt_path = resume_path.parent / "online" / resume_path.name
            if alt_path.exists():
                resume_path = alt_path
            elif resume_path.is_dir():
                for cand in (resume_path / "online" / "latest.pt", resume_path / "latest.pt"):
                    if cand.exists():
                        resume_path = cand
                        break

        ckpt = load_checkpoint(resume_path, map_location=device)
        actor_critic.load_state_dict(ckpt["actor_critic"])
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        if "models" in ckpt:
            for name, m in world_model.items():
                if name in ckpt["models"]:
                    m.load_state_dict(ckpt["models"][name])
        if "wm_optimizer" in ckpt:
            try:
                wm_optimizer.load_state_dict(ckpt["wm_optimizer"])
            except (ValueError, RuntimeError):
                pass
        start_env_steps = int(ckpt.get("env_steps", 0))
        start_episodes = int(ckpt.get("total_episodes", 0))
        print(f"[Online] Resumed from {resume_path} @ env_step={start_env_steps}", flush=True)

    # ---- Environment configs -----------------------------------------------
    env_cfg = configs["env"]
    train_env_cfg = dict(env_cfg)

    eval_cfg_section = train_cfg.get("eval", {})
    eval_start_seed = int(eval_cfg_section.get("start_seed", 10000))
    eval_episodes = int(
        args.eval_episodes or eval_cfg_section.get("num_episodes", 20)
    )
    eval_env_cfg = dict(env_cfg)
    eval_env_cfg["start_seed"] = eval_start_seed
    eval_env_cfg["num_scenarios"] = eval_episodes

    # ---- Create training environment ---------------------------------------
    print("[Online] Initializing MetaDrive environment...", flush=True)
    train_env = make_env(train_env_cfg)

    # ---- Logger ------------------------------------------------------------
    logger = Logger(
        experiment_name=train_cfg.get("experiment_name", "wm_metadrive") + "_online",
        log_dir=paths_cfg.get("log_dir", "logs"),
        wandb_cfg=train_cfg.get("wandb", {}),
    )

    # ---- Build OnlineTrainer and run ---------------------------------------
    trainer = OnlineTrainer(
        configs=configs,
        buffer=buffer,
        world_model=world_model,
        actor_critic=actor_critic,
        wm_optimizer=wm_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        logger=logger,
        device=device,
        train_env=train_env,
        train_env_cfg=train_env_cfg,
        eval_env_cfg=eval_env_cfg,
    )
    trainer.env_steps = start_env_steps
    trainer.total_episodes = start_episodes

    # ---- Training hyperparams (CLI overrides > YAML defaults) --------------
    online_cfg = train_cfg.get("train", {})
    total_steps = int(args.total_steps or online_cfg.get("total_env_steps", 500_000))
    steps_per_iter = int(args.steps_per_iter or 1000)
    wm_updates = int(args.wm_updates or online_cfg.get("world_model_updates", 1))
    ac_updates = int(args.ac_updates or online_cfg.get("actor_critic_updates", 1))
    batch_size = int(args.batch_size or train_cfg.get("wm_train", {}).get("batch_size", 8))
    eval_every = int(args.eval_every or online_cfg.get("eval_every", 10_000))
    ckpt_every = int(args.ckpt_every or online_cfg.get("checkpoint_every", 20_000))

    print(
        f"[Online] total_steps={total_steps}, steps_per_iter={steps_per_iter}, "
        f"wm_updates={wm_updates}, ac_updates={ac_updates}, "
        f"batch_size={batch_size}, eval_every={eval_every}, "
        f"eval_episodes={eval_episodes}, ckpt_every={ckpt_every}, "
        f"device={device}",
        flush=True,
    )

    try:
        trainer.run(
            total_steps=total_steps,
            steps_per_iter=steps_per_iter,
            wm_updates=wm_updates,
            ac_updates=ac_updates,
            batch_size=batch_size,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_start_seed=eval_start_seed,
            checkpoint_every=ckpt_every,
        )
    finally:
        train_env.close()
        logger.finish()


if __name__ == "__main__":
    main()
