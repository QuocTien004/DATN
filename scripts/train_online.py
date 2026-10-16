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
import copy
import json
import os
import subprocess
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
from utils.online_state import resolve_resume_path, resolve_replay_path, validate_replay_step, require_current_contract


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online RL training loop")
    p.add_argument("--config", type=str, default=None,
                   help="Training YAML; on resume defaults to saved settings")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    p.add_argument("--buffer", type=str, default=None,
                   help="Replay buffer .npz (default: data/replay_buffer/bootstrap.npz)")
    p.add_argument("--wm-checkpoint", type=str, default=None,
                   help="World Model checkpoint (default: checkpoints/world_model/latest.pt)")
    p.add_argument("--actor-checkpoint", type=str, default=None,
                   help="Actor-Critic checkpoint to warm-start policy (e.g., BC/warmup checkpoint)")
    p.add_argument("--resume", type=str, default=None,
                   help="Online checkpoint to resume from")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="Output root; online/ is appended (e.g., checkpoints/exp_01)")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Custom experiment name for logging")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--steps-per-iter", type=int, default=None)
    p.add_argument("--wm-updates", type=int, default=None)
    p.add_argument("--ac-updates", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--eval-horizon", type=int, default=None,
                   help="Optional eval-only time limit; bounded diagnostics are not full-policy evaluation")
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--demo-buffer", type=str, default=None, help="Protected bootstrap/demo replay for mixed sampling")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--safe-collect", action="store_true", help="Enable Adaptive Traffic Avoidance Shield during collection")
    p.add_argument("--bc-scale", type=float, default=None, help="Actor BC regularizer loss weight against demo buffer")
    return p.parse_args()


def _resolve_device(name: str | None) -> torch.device:
    requested = name or "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -> using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config or "configs/train.yaml", args.env_config, args.wm_config)
    resume_path = resolve_resume_path(args.resume) if args.resume else None
    resume_ckpt = load_checkpoint(resume_path, map_location="cpu") if resume_path else None
    if resume_ckpt is not None:
        require_current_contract(resume_ckpt)
    if resume_ckpt is not None and args.config is None:
        configs["train"] = copy.deepcopy(resume_ckpt["train_cfg"])
    train_cfg = configs["train"]
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    set_seed(int(train_cfg.get("seed", 0)))
    device = _resolve_device(args.device or train_cfg.get("device", "cpu"))
    paths_cfg = train_cfg.setdefault("paths", {})
    if args.ckpt_dir:
        paths_cfg["checkpoint_dir"] = args.ckpt_dir
    if Path(paths_cfg.get("checkpoint_dir", "checkpoints")).name == "online":
        print("[warn] --ckpt-dir is a run ROOT; script appends online/. This path will produce online/online/.", flush=True)
    if args.exp_name:
        train_cfg["experiment_name"] = args.exp_name
    if args.log_dir:
        paths_cfg["log_dir"] = args.log_dir
    if resume_ckpt is not None:
        train_cfg["actor_critic"] = resume_ckpt["actor_critic_cfg"]
        if args.env_config is None:
            configs["env"] = resume_ckpt["env_cfg"]
    elif (Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "online/latest.pt").exists():
        raise FileExistsError("Online output already exists. Use --resume or a new --ckpt-dir.")

    # ---- Load warm-start Actor-Critic checkpoint if provided ---------------
    ac_ckpt = None
    if args.actor_checkpoint:
        ac_ckpt_path = Path(args.actor_checkpoint)
        if not ac_ckpt_path.exists():
            raise FileNotFoundError(f"Actor-Critic checkpoint not found: {ac_ckpt_path}")
        print(f"[Online] Loading warm-start Actor-Critic checkpoint: {ac_ckpt_path}", flush=True)
        ac_ckpt = load_checkpoint(ac_ckpt_path, map_location="cpu")
        if resume_ckpt is None and "actor_critic_cfg" in ac_ckpt:
            base_ac_cfg = copy.deepcopy(ac_ckpt["actor_critic_cfg"])
            base_ac_cfg.update(train_cfg.get("actor_critic", {}))
            train_cfg["actor_critic"] = base_ac_cfg

    # ---- Load replay buffer ------------------------------------------------
    buffer_path = Path(
        args.buffer
        or Path(paths_cfg.get("buffer_dir", "data/replay_buffer")) / "bootstrap.npz"
    )
    if resume_ckpt is not None:
        buffer_path = resolve_replay_path(resume_path, resume_ckpt, args.buffer)
    if not buffer_path.exists():
        raise FileNotFoundError(
            f"Replay buffer not found: {buffer_path}\n"
            "Run scripts/collect_bootstrap.py first."
        )
    print(f"[Online] Loading initial replay buffer: {buffer_path}", flush=True)
    buffer = ReplayBuffer.load(buffer_path)
    if resume_ckpt is not None:
        validate_replay_step(resume_ckpt, buffer.checkpoint_step)
    expected_image = (int(configs["env"].get("image_height", 64)), int(configs["env"].get("image_width", 64)), 3)
    if buffer.image_shape != expected_image:
        raise ValueError(f"Environment image {expected_image} != replay {buffer.image_shape}")

    # Expand buffer for online data and make writable
    online_cap = int(train_cfg.get("buffer", {}).get("capacity", 25000))
    buffer.make_writable(max(online_cap, len(buffer)))
    buffer.sequence_length = int(
        train_cfg.get("buffer", {}).get("sequence_length", buffer.sequence_length)
    )
    buffer._valid_starts_cache = None
    print(f"[Online] Buffer ready: {buffer.summary()}", flush=True)

    # ---- Load World Model checkpoint ---------------------------------------
    if resume_path is not None:
        wm_ckpt_path = resume_path
    elif args.wm_checkpoint:
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
    wm_ckpt = resume_ckpt if resume_ckpt is not None else load_checkpoint(wm_ckpt_path, map_location="cpu")
    require_current_contract(wm_ckpt)
    validate_world_model_metadata(
        wm_ckpt,
        image_shape=buffer.image_shape,
        state_dim=buffer.state_dim,
        action_dim=buffer.action_dim,
    )
    wm_cfg = wm_ckpt.get("wm_cfg", configs["world_model"])
    configs["world_model"] = wm_cfg
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

    # ---- Connect Control Bridge for Imagination if needed ------------------
    if actor_critic.actor.control_state_dim > 0:
        if "control_state" not in world_model and "decoder" in world_model:
            class DecoderStatePredictor(torch.nn.Module):
                def __init__(self, state_head: torch.nn.Module):
                    super().__init__()
                    self.state_head = state_head

                def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
                    z_flat = z.reshape(z.shape[0], -1) if z.dim() == 3 else z
                    feat = torch.cat([h, z_flat], dim=-1)
                    return self.state_head(feat)

            world_model["control_state"] = DecoderStatePredictor(world_model["decoder"].state_head).to(device)
            print(f"[Online] Bound DecoderStatePredictor to world_model['control_state'] (dim={actor_critic.actor.control_state_dim})", flush=True)
        actor_critic.require_imagination_ready = lambda: None

    # ---- Resume from online checkpoint or warm-start -----------------------
    start_env_steps = 0
    start_episodes = 0
    if resume_ckpt is not None:
        ckpt = resume_ckpt
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
    elif ac_ckpt is not None and "actor_critic" in ac_ckpt:
        print(f"[Online] Initializing Actor-Critic weights from {args.actor_checkpoint}", flush=True)
        actor_critic.load_state_dict(ac_ckpt["actor_critic"], strict=False)

    # ---- Environment configs -----------------------------------------------
    env_cfg = configs["env"]
    train_env_cfg = dict(env_cfg)

    eval_cfg_section = train_cfg.get("eval", {})
    eval_start_seed = int(eval_cfg_section.get("start_seed", 10000))
    eval_episodes = int(
        args.eval_episodes if args.eval_episodes is not None else eval_cfg_section.get("num_episodes", 20)
    )
    eval_env_cfg = dict(env_cfg)
    eval_env_cfg["start_seed"] = eval_start_seed
    eval_env_cfg["num_scenarios"] = eval_episodes
    eval_horizon = args.eval_horizon if args.eval_horizon is not None else eval_cfg_section.get("horizon")
    if eval_horizon is not None:
        if int(eval_horizon) <= 0:
            raise ValueError("eval horizon must be positive")
        eval_env_cfg["horizon"] = int(eval_horizon)
        eval_cfg_section["horizon"] = int(eval_horizon)

    demo_buffer = None
    demo_ratio = float(train_cfg.get("buffer", {}).get("demo_ratio", 0.0))
    if demo_ratio > 0:
        demo_path = Path(args.demo_buffer or (buffer_path if resume_path is None else Path(paths_cfg.get("buffer_dir", "data/replay_buffer")) / "bootstrap.npz"))
        if resume_path is not None and demo_path.resolve() == buffer_path.resolve():
            raise ValueError("--demo-buffer must be a separate protected bootstrap replay")
        demo_buffer = ReplayBuffer.load(demo_path)
        validate_world_model_metadata(wm_ckpt, image_shape=demo_buffer.image_shape, state_dim=demo_buffer.state_dim, action_dim=demo_buffer.action_dim)
        demo_buffer.sequence_length = buffer.sequence_length

    # ---- Create training environment ---------------------------------------
    print("[Online] Initializing MetaDrive environment...", flush=True)
    train_env = make_env(train_env_cfg)

    # ---- Logger ------------------------------------------------------------
    logger = Logger(
        experiment_name=train_cfg.get("experiment_name", "wm_metadrive") + "_online",
        log_dir=paths_cfg.get("log_dir", "logs"),
        wandb_cfg=train_cfg.get("wandb", {}),
    )

    if args.bc_scale is not None:
        train_cfg.setdefault("actor_critic", {})["bc_scale"] = float(args.bc_scale)
    if args.safe_collect:
        train_cfg.setdefault("train", {})["safe_collect"] = True

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
        demo_buffer=demo_buffer,
        safe_collect=bool(train_cfg.get("train", {}).get("safe_collect", False)),
    )
    trainer.env_steps = start_env_steps
    trainer.total_episodes = start_episodes

    # ---- Training hyperparams (CLI overrides > YAML defaults) --------------
    online_cfg = train_cfg.get("train", {})
    total_steps = int(args.total_steps if args.total_steps is not None else online_cfg.get("total_env_steps", 500_000))
    steps_per_iter = int(args.steps_per_iter if args.steps_per_iter is not None else online_cfg.get("steps_per_iter", 1000))
    wm_updates = int(args.wm_updates if args.wm_updates is not None else online_cfg.get("world_model_updates", 1))
    ac_updates = int(args.ac_updates if args.ac_updates is not None else online_cfg.get("actor_critic_updates", 1))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("wm_train", {}).get("batch_size", 8))
    eval_every = int(args.eval_every if args.eval_every is not None else online_cfg.get("eval_every", 10_000))
    ckpt_every = int(args.ckpt_every if args.ckpt_every is not None else online_cfg.get("checkpoint_every", 20_000))
    if steps_per_iter <= 0 or batch_size <= 0 or eval_episodes <= 0 or min(wm_updates, ac_updates, total_steps, eval_every, ckpt_every) < 0:
        raise ValueError("Invalid training counts/intervals")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    # Store effective CLI overrides in checkpoints as well as a readable manifest.
    online_cfg.update(total_env_steps=total_steps, steps_per_iter=steps_per_iter,
                      world_model_updates=wm_updates, actor_critic_updates=ac_updates,
                      eval_every=eval_every, checkpoint_every=ckpt_every)
    train_cfg["train"] = online_cfg
    train_cfg.setdefault("wm_train", {})["batch_size"] = batch_size
    train_cfg.setdefault("eval", {})["num_episodes"] = eval_episodes
    if eval_horizon is not None:
        train_cfg["eval"]["horizon"] = int(eval_horizon)
    train_cfg["device"] = str(device)
    manifest = {"configs": configs, "args": vars(args), "git_revision": revision, "git_dirty": dirty, "transition_contract": 2,
                "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                            "render_backend": os.environ.get("DATN_RENDER_BACKEND", "default"),
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    manifest_path = logger.log_dir / f"run_start_{start_env_steps:06d}.json"
    suffix = 1
    while manifest_path.exists():
        manifest_path = logger.log_dir / f"run_start_{start_env_steps:06d}_{suffix}.json"
        suffix += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (logger.log_dir / "resolved_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        trainer.train_env.close()
        logger.finish()


if __name__ == "__main__":
    main()
