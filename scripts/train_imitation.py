#!/usr/bin/env python
"""Experimental Actor warm-up on pure expert replay, with a frozen World Model.

This is supervised behavior cloning, NOT an RL update or an online resume.
Evaluate the output before using it to initialize any RL experiment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from models.actor_critic import ActorCritic
from models.world_model import build_world_model, load_world_model_state, validate_world_model_metadata
from training.batches import replay_batch_to_torch
from training.imitation import train_imitation_step
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.demonstrations import file_sha256, require_expert_manifest
from utils.online_state import require_current_contract
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-buffer", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_pilot.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--posterior", choices=["alternating", "deterministic", "sampled"], default="alternating")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-state", action="store_true", help="Experimental observed-state BC bridge; RL/imagination remains disabled")
    parser.add_argument("--control-state-source", choices=["observed", "predicted", "continuous"], default="observed",
                        help="Predicted requires --control-state and a WM checkpoint with the learned head")
    args = parser.parse_args()
    if args.control_state_source in {'predicted', 'continuous'} and not args.control_state:
        parser.error("model-state source requires --control-state")
    if min(args.updates, args.batch_size, args.save_every, args.lr) <= 0:
        parser.error("updates, batch-size, save-every and lr must be positive")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty --out-dir, not {out}")
    provenance = require_expert_manifest(args.expert_buffer)
    checkpoint = load_checkpoint(args.wm_checkpoint)
    require_current_contract(checkpoint)
    if checkpoint.get("env_cfg") != provenance.get("env_cfg"):
        raise ValueError("Expert environment config differs from the World Model's environment")
    config = load_config(args.config)
    replay = ReplayBuffer.load(args.expert_buffer)
    if not np.isfinite(replay.actions).all() or (np.abs(replay.actions) > 1).any():
        raise ValueError("Expert replay must contain bounded executed actions, not raw PPO means")
    replay.sequence_length = int(config.get("buffer", {}).get("sequence_length", 32))
    if not 0 <= args.burn_in < replay.sequence_length:
        parser.error("burn-in must be in [0, sequence_length)")
    validate_world_model_metadata(checkpoint, image_shape=replay.image_shape,
                                  state_dim=replay.state_dim, action_dim=replay.action_dim)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
    torch.set_num_threads(2)
    set_seed(args.seed)
    wm = build_world_model(checkpoint["wm_cfg"], replay.image_shape, replay.state_dim,
                           replay.action_dim, device, include_decoder=False)
    load_world_model_state(wm, checkpoint)
    for module in wm.values():
        module.eval()
    ac_cfg = dict(config["actor_critic"])
    if args.control_state:
        vectors = np.asarray(replay.states[:len(replay)])
        if len(vectors) < 2 or not np.isfinite(vectors).all():
            raise ValueError("Control normalization requires at least two finite training states")
        ac_cfg.update(control_state_dim=replay.state_dim,
                      control_state_source=args.control_state_source,
                      control_state_mean=vectors.mean(0).tolist(),
                      control_state_std=np.maximum(vectors.std(0, ddof=1), 1e-4).tolist())
    if args.control_state_source == "predicted" and "control_state" not in wm:
        raise ValueError("Predicted-state BC requires a WM checkpoint with control_state predictor")
    if args.control_state_source == 'continuous' and getattr(wm['rssm'], 'continuous', None) is None:
        raise ValueError('Continuous BC requires a continuous RSSM checkpoint')
    ac = ActorCritic(ac_cfg, wm["rssm"].deter_dim, wm["rssm"].stoch_size, replay.action_dim).to(device)
    optimizer = torch.optim.Adam(ac.actor.parameters(), lr=args.lr)
    out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    diff = subprocess.run(["git", "diff", "HEAD", "--", "models", "training", "scripts", "utils", "configs", "evaluation"], cwd=ROOT, text=True, capture_output=True)
    (out / "source.patch").write_text(diff.stdout, encoding="utf-8")
    # git diff omits untracked new modules; include their hashes explicitly.
    source_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for folder in ("models", "training", "utils", "scripts", "evaluation", "configs", "envs")
        for path in sorted((ROOT / folder).rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    metadata = {
        "method": "behavior_cloning_frozen_world_model", "arguments": vars(args),
        "expert_manifest": provenance, "wm_sha256": file_sha256(args.wm_checkpoint),
        "git_revision": revision.stdout.strip(), "python": sys.version,
        "torch": str(torch.__version__), "actor_critic_cfg": ac_cfg,
        "source_sha256": source_hashes,
        "note": "No expert in evaluation; Critic and policy variance head are not directly trained by BC.",
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for update in range(1, args.updates + 1):
            # Use the replay buffer's terminal-balanced sampling.
            batch = replay_batch_to_torch(replay.sample(args.batch_size), device)
            deterministic = args.posterior == "deterministic" or (args.posterior == "alternating" and update % 2 == 0)
            metrics = train_imitation_step(batch, wm, ac, optimizer,
                deterministic_posterior=deterministic, burn_in=args.burn_in)
            row = {"update": update, **metrics}
            log.write(json.dumps(row) + "\n")
            if update % 25 == 0 or update == args.updates:
                log.flush()
                print(json.dumps(row), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                # Include the original frozen WM, including Decoder, so a future
                # explicitly warm-started RL run can use the same latent basis.
                payload = {key: checkpoint[key] for key in (
                    "models", "wm_cfg", "env_cfg", "image_shape", "state_dim", "action_dim", "transition_contract",
                )}
                kind = "behavior_cloning_control_diagnostic" if ac.actor.control_state_dim else "behavior_cloning"
                if ac.actor.control_state_source == "predicted":
                    kind = "predicted_control_bc_diagnostic"
                if ac.actor.control_state_source == 'continuous':
                    kind = 'continuous_control_bc_diagnostic'
                payload.update(checkpoint_kind=kind, bc_updates=update,
                    control_input=ac.actor.control_state_source if ac.actor.control_state_dim else "latent",
                    actor_critic=ac.state_dict(), actor_critic_cfg=ac_cfg,
                    bc_optimizer=optimizer.state_dict(), imitation=metadata)
                save_checkpoint(out / f"bc_{update:06d}.pt", payload)
                save_checkpoint(out / "latest.pt", payload)


if __name__ == "__main__":
    main()
