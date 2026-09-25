"""Online RL training loop: Collect → Train WM → Train AC → Evaluate."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from envs.metadrive_wrapper import MetaDriveImageEnv, make_env
from evaluation.evaluate import LatentActorPolicy, evaluate_policy
from models.actor_critic import ActorCritic
from training.batches import replay_batch_to_torch
from training.train_agent import posterior_start_states, train_actor_critic_step
from training.train_world_model import train_world_model_step
from utils.checkpoint import save_checkpoint
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer, Transition


class OnlineTrainer:
    """Orchestrates the DreamerV3 online RL loop.

    Lifecycle per iteration:
        1. **Collect** N env steps with stochastic Actor policy
        2. **Train World Model** for M gradient updates
        3. **Train Actor-Critic** for K imagination updates
        4. **Evaluate** periodically on held-out seeds
        5. **Checkpoint** periodically
    """

    def __init__(
        self,
        configs: dict[str, Any],
        buffer: ReplayBuffer,
        world_model: dict[str, nn.Module],
        actor_critic: ActorCritic,
        wm_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        logger: Logger,
        device: torch.device,
        train_env: MetaDriveImageEnv,
        train_env_cfg: dict[str, Any],
        eval_env_cfg: dict[str, Any],
        demo_buffer: ReplayBuffer | None = None,
    ) -> None:
        self.configs = configs
        self.buffer = buffer
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.wm_optimizer = wm_optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.logger = logger
        self.device = device
        self.train_env = train_env
        self.train_env_cfg = dict(train_env_cfg)
        self.eval_env_cfg = dict(eval_env_cfg)
        self.demo_buffer = demo_buffer
        buffer_cfg = configs.get("train", {}).get("buffer", {})
        self.demo_ratio = float(buffer_cfg.get("demo_ratio", 0.0))
        self.term_ratio = float(buffer_cfg.get("term_ratio", 0.25))
        if not 0 <= self.demo_ratio <= 1:
            raise ValueError("buffer.demo_ratio must be in [0, 1]")

        # Counters
        self.env_steps: int = 0
        self.total_episodes: int = 0

        # Running episode state (persists across collect calls)
        self._obs: dict[str, np.ndarray] | None = None
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._is_first: bool = True
        self._policy: LatentActorPolicy | None = None

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------
    def _make_policy(self, *, deterministic: bool = False) -> LatentActorPolicy:
        return LatentActorPolicy(
            self.world_model["encoder"],
            self.world_model["rssm"],
            self.actor_critic,
            device=self.device,
            deterministic=deterministic,
        )

    def _refresh_policy(self) -> None:
        """Recreate the exploration policy so it uses the latest weights."""
        old_latent = getattr(self._policy, "_latent", None)
        old_action = getattr(self._policy, "_previous_action", None)
        self._policy = self._make_policy(deterministic=False)
        # Preserve recurrent latent state if an episode is actively running mid-way
        if not self._is_first and old_latent is not None:
            self._policy._latent = old_latent
            self._policy._previous_action = old_action

    # ------------------------------------------------------------------
    # 1. Collect
    # ------------------------------------------------------------------
    def collect(self, num_steps: int) -> dict[str, float]:
        """Collect *num_steps* transitions using the stochastic Actor."""
        if self._policy is None:
            self._refresh_policy()
        policy = self._policy
        assert policy is not None

        # Lazy first reset
        if self._obs is None:
            self._obs, _ = self.train_env.reset()
            self._is_first = True
            policy.reset()

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        successes = 0
        crashes = 0
        route_completions: list[float] = []
        action_sum = np.zeros(self.buffer.action_dim, dtype=np.float64)
        action_saturated = np.zeros(self.buffer.action_dim, dtype=np.float64)
        diagnostic_sums = {key: 0.0 for key in ("reward_base", "reward_shaping", "lateral_available", "lateral_factor")}
        started = time.monotonic()
        idle_steps = 0

        for _ in range(num_steps):
            action = policy(self._obs)
            # The actor already samples its policy. Extra noise is opt-in.
            noise_scale = float(self.configs["train"].get("train", {}).get("action_noise_std", 0.0))
            noisy_action = np.clip(
                action + np.random.normal(0.0, noise_scale, size=action.shape).astype(np.float32),
                -1.0,
                1.0,
            )
            # Update policy recurrent memory with actual executed action
            policy._previous_action = torch.as_tensor(
                noisy_action, device=self.device, dtype=torch.float32
            ).reshape(1, -1)

            next_obs, reward, terminated, truncated, info = self.train_env.step(
                noisy_action
            )
            done = bool(terminated or truncated)
            idle_steps += float(info.get("velocity", float("inf"))) < 1.0
            action_sum += noisy_action
            action_saturated += np.abs(noisy_action) > 0.9
            for key in diagnostic_sums:
                diagnostic_sums[key] += float(info.get(key, 0.0))

            self.buffer.add(
                Transition(
                    image=self._obs["image"],
                    state=self._obs["state"],
                    action=np.asarray(noisy_action, dtype=np.float32).reshape(-1),
                    reward=float(reward),
                    done=done,
                    next_image=next_obs["image"],
                    next_state=next_obs["state"],
                    terminated=bool(terminated),
                ),
                is_first=self._is_first,
            )

            self._episode_return += float(reward)
            self._episode_length += 1
            self._is_first = False
            self.env_steps += 1

            if done:
                episode_returns.append(self._episode_return)
                episode_lengths.append(self._episode_length)
                if info.get("arrive_dest", False):
                    successes += 1
                if (
                    info.get("crash", False)
                    or info.get("crash_vehicle", False)
                    or info.get("crash_object", False)
                    or info.get("out_of_road", False)
                ):
                    crashes += 1
                route_completions.append(
                    float(info.get("route_completion", 0.0))
                )
                self.total_episodes += 1

                # Episode boundary reset
                self._obs, _ = self.train_env.reset()
                self._is_first = True
                self._episode_return = 0.0
                self._episode_length = 0
                policy.reset()
            else:
                self._obs = next_obs

        n_ep = len(episode_returns)
        return {
            "collected_steps": num_steps,
            "total_env_steps": self.env_steps,
            "completed_episodes": n_ep,
            "total_episodes": self.total_episodes,
            "mean_return": float(np.mean(episode_returns)) if n_ep else 0.0,
            "mean_length": float(np.mean(episode_lengths)) if n_ep else 0.0,
            "success_rate": successes / max(n_ep, 1),
            "crash_rate": crashes / max(n_ep, 1),
            "route_completion": (
                float(np.mean(route_completions)) if route_completions else 0.0
            ),
            "buffer_size": len(self.buffer),
            "idle_fraction": idle_steps / max(num_steps, 1),
            "collection_seconds": time.monotonic() - started,
            "steering_mean": float(action_sum[0] / max(num_steps, 1)),
            "throttle_mean": float(action_sum[1] / max(num_steps, 1)),
            "steering_saturation": float(action_saturated[0] / max(num_steps, 1)),
            "throttle_saturation": float(action_saturated[1] / max(num_steps, 1)),
            **{key + "_mean": value / max(num_steps, 1) for key, value in diagnostic_sums.items()},
        }

    # ------------------------------------------------------------------
    # 2. Train World Model
    # ------------------------------------------------------------------
    def _sample_batch(self, batch_size: int, *, include_next: bool = False) -> dict[str, np.ndarray]:
        n_demo = int(batch_size * self.demo_ratio) if self.demo_buffer is not None else 0
        parts = []
        for source, count in ((self.buffer, batch_size - n_demo), (self.demo_buffer, n_demo)):
            if source is not None and count:
                parts.append(source.sample(count, include_next=include_next, term_ratio=self.term_ratio))
        return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}

    def train_wm(self, num_updates: int, batch_size: int) -> dict[str, float]:
        """Run *num_updates* gradient steps on the World Model."""
        wm_cfg = self.configs["world_model"]
        accum: dict[str, float] = {}

        for module in self.world_model.values():
            module.train()

        for _ in range(num_updates):
            try:
                raw = self._sample_batch(batch_size, include_next=True)
            except MemoryError:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raw = self._sample_batch(batch_size, include_next=True)
            batch = replay_batch_to_torch(raw, self.device)
            del raw
            metrics = train_world_model_step(
                batch, self.world_model, self.wm_optimizer, wm_cfg
            )
            del batch
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v

        # After WM update, refresh collection policy (picks up new weights)
        self._refresh_policy()
        avg = {f"wm/{k}": v / max(num_updates, 1) for k, v in accum.items()}
        return avg

    # ------------------------------------------------------------------
    # 3. Train Actor-Critic
    # ------------------------------------------------------------------
    def train_ac(self, num_updates: int, batch_size: int) -> dict[str, float]:
        """Run *num_updates* latent-imagination AC gradient steps."""
        train_cfg = self.configs.get("train", {})
        ac_cfg = train_cfg.get("actor_critic", train_cfg)
        max_start = int(ac_cfg.get("max_start_states", 64))
        accum: dict[str, float] = {}

        # Freeze WM for imagination
        for module in self.world_model.values():
            module.eval()
        self.actor_critic.train()

        for _ in range(num_updates):
            try:
                raw = self._sample_batch(batch_size)
            except MemoryError:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raw = self._sample_batch(batch_size)
            batch = replay_batch_to_torch(raw, self.device)
            del raw
            start_states = posterior_start_states(
                batch, self.world_model, max_states=max_start
            )
            del batch
            metrics = train_actor_critic_step(
                start_states,
                self.world_model,
                self.actor_critic,
                self.actor_optimizer,
                self.critic_optimizer,
                ac_cfg,
            )
            del start_states
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v

        # Refresh policy with updated actor weights
        self._refresh_policy()
        avg = {f"ac/{k}": v / max(num_updates, 1) for k, v in accum.items()}
        return avg

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        num_episodes: int,
        start_seed: int = 10000,
    ) -> dict[str, float]:
        """Evaluate on held-out seeds using a **separate** MetaDrive instance.

        MetaDrive cannot reliably run two instances concurrently, so we
        close the training env, run evaluation, then reopen the training env.

        The eval env is configured with the matching seed range, then reset with
        each explicit seed so every evaluation covers the same scenarios once.
        """
        # Save and close training env
        self.train_env.close()

        eval_env = make_env(dict(self.eval_env_cfg, start_seed=start_seed, num_scenarios=num_episodes))
        try:
            policy = self._make_policy(deterministic=True)

            # Inline evaluation loop (avoids seed range conflicts)
            from evaluation.metrics import (
                aggregate_episode_metrics,
                compute_episode_metrics,
            )

            episode_metrics: list[dict[str, float]] = []
            for _ep in range(num_episodes):
                obs, info = eval_env.reset(seed=start_seed + _ep)
                policy.reset()
                info_history = [info]
                crashed = False
                out_of_road = False
                crash_vehicle = False
                success = False
                done = False
                episode_return = 0.0
                episode_length = 0

                while not done:
                    action = policy(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(
                        action
                    )
                    info_history.append(info)
                    crashed = crashed or bool(
                        info.get("crash", False)
                        or info.get("crash_vehicle", False)
                        or info.get("crash_object", False)
                        or info.get("out_of_road", False)
                    )
                    out_of_road = out_of_road or bool(
                        info.get("out_of_road", False)
                    )
                    crash_vehicle = crash_vehicle or bool(
                        info.get("crash_vehicle", False)
                        or info.get("crash_object", False)
                    )
                    success = success or bool(
                        info.get("arrive_dest", False)
                    )
                    episode_return += float(reward)
                    episode_length += 1
                    done = bool(terminated or truncated)

                episode_metrics.append(
                    compute_episode_metrics(
                        info_history,
                        crashed=crashed,
                        success=success,
                        out_of_road=out_of_road,
                        crash_vehicle=crash_vehicle,
                        episode_return=episode_return,
                        episode_length=episode_length,
                    )
                )
                episode_metrics[-1]["seed"] = start_seed + _ep
                episode_metrics[-1]["terminated"] = bool(terminated)
                episode_metrics[-1]["truncated"] = bool(truncated)
                print(f"[Eval] seed={start_seed + _ep} steps={episode_length} route={episode_metrics[-1].get('route_completion', 0):.4f} success={success}", flush=True)

            metrics = aggregate_episode_metrics(episode_metrics)
            output = self.logger.log_dir / f"eval_step_{self.env_steps:06d}.json"
            output.write_text(json.dumps({"env_steps": self.env_steps, "horizon": self.eval_env_cfg.get("horizon", 1000), "aggregate": metrics, "episodes": episode_metrics}, indent=2), encoding="utf-8")
        finally:
            eval_env.close()

        # Reopen training env and invalidate running episode
        self.train_env = make_env(self.train_env_cfg)
        self._obs = None
        self._is_first = True
        self._episode_return = 0.0
        self._episode_length = 0
        self._refresh_policy()

        return {f"eval/{k}": v for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # 5. Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> Path:
        """Persist all models, optimizers, and training counters."""
        ckpt_dir = Path(
            self.configs.get("train", {})
            .get("paths", {})
            .get("checkpoint_dir", "checkpoints")
        ) / "online"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        replay_path = ckpt_dir / "online_buffer.npz"
        self.buffer.save(replay_path, checkpoint_step=self.env_steps)

        payload: dict[str, Any] = {
            "env_steps": self.env_steps,
            "transition_contract": 2,
            "replay_path": replay_path.name,
            "train_cfg": self.configs.get("train", {}),
            "actor_critic_cfg": self.actor_critic.cfg,
            "total_episodes": self.total_episodes,
            "models": {
                name: m.state_dict() for name, m in self.world_model.items()
            },
            "actor_critic": self.actor_critic.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "wm_cfg": self.configs.get("world_model", {}),
            "env_cfg": self.configs.get("env", {}),
            "image_shape": self.buffer.image_shape,
            "state_dim": self.buffer.state_dim,
            "action_dim": self.buffer.action_dim,
            "deter_dim": self.actor_critic.deter_dim,
            "stoch_size": self.actor_critic.stoch_size,
        }

        numbered = ckpt_dir / f"online_step_{self.env_steps:06d}.pt"
        save_checkpoint(numbered, payload)
        save_checkpoint(ckpt_dir / "latest.pt", payload)

        # Numbered model snapshots can be evaluated independently. Resume uses
        # latest.pt + online_buffer.npz, whose env_steps must match.

        print(
            f"[Checkpoint] env_step={self.env_steps} -> {numbered}",
            flush=True,
        )
        return numbered

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        total_steps: int,
        steps_per_iter: int = 1000,
        wm_updates: int = 1,
        ac_updates: int = 1,
        batch_size: int = 8,
        eval_every: int = 10000,
        eval_episodes: int = 20,
        eval_start_seed: int = 10000,
        checkpoint_every: int = 20000,
    ) -> None:
        """Run the full online RL training loop."""
        print(
            f"[Online Training] Start: "
            f"steps={self.env_steps}/{total_steps}, "
            f"steps_per_iter={steps_per_iter}, "
            f"wm_updates={wm_updates}, ac_updates={ac_updates}, "
            f"eval_every={eval_every}, ckpt_every={checkpoint_every}",
            flush=True,
        )

        while self.env_steps < total_steps:
            iter_steps = min(steps_per_iter, total_steps - self.env_steps)
            prev_steps = self.env_steps

            # 1. Collect
            rollout = self.collect(iter_steps)

            # 2. Train World Model
            wm_metrics = self.train_wm(wm_updates, batch_size)

            # 3. Train Actor-Critic
            ac_metrics = self.train_ac(ac_updates, batch_size)

            # 4. Log
            all_metrics: dict[str, float] = {}
            all_metrics.update(
                {f"rollout/{k}": v for k, v in rollout.items()}
            )
            all_metrics.update(wm_metrics)
            all_metrics.update(ac_metrics)
            self.logger.log(all_metrics, step=self.env_steps)

            # 5. Evaluate
            if (
                eval_every > 0
                and self.env_steps >= eval_every
                and (prev_steps // eval_every) < (self.env_steps // eval_every)
            ):
                print(
                    f"\n--- Evaluation at env_step={self.env_steps} "
                    f"({eval_episodes} episodes) ---",
                    flush=True,
                )
                eval_metrics = self.evaluate(
                    eval_episodes, start_seed=eval_start_seed
                )
                self.logger.log(eval_metrics, step=self.env_steps)
                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}", flush=True)
                print("--- End Evaluation ---\n", flush=True)

            # 6. Checkpoint
            if (
                checkpoint_every > 0
                and self.env_steps >= checkpoint_every
                and (prev_steps // checkpoint_every)
                < (self.env_steps // checkpoint_every)
            ):
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()
        print(
            f"[Online Training] Done. Total env steps: {self.env_steps}",
            flush=True,
        )
