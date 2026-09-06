from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from models.actor_critic import ActorCritic
from training.batches import replay_batch_to_torch
from training.train_agent import posterior_start_states, train_actor_critic_step
from utils.checkpoint import save_checkpoint
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer


class Trainer:
    """Offline Actor-Critic trainer using a pretrained frozen World Model."""

    def __init__(
        self,
        configs: dict[str, Any],
        buffer: ReplayBuffer,
        logger: Logger,
        device: str = "cpu",
        *,
        world_model: dict[str, torch.nn.Module] | None = None,
        actor_critic: ActorCritic | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        global_step: int = 0,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.configs = configs
        self.buffer = buffer
        self.logger = logger
        self.device = torch.device(device)
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.global_step = int(global_step)
        self.checkpoint_metadata = dict(checkpoint_metadata or {})

    def _require_components(self) -> tuple[
        dict[str, torch.nn.Module],
        ActorCritic,
        torch.optim.Optimizer,
        torch.optim.Optimizer,
    ]:
        missing = [
            name
            for name, value in (
                ("world_model", self.world_model),
                ("actor_critic", self.actor_critic),
                ("actor_optimizer", self.actor_optimizer),
                ("critic_optimizer", self.critic_optimizer),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(f"Trainer is missing required components: {missing}")
        assert self.world_model is not None
        assert self.actor_critic is not None
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None
        return (
            self.world_model,
            self.actor_critic,
            self.actor_optimizer,
            self.critic_optimizer,
        )

    def fit(
        self,
        *,
        updates: int | None = None,
        batch_size: int | None = None,
        log_every: int | None = None,
        checkpoint_every: int | None = None,
    ) -> dict[str, float]:
        """Train Actor-Critic from replay posteriors and latent imagination."""
        world_model, actor_critic, actor_optimizer, critic_optimizer = (
            self._require_components()
        )
        ac_cfg = self.configs["train"].get("actor_critic", {})
        updates = int(updates if updates is not None else ac_cfg.get("updates", 10_000))
        batch_size = int(
            batch_size if batch_size is not None else ac_cfg.get("batch_size", 8)
        )
        log_every = int(
            log_every if log_every is not None else ac_cfg.get("log_every", 20)
        )
        checkpoint_every = int(
            checkpoint_every
            if checkpoint_every is not None
            else ac_cfg.get("checkpoint_every", 500)
        )
        max_start_states = int(ac_cfg.get("max_start_states", 64))
        if updates <= 0 or batch_size <= 0:
            raise ValueError("updates and batch_size must be positive")

        for module in world_model.values():
            module.eval()
        actor_critic.train()
        final_metrics: dict[str, float] = {}
        end_step = self.global_step + updates
        for step in range(self.global_step + 1, end_step + 1):
            try:
                raw_batch = self.buffer.sample(batch_size, include_next=False)
            except MemoryError:
                gc.collect()
                raw_batch = self.buffer.sample(batch_size, include_next=False)
            batch = replay_batch_to_torch(raw_batch, self.device)
            del raw_batch
            start_states = posterior_start_states(
                batch,
                world_model,
                max_states=max_start_states,
            )
            del batch
            final_metrics = train_actor_critic_step(
                start_states,
                world_model,
                actor_critic,
                actor_optimizer,
                critic_optimizer,
                ac_cfg,
            )
            self.global_step = step

            if step == 1 or step % max(log_every, 1) == 0:
                self.logger.log(final_metrics, step=step)
            if checkpoint_every > 0 and step % checkpoint_every == 0:
                self.save(self._checkpoint_dir())

        self.save(self._checkpoint_dir())
        return final_metrics

    def _checkpoint_dir(self) -> Path:
        root = Path(
            self.configs["train"].get("paths", {}).get(
                "checkpoint_dir", "checkpoints"
            )
        )
        return root / "actor_critic"

    def save(self, directory: str | Path) -> Path:
        """Save a numbered Actor-Critic checkpoint and refresh ``latest.pt``."""
        _world_model, actor_critic, actor_optimizer, critic_optimizer = (
            self._require_components()
        )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.global_step,
            "actor_critic": actor_critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "actor_critic_cfg": self.configs["train"].get("actor_critic", {}),
            "deter_dim": actor_critic.deter_dim,
            "stoch_size": actor_critic.stoch_size,
            "action_dim": actor_critic.action_dim,
            **self.checkpoint_metadata,
        }
        numbered = directory / f"agent_step_{self.global_step:06d}.pt"
        save_checkpoint(numbered, payload)
        save_checkpoint(directory / "latest.pt", payload)
        return numbered
