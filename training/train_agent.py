from __future__ import annotations

from typing import Any

import torch


def train_actor_critic_step(
    start_states: Any,
    world_model: dict[str, torch.nn.Module],
    actor_critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """
    One Actor-Critic update via imagination rollouts inside the World Model.

    Phase C: implement imagination horizon loop + actor/critic losses.
    """
    raise NotImplementedError("Implement train_actor_critic_step in Phase C.")
