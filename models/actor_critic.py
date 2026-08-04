from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Policy (Actor) + value (Critic) over latent state (h, z).

    Trained via imagination rollouts from the World Model (Phase C).
    """

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self._dummy = nn.Parameter(torch.zeros(1))
        _ = (deter_dim, stoch_size)

    def actor(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement Actor in Phase C.")

    def critic(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement Critic in Phase C.")
