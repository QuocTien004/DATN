from __future__ import annotations

import torch
import torch.nn as nn


class RewardPredictor(nn.Module):
    """Predict reward from (h_t, z_t)."""

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self._dummy = nn.Parameter(torch.zeros(1))
        _ = (deter_dim, stoch_size)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement RewardPredictor in Phase B.")


class ContinuePredictor(nn.Module):
    """Predict episode continuation probability from (h_t, z_t)."""

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self._dummy = nn.Parameter(torch.zeros(1))
        _ = (deter_dim, stoch_size)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement ContinuePredictor in Phase B.")
