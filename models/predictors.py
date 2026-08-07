from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True)])
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class RewardPredictor(nn.Module):
    """Predict scalar reward from (h_t, z_t). Output: (B, 1)."""

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        hidden = list(cfg.get("predictors", {}).get("reward_hidden", [256, 256]))
        self.net = _mlp(int(deter_dim) + int(stoch_size), hidden, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.reshape(z.shape[0], -1)
        return self.net(torch.cat([h, z], dim=-1))


class ContinuePredictor(nn.Module):
    """
    Predict episode continuation logit from (h_t, z_t).
    Output: (B, 1) logits — use sigmoid for probability of continue=True.
    """

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        hidden = list(cfg.get("predictors", {}).get("continue_hidden", [256, 256]))
        self.net = _mlp(int(deter_dim) + int(stoch_size), hidden, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.reshape(z.shape[0], -1)
        return self.net(torch.cat([h, z], dim=-1))
