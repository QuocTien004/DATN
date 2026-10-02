"""Observation-free readout of continuous control state from RSSM features.

The same readout accepts posterior latents during real interaction and prior
latents during imagination. Having this module is NOT evidence of RL readiness.
"""
from __future__ import annotations

import torch
from torch import nn

from .actor_critic import latent_feature


class ControlStatePredictor(nn.Module):
    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int, state_dim: int):
        super().__init__()
        self.deter_dim, self.stoch_size, self.state_dim = deter_dim, stoch_size, state_dim
        for name, dim, default in (
            ("feature_mean", deter_dim + stoch_size, 0.),
            ("feature_std", deter_dim + stoch_size, 1.),
            ("state_mean", state_dim, 0.), ("state_std", state_dim, 1.),
        ):
            value = torch.as_tensor(cfg.get(name, [default] * dim), dtype=torch.float32)
            if value.shape != (dim,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain {dim} finite values")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, value.clone())
        widths = [deter_dim + stoch_size, *cfg.get("hidden", [256, 256]), state_dim]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def normalized_prediction(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feature = latent_feature(h, z, deter_dim=self.deter_dim, stoch_size=self.stoch_size)
        return self.net((feature - self.feature_mean) / self.feature_std)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.normalized_prediction(h, z) * self.state_std + self.state_mean


def with_predicted_control(state: dict, predictor: nn.Module) -> dict:
    """Ignore any supplied observed control; preserve gradients through h/z."""
    return {"h": state["h"], "z": state["z"],
            "control": predictor(state["h"], state["z"])}
