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


def _feature(h: torch.Tensor, z: torch.Tensor, c: torch.Tensor | None, dim: int) -> torch.Tensor:
    if z.dim() == 3:
        z = z.flatten(1)
    parts = [h, z]
    if dim:
        if c is None or c.shape != (h.shape[0], dim) or not torch.isfinite(c).all():
            raise ValueError('Control-conditioned predictor requires finite model-generated c')
        parts.append(c)
    elif c is not None:
        raise ValueError('Continuous state supplied to a legacy predictor')
    return torch.cat(parts, dim=-1)


def predict_from_state(predictor: nn.Module, state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use the learned RSSM state, never observed/future GT control as fallback."""
    kwargs = {'c': state.get('c')} if getattr(predictor, 'continuous_state_dim', 0) else {}
    return predictor(state['h'], state['z'], **kwargs)


class RewardPredictor(nn.Module):
    """Predict scalar reward from (h_t, z_t). Output: (B, 1)."""

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        hidden = list(cfg.get("predictors", {}).get("reward_hidden", [256, 256]))
        self.continuous_state_dim = int(cfg.get('predictors', {}).get('continuous_state_dim', 0))
        if self.continuous_state_dim < 0:
            raise ValueError('continuous_state_dim must be nonnegative')
        self.net = _mlp(int(deter_dim) + int(stoch_size) + self.continuous_state_dim, hidden, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor, *, c: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(_feature(h, z, c, self.continuous_state_dim))


class ContinuePredictor(nn.Module):
    """
    Predict episode continuation logit from (h_t, z_t).
    Output: (B, 1) logits — use sigmoid for probability of continue=True.
    """

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        hidden = list(cfg.get("predictors", {}).get("continue_hidden", [256, 256]))
        self.continuous_state_dim = int(cfg.get('predictors', {}).get('continuous_state_dim', 0))
        if self.continuous_state_dim < 0:
            raise ValueError('continuous_state_dim must be nonnegative')
        self.net = _mlp(int(deter_dim) + int(stoch_size) + self.continuous_state_dim, hidden, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor, *, c: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(_feature(h, z, c, self.continuous_state_dim))
