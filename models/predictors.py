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


class ModularBoundaryHead(nn.Module):
    """Modular anti-forgetting Continue & Reward head combining frozen `old_hz` with P1/P2 `c`.

    - Preserves `old_continue(h, z)` so visual crash and finish knowledge is never forgotten.
    - Adds a compact physical road-boundary detector on `c` (19-D normalized state).
    - Uses a `RewardPredictor(h, z, c)` trained on `train` for smooth driving rewards, combined with
      a negative failure penalty when either a road-boundary violation or a negative-reward crash is detected.
    """

    def __init__(
        self,
        base_reward_hzc: nn.Module,
        old_reward_hz: nn.Module,
        old_continue_hz: nn.Module,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        super().__init__()
        import copy
        self.reward_hzc = copy.deepcopy(base_reward_hzc)
        self.old_reward = copy.deepcopy(old_reward_hz).eval().requires_grad_(False)
        self.old_continue = copy.deepcopy(old_continue_hz).eval().requires_grad_(False)
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer("state_std", state_std.detach().clone())

        self.alpha_heading = nn.Parameter(torch.tensor(0.16), requires_grad=False)
        self.edge_threshold = nn.Parameter(torch.tensor(0.0725), requires_grad=False)
        self.edge_sharpness = nn.Parameter(torch.tensor(120.0), requires_grad=False)

    def _corner_clearance_and_speed(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = c * self.state_std + self.state_mean
        s0, s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        min_edge = torch.minimum(s0, s1)
        head_dev = (s2 - 0.5).abs()
        corner_clearance = min_edge - self.alpha_heading * head_dev
        return corner_clearance, s3

    def forward(self, h: torch.Tensor, z: torch.Tensor, *, c: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if c is None:
            raise ValueError('ModularBoundaryHead requires c')
        with torch.no_grad():
            old_r = self.old_reward(h, z).squeeze(-1)
            old_c_logit = self.old_continue(h, z).squeeze(-1)

        corner_clearance, s3 = self._corner_clearance_and_speed(c)
        # Stationary vehicles (s3 <= 0.05) cannot cross road boundaries
        moving_mask = (s3 > 0.05).float()
        raw_boundary_logit = self.edge_sharpness * (corner_clearance - self.edge_threshold)
        # If stationary, force boundary continue logit to +10.0 (continue=True)
        boundary_cont_logit = moving_mask * raw_boundary_logit + (1.0 - moving_mask) * 10.0

        # Terminal if EITHER old_hz predicts terminal OR moving vehicle violates road boundary
        cont_logit = torch.minimum(old_c_logit, boundary_cont_logit)

        # Base reward from h/z/c predictor
        if getattr(self.reward_hzc, 'continuous_state_dim', 0) > 0:
            base_r = self.reward_hzc(h, z, c=c).squeeze(-1)
        else:
            base_r = self.reward_hzc(h, z).squeeze(-1)
        
        # If boundary violation occurs (or old_hz detects a crash with old_r < 0), ensure negative reward sign
        is_boundary_fail = (boundary_cont_logit <= 0.0).float()
        is_crash_fail = ((old_c_logit <= 0.0) & (old_r < 0.0)).float()
        fail_flag = torch.maximum(is_boundary_fail, is_crash_fail)
        rew_pred = (1.0 - fail_flag) * base_r + fail_flag * torch.minimum(base_r, torch.tensor(-0.6, device=h.device))
        
        return {"reward": rew_pred, "continue": cont_logit}
