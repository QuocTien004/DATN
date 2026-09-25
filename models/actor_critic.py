from __future__ import annotations

import copy
from dataclasses import dataclass
from math import log
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for width in hidden:
        layers.extend([nn.Linear(current, int(width)), nn.ELU()])
        current = int(width)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


def latent_feature(
    h: torch.Tensor,
    z: torch.Tensor,
    *,
    deter_dim: int,
    stoch_size: int,
) -> torch.Tensor:
    """Flatten categorical ``z`` and concatenate it with deterministic ``h``."""
    if h.ndim != 2:
        raise ValueError(f"h must be (B, deter_dim), got {tuple(h.shape)}")
    if h.shape[-1] != deter_dim:
        raise ValueError(f"Expected h dim {deter_dim}, got {h.shape[-1]}")
    if z.ndim == 3:
        z = z.reshape(z.shape[0], -1)
    elif z.ndim != 2:
        raise ValueError(f"z must be (B, G, C) or (B, G*C), got {tuple(z.shape)}")
    if z.shape[0] != h.shape[0] or z.shape[-1] != stoch_size:
        raise ValueError(
            f"Expected z ({h.shape[0]}, {stoch_size}), got {tuple(z.shape)}"
        )
    return torch.cat([h, z], dim=-1)


@dataclass(frozen=True)
class PolicyOutput:
    """One action sample and the statistics used by imagination training."""

    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class Actor(nn.Module):
    """Tanh-squashed Normal policy over the RSSM latent feature."""

    def __init__(
        self,
        cfg: dict[str, Any],
        deter_dim: int,
        stoch_size: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.deter_dim = int(deter_dim)
        self.stoch_size = int(stoch_size)
        self.action_dim = int(action_dim)
        hidden = [int(x) for x in cfg.get("actor_hidden", [512, 512])]
        self.net = _mlp(
            self.deter_dim + self.stoch_size,
            hidden,
            2 * self.action_dim,
        )

        self.min_std = float(cfg.get("min_std", 0.1))
        self.max_std = float(cfg.get("max_std", 1.0))
        self.mean_transform = str(cfg.get("mean_transform", "clamp"))
        self.std_transform = str(cfg.get("std_transform", "log_clamp"))
        # Keep historical checkpoints resumable; new configs opt into entropy
        # of the executed (tanh-squashed) action, not the unbounded Normal.
        self.entropy_mode = str(cfg.get("entropy_mode", "normal"))
        if self.entropy_mode not in {"normal", "squashed"}:
            raise ValueError("entropy_mode must be normal or squashed")
        if self.mean_transform not in {"clamp", "tanh"} or self.std_transform not in {"log_clamp", "sigmoid"}:
            raise ValueError("Unsupported actor distribution transform")
        if not 0.0 < self.min_std <= self.max_std:
            raise ValueError("Expected 0 < min_std <= max_std")

        low = torch.as_tensor(cfg.get("action_low", -1.0), dtype=torch.float32)
        high = torch.as_tensor(cfg.get("action_high", 1.0), dtype=torch.float32)
        low = low.expand(self.action_dim).clone()
        high = high.expand(self.action_dim).clone()
        if not torch.all(high > low):
            raise ValueError("Each action_high must be greater than action_low")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

        # Begin near zero steering/throttle with a configurable exploration scale.
        head = self.net[-1]
        assert isinstance(head, nn.Linear)
        nn.init.uniform_(head.weight, -1e-3, 1e-3)
        nn.init.zeros_(head.bias)
        init_std = min(max(float(cfg.get("init_std", 0.5)), self.min_std), self.max_std)
        with torch.no_grad():
            if self.std_transform == "sigmoid":
                if self.max_std <= self.min_std:
                    raise ValueError("sigmoid std requires min_std < max_std")
                fraction = min(max((init_std - self.min_std) / (self.max_std - self.min_std), 1e-4), 1.0 - 1e-4)
                head.bias[self.action_dim :].fill_(log(fraction / (1.0 - fraction)))
            else:
                head.bias[self.action_dim :].fill_(log(init_std))

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        feature = latent_feature(
            h,
            z,
            deter_dim=self.deter_dim,
            stoch_size=self.stoch_size,
        )
        mean, log_std = self.net(feature).chunk(2, dim=-1)
        # Legacy checkpoints use hard clamp; fresh policies use a smooth bound.
        mean = 2.0 * torch.tanh(mean / 2.0) if self.mean_transform == "tanh" else torch.clamp(mean, -2.0, 2.0)
        if self.std_transform == "sigmoid":
            std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(log_std)
        else:
            std = log_std.clamp(log(self.min_std), log(self.max_std)).exp()
        distribution = Normal(mean, std)
        pre_tanh = mean if deterministic else distribution.rsample()
        normalized_action = torch.tanh(pre_tanh)

        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        action = bias + scale * normalized_action

        # Change-of-variables correction for tanh and optional action rescaling.
        # Stable log(1 - tanh(x)^2), including near saturated actions.
        log_jacobian = scale.log() + 2.0 * (log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        log_prob = (
            distribution.log_prob(pre_tanh) - log_jacobian
        ).sum(dim=-1)
        # H(tanh(X)) = H(X) + E[log |J|]. The reparameterized sample gives
        # a gradient that discourages mean saturation as well as adjusting std.
        entropy = distribution.entropy().sum(dim=-1)
        if self.entropy_mode == "squashed":
            entropy = entropy + log_jacobian.sum(dim=-1)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            mean=bias + scale * torch.tanh(mean),
            std=std,
        )


class Critic(nn.Module):
    """Scalar state-value network over the same RSSM latent feature as Actor."""

    def __init__(self, cfg: dict[str, Any], deter_dim: int, stoch_size: int) -> None:
        super().__init__()
        self.deter_dim = int(deter_dim)
        self.stoch_size = int(stoch_size)
        hidden = [int(x) for x in cfg.get("critic_hidden", [512, 512])]
        self.net = _mlp(self.deter_dim + self.stoch_size, hidden, 1)

        # Zero-initialize the critic head for stable early value estimation
        head = self.net[-1]
        assert isinstance(head, nn.Linear)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feature = latent_feature(
            h,
            z,
            deter_dim=self.deter_dim,
            stoch_size=self.stoch_size,
        )
        return self.net(feature).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor and Critic operating on RSSM state ``{h, z}``."""

    def __init__(
        self,
        cfg: dict[str, Any],
        deter_dim: int,
        stoch_size: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        ac_cfg = cfg.get("actor_critic", cfg)
        self.cfg = dict(ac_cfg)
        self.deter_dim = int(deter_dim)
        self.stoch_size = int(stoch_size)
        self.action_dim = int(action_dim)
        self.actor = Actor(ac_cfg, deter_dim, stoch_size, action_dim)
        self.critic = Critic(ac_cfg, deter_dim, stoch_size)
        self.slow_critic = copy.deepcopy(self.critic)
        for parameter in self.slow_critic.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("ret_p95", torch.tensor(0.0))
        self.register_buffer("ret_p05", torch.tensor(0.0))
        self.register_buffer("ema_initialized", torch.tensor(0, dtype=torch.uint8))

    def act(
        self,
        state: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        return self.actor(state["h"], state["z"], deterministic=deterministic)

    def value(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(state["h"], state["z"])

    def slow_value(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.slow_critic(state["h"], state["z"])

    def update_slow_critic(self, tau: float = 0.02) -> None:
        """Polyak EMA update for Slow Critic parameters."""
        with torch.no_grad():
            for target_param, source_param in zip(
                self.slow_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        # Backward compatibility for checkpoints saved before slow_critic
        if not any(k.startswith("slow_critic.") for k in state_dict.keys()):
            res = super().load_state_dict(state_dict, strict=False)
            self.slow_critic.load_state_dict(self.critic.state_dict())
            for parameter in self.slow_critic.parameters():
                parameter.requires_grad_(False)
            return res
        res = super().load_state_dict(state_dict, strict=strict)
        for parameter in self.slow_critic.parameters():
            parameter.requires_grad_(False)
        return res
