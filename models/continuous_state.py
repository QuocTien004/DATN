"""Opt-in continuous RSSM branch, with distinct filtering and transition paths.

The coordinates are supervised normalized control state. Observations enter only
through the learned encoder embedding; imagined steps accept only c and action.
This is a structural state variable, not a readout of the categorical latent.
"""
from __future__ import annotations

import torch
from torch import nn


class ContinuousState(nn.Module):
    def __init__(self, cfg: dict, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        self.dim = int(cfg['dim'])
        self.action_dim = int(action_dim)
        self.embed_start = int(cfg.get('embed_start', 0))
        self.embed_stop = int(cfg.get('embed_stop', embed_dim))
        if self.dim <= 0 or not 0 <= self.embed_start < self.embed_stop <= embed_dim:
            raise ValueError('Invalid continuous state dimension or embedding slice')
        width = self.embed_stop - self.embed_start
        for name, size, default in [('embed_mean', width, 0.), ('embed_std', width, 1.),
                                    ('state_mean', self.dim, 0.), ('state_std', self.dim, 1.)]:
            value = torch.as_tensor(cfg.get(name, [default] * size), dtype=torch.float32)
            if value.shape != (size,) or not torch.isfinite(value).all() or ('std' in name and (value <= 0).any()):
                raise ValueError(f'Invalid continuous {name}')
            self.register_buffer(name, value.clone())
        layers = []
        for hidden in cfg.get('posterior_hidden', [128, 128]):
            layers.extend([nn.Linear(width, int(hidden)), nn.ELU()])
            width = int(hidden)
        layers.append(nn.Linear(width, self.dim))
        self.posterior = nn.Sequential(*layers)
        width = self.dim + self.action_dim
        layers = []
        for hidden in cfg.get('prior_hidden', [128, 128]):
            layers.extend([nn.Linear(width, int(hidden)), nn.ELU()])
            width = int(hidden)
        layers.append(nn.Linear(width, self.dim))
        self.transition = nn.Sequential(*layers)
        # Start at persistence; learn changes under executed, bounded actions.
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def observe(self, embedding: torch.Tensor) -> torch.Tensor:
        x = embedding[:, self.embed_start:self.embed_stop]
        return self.posterior((x - self.embed_mean) / self.embed_std)

    def imagine(self, previous: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if previous.ndim != 2 or previous.shape[-1] != self.dim:
            raise ValueError('Expected continuous state (B, dim)')
        if action.shape != (previous.shape[0], self.action_dim):
            raise ValueError('Expected action (B, action_dim)')
        return previous + self.transition(torch.cat([previous, action], -1))

    def decode(self, continuous: torch.Tensor) -> torch.Tensor:
        """Fixed de-normalization, not another learned decoder on h/z."""
        return continuous * self.state_std + self.state_mean

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.state_mean) / self.state_std


def with_continuous_control(state: dict, rssm: nn.Module) -> dict:
    if rssm.continuous is None or 'c' not in state:
        raise ValueError('Model-state control requires the continuous RSSM state; refusing GT fallback')
    return dict(state, control=rssm.continuous.decode(state['c']))
