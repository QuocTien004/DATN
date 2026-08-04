from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Reconstruct observation (image / state) from (h_t, z_t)."""

    def __init__(self, cfg: dict, deter_dim: int, stoch_size: int, image_shape: tuple) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_shape = image_shape
        self._dummy = nn.Parameter(torch.zeros(1))
        _ = (deter_dim, stoch_size)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Implement Decoder in Phase B.")
