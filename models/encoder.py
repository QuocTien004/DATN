from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Dual-branch encoder: CNN(image) + MLP(state) → embedding e_t.

    Phase A skeleton — implement forward() in Phase B.
    """

    def __init__(self, cfg: dict, state_dim: int, image_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        self.image_shape = image_shape  # (H, W, C)
        # TODO(Phase B): build CNN + MLP from cfg["encoder"]
        embed_dim = int(cfg.get("encoder", {}).get("embed_dim", 512))
        self.embed_dim = embed_dim
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : (B, C, H, W) float tensor
        state : (B, state_dim)

        Returns
        -------
        e_t : (B, embed_dim)
        """
        raise NotImplementedError("Implement Encoder in Phase B (World Model).")
