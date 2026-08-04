from __future__ import annotations

from typing import Any

import torch


def train_world_model_step(
    batch: dict[str, torch.Tensor],
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """
    One World Model update on a sequence batch (B, T, ...).

    Phase B: implement reconstruction / reward / continue / KL losses.
    """
    raise NotImplementedError("Implement train_world_model_step in Phase B.")
