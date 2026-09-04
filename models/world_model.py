from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .predictors import ContinuePredictor, RewardPredictor
from .rssm import RSSM


def build_world_model(
    wm_cfg: dict[str, Any],
    image_shape: tuple[int, ...],
    state_dim: int,
    action_dim: int,
    device: torch.device,
    *,
    include_decoder: bool = True,
) -> dict[str, nn.Module]:
    """Build World Model components with the repository's shared dimensions."""
    encoder = Encoder(wm_cfg, state_dim=state_dim, image_shape=image_shape).to(device)
    rssm = RSSM(wm_cfg, embed_dim=encoder.embed_dim, action_dim=action_dim).to(device)
    models: dict[str, nn.Module] = {"encoder": encoder, "rssm": rssm}
    if include_decoder:
        models["decoder"] = Decoder(
            wm_cfg,
            deter_dim=rssm.deter_dim,
            stoch_size=rssm.stoch_size,
            image_shape=image_shape,
            state_dim=state_dim,
        ).to(device)
    # Keep the historical component order so old World Model optimizer
    # checkpoints continue to map optimizer slots to the same parameters.
    models["reward"] = RewardPredictor(
        wm_cfg, rssm.deter_dim, rssm.stoch_size
    ).to(device)
    models["continue"] = ContinuePredictor(
        wm_cfg, rssm.deter_dim, rssm.stoch_size
    ).to(device)
    return models


def load_world_model_state(
    models: dict[str, nn.Module],
    checkpoint: dict[str, Any],
) -> None:
    """Load required component states with an actionable compatibility error."""
    saved = checkpoint.get("models")
    if not isinstance(saved, dict):
        raise KeyError("World Model checkpoint is missing the 'models' mapping")
    missing = sorted(set(models) - set(saved))
    if missing:
        raise KeyError(f"World Model checkpoint is missing components: {missing}")
    for name, module in models.items():
        try:
            module.load_state_dict(saved[name])
        except RuntimeError as exc:
            raise RuntimeError(
                f"World Model component '{name}' is incompatible with this config: {exc}"
            ) from exc


def validate_world_model_metadata(
    checkpoint: dict[str, Any],
    *,
    image_shape: tuple[int, ...] | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
) -> None:
    """Fail early when replay/evaluation dimensions disagree with a checkpoint."""
    expected = {
        "image_shape": tuple(image_shape) if image_shape is not None else None,
        "state_dim": int(state_dim) if state_dim is not None else None,
        "action_dim": int(action_dim) if action_dim is not None else None,
    }
    for key, wanted in expected.items():
        if wanted is None or key not in checkpoint:
            continue
        found: Any = checkpoint[key]
        if key == "image_shape":
            found = tuple(found)
        else:
            found = int(found)
        if found != wanted:
            raise ValueError(
                f"World Model checkpoint {key}={found!r}, but current data expects {wanted!r}"
            )
