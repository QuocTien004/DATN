"""Aligned, observation-free control predictions for short RSSM diagnostics."""
from __future__ import annotations

import torch


def valid_prediction_starts(is_first: torch.Tensor, horizon: int,
                            done: torch.Tensor | None = None) -> torch.Tensor:
    """Indices t whose targets t..t+H exist in one episode.

    action[t] leads from state[t] to state[t+1]. Never train across a reset,
    including timeout resets; terminal next observations absent from the replay
    are deliberately excluded.
    """
    if is_first.ndim != 1 or horizon < 0:
        raise ValueError("Expected flat is_first and nonnegative horizon")
    if done is not None and done.shape != is_first.shape:
        raise ValueError("done must match is_first")
    count = max(0, len(is_first) - horizon)
    starts = torch.arange(count, device=is_first.device)
    valid = torch.ones(count, dtype=torch.bool, device=is_first.device)
    for offset in range(1, horizon + 1):
        valid &= ~is_first[starts + offset].bool()
        if done is not None:
            valid &= ~done[starts + offset - 1].bool()
    return starts[valid]


def predict_control_sequence(start: dict, actions: torch.Tensor, rssm,
                             predictor, *, deterministic: bool = True) -> torch.Tensor:
    """Return (H+1,B,D), decoding the start and each action-conditioned prior.

    There is no observation/state-teacher argument. Do not detach here: callers
    may differentiate through the frozen predictor/RSSM with respect to actions.
    """
    if actions.ndim != 3 or actions.shape[1] != start["h"].shape[0]:
        raise ValueError("actions must have shape (H, B, A)")
    current = {"h": start["h"], "z": start["z"]}
    predictions = [predictor(current["h"], current["z"])]
    for action in actions:
        current, _ = rssm.imagine_step(current, action, deterministic=deterministic)
        predictions.append(predictor(current["h"], current["z"]))
    return torch.stack(predictions)
