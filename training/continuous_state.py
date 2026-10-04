"""Open-loop branch fitting: future observations/GT are loss targets only."""
from __future__ import annotations

import torch


def continuous_rollout(branch, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """actions[t] takes c[t] to c[t+1]; never resets to a future posterior."""
    if actions.ndim != 3 or actions.shape[1] != initial.shape[0]:
        raise ValueError('Expected actions=(H,B,A), initial=(B,D)')
    sequence = [initial]
    for action in actions.unbind(0):
        sequence.append(branch.imagine(sequence[-1], action))
    return torch.stack(sequence)


def continuous_prediction_loss(branch, embedding, actions, targets, *, horizons=(1, 5, 15)):
    """targets=(H+1,B,D) supervised only; embedding contains current observation.

    Caller must exclude reset/terminal crossings. Freeze posterior parameters
    for a transition-only stage, or retain gradients for joint branch fitting.
    """
    prediction = continuous_rollout(branch, branch.observe(embedding), actions)
    if prediction.shape != targets.shape or not horizons or min(horizons) < 1 or max(horizons) >= len(prediction):
        raise ValueError('Targets/horizons must match the open-loop sequence')
    normalized = branch.normalize(targets)
    post_loss = (prediction[0] - normalized[0]).square().mean()
    prior_loss = torch.stack([(prediction[h] - normalized[h]).square().mean() for h in horizons]).mean()
    return post_loss + prior_loss, {'posterior_mse': post_loss.detach(), 'prior_mse': prior_loss.detach()}


def continuous_transition_loss(branch, initial, executed_actions, targets, valid, *, horizons=(1, 5, 15)):
    """Fit only dynamics from a frozen, cached posterior; no future teacher forcing.

    Actions are (H,B,A), physical-state targets (H+1,B,D), and valid (H,B).
    valid marks real transitions, INCLUDING the final terminal successor, then
    a false suffix for padding. Callers must never concatenate across resets.
    Ground truth is a loss target only. Initial posterior and targets are
    detached; the recursive transition graph is retained through every step.
    Freeze all other parameters and give the optimizer only branch.transition.
    """
    if initial.ndim != 2 or executed_actions.ndim != 3:
        raise ValueError('Expected initial=(B,D), executed_actions=(H,B,A)')
    h, batch, _ = executed_actions.shape
    if (initial.shape[0] != batch or targets.shape != (h + 1, *initial.shape)
            or valid.shape != (h, batch) or valid.dtype != torch.bool):
        raise ValueError('Targets and boolean valid mask must align with actions')
    if not horizons or min(horizons) < 1 or max(horizons) > h or len(set(horizons)) != len(horizons):
        raise ValueError('Invalid transition loss horizons')
    if (valid[1:] & ~valid[:-1]).any():
        raise ValueError('Valid transitions must be a prefix; cannot cross reset/padding')
    if not torch.isfinite(initial).all() or not torch.isfinite(executed_actions).all():
        raise ValueError('Initial state/actions must be finite, including padding')
    if not torch.isfinite(targets[1:][valid]).all():
        raise ValueError('Real successor targets must be finite')
    prediction = continuous_rollout(branch, initial.detach(), executed_actions.detach())
    terms = []
    counts = {}
    for horizon in horizons:
        mask = valid[horizon - 1]
        counts[horizon] = int(mask.sum())
        if counts[horizon]:
            # Index before normalization: invalid padded targets may contain NaN.
            target = branch.normalize(targets[horizon, mask].detach())
            terms.append((prediction[horizon, mask] - target).square().mean())
    if not terms:
        raise ValueError('No valid targets at requested horizons')
    loss = torch.stack(terms).mean()
    return loss, {'prior_mse': loss.detach(), 'valid_counts': counts}
