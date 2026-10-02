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
