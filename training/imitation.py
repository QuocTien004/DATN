"""Optional expert imitation; distinct from the imagination/RL objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.actor_critic import ActorCritic
from models.control_state import with_predicted_control
from training.train_agent import posterior_start_states, _validate_optimizer_scope


def train_imitation_step(
    batch: dict[str, torch.Tensor],
    world_model: dict[str, torch.nn.Module],
    actor_critic: ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    deterministic_posterior: bool = True,
    burn_in: int = 0,
    grad_clip: float = 10.,
) -> dict[str, float]:
    """Fit Actor means to verified expert actions; WM and Critics stay unchanged.

    Features and labels remain in identical B,T order. No random state subsampling
    is allowed here. ``burn_in`` excludes the start of randomly cut replay chunks
    from the loss, while still using them to initialize the recurrent state.
    This loss does not directly train the policy's exploration-variance output.
    """
    _validate_optimizer_scope(optimizer, actor_critic.actor, name="imitation_optimizer")
    target = batch["action"]
    if target.ndim != 3 or target.shape[-1] != actor_critic.action_dim or not 0 <= burn_in < target.shape[1]:
        raise ValueError("Expected action=(B,T,A) and 0 <= burn_in < T")
    if not torch.isfinite(target).all():
        raise ValueError("Expert actions must be finite")
    if ((target < actor_critic.actor.action_low) | (target > actor_critic.actor.action_high)).any():
        raise ValueError("Expert actions exceed the Actor action bounds")
    original_modes = {name: module.training for name, module in world_model.items()}
    predicted_control = actor_critic.actor.control_state_source == "predicted"
    if predicted_control and "control_state" not in world_model:
        raise ValueError("Predicted-state BC requires the learned control_state head")
    try:
        for module in world_model.values():
            module.eval()
        states = posterior_start_states(
            batch, world_model, deterministic=deterministic_posterior,
            include_control=actor_critic.actor.control_state_dim > 0 and not predicted_control,
        )
        if predicted_control:
            with torch.no_grad():
                states = with_predicted_control(states, world_model["control_state"])
    finally:
        for name, module in world_model.items():
            module.train(original_modes[name])
    prediction = actor_critic.act(states, deterministic=True).mean.reshape_as(target)
    error = F.mse_loss(prediction[:, burn_in:], target[:, burn_in:], reduction="none")
    loss = error.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {
        "bc_mse": float(loss.detach()),
        "steering_mse": float(error[..., 0].mean().detach()),
        "throttle_mse": float(error[..., 1].mean().detach()),
        "actor_grad_norm": float(norm),
    }
