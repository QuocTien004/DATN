"""Optional expert imitation and Critic-guided offline Actor training."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.actor_critic import ActorCritic
from models.control_state import with_predicted_control
from models.continuous_state import with_continuous_control
from training.train_agent import posterior_start_states, _validate_optimizer_scope


def _compute_chunk_discounted_symlog_returns(
    rewards: torch.Tensor,
    terminals: torch.Tensor | None,
    bootstrapped_value_symlog: torch.Tensor,
    gamma: float = 0.997,
) -> torch.Tensor:
    """Compute discounted returns G_t along replay chunks with Critic bootstrap at T."""
    B, T = rewards.shape[:2]
    rewards_2d = rewards.reshape(B, T)
    if terminals is not None:
        cont = (1.0 - terminals.reshape(B, T).float()).clamp(0.0, 1.0)
    else:
        cont = torch.ones_like(rewards_2d)

    # Convert Critic symlog prediction at the final step back to linear return space
    v_last_sym = bootstrapped_value_symlog.reshape(B, T)[:, -1]
    acc = torch.sign(v_last_sym) * torch.expm1(torch.abs(v_last_sym).clamp(max=15.0))
    returns = torch.zeros_like(rewards_2d)
    for t in reversed(range(T)):
        acc = rewards_2d[:, t] + gamma * cont[:, t] * acc
        returns[:, t] = acc
    return torch.sign(returns) * torch.log1p(torch.abs(returns))


def train_imitation_step(
    batch: dict[str, torch.Tensor],
    world_model: dict[str, torch.nn.Module],
    actor_critic: ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    deterministic_posterior: bool = True,
    burn_in: int = 0,
    grad_clip: float = 10.,
    critic_guided: bool = False,
    critic_temp: float = 2.0,
    gamma: float = 0.997,
) -> dict[str, float]:
    """Fit Actor means to verified expert actions (optionally weighted by Critic advantage).

    Features and labels remain in identical B,T order. No random state subsampling
    is allowed here. ``burn_in`` excludes the start of randomly cut replay chunks
    from the loss, while still using them to initialize the recurrent state.
    When ``critic_guided=True``, the pretrained Critic evaluates V_psi(s_t) and
    computes normalized advantage weights w_t = exp((G_sym_t - V_psi(s_t)) / beta)
    to weight the Actor behavioral cloning loss.
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
    continuous_control = actor_critic.actor.control_state_source == 'continuous'
    if continuous_control and getattr(world_model.get('rssm'), 'continuous', None) is None:
        raise ValueError('Continuous BC requires the learned RSSM branch')
    if predicted_control and "control_state" not in world_model:
        raise ValueError("Predicted-state BC requires the learned control_state head")
    try:
        for module in world_model.values():
            module.eval()
        states = posterior_start_states(
            batch, world_model, deterministic=deterministic_posterior,
            include_control=actor_critic.actor.control_state_dim > 0 and not (predicted_control or continuous_control),
        )
        if predicted_control:
            with torch.no_grad():
                states = with_predicted_control(states, world_model["control_state"])
        if continuous_control:
            with torch.no_grad():
                states = with_continuous_control(states, world_model['rssm'])
    finally:
        for name, module in world_model.items():
            module.train(original_modes[name])

    prediction = actor_critic.act(states, deterministic=True).mean.reshape_as(target)
    error = F.mse_loss(prediction[:, burn_in:], target[:, burn_in:], reduction="none")

    weight_mean = 1.0
    adv_mean = 0.0
    if critic_guided and "reward" in batch:
        with torch.no_grad():
            v_pred = actor_critic.value(states).reshape(target.shape[0], target.shape[1])
            g_sym = _compute_chunk_discounted_symlog_returns(
                batch["reward"],
                batch.get("terminal", batch.get("done")),
                v_pred,
                gamma=gamma,
            )
            adv = (g_sym[:, burn_in:] - v_pred[:, burn_in:]).clamp(-4.0, 4.0)
            raw_w = torch.exp(adv / max(critic_temp, 1e-3)).clamp(0.5, 2.5)
            weights = (raw_w / raw_w.mean().clamp(min=1e-6)).unsqueeze(-1)
            weight_mean = float(raw_w.mean().detach())
            adv_mean = float(adv.mean().detach())
        loss = (weights * error).mean()
    else:
        loss = error.mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    metrics = {
        "bc_mse": float(error.mean().detach()),
        "actor_loss": float(loss.detach()),
        "steering_mse": float(error[..., 0].mean().detach()),
        "throttle_mse": float(error[..., 1].mean().detach()),
        "actor_grad_norm": float(norm),
    }
    if critic_guided:
        metrics["critic_weight_mean"] = weight_mean
        metrics["critic_advantage_mean"] = adv_mean
    return metrics
