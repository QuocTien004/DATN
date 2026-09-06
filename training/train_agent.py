from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor_critic import ActorCritic


LatentState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ImaginedTrajectory:
    """Time-major tensors produced by a latent imagination rollout."""

    h: torch.Tensor  # (H + 1, B, deter_dim)
    z: torch.Tensor  # (H + 1, B, stoch_dim, stoch_classes)
    action: torch.Tensor  # (H, B, action_dim)
    reward: torch.Tensor  # (H, B)
    discount: torch.Tensor  # (H, B), gamma * P(continue)
    value: torch.Tensor  # (H + 1, B)
    log_prob: torch.Tensor  # (H, B)
    entropy: torch.Tensor  # (H, B)


def _unique_modules(modules: Iterable[nn.Module]) -> list[nn.Module]:
    result: list[nn.Module] = []
    seen: set[int] = set()
    for module in modules:
        if id(module) not in seen:
            result.append(module)
            seen.add(id(module))
    return result


@contextmanager
def freeze_parameters(modules: Iterable[nn.Module]) -> Iterator[None]:
    """Freeze parameter leaves without disabling gradients through module inputs."""
    unique = _unique_modules(modules)
    parameters = [parameter for module in unique for parameter in module.parameters()]
    original = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original):
            parameter.requires_grad_(requires_grad)


def _validate_start_states(start_states: LatentState) -> None:
    if not {"h", "z"}.issubset(start_states):
        raise KeyError("start_states must contain 'h' and 'z'")
    h, z = start_states["h"], start_states["z"]
    if h.ndim != 2 or z.ndim != 3 or h.shape[0] != z.shape[0]:
        raise ValueError(
            "Expected h=(B,D), z=(B,G,C); got "
            f"h={tuple(h.shape)}, z={tuple(z.shape)}"
        )


def posterior_start_states(
    batch: dict[str, torch.Tensor],
    world_model: dict[str, nn.Module],
    *,
    max_states: int | None = None,
) -> LatentState:
    """Encode a replay sequence and return detached posterior states for imagination."""
    encoder = world_model["encoder"]
    rssm = world_model["rssm"]
    image, vector, action = batch["image"], batch["state"], batch["action"]
    if image.ndim != 5 or vector.ndim != 3 or action.ndim != 3:
        raise ValueError("Expected replay tensors image=(B,T,C,H,W), state/action=(B,T,D)")
    batch_size, sequence_length = image.shape[:2]
    if vector.shape[:2] != (batch_size, sequence_length) or action.shape[:2] != (
        batch_size,
        sequence_length,
    ):
        raise ValueError("Replay tensors must share batch and time dimensions")
    if "done" in batch and torch.any(batch["done"][:, :-1]):
        raise ValueError("Replay sequence crosses an episode boundary")

    with torch.no_grad():
        embedding = encoder(
            image.reshape(batch_size * sequence_length, *image.shape[2:]),
            vector.reshape(batch_size * sequence_length, vector.shape[-1]),
        ).reshape(batch_size, sequence_length, -1)
        current = rssm.initial_state(batch_size, image.device)
        zero_action = torch.zeros_like(action[:, 0])
        all_h: list[torch.Tensor] = []
        all_z: list[torch.Tensor] = []
        for time in range(sequence_length):
            previous_action = zero_action if time == 0 else action[:, time - 1]
            current, _stats = rssm.observe_step(
                current,
                previous_action,
                embedding[:, time],
            )
            all_h.append(current["h"])
            all_z.append(current["z"])

        h = torch.stack(all_h, dim=1).reshape(batch_size * sequence_length, -1)
        z = torch.stack(all_z, dim=1).reshape(
            batch_size * sequence_length,
            *all_z[0].shape[1:],
        )
        if max_states is not None and max_states > 0 and h.shape[0] > max_states:
            indices = torch.randperm(h.shape[0], device=h.device)[:max_states]
            h, z = h[indices], z[indices]
    return {"h": h.detach(), "z": z.detach()}


def imagine_rollout(
    start_states: LatentState,
    world_model: dict[str, nn.Module],
    actor_critic: ActorCritic,
    *,
    horizon: int,
    gamma: float,
) -> ImaginedTrajectory:
    """Roll the frozen RSSM prior forward under stochastic Actor actions."""
    _validate_start_states(start_states)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    rssm = world_model["rssm"]
    reward_predictor = world_model["reward"]
    continue_predictor = world_model["continue"]
    current = {"h": start_states["h"], "z": start_states["z"]}
    all_h = [current["h"]]
    all_z = [current["z"]]
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    discounts: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for _time in range(horizon):
        values.append(actor_critic.value(current))
        policy = actor_critic.act(current, deterministic=False)
        next_state, _stats = rssm.imagine_step(current, policy.action)
        reward = reward_predictor(next_state["h"], next_state["z"]).squeeze(-1)
        continue_logit = continue_predictor(
            next_state["h"], next_state["z"]
        ).squeeze(-1)
        discount = gamma * torch.sigmoid(continue_logit)

        actions.append(policy.action)
        rewards.append(reward)
        discounts.append(discount)
        log_probs.append(policy.log_prob)
        entropies.append(policy.entropy)
        all_h.append(next_state["h"])
        all_z.append(next_state["z"])
        current = next_state

    values.append(actor_critic.value(current))
    trajectory = ImaginedTrajectory(
        h=torch.stack(all_h, dim=0),
        z=torch.stack(all_z, dim=0),
        action=torch.stack(actions, dim=0),
        reward=torch.stack(rewards, dim=0),
        discount=torch.stack(discounts, dim=0),
        value=torch.stack(values, dim=0),
        log_prob=torch.stack(log_probs, dim=0),
        entropy=torch.stack(entropies, dim=0),
    )
    for name in ("action", "reward", "discount", "value", "log_prob", "entropy"):
        tensor = getattr(trajectory, name)
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values in imagined {name}")
    return trajectory


def lambda_return(
    reward: torch.Tensor,
    discount: torch.Tensor,
    value: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Generalized lambda-return for time-major tensors ``(H, B)``."""
    if reward.shape != discount.shape or reward.shape != value.shape:
        raise ValueError("reward, discount, and value must have the same (H, B) shape")
    if bootstrap.shape != reward.shape[1:]:
        raise ValueError(
            f"bootstrap must have shape {tuple(reward.shape[1:])}, got {tuple(bootstrap.shape)}"
        )
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    next_return = bootstrap
    returns: list[torch.Tensor] = []
    for time in range(reward.shape[0] - 1, -1, -1):
        next_return = reward[time] + discount[time] * (
            (1.0 - lambda_) * value[time] + lambda_ * next_return
        )
        returns.append(next_return)
    returns.reverse()
    return torch.stack(returns, dim=0)


def _continuation_weights(discount: torch.Tensor) -> torch.Tensor:
    """Weight step t by survival probability before reaching that step."""
    first = torch.ones_like(discount[:1])
    return torch.cat([first, torch.cumprod(discount[:-1], dim=0)], dim=0)


def _validate_optimizer_scope(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
    *,
    name: str,
) -> None:
    expected = {id(parameter) for parameter in module.parameters()}
    configured = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing = expected - configured
    unexpected = configured - expected
    if missing or unexpected:
        raise ValueError(
            f"{name} must contain exactly the parameters of its matching module "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )


def train_actor_critic_step(
    start_states: LatentState,
    world_model: dict[str, nn.Module],
    actor_critic: ActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run one differentiable imagination update for Actor and Critic."""
    ac_cfg = cfg.get("actor_critic", cfg)
    horizon = int(ac_cfg.get("imagination_horizon", 15))
    gamma = float(ac_cfg.get("gamma", 0.997))
    lambda_ = float(ac_cfg.get("lambda", 0.95))
    entropy_scale = float(ac_cfg.get("entropy_scale", 1e-4))
    grad_clip = float(ac_cfg.get("grad_clip", 100.0))
    _validate_optimizer_scope(actor_optimizer, actor_critic.actor, name="actor_optimizer")
    _validate_optimizer_scope(
        critic_optimizer, actor_critic.critic, name="critic_optimizer"
    )

    frozen_world_modules = list(world_model.values())
    with freeze_parameters(frozen_world_modules):
        # Critic acts as a differentiable scoring function for latent states, but
        # its weights must not receive the Actor update.
        with freeze_parameters([actor_critic.critic]):
            trajectory = imagine_rollout(
                start_states,
                world_model,
                actor_critic,
                horizon=horizon,
                gamma=gamma,
            )
            returns = lambda_return(
                trajectory.reward,
                trajectory.discount,
                trajectory.value[1:],
                trajectory.value[-1],
                lambda_,
            )
            weights = _continuation_weights(trajectory.discount).detach()
            weight_sum = weights.sum().clamp_min(1.0)
            actor_objective = (weights * returns).sum() / weight_sum
            entropy = (weights * trajectory.entropy).sum() / weight_sum
            actor_loss = -actor_objective - entropy_scale * entropy

            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.actor.parameters(), grad_clip, error_if_nonfinite=True
            )
            actor_optimizer.step()

        # Detaching both states and targets confines this backward pass to Critic.
        horizon_size, batch_size = trajectory.reward.shape
        critic_value = actor_critic.critic(
            trajectory.h[:-1].detach().reshape(horizon_size * batch_size, -1),
            trajectory.z[:-1].detach().reshape(
                horizon_size * batch_size,
                *trajectory.z.shape[2:],
            ),
        ).reshape(horizon_size, batch_size)
        critic_target = returns.detach()
        critic_weights = weights.detach()
        critic_loss = (
            critic_weights * F.mse_loss(critic_value, critic_target, reduction="none")
        ).sum() / critic_weights.sum().clamp_min(1.0)

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_critic.critic.parameters(), grad_clip, error_if_nonfinite=True
        )
        critic_optimizer.step()

    return {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "imagined_reward_mean": float(trajectory.reward.detach().mean()),
        "continue_mean": float((trajectory.discount.detach() / gamma).mean())
        if gamma > 0.0
        else 0.0,
        "value_mean": float(critic_value.detach().mean()),
        "lambda_return_mean": float(critic_target.mean()),
        "entropy": float(trajectory.entropy.detach().mean()),
        "actor_grad_norm": float(actor_grad_norm.detach()),
        "critic_grad_norm": float(critic_grad_norm.detach()),
    }
