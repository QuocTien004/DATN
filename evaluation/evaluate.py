from __future__ import annotations

from typing import Callable
import json
from pathlib import Path

import numpy as np
import torch

from envs.metadrive_wrapper import MetaDriveImageEnv
from evaluation.metrics import aggregate_episode_metrics, compute_episode_metrics
from evaluation.safety_shield import AdaptiveTrafficAvoidanceShield, SafeLatentActorPolicy
from models.actor_critic import ActorCritic
from models.control_state import with_predicted_control
from models.continuous_state import with_continuous_control


class LatentActorPolicy:
    """Observation policy that maintains the recurrent RSSM posterior state."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        rssm: torch.nn.Module,
        actor_critic: ActorCritic,
        *,
        device: torch.device,
        deterministic: bool = True,
        control_predictor: torch.nn.Module | None = None,
    ) -> None:
        self.encoder = encoder
        self.rssm = rssm
        self.actor_critic = actor_critic
        self.device = device
        self.deterministic = bool(deterministic)
        self.control_predictor = control_predictor
        if actor_critic.actor.control_state_source == 'continuous':
            if getattr(rssm, 'continuous', None) is None or control_predictor is not None:
                raise ValueError('Continuous Actor requires continuous RSSM, not a readout or GT fallback')
            if rssm.continuous.dim != actor_critic.actor.control_state_dim:
                raise ValueError('Continuous Actor/RSSM dimension mismatch')
        if actor_critic.actor.control_state_source == "predicted" and control_predictor is None:
            raise ValueError("Predicted-state Actor requires its control predictor; refusing GT fallback")
        if control_predictor is not None:
            if not actor_critic.actor.control_state_dim:
                raise ValueError("control_predictor requires an Actor control bridge")
            control_predictor.eval()
        self._latent: dict[str, torch.Tensor] | None = None
        self._previous_action: torch.Tensor | None = None
        self.encoder.eval()
        self.rssm.eval()
        self.actor_critic.eval()

    def reset(self) -> None:
        """Reset recurrent state at the beginning of every episode."""
        self._latent = self.rssm.initial_state(1, self.device)
        self._previous_action = torch.zeros(
            1, self.actor_critic.action_dim, device=self.device
        )

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self._latent is None or self._previous_action is None:
            self.reset()
        image_np = np.asarray(observation["image"])
        state_np = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        if image_np.ndim != 3:
            raise ValueError(f"Expected HWC image, got {image_np.shape}")
        image = torch.as_tensor(image_np, device=self.device, dtype=torch.float32)
        image = image.permute(2, 0, 1).unsqueeze(0)
        if image_np.dtype == np.uint8 or float(image.max()) > 1.0:
            image = image / 255.0
        vector = torch.as_tensor(state_np, device=self.device).unsqueeze(0)

        assert self._latent is not None and self._previous_action is not None
        with torch.no_grad():
            embedding = self.encoder(image, vector)
            self._latent, _stats = self.rssm.observe_step(
                self._latent,
                self._previous_action,
                embedding,
                deterministic=self.deterministic,
            )
            if self.actor_critic.actor.control_state_dim:
                if self.actor_critic.actor.control_state_source == 'continuous':
                    self._latent = with_continuous_control(self._latent, self.rssm)
                elif self.control_predictor is None:
                    self._latent["control"] = vector
                else:
                    self._latent = with_predicted_control(self._latent, self.control_predictor)
            policy = self.actor_critic.act(
                self._latent,
                deterministic=self.deterministic,
            )
            self._previous_action = policy.action
        return policy.action.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def evaluate_policy(
    env: MetaDriveImageEnv,
    policy_fn: Callable[[dict], np.ndarray],
    *,
    num_episodes: int = 20,
    start_seed: int = 10000,
    output_path: str | Path | None = None,
    shield: AdaptiveTrafficAvoidanceShield | None = None,
) -> dict[str, float]:
    """
    Run policy in MetaDrive for `num_episodes` on hold-out seeds.

    Stateful policies may expose ``reset()``; it is called after each env reset.
    If ``shield`` is provided, policy_fn is wrapped with SafeLatentActorPolicy.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    if shield is not None and not isinstance(policy_fn, SafeLatentActorPolicy):
        policy_fn = SafeLatentActorPolicy(policy_fn, shield=shield, env=env)
    elif hasattr(policy_fn, "set_env"):
        policy_fn.set_env(env)

    episode_metrics: list[dict[str, float]] = []

    for i in range(num_episodes):
        obs, info = env.reset(seed=start_seed + i)
        reset_policy = getattr(policy_fn, "reset", None)
        if callable(reset_policy):
            reset_policy()
        info_history = [info]
        crashed = False
        out_of_road = False
        crash_vehicle = False
        success = False
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            info_history.append(info)
            crashed = crashed or bool(
                info.get("crash", False)
                or info.get("crash_vehicle", False)
                or info.get("crash_object", False)
                or info.get("out_of_road", False)
            )
            out_of_road = out_of_road or bool(info.get("out_of_road", False))
            crash_vehicle = crash_vehicle or bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
            success = success or bool(info.get("arrive_dest", False))
            episode_return += float(reward)
            episode_length += 1
            done = bool(terminated or truncated)

        episode_metrics.append(
            compute_episode_metrics(
                info_history,
                crashed=crashed,
                success=success,
                out_of_road=out_of_road,
                crash_vehicle=crash_vehicle,
                episode_return=episode_return,
                episode_length=episode_length,
            )
        )

        episode_metrics[-1]["seed"] = start_seed + i
        episode_metrics[-1]["terminated"] = bool(terminated)
        episode_metrics[-1]["truncated"] = bool(truncated)

    metrics = aggregate_episode_metrics(episode_metrics)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "horizon": getattr(env, "env_cfg", {}).get("horizon"),
            "aggregate": metrics, "episodes": episode_metrics,
        }, indent=2), encoding="utf-8")
    return metrics
