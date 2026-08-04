from __future__ import annotations

from typing import Any, Callable

import numpy as np

from envs.metadrive_wrapper import MetaDriveImageEnv
from evaluation.metrics import aggregate_episode_metrics, compute_episode_metrics


def evaluate_policy(
    env: MetaDriveImageEnv,
    policy_fn: Callable[[dict], np.ndarray],
    *,
    num_episodes: int = 20,
    start_seed: int = 10000,
) -> dict[str, float]:
    """
    Run policy in MetaDrive for `num_episodes` on hold-out seeds.

    Phase A: works with any callable policy (e.g. random).
    Phase C: pass Actor that maps obs -> action (or latent policy + encoder).
    """
    episode_metrics: list[dict[str, float]] = []

    for i in range(num_episodes):
        obs, info = env.reset(seed=start_seed + i)
        info_history = [info]
        crashed = False
        success = False
        done = False

        while not done:
            action = policy_fn(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            info_history.append(info)
            crashed = crashed or bool(info.get("crash", False) or info.get("crash_vehicle", False))
            success = bool(info.get("arrive_dest", False))
            done = bool(terminated or truncated)

        episode_metrics.append(
            compute_episode_metrics(info_history, crashed=crashed, success=success)
        )

    return aggregate_episode_metrics(episode_metrics)
