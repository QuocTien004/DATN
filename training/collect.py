from __future__ import annotations

from typing import Callable

import numpy as np
from tqdm import tqdm

from envs.metadrive_wrapper import MetaDriveImageEnv
from utils.replay_buffer import ReplayBuffer, Transition


def random_action(action_space) -> np.ndarray:
    """Sample a random action from the env action space."""
    return np.asarray(action_space.sample(), dtype=np.float32)


def collect_steps(
    env: MetaDriveImageEnv,
    buffer: ReplayBuffer,
    num_steps: int,
    policy_fn: Callable[[dict], np.ndarray] | None = None,
    *,
    show_progress: bool = True,
) -> dict:
    """
    Interact with MetaDrive and append transitions to `buffer`.

    Parameters
    ----------
    policy_fn : optional
        Maps obs dict -> action. Defaults to random policy.
    """
    if policy_fn is None:
        policy_fn = lambda _obs: random_action(env.action_space)

    obs, _info = env.reset()
    is_first = True
    ep_return = 0.0
    ep_len = 0
    returns: list[float] = []

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="collect", leave=True)

    for _ in iterator:
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        buffer.add(
            Transition(
                image=obs["image"],
                state=obs["state"],
                action=np.asarray(action, dtype=np.float32).reshape(-1),
                reward=float(reward),
                done=done,
                next_image=next_obs["image"],
                next_state=next_obs["state"],
            ),
            is_first=is_first,
        )

        ep_return += float(reward)
        ep_len += 1
        is_first = False

        if done:
            returns.append(ep_return)
            obs, _info = env.reset()
            is_first = True
            ep_return = 0.0
            ep_len = 0
        else:
            obs = next_obs

    return {
        "steps": num_steps,
        "episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "buffer": buffer.summary(),
    }
