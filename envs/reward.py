from __future__ import annotations

from typing import Any


def apply_custom_reward(
    info: dict[str, Any],
    reward_cfg: dict[str, Any],
    last_route_completion: float = 0.0,
) -> float:
    """
    Reward shaping from MetaDrive `info` dict.

    1. Progress reward: W_p * max(0.0, route_t - route_{t-1})
    2. Idle / low-speed penalty: penalize when velocity < threshold and not arrived
    3. Success bonus: bonus reward upon reaching destination
    4. Crash extra penalty: additional penalty if configured (default 0.0 to rely on MetaDrive's -5.0)
    """
    shaped_reward = 0.0

    # 1. Progress reward gated by lane centering
    curr_route = float(info.get("route_completion", 0.0) or 0.0)
    delta_route = max(0.0, curr_route - float(last_route_completion or 0.0))
    progress_weight = float(reward_cfg.get("progress_weight", 25.0))
    lateral_factor = float(info.get("lateral_factor", 1.0))
    # Gated progress: vehicle ONLY earns progress reward if it remains within the lane!
    shaped_reward += progress_weight * delta_route * lateral_factor

    # 2. Early lateral deviation penalty (rumble strip effect before crash)
    lateral_penalty_weight = float(reward_cfg.get("lateral_penalty_weight", 0.5))
    if lateral_factor < 0.5:
        shaped_reward -= lateral_penalty_weight * (0.5 - lateral_factor)

    # 3. Idle penalty (optional, disabled by default to avoid RL suicide trap)
    idle_penalty = float(reward_cfg.get("idle_penalty", 0.0))
    if idle_penalty != 0.0:
        vel = float(info.get("velocity", 0.0) or 0.0)
        idle_threshold = float(reward_cfg.get("idle_velocity_threshold", 1.0))
        if vel < idle_threshold and not info.get("arrive_dest", False):
            shaped_reward += idle_penalty

    # 4. Success bonus
    if info.get("arrive_dest", False):
        shaped_reward += float(reward_cfg.get("success_bonus", 25.0))

    # 5. Crash extra penalty (if configured)
    if (
        info.get("out_of_road", False)
        or info.get("crash_vehicle", False)
        or info.get("crash_object", False)
        or info.get("crash_sidewalk", False)
        or info.get("crash", False)
    ):
        shaped_reward += float(reward_cfg.get("crash_extra_penalty", 0.0))

    return shaped_reward
