from __future__ import annotations

from typing import Any


def apply_custom_reward(info: dict[str, Any], reward_cfg: dict[str, Any]) -> float:
    """
    Optional reward shaping from MetaDrive `info` dict.

    Phase A: skeleton only — returns 0.0 so callers can fall back to env reward.
    Fill in when you enable `use_custom_reward` in configs/env_metadrive.yaml.
    """
    # Example (uncomment / adapt later):
    # r = 0.0
    # r += reward_cfg.get("driving_reward", 1.0) * info.get("route_completion", 0.0)
    # if info.get("crash", False):
    #     r += reward_cfg.get("crash_vehicle", -5.0)
    # return r
    _ = (info, reward_cfg)
    return 0.0
