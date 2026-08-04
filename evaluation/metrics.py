from __future__ import annotations

from typing import Any


def compute_episode_metrics(info_history: list[dict[str, Any]], *, crashed: bool, success: bool) -> dict[str, float]:
    """Compute per-episode metrics from MetaDrive info dicts."""
    route = 0.0
    if info_history:
        route = float(info_history[-1].get("route_completion", 0.0) or 0.0)
    return {
        "success": 1.0 if success else 0.0,
        "collision": 1.0 if crashed else 0.0,
        "route_completion": route,
        "episode_length": float(len(info_history)),
    }


def aggregate_episode_metrics(episodes: list[dict[str, float]]) -> dict[str, float]:
    """Mean metrics over evaluation episodes."""
    if not episodes:
        return {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "mean_route_completion": 0.0,
            "mean_episode_length": 0.0,
            "num_episodes": 0.0,
        }
    n = len(episodes)
    return {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "collision_rate": sum(e["collision"] for e in episodes) / n,
        "mean_route_completion": sum(e["route_completion"] for e in episodes) / n,
        "mean_episode_length": sum(e["episode_length"] for e in episodes) / n,
        "num_episodes": float(n),
    }
