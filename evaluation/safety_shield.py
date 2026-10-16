"""Safety-Critical Traffic Avoidance Shield and Safe Policy Wrapper.

Provides an Adaptive Cruise Control (ACC) and Autonomous Emergency Braking (AEB)
safety shield for MetaDrive agents. This layer operates alongside latent world model
policies, monitoring the forward ego corridor to prevent collisions with surrounding
traffic without disrupting lateral steering control.
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np


class AdaptiveTrafficAvoidanceShield:
    """Longitudinal Safe Control & Emergency Collision Shield for MetaDrive."""

    def __init__(
        self,
        *,
        corridor_width: float = 2.4,
        threat_distance: float = 28.0,
        desired_time_gap: float = 1.2,
        min_gap: float = 5.0,
        max_decel: float = -1.0,
    ) -> None:
        self.corridor_width = float(corridor_width)
        self.threat_distance = float(threat_distance)
        self.desired_time_gap = float(desired_time_gap)
        self.min_gap = float(min_gap)
        self.max_decel = float(max_decel)
        self.stuck_counter = 0

    def modulate_action(
        self,
        base_action: np.ndarray,
        vehicle: Any,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Apply longitudinal safety shield with anti-stuck and proactive intersection protection."""
        action = np.copy(base_action)
        diag = {
            "threat_detected": 0.0,
            "min_lead_dist": 999.0,
            "throttle_override": 0.0,
        }

        if vehicle is None:
            return action, diag

        ego_speed_ms = float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6
        steer_abs = abs(float(action[0]))

        # 1. Smooth high-speed turn regulation: proportionally cap throttle when turning at speed
        if ego_speed_ms > 4.5 and steer_abs > 0.20:
            max_turn_throttle = max(0.05, 0.35 - (steer_abs - 0.20) * 1.2)
            action[1] = min(float(action[1]), max_turn_throttle)

        # 2. Identify intersection / roundabout / ramp approach zones
        lane_idx = getattr(vehicle, "lane_index", None)
        is_intersection = False
        if isinstance(lane_idx, tuple) and len(lane_idx) >= 2:
            is_intersection = any(x in str(lane_idx[0]) or x in str(lane_idx[1]) for x in ("X", "O", "I", "R", "r"))

        # 3. Obstacle & Cross-Traffic Scanning
        objs = vehicle.lidar.get_surrounding_objects(vehicle) if (hasattr(vehicle, "lidar") and vehicle.lidar is not None) else []
        ego_heading = float(vehicle.heading_theta)
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)
        ego_pos = vehicle.position

        threat_dist = 999.0
        threat_obj = None
        threat_is_same_dir = False
        threat_vx = 0.0

        for obj in objs:
            rel = obj.position - ego_pos
            # Project into ego coordinate frame: dx = longitudinal forward, dy = lateral side
            dx = float(rel[0] * cos_h + rel[1] * sin_h)
            dy = float(-rel[0] * sin_h + rel[1] * cos_h)

            if not (0.5 < dx < self.threat_distance):
                continue

            obj_heading = float(getattr(obj, "heading_theta", ego_heading))
            heading_diff = obj_heading - ego_heading
            heading_cos = float(np.cos(heading_diff))
            heading_sin = float(np.sin(heading_diff))
            obj_speed_ms = float(getattr(obj, "speed_km_h", 0.0)) / 3.6
            v_x = obj_speed_ms * heading_cos
            v_y = obj_speed_ms * heading_sin

            # 1. Ignore oncoming vehicles in opposing lanes (prevents sudden phantom braking when passing oncoming cars)
            if heading_cos < -0.40 and abs(dy) >= 1.55:
                continue

            is_candidate = False
            # 2. Vehicle directly ahead in ego lane corridor
            if abs(dy) < 2.25:
                is_candidate = True
            # 3. Same-direction vehicle in adjacent/merging lane within close proximity or converging
            elif heading_cos > 0.45:
                if abs(dy) < 3.5 and dx < 12.5 and (dy * v_y < -0.15 or is_intersection or dx < 7.5):
                    is_candidate = True
            # 4. Cross-traffic / roundabout vehicle inside or approaching our path (ignore if moving away)
            else:
                if is_intersection and abs(dy) < 7.5 and dx < 20.0 and (dy * v_y < 0.15):
                    is_candidate = True
                elif abs(dy) < 3.6 and dx < 13.0 and (dy * v_y < -0.20):
                    is_candidate = True

            if is_candidate and dx < threat_dist:
                threat_dist = dx
                threat_obj = obj
                threat_is_same_dir = bool(heading_cos > 0.45)
                threat_vx = max(0.0, v_x)

        # Smooth proportional intersection speed governor around ~18.5 km/h (5.1 m/s)
        # Avoids bang-bang braking oscillation while ensuring safe entry speed into intersections
        if is_intersection:
            int_max_throttle = float(np.clip(0.10 - (ego_speed_ms - 5.1) * 0.30, -0.25, 0.30))
            action[1] = min(float(action[1]), int_max_throttle)

        if threat_obj is not None:
            diag["threat_detected"] = 1.0
            diag["min_lead_dist"] = threat_dist

            safe_buffer = self.min_gap
            if threat_is_same_dir:
                # Smooth Adaptive Cruise Control (ACC) for same-direction vehicles:
                # Center-to-center distance includes ~4.5m vehicle length, so desired_gap starts at 8.0m (~3.5m bumper gap)
                desired_gap = (safe_buffer + 3.0) + ego_speed_ms * 0.55
                target_speed_ms = max(0.0, threat_vx + (threat_dist - desired_gap) * 0.55)

                if threat_dist < 5.8 and ego_speed_ms > max(0.5, threat_vx - 0.5):
                    action[1] = self.max_decel
                    diag["throttle_override"] = 1.0
                elif ego_speed_ms > target_speed_ms:
                    speed_excess = ego_speed_ms - target_speed_ms
                    decel = -min(0.85, max(0.08, speed_excess / 3.0))
                    action[1] = min(float(action[1]), float(decel))
                    diag["throttle_override"] = 1.0
            else:
                # Cross-traffic / stationary obstacle stopping profile
                if threat_dist <= safe_buffer + 1.5:
                    action[1] = self.max_decel
                    diag["throttle_override"] = 1.0
                else:
                    target_speed_ms = max(0.0, (threat_dist - (safe_buffer + 2.5)) * 0.35)
                    if ego_speed_ms > target_speed_ms:
                        speed_excess = ego_speed_ms - target_speed_ms
                        decel = -min(1.0, max(0.15, speed_excess / 2.5))
                        action[1] = min(float(action[1]), float(decel))
                        diag["throttle_override"] = 1.0

        # 4. Continuous Minimum Cruise Speed Floor when road ahead is clear (prevents turn/roundabout hesitation)
        if (threat_dist > 15.0 or (threat_is_same_dir and threat_dist > 8.5 and threat_vx > 3.5)) and ego_speed_ms < 3.0:
            min_throttle = 0.24 + 0.14 * (1.0 - ego_speed_ms / 3.0)
            if float(action[1]) < min_throttle:
                action[1] = min_throttle
                if steer_abs > 0.55:
                    action[0] = float(action[0]) * 0.78
                diag["throttle_override"] = 1.0
            self.stuck_counter = 0
        elif ego_speed_ms < 0.5 and threat_dist > 13.5:
            self.stuck_counter += 1
            if self.stuck_counter > 5:
                action[1] = max(float(action[1]), 0.35)
                if steer_abs > 0.5:
                    action[0] = float(action[0]) * 0.7
                diag["throttle_override"] = 1.0
        else:
            self.stuck_counter = 0

        return action, diag



class SafeLatentActorPolicy:
    """Wrapper that couples a base latent policy with the Adaptive Traffic Avoidance Shield."""

    def __init__(
        self,
        base_policy: Callable[[dict[str, np.ndarray]], np.ndarray],
        shield: AdaptiveTrafficAvoidanceShield | None = None,
        env: Any = None,
    ) -> None:
        self.base_policy = base_policy
        self.shield = shield or AdaptiveTrafficAvoidanceShield()
        self.env = env
        self.last_shield_diag: dict[str, float] = {}

    def set_env(self, env: Any) -> None:
        """Bind environment reference to query surrounding objects during rollout."""
        self.env = env

    def reset(self) -> None:
        """Delegate reset to the base policy if supported."""
        self.last_shield_diag = {}
        reset_fn = getattr(self.base_policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    @property
    def _latent(self):
        return getattr(self.base_policy, "_latent", None)

    @_latent.setter
    def _latent(self, val):
        if hasattr(self.base_policy, "_latent"):
            self.base_policy._latent = val

    @property
    def _previous_action(self):
        return getattr(self.base_policy, "_previous_action", None)

    @_previous_action.setter
    def _previous_action(self, val):
        if hasattr(self.base_policy, "_previous_action"):
            self.base_policy._previous_action = val

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        base_action = self.base_policy(observation)
        if self.shield is None or self.env is None:
            return base_action

        # Extract vehicle from MetaDrive wrapper or raw env
        raw_env = getattr(self.env, "_env", self.env)
        vehicle = (
            getattr(raw_env, "agent", None)
            or getattr(raw_env, "current_track_vehicle", None)
            or getattr(raw_env, "vehicle", None)
        )

        action, diag = self.shield.modulate_action(base_action, vehicle)
        self.last_shield_diag = diag
        return action
