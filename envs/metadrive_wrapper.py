from __future__ import annotations

from typing import Any
import os
import warnings

import numpy as np

from .reward import apply_custom_reward


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize MetaDrive image arrays to uint8 HWC (H, W, C)."""
    img = np.asarray(image)
    # MetaDrive image_observation often returns (H, W, C, stack).
    if img.ndim == 4:
        # Prefer last stack frame along the smallest trailing dim.
        if img.shape[-1] <= 8 and img.shape[0] > 8 and img.shape[1] > 8:
            img = img[..., -1]
        else:
            img = img[-1]
    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        max_v = float(np.max(img)) if img.size else 0.0
        if max_v <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return img


class MetaDriveImageEnv:
    """
    Thin wrapper around MetaDriveEnv.

    Forces image + vector observations for the dual-branch Encoder (CNN + MLP).
    Observation dict:
        {
          "image": uint8 array (H, W, C),
          "state": float32 vector,
        }
    """

    def __init__(self, env_cfg: dict[str, Any]) -> None:
        self.env_cfg = env_cfg
        self._env = None
        self._last_route_completion = 0.0
        self._warned_lateral = False
        self._build_env()

    def _build_env(self) -> None:
        backend = os.environ.get("DATN_RENDER_BACKEND", "default")
        if backend not in {"default", "egl"}:
            raise ValueError("DATN_RENDER_BACKEND must be default or egl")
        if backend == "egl":
            if self.env_cfg.get("use_render", False):
                raise ValueError("EGL backend requires use_render=false")
            from panda3d.core import loadPrcFileData
            loadPrcFileData("", "load-display p3headlessgl\naux-display p3headlessgl\nsync-video false")
        try:
            from metadrive.component.sensors.rgb_camera import RGBCamera
            from metadrive.envs.metadrive_env import MetaDriveEnv
        except ImportError as exc:
            raise ImportError(
                "MetaDrive is not installed. Run: pip install -e ./metadrive"
            ) from exc

        h = int(self.env_cfg.get("image_height", 84))
        w = int(self.env_cfg.get("image_width", 84))
        config = {
            "use_render": bool(self.env_cfg.get("use_render", False)),
            "image_observation": bool(self.env_cfg.get("image_observation", True)),
            "num_scenarios": int(self.env_cfg.get("num_scenarios", 100)),
            "start_seed": int(self.env_cfg.get("start_seed", 0)),
            "horizon": int(self.env_cfg.get("horizon", 1000)),
            "traffic_density": float(self.env_cfg.get("traffic_density", 0.1)),
            "accident_prob": float(self.env_cfg.get("accident_prob", 0.0)),
            "manual_control": bool(self.env_cfg.get("manual_control", False)),
            "decision_repeat": int(self.env_cfg.get("decision_repeat", 5)),
            "disable_model_compression": bool(
                self.env_cfg.get("disable_model_compression", True)
            ),
            "show_skybox": bool(self.env_cfg.get("show_skybox", True)),
            "show_terrain": bool(self.env_cfg.get("show_terrain", True)),
            "vehicle_config": {
                "image_source": self.env_cfg.get("image_source", "rgb_camera"),
            },
            "sensors": {
                "rgb_camera": (RGBCamera, w, h),
            },
            "stack_size": int(self.env_cfg.get("stack_size", 1)),
        }
        for k in ("driving_reward", "speed_reward", "use_lateral_reward", "map"):
            if k in self.env_cfg and self.env_cfg[k] is not None:
                config[k] = self.env_cfg[k]
        self._env = MetaDriveEnv(config)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        if seed is not None:
            obs, info = self._env.reset(seed=seed)
        else:
            obs, info = self._env.reset()
        self._last_route_completion = float(info.get("route_completion", 0.0) or 0.0)
        return self._format_obs(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        info["reward_base"] = float(reward)
        info["lateral_available"] = False
        vehicle = getattr(self._env, "agent", None) or getattr(self._env, "current_track_vehicle", None) or getattr(self._env, "vehicle", None)
        if vehicle is not None and hasattr(vehicle, "navigation") and vehicle.navigation is not None:
            try:
                ref_lanes = vehicle.navigation.current_ref_lanes
                current_lane = vehicle.lane if (hasattr(vehicle, "lane") and vehicle.lane in ref_lanes) else (ref_lanes[0] if ref_lanes else None)
                if current_lane is not None:
                    _, lat_now = current_lane.local_coordinates(vehicle.position)
                    lane_w = float(vehicle.navigation.get_current_lane_width())
                    lat_factor = max(0.0, 1.0 - 2.0 * abs(float(lat_now)) / max(lane_w, 1.0))
                    info["lateral_factor"] = float(lat_factor)
                    info["lateral_dist"] = float(abs(lat_now))
                    info["lateral_available"] = True
            except Exception:
                pass

        if self.env_cfg.get("use_custom_reward", True) and not info["lateral_available"] and not self._warned_lateral:
            warnings.warn("Lane coordinates unavailable: custom lateral reward is inactive", RuntimeWarning)
            self._warned_lateral = True

        shaped = 0.0
        if self.env_cfg.get("use_custom_reward", True):
            shaped = apply_custom_reward(
                info,
                self.env_cfg.get("reward", {}),
                last_route_completion=self._last_route_completion,
            )
            if shaped != 0.0:
                reward = float(reward) + shaped
        info["reward_shaping"] = shaped
        info["reward_total"] = float(reward)
        self._last_route_completion = float(info.get("route_completion", 0.0) or 0.0)
        done = bool(terminated or truncated)
        return self._format_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self, *args, **kwargs):
        if self._env is not None:
            return self._env.render(*args, **kwargs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()

    def _format_obs(self, obs: Any) -> dict[str, np.ndarray]:
        """
        Convert MetaDrive obs into {"image", "state"}.

        MetaDrive may return a dict when image_observation=True, or a flat
        vector when images are disabled. We always expose both keys.
        """
        if isinstance(obs, dict):
            # Typical keys vary by MetaDrive version: image / rgb_camera / top_down...
            image = None
            for key in ("image", "rgb_camera", "rgb"):
                if key in obs:
                    image = obs[key]
                    break
            if image is None:
                # Fallback: first ndarray with ndim >= 3
                for v in obs.values():
                    if isinstance(v, np.ndarray) and v.ndim >= 3:
                        image = v
                        break
            if image is None:
                raise KeyError(f"No image found in obs keys={list(obs.keys())}")

            # State: prefer explicit vector fields; else concatenate non-image arrays
            if "state" in obs:
                state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            else:
                parts = []
                for k, v in obs.items():
                    if k in ("image", "rgb_camera", "rgb"):
                        continue
                    if isinstance(v, np.ndarray) and v.ndim <= 2:
                        parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
                state = (
                    np.concatenate(parts, axis=0)
                    if parts
                    else np.zeros((1,), dtype=np.float32)
                )

            target_dim = int(self.env_cfg.get("state_dim", 23))
            if target_dim == 23 and state.shape[0] == 19:
                c4 = self._extract_surrounding_features()
                state = np.concatenate([state, c4], axis=0).astype(np.float32)

            return {"image": _to_hwc_uint8(image), "state": state}

        # Flat vector observation (camera off)
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        target_dim = int(self.env_cfg.get("state_dim", 23))
        if target_dim == 23 and flat.shape[0] == 19:
            c4 = self._extract_surrounding_features()
            flat = np.concatenate([flat, c4], axis=0).astype(np.float32)

        h = int(self.env_cfg.get("image_height", 84))
        w = int(self.env_cfg.get("image_width", 84))
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        return {"image": dummy, "state": flat}

    def _extract_surrounding_features(self) -> np.ndarray:
        """Extract 4 dynamic obstacle features in ego frame (lead dist, rel v, lat offset, intersection threat)."""
        vehicle = (
            getattr(self._env, "agent", None)
            or getattr(self._env, "current_track_vehicle", None)
            or getattr(self._env, "vehicle", None)
        )
        if vehicle is None or not hasattr(vehicle, "lidar") or vehicle.lidar is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        objs = vehicle.lidar.get_surrounding_objects(vehicle)
        if not objs:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        ego_heading = float(vehicle.heading_theta)
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)
        ego_speed_ms = float(vehicle.speed_km_h) / 3.6
        ego_pos = vehicle.position

        lead_dist = 50.0
        lead_rel_v = 0.0
        lead_lat = 0.0
        lead_obj = None

        intersection_threat = 0.0
        lane_idx = getattr(vehicle, "lane_index", None)
        is_intersection_approach = False
        if isinstance(lane_idx, tuple) and len(lane_idx) >= 2:
            is_intersection_approach = "X" in str(lane_idx[0]) or "X" in str(lane_idx[1])

        for obj in objs:
            rel = obj.position - ego_pos
            dx = float(rel[0] * cos_h + rel[1] * sin_h)
            dy = float(-rel[0] * sin_h + rel[1] * cos_h)

            obj_heading = float(getattr(obj, "heading_theta", ego_heading))
            heading_diff = obj_heading - ego_heading
            heading_cos = float(np.cos(heading_diff))
            heading_sin = float(np.sin(heading_diff))
            obj_speed_ms = float(getattr(obj, "speed_km_h", 0.0)) / 3.6
            v_y = obj_speed_ms * heading_sin

            # Ignore oncoming traffic in opposing lanes
            if heading_cos < -0.40 and abs(dy) >= 1.5:
                continue

            # 1. Lead vehicle detection in ego lane corridor
            if 0.5 < dx < 50.0:
                if abs(dy) < 2.15:
                    if dx < lead_dist:
                        lead_dist = dx
                        lead_obj = obj
                        lead_lat = dy

            # 2. Cross-traffic detection at intersections (only vehicles inside or moving toward ego path)
            if is_intersection_approach and 1.0 < dx < 22.0 and abs(dy) < 6.0:
                if abs(dy) < 2.2 or (dy * v_y < -0.2):
                    dist = np.linalg.norm(rel)
                    threat = float(np.clip(1.0 - (dist / 18.0), 0.0, 1.0))
                    if threat > intersection_threat:
                        intersection_threat = threat

        if lead_obj is not None:
            lead_heading = float(getattr(lead_obj, "heading_theta", ego_heading))
            lead_vx = (float(getattr(lead_obj, "speed_km_h", 0.0)) / 3.6) * max(0.0, float(np.cos(lead_heading - ego_heading)))
            lead_rel_v = float(np.clip((lead_vx - ego_speed_ms) / 20.0, -1.0, 1.0))
            lead_lat = float(np.clip(lead_lat / 3.5, -1.0, 1.0))

        norm_lead_dist = float(np.clip(lead_dist / 50.0, 0.0, 1.0))

        return np.array(
            [norm_lead_dist, lead_rel_v, lead_lat, intersection_threat],
            dtype=np.float32,
        )



def make_env(env_cfg: dict[str, Any]) -> MetaDriveImageEnv:
    """Factory used by scripts/training."""
    return MetaDriveImageEnv(env_cfg)
