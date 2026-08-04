from __future__ import annotations

from typing import Any

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
        self._build_env()

    def _build_env(self) -> None:
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
        return self._format_obs(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self.env_cfg.get("use_custom_reward", False):
            shaped = apply_custom_reward(info, self.env_cfg.get("reward", {}))
            # Keep env reward unless custom returns non-zero (skeleton returns 0).
            if shaped != 0.0:
                reward = shaped
        done = bool(terminated or truncated)
        return self._format_obs(obs), float(reward), bool(terminated), bool(truncated), info

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
            return {"image": _to_hwc_uint8(image), "state": state}

        # Flat vector observation (camera off)
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        h = int(self.env_cfg.get("image_height", 84))
        w = int(self.env_cfg.get("image_width", 84))
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        return {"image": dummy, "state": flat}


def make_env(env_cfg: dict[str, Any]) -> MetaDriveImageEnv:
    """Factory used by scripts/training."""
    return MetaDriveImageEnv(env_cfg)
