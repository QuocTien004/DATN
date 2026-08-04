from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge configs; later dicts override earlier keys."""
    merged: dict[str, Any] = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


def load_experiment_configs(
    train_cfg_path: str | Path,
    env_cfg_path: str | Path | None = None,
    wm_cfg_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load train config and optionally attach env / world_model configs.

    Returns
    -------
    dict with keys: train, env, world_model
    """
    root = Path(train_cfg_path).resolve().parent
    train = load_config(train_cfg_path)
    env_path = env_cfg_path or root / "env_metadrive.yaml"
    wm_path = wm_cfg_path or root / "world_model.yaml"
    return {
        "train": train,
        "env": load_config(env_path),
        "world_model": load_config(wm_path),
    }
