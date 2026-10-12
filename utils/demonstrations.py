"""Content-bound provenance for expert imitation data (not arbitrary replay)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np


def bounded_expert_action(action, action_dim: int = 2) -> np.ndarray:
    """Record the bounded action actually sent to MetaDrive, not a raw PPO mean."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (action_dim,) or not np.isfinite(action).all():
        raise ValueError("Expert returned invalid action; no random fallback")
    return np.clip(action, -1., 1.)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_collection_manifest(path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    destination = path.with_suffix(".manifest.json")
    destination.write_text(json.dumps({
        **metadata, "schema_version": 1, "replay_sha256": file_sha256(path),
    }, indent=2), encoding="utf-8")
    return destination


def require_expert_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Expert manifest missing. Collect --policy pure_expert; do not relabel mixed/random replay.")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("policy") != "pure_expert" or metadata.get("expert_deterministic") is not True:
        raise ValueError("Imitation requires deterministic pure_expert actions, not mixed/random replay")
    if metadata.get("action_transform") != "clip[-1,1]":
        raise ValueError("Expert manifest must certify bounded executed actions: action_transform=clip[-1,1]")
    if metadata.get("replay_sha256") != file_sha256(path):
        raise ValueError("Expert replay does not match its provenance checksum")
    return metadata
