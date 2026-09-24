"""Resolve and validate model/replay pairs before allocating environments."""
from pathlib import Path
from typing import Any


def resolve_resume_path(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path / "online/latest.pt", path / "latest.pt"] if path.is_dir() else [path, path.parent / "online" / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Online checkpoint not found: {path}")


def resolve_replay_path(checkpoint_path: Path, checkpoint: dict[str, Any], explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
    elif checkpoint.get("replay_path"):
        path = checkpoint_path.parent / checkpoint["replay_path"]
    else:
        raise ValueError("Checkpoint has no replay link. Pass its matching --buffer; never resume from bootstrap data.")
    if not path.is_file():
        raise FileNotFoundError(f"Matching online replay missing: {path}")
    return path


def validate_replay_step(checkpoint: dict[str, Any], buffer_step: int | None) -> None:
    step = int(checkpoint["env_steps"])
    if buffer_step != step:
        raise ValueError(f"Checkpoint step {step} != replay step {buffer_step}. Restore a matching latest.pt + online_buffer.npz pair.")


def require_current_contract(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("transition_contract") != 2:
        raise ValueError("Legacy WM uses misaligned transition targets. Evaluate it with scripts/eval.py; pretrain a new WM before retraining/resuming with this code.")
