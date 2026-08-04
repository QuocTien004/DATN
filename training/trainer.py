from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer


class Trainer:
    """
    Online training loop orchestrator (collect ↔ train WM ↔ train AC ↔ eval).

    Phase A: only holds config / buffer / logger references.
    Phase B/C: fill `fit()`.
    """

    def __init__(
        self,
        configs: dict[str, Any],
        buffer: ReplayBuffer,
        logger: Logger,
        device: str = "cpu",
    ) -> None:
        self.configs = configs
        self.buffer = buffer
        self.logger = logger
        self.device = device
        self.global_step = 0

    def fit(self) -> None:
        """Main training loop — implement after World Model + Actor-Critic exist."""
        raise NotImplementedError(
            "Trainer.fit() is for Phase B/C. "
            "Use scripts/collect_bootstrap.py for Phase A data collection."
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # TODO(Phase B): save model weights + optimizer + step
        self.buffer.save(directory / "buffer.npz")
