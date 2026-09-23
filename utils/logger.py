from __future__ import annotations

from pathlib import Path
from typing import Any


class Logger:
    """
    Thin logging wrapper.

    - Always prints key scalars.
    - Optionally logs to Weights & biases when enabled in config.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str | Path,
        wandb_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._wandb = None

        wandb_cfg = wandb_cfg or {}
        if wandb_cfg.get("enabled"):
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=wandb_cfg.get("project", "worldmodel-metadrive"),
                    entity=wandb_cfg.get("entity"),
                    name=experiment_name,
                    dir=str(self.log_dir),
                )
            except Exception as exc:  # noqa: BLE001 — keep training usable offline
                print(f"[Logger] wandb init failed, continuing without it: {exc}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import json
        msg = " | ".join(f"{k}={v}" for k, v in metrics.items())
        prefix = f"step={step} | " if step is not None else ""
        print(f"[log] {prefix}{msg}", flush=True)

        entry = dict(metrics)
        if step is not None:
            entry["step"] = step
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            pass

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
