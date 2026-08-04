#!/usr/bin/env python
"""
Main training entrypoint (Phase B/C).

Phase A: validates configs and reminds you to collect bootstrap data first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

from utils.config import load_experiment_configs
from utils.logger import Logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train World Model + Actor-Critic")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default=None)
    p.add_argument("--wm-config", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_experiment_configs(args.config, args.env_config, args.wm_config)
    train_cfg = configs["train"]
    set_seed(int(train_cfg.get("seed", 0)))

    logger = Logger(
        experiment_name=train_cfg.get("experiment_name", "wm_metadrive"),
        log_dir=train_cfg.get("paths", {}).get("log_dir", "logs"),
        wandb_cfg=train_cfg.get("wandb", {}),
    )
    logger.log(
        {
            "status": "skeleton",
            "message": "Implement Trainer.fit() after World Model (Phase B) and Actor-Critic (Phase C).",
            "hint": "Run: python scripts/collect_bootstrap.py --config configs/train.yaml",
        }
    )
    logger.finish()
    raise SystemExit(
        "training/trainer.py is not implemented yet. "
        "Finish Phase A collection, then implement Phase B World Model training."
    )


if __name__ == "__main__":
    main()
