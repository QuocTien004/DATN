from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.actor_critic import ActorCritic
from models.world_model import build_world_model
from training.trainer import Trainer
from utils.checkpoint import load_checkpoint
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer, Transition


class TrainerSmokeTest(unittest.TestCase):
    def test_one_offline_update_and_checkpoint(self) -> None:
        torch.manual_seed(9)
        wm_cfg = {
            "encoder": {
                "cnn_channels": [4, 8],
                "mlp_hidden": [8],
                "embed_dim": 12,
            },
            "rssm": {
                "deter_dim": 8,
                "stoch_dim": 2,
                "stoch_classes": 3,
                "hidden_dim": 12,
            },
            "predictors": {
                "reward_hidden": [8],
                "continue_hidden": [8],
            },
        }
        ac_cfg = {
            "actor_hidden": [8],
            "critic_hidden": [8],
            "imagination_horizon": 2,
            "gamma": 0.99,
            "lambda": 0.95,
            "max_start_states": 2,
            "batch_size": 1,
            "updates": 1,
            "checkpoint_every": 0,
        }
        buffer = ReplayBuffer(
            capacity=3,
            image_shape=(32, 32, 3),
            state_dim=4,
            action_dim=2,
            sequence_length=2,
        )
        for index in range(3):
            image = np.full((32, 32, 3), index, dtype=np.uint8)
            state = np.full(4, index, dtype=np.float32)
            buffer.add(
                Transition(
                    image=image,
                    state=state,
                    action=np.zeros(2, dtype=np.float32),
                    reward=0.0,
                    done=False,
                    next_image=image,
                    next_state=state,
                ),
                is_first=index == 0,
            )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_root = Path(temporary) / "checkpoints"
            configs = {
                "train": {
                    "paths": {
                        "checkpoint_dir": str(checkpoint_root),
                        "log_dir": str(Path(temporary) / "logs"),
                    },
                    "actor_critic": ac_cfg,
                },
                "world_model": wm_cfg,
                "env": {},
            }
            device = torch.device("cpu")
            world_model = build_world_model(
                wm_cfg, (32, 32, 3), 4, 2, device, include_decoder=False
            )
            for module in world_model.values():
                module.requires_grad_(False)
            rssm = world_model["rssm"]
            actor_critic = ActorCritic(ac_cfg, rssm.deter_dim, rssm.stoch_size, 2)
            actor_optimizer = torch.optim.Adam(
                actor_critic.actor.parameters(), lr=1e-3
            )
            critic_optimizer = torch.optim.Adam(
                actor_critic.critic.parameters(), lr=1e-3
            )
            logger = Logger("smoke", Path(temporary) / "logs", {"enabled": False})
            trainer = Trainer(
                configs,
                buffer,
                logger,
                "cpu",
                world_model=world_model,
                actor_critic=actor_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
            )
            metrics = trainer.fit(updates=1, batch_size=1, checkpoint_every=0)
            checkpoint_path = checkpoint_root / "actor_critic" / "latest.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint["step"], 1)
            self.assertIn("actor_critic", checkpoint)
            self.assertTrue(np.isfinite(list(metrics.values())).all())


if __name__ == "__main__":
    unittest.main()
