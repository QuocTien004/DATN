from __future__ import annotations

import unittest

import numpy as np
import torch

from evaluation.evaluate import LatentActorPolicy, evaluate_policy
from models.actor_critic import ActorCritic
from models.encoder import Encoder
from models.rssm import RSSM


class ShortEpisodeEnv:
    def __init__(self) -> None:
        self.time = 0

    def reset(self, seed: int | None = None):
        self.time = 0
        return {"image": np.zeros((4, 4, 3)), "state": np.zeros(2)}, {
            "seed": seed
        }

    def step(self, _action: np.ndarray):
        self.time += 1
        done = self.time == 2
        return (
            {"image": np.zeros((4, 4, 3)), "state": np.zeros(2)},
            1.5,
            done,
            False,
            {
                "arrive_dest": done,
                "route_completion": self.time / 2,
                "crash": False,
            },
        )


class ResettablePolicy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def __call__(self, _observation: dict) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_policy_resets_state_and_aggregates_returns(self) -> None:
        policy = ResettablePolicy()
        metrics = evaluate_policy(
            ShortEpisodeEnv(), policy, num_episodes=3, start_seed=100
        )
        self.assertEqual(policy.resets, 3)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["crash_rate"], 0.0)
        self.assertEqual(metrics["mean_episode_return"], 3.0)
        self.assertEqual(metrics["mean_episode_length"], 2.0)

    def test_latent_policy_observation_to_action_smoke(self) -> None:
        torch.manual_seed(5)
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
        }
        encoder = Encoder(wm_cfg, state_dim=4, image_shape=(32, 32, 3))
        rssm = RSSM(wm_cfg, embed_dim=encoder.embed_dim, action_dim=2)
        actor_critic = ActorCritic(
            {"actor_hidden": [8], "critic_hidden": [8]}, 8, 6, 2
        )
        policy = LatentActorPolicy(
            encoder,
            rssm,
            actor_critic,
            device=torch.device("cpu"),
            deterministic=True,
        )
        observation = {
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "state": np.zeros(4, dtype=np.float32),
        }
        policy.reset()
        first = policy(observation)
        second = policy(observation)
        self.assertEqual(first.shape, (2,))
        self.assertEqual(second.shape, (2,))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first >= -1.0).all() and (first <= 1.0).all())
        policy.reset()
        repeated_first = policy(observation)
        np.testing.assert_array_equal(first, repeated_first)


if __name__ == "__main__":
    unittest.main()
