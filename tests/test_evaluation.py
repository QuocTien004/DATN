from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate import LatentActorPolicy, evaluate_policy
from evaluation.safety_shield import AdaptiveTrafficAvoidanceShield, SafeLatentActorPolicy
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
    def test_eval_distinguishes_timeout_from_terminal(self) -> None:
        class TimeoutEnv(ShortEpisodeEnv):
            env_cfg = {"horizon": 2}
            def step(self, action):
                obs, reward, done, _, info = super().step(action)
                info["arrive_dest"] = False
                return obs, reward, False, done, info
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timeout.json"
            evaluate_policy(TimeoutEnv(), ResettablePolicy(), num_episodes=1, output_path=output)
            result = json.loads(output.read_text())
            self.assertEqual(result["horizon"], 2)
            self.assertTrue(result["episodes"][0]["truncated"])
            self.assertFalse(result["episodes"][0]["terminated"])
            self.assertEqual(result["aggregate"]["success_rate"], 0.)

    def test_eval_rejects_empty_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate_policy(ShortEpisodeEnv(), ResettablePolicy(), num_episodes=0)

    def test_collision_breakdown_and_per_seed_output(self) -> None:
        class CrashEnv(ShortEpisodeEnv):
            def step(self, action):
                obs, reward, done, trunc, info = super().step(action)
                info.update(arrive_dest=False, crash_vehicle=done)
                return obs, reward, done, trunc, info
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval.json"
            metrics = evaluate_policy(CrashEnv(), ResettablePolicy(), num_episodes=2, start_seed=50, output_path=output)
            self.assertEqual(metrics["crash_vehicle_rate"], 1.0)
            self.assertEqual(metrics["out_of_road_rate"], 0.0)
            self.assertEqual([r["seed"] for r in json.loads(output.read_text())["episodes"]], [50, 51])

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

    def test_safety_shield_modulation(self) -> None:
        shield = AdaptiveTrafficAvoidanceShield(
            corridor_width=2.4, threat_distance=28.0, min_gap=5.0, desired_time_gap=1.2
        )
        base_act = np.array([0.0, 0.8], dtype=np.float32)

        # 1. No vehicle -> untouched
        act, diag = shield.modulate_action(base_act, None)
        np.testing.assert_array_equal(act, base_act)
        self.assertEqual(diag["threat_detected"], 0.0)

        # 2. Mock vehicle with lidar and mock lead obstacle
        class MockObstacle:
            def __init__(self, pos, speed=0.0):
                self.position = np.array(pos, dtype=np.float32)
                self.speed_km_h = speed

        class MockLidar:
            def __init__(self, objs):
                self.objs = objs
            def get_surrounding_objects(self, _v):
                return self.objs

        class MockVehicle:
            def __init__(self, objs, speed=36.0, heading=0.0):
                self.position = np.array([0.0, 0.0], dtype=np.float32)
                self.speed_km_h = speed
                self.heading_theta = heading
                self.lidar = MockLidar(objs)

        # 2a. Obstacle outside corridor (dy=5.0) -> untouched
        v_far = MockVehicle([MockObstacle([10.0, 5.0])])
        act, diag = shield.modulate_action(base_act, v_far)
        np.testing.assert_array_equal(act, base_act)

        # 2b. Obstacle inside corridor (dx=10.0, dy=0.0), ego speed=36km/h (10m/s), desired gap = 5 + 12 = 17m
        # Gap ratio = 10 / 17 -> progressive braking
        v_lead = MockVehicle([MockObstacle([10.0, 0.0])])
        act, diag = shield.modulate_action(base_act, v_lead)
        self.assertEqual(diag["threat_detected"], 1.0)
        self.assertLess(act[1], 0.0)

        # 2c. Obstacle within emergency min gap (dx=3.0, dy=0.0) -> max decel (-1.0)
        v_close = MockVehicle([MockObstacle([3.0, 0.0])])
        act, diag = shield.modulate_action(base_act, v_close)
        self.assertEqual(act[1], -1.0)
        self.assertEqual(diag["throttle_override"], 1.0)

    def test_safe_latent_actor_policy_wrapper(self) -> None:
        policy = ResettablePolicy()
        shield = AdaptiveTrafficAvoidanceShield()
        safe_policy = SafeLatentActorPolicy(policy, shield=shield, env=ShortEpisodeEnv())
        safe_policy.reset()
        self.assertEqual(policy.resets, 1)
        action = safe_policy({"image": np.zeros((4, 4, 3)), "state": np.zeros(2)})
        self.assertEqual(action.shape, (2,))

    def test_evaluate_policy_with_shield(self) -> None:
        policy = ResettablePolicy()
        shield = AdaptiveTrafficAvoidanceShield()
        metrics = evaluate_policy(
            ShortEpisodeEnv(), policy, num_episodes=2, start_seed=100, shield=shield
        )
        self.assertEqual(policy.resets, 2)
        self.assertEqual(metrics["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

