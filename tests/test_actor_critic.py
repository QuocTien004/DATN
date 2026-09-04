from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.encoder import Encoder
from models.predictors import ContinuePredictor, RewardPredictor
from models.rssm import RSSM
from training.train_agent import (
    imagine_rollout,
    lambda_return,
    posterior_start_states,
    train_actor_critic_step,
)


def _config() -> dict:
    return {
        "actor_hidden": [16, 16],
        "critic_hidden": [16, 16],
        "min_std": 0.1,
        "max_std": 1.0,
        "init_std": 0.4,
        "imagination_horizon": 4,
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_scale": 1e-4,
        "grad_clip": 10.0,
    }


def _state(batch_size: int, deter_dim: int, groups: int, classes: int) -> dict:
    z = torch.zeros(batch_size, groups, classes)
    z[..., 0] = 1.0
    return {"h": torch.randn(batch_size, deter_dim), "z": z}


class FakeRSSM(nn.Module):
    def __init__(self, deter_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_projection = nn.Linear(action_dim, deter_dim, bias=False)
        nn.init.constant_(self.action_projection.weight, 0.25)

    def imagine_step(self, previous: dict, action: torch.Tensor):
        h = torch.tanh(previous["h"] + self.action_projection(action))
        return {"h": h, "z": previous["z"]}, {"prior_logits": previous["z"]}


class FakeReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, _z: torch.Tensor) -> torch.Tensor:
        return self.scale * h.mean(dim=-1, keepdim=True)


class FakeContinue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(2.0))

    def forward(self, h: torch.Tensor, _z: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(h.shape[0], 1)


class ActorCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch_size = 5
        self.deter_dim = 6
        self.groups = 2
        self.classes = 3
        self.stoch_size = self.groups * self.classes
        self.action_dim = 2
        self.actor_critic = ActorCritic(
            _config(), self.deter_dim, self.stoch_size, self.action_dim
        )
        self.start = _state(
            self.batch_size, self.deter_dim, self.groups, self.classes
        )

    def test_actor_and_critic_shapes_ranges_and_finiteness(self) -> None:
        stochastic = self.actor_critic.act(self.start)
        deterministic_a = self.actor_critic.act(self.start, deterministic=True)
        deterministic_b = self.actor_critic.act(self.start, deterministic=True)
        value = self.actor_critic.value(self.start)

        self.assertEqual(stochastic.action.shape, (self.batch_size, self.action_dim))
        self.assertEqual(stochastic.log_prob.shape, (self.batch_size,))
        self.assertEqual(stochastic.entropy.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(stochastic.action).all())
        self.assertTrue(torch.all(stochastic.action >= -1.0))
        self.assertTrue(torch.all(stochastic.action <= 1.0))
        self.assertTrue(torch.equal(deterministic_a.action, deterministic_b.action))
        self.assertEqual(value.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(value).all())

    def test_lambda_return_hand_computed_case(self) -> None:
        reward = torch.tensor([[1.0], [2.0]])
        discount = torch.full((2, 1), 0.5)
        next_value = torch.tensor([[3.0], [4.0]])
        result = lambda_return(
            reward, discount, next_value, bootstrap=torch.tensor([4.0]), lambda_=0.5
        )
        expected = torch.tensor([[2.75], [4.0]])
        self.assertEqual(result.shape, (2, 1))
        self.assertTrue(torch.allclose(result, expected))

    def test_imagination_and_update_change_only_actor_critic(self) -> None:
        world_model = {
            "rssm": FakeRSSM(self.deter_dim, self.action_dim),
            "reward": FakeReward(),
            "continue": FakeContinue(),
        }
        trajectory = imagine_rollout(
            self.start,
            world_model,
            self.actor_critic,
            horizon=3,
            gamma=0.99,
        )
        self.assertEqual(trajectory.h.shape, (4, self.batch_size, self.deter_dim))
        self.assertEqual(
            trajectory.z.shape,
            (4, self.batch_size, self.groups, self.classes),
        )
        self.assertEqual(trajectory.action.shape, (3, self.batch_size, self.action_dim))
        self.assertEqual(trajectory.reward.shape, (3, self.batch_size))
        self.assertEqual(trajectory.discount.shape, (3, self.batch_size))
        self.assertEqual(trajectory.value.shape, (4, self.batch_size))

        actor_before = [p.detach().clone() for p in self.actor_critic.actor.parameters()]
        critic_before = [p.detach().clone() for p in self.actor_critic.critic.parameters()]
        world_before = {
            name: [p.detach().clone() for p in module.parameters()]
            for name, module in world_model.items()
        }
        actor_optimizer = torch.optim.Adam(
            self.actor_critic.actor.parameters(), lr=1e-2
        )
        critic_optimizer = torch.optim.Adam(
            self.actor_critic.critic.parameters(), lr=1e-2
        )
        metrics = train_actor_critic_step(
            self.start,
            world_model,
            self.actor_critic,
            actor_optimizer,
            critic_optimizer,
            _config(),
        )

        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(actor_before, self.actor_critic.actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(critic_before, self.actor_critic.critic.parameters())
            )
        )
        for name, module in world_model.items():
            for before, after in zip(world_before[name], module.parameters()):
                self.assertTrue(torch.equal(before, after))

    def test_real_rssm_imagination_smoke(self) -> None:
        wm_cfg = {
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
        rssm = RSSM(wm_cfg, embed_dim=10, action_dim=2)
        actor_critic = ActorCritic(_config(), 8, 6, 2)
        world_model = {
            "rssm": rssm,
            "reward": RewardPredictor(wm_cfg, 8, 6),
            "continue": ContinuePredictor(wm_cfg, 8, 6),
        }
        trajectory = imagine_rollout(
            rssm.initial_state(3),
            world_model,
            actor_critic,
            horizon=2,
            gamma=0.99,
        )
        self.assertEqual(trajectory.action.shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(trajectory.reward).all())

    def test_posterior_start_state_extraction(self) -> None:
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
        batch = {
            "image": torch.rand(2, 3, 3, 32, 32),
            "state": torch.rand(2, 3, 4),
            "action": torch.rand(2, 3, 2) * 2.0 - 1.0,
            "done": torch.zeros(2, 3, dtype=torch.bool),
        }
        starts = posterior_start_states(
            batch, {"encoder": encoder, "rssm": rssm}, max_states=4
        )
        self.assertEqual(starts["h"].shape, (4, 8))
        self.assertEqual(starts["z"].shape, (4, 2, 3))
        self.assertFalse(starts["h"].requires_grad)


if __name__ == "__main__":
    unittest.main()
