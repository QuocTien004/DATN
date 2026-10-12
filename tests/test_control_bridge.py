"""The observed-state bridge is BC-only until a learned prior is validated."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluation.evaluate import LatentActorPolicy
from models.actor_critic import ActorCritic
from models.world_model import build_world_model
from training.imitation import train_imitation_step
from training.train_agent import posterior_start_states, imagine_rollout, train_actor_critic_step
from training.trainer import Trainer
from training.online_trainer import OnlineTrainer


class ControlBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.cfg = dict(actor_hidden=[8], critic_hidden=[8], control_state_dim=3,
                        control_state_mean=[.1, .2, .3], control_state_std=[.2, .3, .4],
                        mean_transform="identity")
        self.ac = ActorCritic(self.cfg, 4, 4, 2)
        self.wm = build_world_model({
            "encoder": {"cnn_channels": [4], "mlp_hidden": [4], "embed_dim": 4},
            "rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 4},
            "predictors": {"reward_hidden": [4], "continue_hidden": [4]},
        }, (16, 16, 3), 3, 2, torch.device("cpu"), include_decoder=False)
        self.batch = dict(image=torch.rand(2, 4, 3, 16, 16), state=torch.rand(2, 4, 3),
                          action=torch.full((2, 4, 2), .3),
                          is_first=torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0]]))

    def test_default_keeps_legacy_parameter_shapes_and_checkpoint_keys(self):
        legacy = ActorCritic(dict(actor_hidden=[8], critic_hidden=[8]), 4, 4, 2)
        self.assertEqual(legacy.actor.net[0].weight.shape, (8, 8))
        self.assertFalse(any("control" in key for key in legacy.state_dict()))
        legacy.require_imagination_ready()

    def test_feature_order_and_normalization_match_diagnostic(self):
        state = posterior_start_states(self.batch, self.wm, deterministic=True, include_control=True)
        x = torch.cat([state["h"], state["z"].flatten(1),
                       (state["control"]-self.ac.actor.control_state_mean)/self.ac.actor.control_state_std], 1)
        expected = torch.tanh(self.ac.actor.net(x)[:, :2])
        actual = self.ac.act(state, deterministic=True).action
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state["control"], self.batch["state"].flatten(0, 1), rtol=0, atol=0)
        self.assertFalse(state["control"].requires_grad)

    def test_subsample_preserves_control_alignment(self):
        full = posterior_start_states(self.batch, self.wm, deterministic=True, include_control=True)
        order = torch.arange(7, -1, -1)
        with patch("training.train_agent.torch.randperm", return_value=order):
            few = posterior_start_states(self.batch, self.wm, deterministic=True, include_control=True, max_states=3)
        for key in full:
            torch.testing.assert_close(few[key], full[key][order[:3]])

    def test_bad_stats_and_missing_control_are_rejected(self):
        for change in (dict(control_state_std=[0., 1., 1.]), dict(control_state_mean=[0.]),
                       dict(control_state_std=[float("nan"), 1., 1.]), dict(control_state_dim=-1)):
            with self.assertRaises(ValueError):
                ActorCritic(dict(self.cfg, **change), 4, 4, 2)
        state = posterior_start_states(self.batch, self.wm, deterministic=True)
        with self.assertRaisesRegex(ValueError, "control_state"):
            self.ac.act(state, deterministic=True)
        for control in (torch.zeros(8, 2), torch.full((8, 3), float("inf"))):
            with self.assertRaisesRegex(ValueError, "control_state"):
                self.ac.act(dict(state, control=control), deterministic=True)

    def test_bc_updates_actor_only_with_matching_current_controls(self):
        before = {name: copy.deepcopy(module.state_dict()) for name, module in self.wm.items()}
        critics = copy.deepcopy(self.ac.critic.state_dict())
        opt = torch.optim.Adam(self.ac.actor.parameters(), lr=.01)
        losses = [train_imitation_step(self.batch, self.wm, self.ac, opt)["bc_mse"] for _ in range(10)]
        self.assertLess(losses[-1], losses[0])
        for name, module in self.wm.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
        for key, value in self.ac.critic.state_dict().items():
            torch.testing.assert_close(value, critics[key], rtol=0, atol=0)

    def test_imagination_and_rl_fail_before_any_model_update(self):
        state = posterior_start_states(self.batch, self.wm, deterministic=True, include_control=True)
        before = copy.deepcopy(self.ac.state_dict())
        with self.assertRaisesRegex(ValueError, "no validated.*prior"):
            imagine_rollout(state, {}, self.ac, horizon=3, gamma=.99)
        with self.assertRaisesRegex(ValueError, "no validated.*prior"):
            train_actor_critic_step(state, {}, self.ac, torch.optim.Adam(self.ac.actor.parameters()),
                                    torch.optim.Adam(self.ac.critic.parameters()), {})
        with self.assertRaisesRegex(ValueError, "no validated.*prior"):
            Trainer({}, None, None, actor_critic=self.ac)
        with self.assertRaisesRegex(ValueError, "no validated.*prior"):
            OnlineTrainer({}, None, {}, self.ac, None, None, None, None, torch.device("cpu"), None, {}, {})
        for key, value in self.ac.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_policy_uses_current_observation_and_resets_memory(self):
        policy = LatentActorPolicy(self.wm["encoder"], self.wm["rssm"], self.ac, device=torch.device("cpu"))
        obs = dict(image=np.zeros((16, 16, 3), dtype=np.uint8), state=np.array([.4, .5, .6], dtype=np.float32))
        with patch.object(self.ac.actor, "forward", wraps=self.ac.actor.forward) as forward:
            first = policy(obs)
            torch.testing.assert_close(forward.call_args.kwargs["control_state"], torch.tensor([[.4, .5, .6]]))
            policy(dict(obs, state=np.zeros(3, dtype=np.float32)))
            torch.testing.assert_close(forward.call_args.kwargs["control_state"], torch.zeros(1, 3))
        policy.reset()
        np.testing.assert_array_equal(first, policy(obs))

    def test_checkpoint_roundtrip_preserves_bridge_and_bounded_identity_mean(self):
        state = posterior_start_states(self.batch, self.wm, deterministic=True, include_control=True)
        with torch.no_grad():
            self.ac.actor.net[-1].bias[:2].fill_(30.)
        loaded = ActorCritic(self.cfg, 4, 4, 2)
        loaded.load_state_dict(copy.deepcopy(self.ac.state_dict()))
        a = loaded.act(state, deterministic=True).action
        torch.testing.assert_close(a, self.ac.act(state, deterministic=True).action, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a).all() and (a.abs() <= 1).all())


if __name__ == "__main__":
    unittest.main()
