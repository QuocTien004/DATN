from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from models.actor_critic import ActorCritic
from models.world_model import build_world_model
from training.train_world_model import train_world_model_step


class JointWorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.cfg = {
            "encoder": {"cnn_channels": [4], "mlp_hidden": [4], "embed_dim": 4},
            "rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 4},
            "decoder": {"cnn_channels": [4, 4]},
            "predictors": {"reward_hidden": [4], "continue_hidden": [4]},
            "training": {"grad_clip": 1e9},
        }
        self.wm = build_world_model(self.cfg, (16, 16, 3), 3, 2, torch.device("cpu"))
        self.ac = ActorCritic({"actor_hidden": [8], "critic_hidden": [8]}, 4, 4, 2)
        self.parameters = [p for m in self.wm.values() for p in m.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters + list(self.ac.actor.parameters()), lr=.001)
        self.batch = {
            "image": torch.rand(2, 3, 3, 16, 16), "state": torch.rand(2, 3, 3),
            "next_image": torch.rand(2, 3, 3, 16, 16), "next_state": torch.rand(2, 3, 3),
            "action": torch.linspace(-.8, .8, 12).reshape(2, 3, 2), "reward": torch.rand(2, 3),
            "done": torch.zeros(2, 3), "is_first": torch.tensor([[1, 0, 0], [1, 0, 0]]),
        }

    def test_joint_updates_representation_actor_and_heads_but_not_critics(self):
        old = {name: copy.deepcopy(module.state_dict()) for name, module in self.wm.items()}
        old_ac = copy.deepcopy(self.ac.state_dict())
        row = train_world_model_step(self.batch, self.wm, self.optimizer, self.cfg,
                                    imitation_actor=self.ac.actor, imitation_weight=10.)
        self.assertGreater(row["loss_imitation"], 0)
        for name, module in self.wm.items():
            self.assertTrue(any(not torch.equal(value, old[name][key]) for key, value in module.state_dict().items()), name)
        self.assertTrue(any(not torch.equal(value, old_ac["actor."+key]) for key, value in self.ac.actor.state_dict().items()))
        for prefix, module in (("critic.", self.ac.critic), ("slow_critic.", self.ac.slow_critic)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, old_ac[prefix+key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_joint_loss_uses_current_features_and_matching_action_order(self):
        optimizer = torch.optim.SGD(self.parameters + list(self.ac.actor.parameters()), lr=0.)
        torch.manual_seed(27)
        base = train_world_model_step(self.batch, self.wm, optimizer, self.cfg)
        representation_parameters = [p for name in ("encoder", "rssm") for p in self.wm[name].parameters()]
        base_representation_grad = [p.grad.clone() for p in representation_parameters]
        torch.manual_seed(27)
        captured = {}
        original_forward = self.ac.actor.forward
        def observe(h, z, **kwargs):
            captured["h"] = h
            result = original_forward(h, z, **kwargs)
            captured["mean"] = result.mean.detach()
            return result
        with patch.object(self.ac.actor, "forward", side_effect=observe):
            joint = train_world_model_step(self.batch, self.wm, optimizer, self.cfg,
                                          imitation_actor=self.ac.actor, imitation_weight=7.)
        expected = (captured["mean"] - self.batch["action"].flatten(0, 1)).square().mean()
        self.assertTrue(captured["h"].requires_grad)
        self.assertAlmostEqual(joint["loss_imitation"], expected.item(), places=6)
        self.assertAlmostEqual(joint["loss"]-base["loss"], 7*expected.item(), places=5)
        self.assertTrue(any(not torch.equal(old, p.grad)
                            for old, p in zip(base_representation_grad, representation_parameters)))
        for key in ("loss_recon_img", "loss_recon_state", "loss_reward", "loss_continue", "loss_kl"):
            self.assertEqual(base[key], joint[key])

    def test_bootstrap_update_clears_old_actor_gradients_and_leaves_actor_fixed(self):
        train_world_model_step(self.batch, self.wm, self.optimizer, self.cfg,
                               imitation_actor=self.ac.actor, imitation_weight=10.)
        before = copy.deepcopy(self.ac.actor.state_dict())
        row = train_world_model_step(self.batch, self.wm, self.optimizer, self.cfg)
        self.assertEqual(row["loss_imitation"], 0.)
        for key, value in self.ac.actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.ac.actor.parameters()))

    def test_rejects_missing_actor_bad_weight_labels_and_critic_optimizer(self):
        for kwargs in ({"imitation_weight": 1.}, {"imitation_actor": self.ac.actor},
                       {"imitation_actor": self.ac.actor, "imitation_weight": -1.},
                       {"imitation_actor": self.ac.actor, "imitation_weight": float("nan")},
                       {"imitation_actor": self.ac.actor, "imitation_weight": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_world_model_step(self.batch, self.wm, self.optimizer, self.cfg, **kwargs)
        bad_opt = torch.optim.Adam(self.parameters + list(self.ac.parameters()), lr=.001)
        with self.assertRaisesRegex(ValueError, "exactly"):
            train_world_model_step(self.batch, self.wm, bad_opt, self.cfg,
                                   imitation_actor=self.ac.actor, imitation_weight=1.)
        for value in (1.1, float("nan")):
            self.batch["action"][0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "bounded"):
                train_world_model_step(self.batch, self.wm, self.optimizer, self.cfg,
                                       imitation_actor=self.ac.actor, imitation_weight=1.)


if __name__ == "__main__":
    unittest.main()
