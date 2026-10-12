from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from models.actor_critic import ActorCritic
from models.world_model import build_world_model
from training.imitation import train_imitation_step
from training.train_agent import posterior_start_states
from utils.demonstrations import require_expert_manifest, write_collection_manifest, bounded_expert_action


class ImitationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        self.wm = build_world_model({
            "encoder": {"cnn_channels": [4], "mlp_hidden": [4], "embed_dim": 4},
            "rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 4},
            "predictors": {"reward_hidden": [4], "continue_hidden": [4]},
        }, (16, 16, 3), 1, 2, torch.device("cpu"), include_decoder=False)
        self.ac = ActorCritic({"actor_hidden": [8], "critic_hidden": [8]}, 4, 4, 2)
        self.optimizer = torch.optim.Adam(self.ac.actor.parameters(), lr=.01)
        self.batch = {
            "image": torch.rand(2, 4, 3, 16, 16), "state": torch.rand(2, 4, 1),
            "action": torch.full((2, 4, 2), .6),
            "is_first": torch.tensor([[True, False, True, False], [True, False, False, False]]),
        }

    def test_updates_only_actor_and_restores_world_model_modes(self):
        before = {name: {key: value.clone() for key, value in m.state_dict().items()} for name, m in self.wm.items()}
        critic_before = {key: value.clone() for key, value in self.ac.critic.state_dict().items()}
        slow_before = {key: value.clone() for key, value in self.ac.slow_critic.state_dict().items()}
        self.wm["rssm"].eval()
        losses = [train_imitation_step(self.batch, self.wm, self.ac, self.optimizer)["bc_mse"] for _ in range(15)]
        self.assertLess(losses[-1], losses[0])
        for name, module in self.wm.items():
            self.assertEqual(module.training, name != "rssm")
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        for module, saved in ((self.ac.critic, critic_before), (self.ac.slow_critic, slow_before)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_posterior_order_matches_separately_filtered_sequences(self):
        together = posterior_start_states(self.batch, self.wm, deterministic=True)
        for i in range(2):
            separate = posterior_start_states({k: v[i:i+1] for k, v in self.batch.items()}, self.wm, deterministic=True)
            for key in ("h", "z"):
                torch.testing.assert_close(together[key][i*4:(i+1)*4], separate[key])
                self.assertFalse(together[key].requires_grad)

    def test_burn_in_is_excluded_and_features_are_not_subsampled(self):
        self.batch["action"][:, 0] = -.8
        states = posterior_start_states(self.batch, self.wm, deterministic=True)
        with torch.no_grad():
            prediction = self.ac.act(states, deterministic=True).mean.reshape(2, 4, 2)
            expected = (prediction[:, 1:] - self.batch["action"][:, 1:]).square().mean()
        with patch("training.imitation.posterior_start_states", return_value=states) as observe:
            result = train_imitation_step(self.batch, self.wm, self.ac, self.optimizer, burn_in=1)
        self.assertAlmostEqual(result["bc_mse"], expected.item(), places=6)
        self.assertNotIn("max_states", observe.call_args.kwargs)

    def test_invalid_labels_or_optimizer_fail(self):
        with self.assertRaisesRegex(ValueError, "burn_in"):
            train_imitation_step(self.batch, self.wm, self.ac, self.optimizer, burn_in=4)
        self.batch["action"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            train_imitation_step(self.batch, self.wm, self.ac, self.optimizer)
        self.batch["action"][0, 0, 0] = 2.
        with self.assertRaisesRegex(ValueError, "bounds"):
            train_imitation_step(self.batch, self.wm, self.ac, self.optimizer)
        with self.assertRaisesRegex(ValueError, "exactly"):
            train_imitation_step(self.batch, self.wm, self.ac, torch.optim.Adam(self.ac.parameters()))

    def test_provenance_rejects_mixed_missing_and_changed_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "expert.npz"
            replay.write_bytes(b"fixture-replay")
            with self.assertRaisesRegex(ValueError, "missing"):
                require_expert_manifest(replay)
            write_collection_manifest(replay, {"policy": "mixed_expert", "expert_deterministic": False})
            with self.assertRaisesRegex(ValueError, "pure_expert"):
                require_expert_manifest(replay)
            write_collection_manifest(replay, {"policy": "pure_expert", "expert_deterministic": True})
            with self.assertRaisesRegex(ValueError, "executed"):
                require_expert_manifest(replay)
            write_collection_manifest(replay, {"policy": "pure_expert", "expert_deterministic": True, "action_transform": "clip[-1,1]"})
            self.assertEqual(require_expert_manifest(replay)["policy"], "pure_expert")
            replay.write_bytes(b"different-replay")
            with self.assertRaisesRegex(ValueError, "checksum"):
                require_expert_manifest(replay)

    def test_expert_mean_is_clipped_before_execution_and_storage(self):
        import numpy as np
        result = bounded_expert_action([-.3, 1.7])
        np.testing.assert_allclose(result, [-.3, 1.])
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "invalid"):
            bounded_expert_action([float("nan"), .2])
        with self.assertRaisesRegex(ValueError, "invalid"):
            bounded_expert_action([[.1, .2]])


if __name__ == "__main__":
    unittest.main()
