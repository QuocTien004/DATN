from __future__ import annotations

import copy
import unittest

import torch

from models.world_model import build_world_model
from training.train_world_model import train_world_model_step


class WorldModelLossScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.cfg = {
            "encoder": {"cnn_channels": [4], "mlp_hidden": [4], "embed_dim": 4},
            "rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 4},
            "decoder": {"cnn_channels": [4, 4]},
            "predictors": {"reward_hidden": [4], "continue_hidden": [4]},
            "training": {"grad_clip": 1e9},
        }
        self.wm = build_world_model(self.cfg, (16, 16, 3), 3, 2, torch.device("cpu"))
        self.optimizer = torch.optim.SGD([p for m in self.wm.values() for p in m.parameters()], lr=0.)
        self.batch = {
            "image": torch.rand(2, 3, 3, 16, 16), "state": torch.rand(2, 3, 3),
            "next_image": torch.rand(2, 3, 3, 16, 16), "next_state": torch.rand(2, 3, 3),
            "action": torch.rand(2, 3, 2), "reward": torch.rand(2, 3),
            "done": torch.zeros(2, 3), "is_first": torch.tensor([[1, 0, 0], [1, 0, 0]]),
        }

    def run_step(self, scale=None, *, image_scale=None, state_weights=None):
        cfg = copy.deepcopy(self.cfg)
        if scale is not None:
            cfg["training"]["state_reconstruction_scale"] = scale
        if image_scale is not None:
            cfg["training"]["image_reconstruction_scale"] = image_scale
        if state_weights is not None:
            cfg["training"]["state_reconstruction_weights"] = state_weights
        torch.manual_seed(47)
        metrics = train_world_model_step(self.batch, self.wm, self.optimizer, cfg)
        gradients = [p.grad.clone() for p in self.wm["decoder"].state_head.parameters()]
        return metrics, gradients

    def test_default_preserves_legacy_loss_and_gradients(self):
        default, before = self.run_step()
        explicit, after = self.run_step(1., image_scale=1.)
        self.assertEqual(default, explicit)
        self.assertEqual(default["loss_recon_state"], default["loss_recon_state_scaled"])
        self.assertEqual(default["loss_recon_img"], default["loss_recon_img_scaled"])
        self.assertAlmostEqual(default["loss"], sum(default[k] for k in (
            "loss_recon_img", "loss_recon_state", "loss_reward", "loss_continue", "loss_kl")), places=5)
        for first, second in zip(before, after):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_scale_changes_state_loss_and_decoder_gradient_only_as_expected(self):
        base, grad = self.run_step(1.)
        scaled, scaled_grad = self.run_step(19.)
        self.assertAlmostEqual(scaled["loss"] - base["loss"], 18 * base["loss_recon_state"], places=5)
        for key in ("loss_recon_img", "loss_recon_state", "loss_reward", "loss_continue", "loss_kl"):
            self.assertEqual(base[key], scaled[key])
        for first, second in zip(grad, scaled_grad):
            torch.testing.assert_close(second, 19 * first)

    def test_image_scale_changes_image_gradient_without_scaling_state_head(self):
        base, state_grad = self.run_step(image_scale=1.)
        image_grad = [p.grad.clone() for p in self.wm["decoder"].image_head.parameters()]
        scaled, state_scaled = self.run_step(image_scale=7.)
        self.assertAlmostEqual(scaled["loss"] - base["loss"], 6 * base["loss_recon_img"], places=5)
        for key in ("loss_recon_img", "loss_recon_state", "loss_reward", "loss_continue", "loss_kl"):
            self.assertEqual(base[key], scaled[key])
        for first, second in zip(image_grad, self.wm["decoder"].image_head.parameters()):
            torch.testing.assert_close(second.grad, 7 * first)
        for first, second in zip(state_grad, state_scaled):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_scale_fails_before_training(self):
        for scale in (0., -1., float("nan"), float("inf")):
            for kwargs in ({"scale": scale}, {"image_scale": scale}):
                with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.run_step(**kwargs)

    def test_all_one_feature_weights_preserve_loss_and_gradients(self):
        base, before = self.run_step()
        weighted, after = self.run_step(state_weights=[1., 1., 1.])
        for key in base:
            self.assertAlmostEqual(base[key], weighted[key], places=6)
        for first, second in zip(before, after):
            torch.testing.assert_close(first, second)

    def test_feature_weights_scale_corresponding_output_rows(self):
        base, grad = self.run_step()
        image_grad = [p.grad.clone() for p in self.wm["decoder"].image_head.parameters()]
        weights = torch.tensor([.5, 2., 3.])
        weighted, after = self.run_step(state_weights=weights.tolist())
        self.assertEqual(base["loss_recon_state"], weighted["loss_recon_state"])
        self.assertAlmostEqual(weighted["loss"] - base["loss"],
            weighted["loss_recon_state_weighted"] - base["loss_recon_state"], places=5)
        # Final state-head Linear rows each predict one independent field.
        torch.testing.assert_close(after[-2], grad[-2] * weights[:, None])
        torch.testing.assert_close(after[-1], grad[-1] * weights)
        for first, second in zip(image_grad, self.wm["decoder"].image_head.parameters()):
            torch.testing.assert_close(first, second.grad, rtol=0, atol=0)

    def test_invalid_feature_weights_fail_before_model_updates(self):
        for weights in ([], [1.], [[1., 1., 1.]], [1., 0., 1.],
                        [1., -1., 1.], [1., float("nan"), 1.], [1., float("inf"), 1.]):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "per state field"):
                self.run_step(state_weights=weights)


if __name__ == "__main__":
    unittest.main()
