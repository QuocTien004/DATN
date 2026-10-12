from __future__ import annotations

import gc
import tempfile
import unittest
from types import SimpleNamespace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.actor_critic import ActorCritic
from models.rssm import RSSM
from models.world_model import build_world_model
from training.batches import replay_batch_to_torch
from training.online_trainer import OnlineTrainer
from training.train_world_model import train_world_model_step
from training.train_agent import symexp
from utils.online_state import resolve_replay_path, validate_replay_step, require_current_contract
from utils.replay_buffer import ReplayBuffer, Transition
from scripts.setup_colab_egl import select_gl_package


def transition(value: int, *, done: bool = False, terminated: bool = False) -> Transition:
    return Transition(np.full((16, 16, 3), value, np.uint8), np.array([value], np.float32),
                      np.zeros(2, np.float32), float(value), done,
                      np.full((16, 16, 3), value + 1, np.uint8), np.array([value + 1], np.float32), terminated)


def buffer(capacity=8, length=2):
    return ReplayBuffer(capacity, (16, 16, 3), 1, 2, length)


class ReplayRegressionTests(unittest.TestCase):
    def test_ring_roundtrip_preserves_write_pointer_and_order(self):
        replay = buffer(5, 2)
        for i in range(8):
            replay.add(transition(i), is_first=i == 0)
        with tempfile.TemporaryDirectory() as directory:
            path = replay.save(Path(directory) / "replay.npz", checkpoint_step=8)
            loaded = ReplayBuffer.load(path)
            self.assertEqual(loaded.idx, 3)
            self.assertEqual(loaded.checkpoint_step, 8)
            self.assertEqual(loaded._valid_start_indices().tolist(), [0, 1, 3])
            loaded.make_writable(8)
            np.testing.assert_array_equal(loaded.states[:5, 0], [3, 4, 5, 6, 7])
            loaded.add(transition(8))
            np.testing.assert_array_equal(loaded.states[:6, 0], [3, 4, 5, 6, 7, 8])

    def test_latest_next_observation_and_timeout_survive(self):
        replay = buffer(2, 2)
        replay.add(transition(1), is_first=True)
        replay.add(transition(2, done=True, terminated=False))
        sample = replay.sample(1, include_next=True)
        np.testing.assert_array_equal(sample["next_state"][0, :, 0], [2, 3])
        self.assertTrue(sample["done"][0, -1])
        self.assertFalse(sample["terminated"].any())

    def test_reset_preserves_interrupted_successor(self):
        replay = buffer()
        replay.add(transition(2), is_first=True)
        replay.add(transition(10), is_first=True)
        sample = replay.sample(1, include_next=True)
        self.assertEqual(sample["next_state"][0, 0, 0], 3)
        self.assertTrue(sample["done"][0, 0])
        self.assertFalse(sample["terminated"][0, 0])

    def test_same_size_archive_does_not_reuse_stale_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay = buffer(2, 2)
            replay.add(transition(1)); replay.add(transition(2))
            replay.save(path)
            old = ReplayBuffer.load(path)
            self.assertEqual(old.images[0, 0, 0, 0], 1)
            del old
            gc.collect()
            replay.images.fill(99)
            replay.save(path)
            fresh = ReplayBuffer.load(path)
            self.assertEqual(fresh.images[0, 0, 0, 0], 99)
            del fresh
            gc.collect()

    def test_image_shape_mismatch_fails_before_writing(self):
        replay = buffer()
        row = transition(0)
        row.image = np.zeros((64, 64, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "image shape"):
            replay.add(row)
        self.assertEqual(len(replay), 0)


class TrainingRegressionTests(unittest.TestCase):
    def test_online_eval_saves_seed_and_time_limit_and_resets_episode(self):
        class Env:
            def reset(self, seed=None):
                self.steps = 0
                return {}, {}
            def step(self, action):
                self.steps += 1
                return {}, 1., False, self.steps == 2, {"route_completion": .1}
            def close(self):
                pass
        class Policy:
            def reset(self):
                pass
            def __call__(self, obs):
                return np.zeros(2)
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(OnlineTrainer)
            trainer.train_env = Env()
            trainer.train_env_cfg = {}
            trainer.eval_env_cfg = {"horizon": 2}
            trainer.logger = SimpleNamespace(log_dir=Path(directory))
            trainer.env_steps = 250
            trainer._episode_return = 999.
            trainer._episode_length = 999
            trainer._make_policy = lambda **kwargs: Policy()
            trainer._refresh_policy = lambda: None
            with patch('training.online_trainer.make_env', side_effect=lambda cfg: Env()):
                metrics = trainer.evaluate(2, start_seed=10000)
            saved = json.loads((Path(directory)/'eval_step_000250.json').read_text())
            self.assertEqual(saved['horizon'], 2)
            self.assertEqual([e['seed'] for e in saved['episodes']], [10000, 10001])
            self.assertTrue(all(e['truncated'] for e in saved['episodes']))
            self.assertEqual(metrics['eval/mean_episode_length'], 2)
            self.assertEqual(trainer._episode_return, 0.)
            self.assertEqual(trainer._episode_length, 0)

    def test_egl_setup_requires_exact_driver_version(self):
        listing = "libnvidia-gl-580 | 580.99.01-1ubuntu1 | repo\nlibnvidia-gl-580 | 580.82.07-0ubuntu1 | repo"
        self.assertEqual(select_gl_package("580.82.07", listing), "580.82.07-0ubuntu1")
        with self.assertRaisesRegex(RuntimeError, "No exact"):
            select_gl_package("580.82.06", listing)

    def test_symexp_has_unit_gradient_at_zero(self):
        values = torch.tensor([-.1, 0., .1], requires_grad=True)
        symexp(values, 3.26).sum().backward()
        torch.testing.assert_close(values.grad, values.detach().abs().exp())

    def test_squashed_entropy_pushes_saturated_mean_inward(self):
        ac = ActorCritic({"actor_hidden": [4], "critic_hidden": [4],
                          "mean_transform": "tanh", "entropy_mode": "squashed"}, 4, 4, 2)
        with torch.no_grad():
            ac.actor.net[-1].weight.zero_()
            ac.actor.net[-1].bias[:2].copy_(torch.tensor([3., -3.]))
        out = ac.act({"h": torch.zeros(1, 4), "z": torch.zeros(1, 2, 2)}, deterministic=True)
        out.entropy.sum().backward()
        grad = ac.actor.net[-1].bias.grad[:2]
        self.assertLess(grad[0].item(), 0.)
        self.assertGreater(grad[1].item(), 0.)
        self.assertTrue(torch.isfinite(out.log_prob).all())

    def test_log_prob_matches_torch_transformed_distribution(self):
        ac = ActorCritic({"actor_hidden": [4], "critic_hidden": [4],
                          "action_low": [-2., -1.], "action_high": [2., 3.]}, 4, 4, 2)
        latent = {"h": torch.zeros(16, 4), "z": torch.zeros(16, 2, 2)}
        out = ac.act(latent)
        feature = torch.cat([latent['h'], latent['z'].flatten(1)], -1)
        mean, log_std = ac.actor.net(feature).chunk(2, -1)
        normal = torch.distributions.Normal(mean.clamp(-2, 2), log_std.clamp(np.log(.1), 0).exp())
        dist = torch.distributions.TransformedDistribution(normal, [
            torch.distributions.TanhTransform(cache_size=1),
            torch.distributions.AffineTransform(torch.tensor([0., 1.]), torch.tensor([2., 2.]))])
        torch.testing.assert_close(out.log_prob, dist.log_prob(out.action).sum(-1), atol=1e-5, rtol=1e-5)

    def test_conversion_keeps_resets_and_true_terminals(self):
        replay = buffer(2, 2)
        replay.add(transition(1), is_first=True)
        replay.add(transition(2, done=True, terminated=False))
        batch = replay_batch_to_torch(replay.sample(1, include_next=True), torch.device('cpu'))
        self.assertEqual(batch["is_first"].tolist(), [[True, False]])
        self.assertFalse(batch["terminated"].any())
        self.assertEqual(tuple(batch["next_image"].shape), (1, 2, 3, 16, 16))

    def test_rssm_reset_cannot_carry_previous_episode(self):
        rssm = RSSM({"rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 4}}, 3, 2)
        initial = rssm.initial_state(1)
        previous = {"h": torch.randn(1, 4), "z": torch.randn(1, 2, 2)}
        embedding = torch.randn(1, 3)
        expected, _ = rssm.observe_step(initial, torch.zeros(1, 2), embedding, deterministic=True)
        actual, _ = rssm.observe_step(previous, torch.ones(1, 2), embedding, is_first=torch.ones(1), deterministic=True)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])

    def test_wm_heads_receive_successor_posterior(self):
        cfg = {"encoder": {"cnn_channels": [4], "mlp_hidden": [4], "embed_dim": 4},
               "rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 4},
               "decoder": {"cnn_channels": [4, 4]},
               "predictors": {"reward_hidden": [4], "continue_hidden": [4]}}
        wm = build_world_model(cfg, (16, 16, 3), 1, 2, torch.device('cpu'))
        replay = buffer(2, 2)
        replay.add(transition(1), is_first=True)
        replay.add(transition(2, done=True, terminated=False))
        batch = replay_batch_to_torch(replay.sample(1, include_next=True), torch.device('cpu'))
        observations, reward_inputs = [], []
        real_step = wm['rssm'].observe_step
        def observe(*args, **kwargs):
            result = real_step(*args, **kwargs)
            observations.append(result[0]['h'].detach().clone())
            return result
        hook = wm['reward'].register_forward_pre_hook(lambda _m, args: reward_inputs.append(args[0].detach().clone()))
        optimizer = torch.optim.Adam([p for m in wm.values() for p in m.parameters()], lr=1e-3)
        with patch.object(wm['rssm'], 'observe_step', side_effect=observe):
            metrics = train_world_model_step(batch, wm, optimizer, cfg)
        hook.remove()
        torch.testing.assert_close(reward_inputs[0], observations[-1])
        self.assertEqual(observations[-1].shape[0], 2)
        self.assertEqual(metrics['terminal_fraction'], 0.0)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_smooth_actor_has_gradient_beyond_old_clamp(self):
        ac = ActorCritic({"actor_hidden": [4], "critic_hidden": [4], "mean_transform": "tanh", "std_transform": "sigmoid", "min_std": .1, "max_std": .5, "init_std": .3}, 4, 4, 2)
        with torch.no_grad():
            ac.actor.net[-1].weight.zero_()
            ac.actor.net[-1].bias[:2].fill_(3.0)
            ac.actor.net[-1].bias[2:].fill_(-3.0)
        state = {"h": torch.zeros(1, 4), "z": torch.zeros(1, 2, 2)}
        output = ac.act(state, deterministic=True)
        (output.action.sum() + output.std.sum()).backward()
        self.assertTrue((ac.actor.net[-1].bias.grad.abs() > 0).all())

    def test_demo_sampling_retains_protected_data(self):
        trainer = object.__new__(OnlineTrainer)
        trainer.buffer = buffer(2, 2); trainer.demo_buffer = buffer(2, 2)
        for i in range(2):
            trainer.buffer.add(transition(i))
            trainer.demo_buffer.add(transition(i+50))
        trainer.demo_ratio = .5; trainer.term_ratio = 0
        mixed = trainer._sample_batch(4)
        self.assertEqual(int((mixed['state'][:, 0, 0] >= 50).sum()), 2)

    def test_resume_rejects_missing_or_mismatched_replay(self):
        with self.assertRaisesRegex(ValueError, 'no replay link'):
            resolve_replay_path(Path('latest.pt'), {'env_steps': 20})
        with self.assertRaisesRegex(ValueError, '!= replay'):
            validate_replay_step({'env_steps': 20}, 10)
        validate_replay_step({'env_steps': 20}, 20)
        with self.assertRaisesRegex(ValueError, 'Legacy WM'):
            require_current_contract({})


if __name__ == '__main__':
    unittest.main()
