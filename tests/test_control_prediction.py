import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.control_state import ControlStatePredictor, with_predicted_control
from models.world_model import build_world_model, load_world_model_state
from models.actor_critic import ActorCritic
from training.control_state import valid_prediction_starts, predict_control_sequence
from training.train_agent import imagine_rollout, freeze_parameters
from evaluation.evaluate import LatentActorPolicy
from training.imitation import train_imitation_step
from training.train_world_model import train_world_model_step


class ControlPredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        self.cfg = {
            "encoder": {"cnn_channels": [4], "mlp_hidden": [4], "embed_dim": 4},
            "rssm": {"deter_dim": 4, "stoch_dim": 2, "stoch_classes": 2, "hidden_dim": 8},
            "predictors": {"reward_hidden": [4], "continue_hidden": [4]},
            "control_state_predictor": {"hidden": [8], "state_mean": [.2, .3, .4], "state_std": [.1, .2, .3]},
        }
        self.wm = build_world_model(self.cfg, (16, 16, 3), 3, 2, torch.device('cpu'), include_decoder=False)
        self.head = self.wm['control_state']
        self.ac = ActorCritic(dict(actor_hidden=[8], critic_hidden=[8], control_state_dim=3,
                                   control_state_mean=[0., 0., 0.], control_state_std=[1., 1., 1.]), 4, 4, 2)
        self.start = self.wm['rssm'].initial_state(2, 'cpu')

    def test_normalization_and_checkpoint_roundtrip(self):
        h, z = self.start['h'], self.start['z']
        torch.testing.assert_close(self.head(h, z), self.head.normalized_prediction(h, z)*self.head.state_std+self.head.state_mean)
        restored = build_world_model(self.cfg, (16, 16, 3), 3, 2, torch.device('cpu'), include_decoder=False)
        load_world_model_state(restored, {'models': {k:copy.deepcopy(v.state_dict()) for k,v in self.wm.items()}})
        torch.testing.assert_close(restored['control_state'](h,z), self.head(h,z), rtol=0, atol=0)

    def test_invalid_statistics_rejected(self):
        for cfg in ({'state_std': [0., 1., 1.]}, {'state_mean':[float('nan')]*3}, {'feature_std':[1.]}):
            with self.assertRaises(ValueError):
                ControlStatePredictor(cfg, 4, 4, 3)

    def test_reset_and_terminal_boundaries(self):
        first = torch.tensor([1,0,0,1,0,0,0])
        done = torch.tensor([0,0,1,0,0,1,0])
        self.assertEqual(valid_prediction_starts(first, 2, done).tolist(), [0,3])
        self.assertEqual(valid_prediction_starts(first, 0, done).tolist(), list(range(7)))
        self.assertEqual(valid_prediction_starts(first, 15, done).numel(), 0)
        with self.assertRaises(ValueError):
            valid_prediction_starts(first, -1)

    def test_future_alignment_and_no_teacher_forcing(self):
        class AddRSSM:
            def imagine_step(self, state, action, **kwargs):
                return dict(h=state['h']+action, z=state['z']), {}
        state = dict(h=torch.zeros(1, 1), z=torch.zeros(1, 1), control=torch.full((1,1),999.))
        actions = torch.tensor([[[1.]], [[2.]], [[4.]]])
        pred = predict_control_sequence(state, actions, AddRSSM(), lambda h,z:h)
        self.assertEqual(pred.flatten().tolist(), [0.,1.,3.,7.])

    def test_frozen_models_preserve_action_gradients(self):
        actions = torch.randn(3,2,2,requires_grad=True)
        with freeze_parameters(self.wm.values()):
            pred = predict_control_sequence(self.start, actions, self.wm['rssm'], self.head)
            pred[1:].square().sum().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(actions.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for m in self.wm.values() for p in m.parameters()))

    def test_diagnostic_imagination_replaces_even_initial_gt_and_keeps_rl_guard(self):
        state = dict(self.start, control=torch.full((2,3),float('nan')))
        with self.assertRaisesRegex(ValueError, 'no validated'):
            imagine_rollout(state, self.wm, self.ac, horizon=3, gamma=.99)
        traj = imagine_rollout(state, self.wm, self.ac, horizon=3, gamma=.99, diagnostic_control=True)
        self.assertEqual(traj.control.shape, (4,2,3))
        torch.testing.assert_close(traj.control[0], self.head(state['h'],state['z']))
        self.assertTrue(torch.isfinite(traj.control).all())
        with self.assertRaisesRegex(ValueError, 'requires'):
            imagine_rollout(state, {}, self.ac, horizon=3, gamma=.99, diagnostic_control=True)
        with self.assertRaisesRegex(ValueError, 'no validated'):
            self.ac.require_imagination_ready()

    def test_evaluation_uses_predicted_control_and_reset(self):
        policy = LatentActorPolicy(self.wm['encoder'], self.wm['rssm'], self.ac,
                                   device=torch.device('cpu'), control_predictor=self.head)
        obs = dict(image=np.zeros((16,16,3),dtype=np.uint8), state=np.zeros(3,dtype=np.float32))
        with patch.object(self.ac.actor,'forward',wraps=self.ac.actor.forward) as act:
            first = policy(obs)
            torch.testing.assert_close(act.call_args.kwargs['control_state'],
                                       self.head(policy._latent['h'],policy._latent['z']))
            self.assertFalse(torch.equal(act.call_args.kwargs['control_state'],torch.zeros(1,3)))
        policy(obs); policy.reset()
        np.testing.assert_array_equal(first,policy(obs))

    def test_with_predicted_control_ignores_observed_control(self):
        a = with_predicted_control(dict(self.start,control=torch.zeros(2,3)), self.head)
        b = with_predicted_control(dict(self.start,control=torch.ones(2,3)*1000), self.head)
        torch.testing.assert_close(a['control'],b['control'],rtol=0,atol=0)

    def test_predicted_checkpoint_fails_closed_without_head(self):
        self.ac.actor.control_state_source = 'predicted'
        with self.assertRaisesRegex(ValueError,'GT fallback'):
            LatentActorPolicy(self.wm['encoder'], self.wm['rssm'], self.ac, device=torch.device('cpu'))
        with self.assertRaisesRegex(ValueError,'Joint WM imitation'):
            train_world_model_step({}, {}, None, {}, imitation_actor=self.ac.actor)

    def test_predicted_bc_uses_frozen_head_not_gt(self):
        self.ac.actor.control_state_source = 'predicted'
        batch = dict(image=torch.rand(2,3,3,16,16),state=torch.rand(2,3,3),action=torch.zeros(2,3,2))
        before = copy.deepcopy(self.head.state_dict())
        opt = torch.optim.Adam(self.ac.actor.parameters(),lr=.001)
        with patch.object(self.head,'forward',wraps=self.head.forward) as pred, patch.object(self.ac.actor,'forward',wraps=self.ac.actor.forward) as act:
            train_imitation_step(batch,self.wm,self.ac,opt)
            self.assertEqual(pred.call_count,1)
            torch.testing.assert_close(act.call_args.kwargs['control_state'],self.head(*pred.call_args.args))
        for key,value in self.head.state_dict().items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in self.head.parameters()))


if __name__ == '__main__':
    unittest.main()
