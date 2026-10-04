import copy
import unittest
from unittest.mock import patch

import torch

from models.continuous_state import ContinuousState
from training.continuous_state import continuous_rollout, continuous_transition_loss


class ContinuousTransitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.branch = ContinuousState(dict(dim=2, posterior_hidden=[4], prior_hidden=[4]), 3, 2)
        self.branch.requires_grad_(False)
        self.branch.transition.requires_grad_(True)
        self.initial = torch.randn(2, 2, requires_grad=True)
        self.actions = torch.randn(3, 2, 2)
        self.targets = torch.randn(4, 2, 2, requires_grad=True)
        self.valid = torch.tensor([[True, True], [True, False], [True, False]])

    def loss(self, **kw):
        args = dict(branch=self.branch, initial=self.initial, executed_actions=self.actions,
                    targets=self.targets, valid=self.valid, horizons=(1, 3))
        args.update(kw)
        return continuous_transition_loss(**args)

    def test_terminal_successor_included_and_padding_ignored(self):
        loss, info = self.loss()
        self.assertEqual(info['valid_counts'], {1: 2, 3: 1})
        targets = self.targets.detach().clone()
        targets[2:, 1] = float('nan')
        torch.testing.assert_close(loss, self.loss(targets=targets)[0])
        targets[1, 1] += 10  # The only real transition of episode 2 is terminal.
        self.assertNotEqual(float(loss), float(self.loss(targets=targets)[0]))

    def test_only_transition_updates_and_initial_target_are_detached(self):
        before = copy.deepcopy(self.branch.state_dict())
        optimizer = torch.optim.Adam(self.branch.transition.parameters(), lr=.001)
        self.loss()[0].backward()
        self.assertIsNone(self.initial.grad)
        self.assertIsNone(self.targets.grad)
        self.assertTrue(all(p.grad is None for p in self.branch.posterior.parameters()))
        optimizer.step()
        changed = [k for k, v in self.branch.state_dict().items() if not torch.equal(v, before[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith('transition.') for k in changed))

    def test_no_future_posterior_or_teacher_forcing(self):
        with patch.object(self.branch, 'observe', side_effect=AssertionError('future observation')):
            with patch.object(self.branch, 'imagine', wraps=self.branch.imagine) as call:
                self.loss()
                inputs = [c.args[0].detach().clone() for c in call.call_args_list]
            with patch.object(self.branch, 'imagine', wraps=self.branch.imagine) as call:
                self.loss(targets=self.targets + 100)
                for old, new in zip(inputs, call.call_args_list):
                    torch.testing.assert_close(old, new.args[0])

    def test_gradient_through_all_recursive_steps(self):
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(2.))
            def imagine(self, c, a): return self.w * c + a
            def normalize(self, y): return y
        branch = Scale()
        loss, _ = continuous_transition_loss(branch, torch.ones(1, 1), torch.zeros(3, 1, 1),
            torch.zeros(4, 1, 1), torch.ones(3, 1, dtype=torch.bool), horizons=(3,))
        loss.backward()
        self.assertEqual(float(branch.w.grad), 192.)  # d(w**6)/dw, w=2.

    def test_action_timing_and_executed_not_bc(self):
        class Add:
            def imagine(self, c, a): return c + a
            def normalize(self, y): return y
        actions = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]])
        target = torch.tensor([[[0., 0.]], [[1., 2.]], [[4., 6.]], [[9., 12.]]])
        args = dict(branch=Add(), initial=target[0], executed_actions=actions, targets=target,
                    valid=torch.ones(3, 1, dtype=torch.bool), horizons=(1, 3))
        self.assertEqual(float(continuous_transition_loss(**args)[0]), 0.)
        args['executed_actions'] = torch.zeros_like(actions)  # Wrong BC proposal.
        self.assertGreater(float(continuous_transition_loss(**args)[0]), 0.)

    def test_mask_cannot_resume_after_reset(self):
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.loss(valid=torch.tensor([[True, True], [True, False], [True, True]]))

    def test_empty_invalid_horizons_and_nonfinite_rejected(self):
        for horizons in [(), (0,), (4,), (1, 1)]:
            with self.assertRaises(ValueError): self.loss(horizons=horizons)
        with self.assertRaisesRegex(ValueError, 'No valid'):
            self.loss(valid=torch.zeros_like(self.valid))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.loss(valid=self.valid.float())
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.loss(executed_actions=self.actions * float('nan'))


if __name__ == '__main__':
    unittest.main()
