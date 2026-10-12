import unittest
import numpy as np
from training.reward_diagnostics import horizon_pairs,reward_terminal_metrics,reward_terminal_gate


class RewardDiagnosticTests(unittest.TestCase):
    def test_terminal_successor_retained(self):
        for h in [1,5,15]:
            starts,targets=horizon_pairs(20,h)
            self.assertEqual(targets[-1],19)
            np.testing.assert_array_equal(starts+h,targets+1)
            self.assertEqual(len(starts),21-h)

    def test_short_episode_no_reset_crossing(self):
        self.assertEqual(len(horizon_pairs(3,5)[0]),0)
        with self.assertRaises(ValueError):horizon_pairs(3,0)

    def test_timeout_remains_nonterminal(self):
        reward=np.array([0.,-20.,.5]);terminal=np.array([False,True,False])
        pred=np.sign(reward)*np.log1p(abs(reward))
        m=reward_terminal_metrics(reward,terminal,pred,[.99,.01,.99],0.)
        self.assertTrue(reward_terminal_gate(m));self.assertEqual(m['nonterminal_n'],2)
        wrong=reward_terminal_metrics(reward,terminal,pred,[.99,.01,.01],0.)
        self.assertEqual(wrong['false_positive_rate'],.5);self.assertFalse(reward_terminal_gate(wrong))

    def test_missing_failure_is_not_a_gate_pass(self):
        m=reward_terminal_metrics([1.,1.],[False,False],[.69,.69],[.99,.99],0.)
        self.assertFalse(reward_terminal_gate(m))

    def test_nonfinite_and_misaligned_rejected(self):
        with self.assertRaises(ValueError):reward_terminal_metrics([1],[True],[float('nan')],[.5],0.)
        with self.assertRaises(ValueError):reward_terminal_metrics([1],[True],[.1,.2],[.5],0.)


if __name__=='__main__':unittest.main()
