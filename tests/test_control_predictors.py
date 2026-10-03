import unittest
import torch
from models.predictors import RewardPredictor, ContinuePredictor, predict_from_state


class ControlPredictorTests(unittest.TestCase):
    def test_legacy_checkpoint_and_values_unchanged(self):
        for cls in [RewardPredictor, ContinuePredictor]:
            a=cls({},4,6);b=cls({},4,6);b.load_state_dict(a.state_dict())
            h=torch.randn(2,4);z=torch.randn(2,2,3)
            torch.testing.assert_close(a(h,z),b(h,z))
            torch.testing.assert_close(a(h,z),a.net(torch.cat([h,z.flatten(1)],1)))

    def test_model_state_required_no_gt_fallback(self):
        m=RewardPredictor({'predictors':{'continuous_state_dim':3}},4,6)
        s=dict(h=torch.randn(2,4),z=torch.randn(2,2,3),control=torch.randn(2,3))
        with self.assertRaises(ValueError):predict_from_state(m,s)
        s['c']=torch.randn(2,3)
        self.assertEqual(predict_from_state(m,s).shape,(2,1))

    def test_control_gradient_and_invalid_shapes(self):
        for cls in [RewardPredictor, ContinuePredictor]:
            m=cls({'predictors':{'continuous_state_dim':3}},4,6)
            h=torch.randn(2,4);z=torch.randn(2,2,3);c=torch.randn(2,3,requires_grad=True)
            m(h,z,c=c).sum().backward();self.assertGreater(c.grad.abs().sum(),0)
            for bad in [None,torch.zeros(1,3),torch.full((2,3),float('nan'))]:
                with self.assertRaises(ValueError):m(h,z,c=bad)

    def test_zero_extra_columns_match_warm_start(self):
        for cls in [RewardPredictor, ContinuePredictor]:
            old=cls({},4,6);new=cls({'predictors':{'continuous_state_dim':3}},4,6)
            state=old.state_dict();state['net.0.weight']=torch.cat([state['net.0.weight'],torch.zeros(256,3)],1)
            new.load_state_dict(state)
            h=torch.randn(2,4);z=torch.randn(2,2,3);c=torch.randn(2,3)
            torch.testing.assert_close(old(h,z),new(h,z,c=c))

    def test_legacy_rejects_accidental_control(self):
        with self.assertRaises(ValueError):RewardPredictor({},4,6)(torch.zeros(2,4),torch.zeros(2,6),c=torch.zeros(2,3))


if __name__=='__main__':unittest.main()
