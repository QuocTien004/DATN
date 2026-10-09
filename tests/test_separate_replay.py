import copy
import unittest
from unittest.mock import patch
import torch
from models.actor_critic import Actor
from models.world_model import build_world_model
from training.continuous_posterior import separate_replay_constraints, train_continuous_posterior_step


class SeparateReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        cfg = dict(encoder=dict(cnn_channels=[4], mlp_hidden=[4], embed_dim=6, control_embedding=dict(dim=4)),
                   rssm=dict(deter_dim=4, stoch_dim=2, stoch_classes=2, hidden_dim=8, categorical_embed_dim=6,
                             continuous_state=dict(dim=19, embed_start=6, embed_stop=10,
                                                   posterior_hidden=[], prior_hidden=[4])))
        self.wm = build_world_model(cfg, (16,16,3), 19, 2, torch.device('cpu'), include_decoder=False)
        self.b = self.wm['rssm'].continuous
        self.actor = Actor(dict(control_state_dim=19, control_state_source='continuous',
            control_state_mean=[0.]*19, control_state_std=[1.]*19, actor_hidden=[8]),4,4,2).eval().requires_grad_(False)
        self.pred = torch.randn(4,19,requires_grad=True)
        self.ref = torch.randn(4,19,requires_grad=True)
        self.target = torch.randn(4,19,requires_grad=True)
        self.h = torch.randn(4,4,requires_grad=True)
        self.z = torch.randn(4,4,requires_grad=True)
        self.mask = torch.tensor([True,True,False,False])

    def loss(self, limits):
        return separate_replay_constraints(self.actor,self.b,self.pred,self.ref,self.target,
                                           self.h,self.z,self.mask,limits)

    def test_formula_no_cross_axis_compensation(self):
        # Steering violates its budget; perfect throttle cannot cancel it.
        with patch('training.continuous_posterior.frozen_actor_replay_mse',return_value=torch.tensor([2.,0.])):
            self.target = self.pred.detach().clone()
            loss, metrics = self.loss([1.,1.,1.,1.])
            self.assertEqual(float(loss),1.)
            self.assertEqual(float(metrics['clean_steering_budget_ratio']),2.)

    def test_clean_only_gradients_and_detached_targets_limits(self):
        limits = torch.full((4,),1e-5,requires_grad=True)
        loss,_ = self.loss(limits);loss.backward()
        self.assertGreater(float(self.pred.grad[self.mask].abs().sum()),0.)
        self.assertEqual(float(self.pred.grad[~self.mask].abs().sum()),0.)
        for x in (self.ref,self.target,self.h,self.z,limits):self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in self.actor.parameters()))

    def test_offbc_nan_ignored_and_satisfied_penalty_zero(self):
        self.ref = self.pred.detach().clone();self.target = self.pred.detach().clone()
        with torch.no_grad():
            for x in (self.pred,self.ref,self.target,self.h,self.z):x[~self.mask]=float('nan')
        loss,_=self.loss([1.,1.,1.,1.]);self.assertEqual(float(loss),0.)

    def test_invalid_bounds_rejected(self):
        for limits in ([1,1,1], [1,1,1,0], [1,1,1,float('nan')], [1,1,1,-1]):
            with self.assertRaises(ValueError):self.loss(limits)

    def test_only_posterior_weights_change(self):
        enc=self.wm['encoder'];before={k:copy.deepcopy(v.state_dict()) for k,v in self.wm.items()}
        actor_before=copy.deepcopy(self.actor.state_dict())
        opt=torch.optim.Adam(list(enc.control_embedding.parameters())+list(self.b.posterior.parameters()),lr=.001)
        with patch.object(self.b,'imagine',side_effect=AssertionError('prior called')):
            out=train_continuous_posterior_step(self.wm,opt,self.target,self.ref,self.mask,
                frozen_actor=self.actor,latent_h=self.h,latent_z=self.z,
                separate_limits=[1e-3]*4,separate_weight=.001)
        self.assertIn('clean_memory6_budget_ratio',out)
        changed=[]
        for n,m in self.wm.items():
            for k,v in m.state_dict().items():
                if not torch.equal(v,before[n][k]):changed.append((n,k))
        self.assertTrue(changed)
        self.assertTrue(all(n=='encoder' and k.startswith('control_embedding.') or
                            n=='rssm' and k.startswith('continuous.posterior.') for n,k in changed))
        for k,v in self.actor.state_dict().items():self.assertTrue(torch.equal(v,actor_before[k]))

    def test_invalid_options_restore_flags(self):
        enc=self.wm['encoder'];opt=torch.optim.Adam(list(enc.control_embedding.parameters())+list(self.b.posterior.parameters()))
        for kwargs in [dict(separate_weight=1.),dict(separate_limits=[1.]*4),
                       dict(separate_weight=-1.),dict(separate_weight=1.,separate_limits=[1.]*4,action_weight=1.)]:
            with self.assertRaises(ValueError):
                train_continuous_posterior_step(self.wm,opt,self.target,self.ref,self.mask,**kwargs)
        self.actor.train().requires_grad_(True)
        with self.assertRaises(ValueError):
            train_continuous_posterior_step(self.wm,opt,self.target,self.ref,self.mask,
                frozen_actor=self.actor,latent_h=self.h,latent_z=self.z,separate_limits=[0.]*4,separate_weight=1.)
        self.assertTrue(self.actor.training)
        self.assertTrue(all(p.requires_grad for p in self.actor.parameters()))


if __name__ == '__main__':unittest.main()
