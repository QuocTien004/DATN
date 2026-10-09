import copy
import unittest
from unittest.mock import patch
import torch
from models.world_model import build_world_model
from training.continuous_posterior import control_posterior, posterior_replay_loss, train_continuous_posterior_step, frozen_actor_replay_loss
from models.actor_critic import Actor


class ContinuousPosteriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(7)
        cfg=dict(encoder=dict(cnn_channels=[4],mlp_hidden=[4],embed_dim=6,control_embedding=dict(dim=4)),
                 rssm=dict(deter_dim=4,stoch_dim=2,stoch_classes=2,hidden_dim=8,categorical_embed_dim=6,
                           continuous_state=dict(dim=3,embed_start=6,embed_stop=10,posterior_hidden=[],prior_hidden=[4])))
        self.wm=build_world_model(cfg,(16,16,3),3,2,torch.device('cpu'),include_decoder=False)
        self.e=self.wm['encoder']; self.b=self.wm['rssm'].continuous
        self.x=torch.randn(4,3); self.reference=torch.randn(4,3); self.mask=torch.tensor([True,True,False,False])
        self.opt=torch.optim.Adam(list(self.e.control_embedding.parameters())+list(self.b.posterior.parameters()),lr=.001)

    def test_full_embedding_parity_and_legacy_unchanged(self):
        image=torch.rand(4,3,16,16)
        torch.testing.assert_close(control_posterior(self.e,self.b,self.x), self.b.observe(self.e(image,self.x)),rtol=0,atol=0)

    def test_only_allowed_weights_change_and_flags_restore(self):
        before={k:copy.deepcopy(m.state_dict()) for k,m in self.wm.items()}
        flags=[p.requires_grad for m in self.wm.values() for p in m.parameters()]
        with patch.object(self.b,'imagine',side_effect=AssertionError('prior called')):
            out=train_continuous_posterior_step(self.wm,self.opt,self.x,self.reference,self.mask)
        self.assertTrue(out['loss']>=0)
        changed=[]
        for name,m in self.wm.items():
            for k,v in m.state_dict().items():
                if not torch.equal(v,before[name][k]):changed.append((name,k))
        self.assertTrue(any(n=='encoder' for n,k in changed))
        self.assertTrue(any(n=='rssm' for n,k in changed))
        self.assertTrue(all(n=='encoder' and k.startswith('control_embedding.') or n=='rssm' and k.startswith('continuous.posterior.') for n,k in changed))
        self.assertEqual(flags,[p.requires_grad for m in self.wm.values() for p in m.parameters()])

    def test_reject_optimizer_with_prior_before_changes(self):
        opt=torch.optim.Adam(self.wm['rssm'].parameters())
        with self.assertRaisesRegex(ValueError,'exactly'):
            train_continuous_posterior_step(self.wm,opt,self.x,self.reference,self.mask)

    def test_detached_supervision_and_replay_formula(self):
        prediction=torch.zeros(4,3,requires_grad=True); target=torch.ones(4,3,requires_grad=True)
        teacher=torch.full((4,3),2.,requires_grad=True)
        loss,_=posterior_replay_loss(self.b,prediction,target,teacher,self.mask,anchor_weight=.5)
        self.assertEqual(float(loss),3.)
        loss.backward(); self.assertIsNone(target.grad);self.assertIsNone(teacher.grad)
        self.assertIsNotNone(prediction.grad)

    def test_only_clean_anchor_rows_matter(self):
        pred=torch.randn(4,3); ref=self.reference.clone()
        a=posterior_replay_loss(self.b,pred,self.x,ref,self.mask)[0]
        ref[~self.mask]=float('nan')
        torch.testing.assert_close(a,posterior_replay_loss(self.b,pred,self.x,ref,self.mask)[0])

    def test_missing_clean_anchor_invalid_shape_weight_rejected(self):
        for mask in [torch.zeros(4,dtype=torch.bool),torch.ones(4),torch.ones(3,dtype=torch.bool)]:
            with self.assertRaises(ValueError):posterior_replay_loss(self.b,self.x,self.x,self.reference,mask)
        for w in [-1.,float('nan')]:
            with self.assertRaises(ValueError):posterior_replay_loss(self.b,self.x,self.x,self.reference,self.mask,anchor_weight=w)

    def test_refuse_legacy_slice_and_restore_after_error(self):
        self.b.embed_start=0
        flags=[p.requires_grad for m in self.wm.values() for p in m.parameters()]
        with self.assertRaisesRegex(ValueError,'exactly the separate'):
            train_continuous_posterior_step(self.wm,self.opt,self.x,self.reference,self.mask)
        self.assertEqual(flags,[p.requires_grad for m in self.wm.values() for p in m.parameters()])

    def action_fixture(self):
        actor=Actor(dict(control_state_dim=3,control_state_source='continuous',control_state_mean=[0.]*3,
                         control_state_std=[1.]*3,actor_hidden=[8]),4,4,2)
        return actor,torch.randn(4,4,requires_grad=True),torch.randn(4,2,2,requires_grad=True)

    def test_action_loss_backpropagates_only_clean_student_not_teacher_or_actor(self):
        actor,h,z=self.action_fixture();actor.eval().requires_grad_(False)
        p=torch.randn(4,3,requires_grad=True);ref=self.reference.clone().requires_grad_()
        loss=frozen_actor_replay_loss(actor,self.b,p,ref,h,z,self.mask);loss.backward()
        self.assertGreater(float(p.grad[self.mask].abs().sum()),0.)
        self.assertEqual(float(p.grad[~self.mask].abs().sum()),0.)
        self.assertIsNone(ref.grad);self.assertIsNone(h.grad);self.assertIsNone(z.grad)
        self.assertTrue(all(a.grad is None for a in actor.parameters()))

    def test_action_loss_ignores_offbc_and_zero_for_equal_control(self):
        actor,h,z=self.action_fixture();actor.eval().requires_grad_(False)
        p=self.reference.clone();ref=p.clone()
        p[~self.mask]=float('nan');ref[~self.mask]=float('nan');h.data[~self.mask]=float('nan')
        self.assertEqual(float(frozen_actor_replay_loss(actor,self.b,p,ref,h,z,self.mask)),0.)

    def test_action_step_keeps_actor_prior_frozen_and_restores_flags(self):
        actor,h,z=self.action_fixture();before=copy.deepcopy(actor.state_dict());prior=copy.deepcopy(self.b.transition.state_dict())
        out=train_continuous_posterior_step(self.wm,self.opt,self.x,self.reference,self.mask,
            frozen_actor=actor,latent_h=h,latent_z=z,action_weight=1.)
        self.assertIn('clean_action_mse',out);self.assertTrue(actor.training)
        self.assertTrue(all(p.requires_grad and p.grad is None for p in actor.parameters()))
        for k,v in actor.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        for k,v in self.b.transition.state_dict().items():torch.testing.assert_close(v,prior[k],rtol=0,atol=0)

    def test_action_weight_and_missing_inputs_rejected(self):
        for weight in [-1.,float('nan'),1.]:
            with self.assertRaises(ValueError):
                train_continuous_posterior_step(self.wm,self.opt,self.x,self.reference,self.mask,action_weight=weight)

    def test_action_loss_refuses_trainable_actor(self):
        actor,h,z=self.action_fixture()
        with self.assertRaisesRegex(ValueError,'frozen Actor'):
            frozen_actor_replay_loss(actor,self.b,self.x,self.reference,h,z,self.mask)

    def test_action_error_restores_actor_flags(self):
        actor,h,z=self.action_fixture()
        with self.assertRaises(ValueError):
            train_continuous_posterior_step(self.wm,self.opt,self.x,self.reference,self.mask,
                frozen_actor=actor,latent_h=h[:2],latent_z=z,action_weight=1.)
        self.assertTrue(actor.training);self.assertTrue(all(p.requires_grad for p in actor.parameters()))


if __name__=='__main__':unittest.main()
