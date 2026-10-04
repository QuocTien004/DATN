import copy
import unittest
from unittest.mock import patch
import torch
from models.world_model import build_world_model
from training.continuous_posterior import control_posterior, posterior_replay_loss, train_continuous_posterior_step


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


if __name__=='__main__':unittest.main()
