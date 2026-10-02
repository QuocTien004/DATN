import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.rssm import RSSM
from models.world_model import build_world_model, load_world_model_state
from models.actor_critic import ActorCritic
from models.continuous_state import ContinuousState
from training.continuous_state import continuous_rollout, continuous_prediction_loss
from training.train_agent import posterior_start_states, imagine_rollout, freeze_parameters, train_actor_critic_step
from training.imitation import train_imitation_step
from training.train_world_model import train_world_model_step
from evaluation.evaluate import LatentActorPolicy


class ContinuousStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.legacy = dict(encoder=dict(cnn_channels=[4], mlp_hidden=[4], embed_dim=6),
                           rssm=dict(deter_dim=4, stoch_dim=2, stoch_classes=2, hidden_dim=8))
        self.cfg = copy.deepcopy(self.legacy)
        self.cfg['rssm']['continuous_state'] = dict(dim=3, embed_start=3, posterior_hidden=[8],
            prior_hidden=[8], state_mean=[.1,.2,.3], state_std=[.2,.3,.4])
        self.wm = build_world_model(self.cfg, (16,16,3), 3, 2, torch.device('cpu'), include_decoder=False)
        self.rssm = self.wm['rssm']
        self.ac = ActorCritic(dict(actor_hidden=[8], critic_hidden=[8], control_state_dim=3,
            control_state_source='continuous', control_state_mean=[0.]*3, control_state_std=[1.]*3), 4,4,2)
        self.batch = dict(image=torch.rand(2,3,3,16,16), state=torch.rand(2,3,3), action=torch.zeros(2,3,2),
                          is_first=torch.tensor([[1,0,0],[1,0,1]]))

    def test_legacy_keys_and_explicit_migration(self):
        old = RSSM(self.legacy,6,2)
        self.assertEqual(set(old.initial_state(2)), {'h','z'})
        self.assertFalse(any(k.startswith('continuous.') for k in old.state_dict()))
        self.rssm.warm_start_categorical(old.state_dict())
        before = old.initial_state(2); after = self.rssm.initial_state(2)
        e = torch.randn(2,6); a = torch.randn(2,2)
        before,_ = old.observe_step(before,a,e,deterministic=True)
        after,_ = self.rssm.observe_step(after,a,e,deterministic=True)
        for k in ['h','z']: torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
        broken = dict(old.state_dict()); broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError): self.rssm.warm_start_categorical(broken)
        with self.assertRaises(ValueError): self.rssm.warm_start_categorical(self.rssm.state_dict())

    def test_strict_reload_and_missing_branch_rejected(self):
        saved = {k:copy.deepcopy(v.state_dict()) for k,v in self.wm.items()}
        load_world_model_state(self.wm, {'models':saved})
        saved['rssm'] = {k:v for k,v in saved['rssm'].items() if not k.startswith('continuous.')}
        with self.assertRaises(RuntimeError): load_world_model_state(self.wm, {'models':saved})

    def test_observation_slice_and_no_categorical_readout(self):
        e = torch.randn(2,6); altered=e.clone(); altered[:,:3] += 100
        torch.testing.assert_close(self.rssm.continuous.observe(e),self.rssm.continuous.observe(altered))
        state = self.rssm.initial_state(2)
        output,_ = self.rssm.observe_step(state,torch.zeros(2,2),e)
        torch.testing.assert_close(output['c'],self.rssm.continuous.observe(e))

    def test_reset_masks_previous_continuous_state_and_action(self):
        state = self.rssm.initial_state(2); state['c'].fill_(100)
        a=torch.ones(2,2); e=torch.randn(2,6)
        _,stats=self.rssm.observe_step(state,a,e,is_first=torch.tensor([1,0]))
        expected=self.rssm.continuous.imagine(torch.tensor([[0.]*3,[100.]*3]),torch.tensor([[0.,0.],[1.,1.]]))
        torch.testing.assert_close(stats['continuous_prior'],expected)

    def test_prior_is_recursive_and_no_future_observation_or_gt(self):
        class Add:
            def imagine(self,c,a): return c+a
        seq=continuous_rollout(Add(),torch.zeros(1,2),torch.tensor([[[1.,2.]],[[3.,4.]]]))
        self.assertEqual(seq[:,0].tolist(),[[0.,0.],[1.,2.],[4.,6.]])
        start=self.rssm.initial_state(2)
        with patch.object(self.rssm.continuous,'observe',side_effect=AssertionError('future observation')):
            end,_=self.rssm.imagine_step(dict(start,control=torch.full((2,3),float('nan'))),torch.ones(2,2))
        self.assertTrue(torch.isfinite(end['c']).all())

    def test_future_targets_only_change_loss_not_rollout(self):
        branch=self.rssm.continuous; e=torch.randn(2,6); actions=torch.randn(2,2,2)
        target=torch.randn(3,2,3)
        with patch.object(branch,'imagine',wraps=branch.imagine) as call:
            loss,_=continuous_prediction_loss(branch,e,actions,target,horizons=(1,2))
            arguments=[x.args[0].detach().clone() for x in call.call_args_list]
        with patch.object(branch,'imagine',wraps=branch.imagine) as call:
            other,_=continuous_prediction_loss(branch,e,actions,target+10,horizons=(1,2))
            for x,y in zip(arguments,call.call_args_list): torch.testing.assert_close(x,y.args[0])
        self.assertNotEqual(float(loss),float(other))

    def test_stacked_posterior_preserves_c_and_subsample_alignment(self):
        states=posterior_start_states(self.batch,self.wm,deterministic=True)
        embedding=self.wm['encoder'](self.batch['image'].flatten(0,1),self.batch['state'].flatten(0,1))
        torch.testing.assert_close(states['c'],self.rssm.continuous.observe(embedding))
        self.assertNotIn('control',states)
        torch.manual_seed(6); small=posterior_start_states(self.batch,self.wm,deterministic=True,max_states=3)
        torch.manual_seed(6); indices=torch.randperm(6)[:3]
        torch.testing.assert_close(small['c'],states['c'][indices])

    def test_bc_and_live_policy_never_fall_back_to_gt(self):
        original=copy.deepcopy(self.rssm.state_dict())
        with patch.object(self.ac.actor,'forward',wraps=self.ac.actor.forward) as call:
            train_imitation_step(self.batch,self.wm,self.ac,torch.optim.Adam(self.ac.actor.parameters(),lr=.001))
            states=posterior_start_states(self.batch,self.wm,deterministic=True)
            torch.testing.assert_close(call.call_args.kwargs['control_state'],self.rssm.continuous.decode(states['c']))
        for k,v in original.items(): torch.testing.assert_close(self.rssm.state_dict()[k],v,rtol=0,atol=0)
        policy=LatentActorPolicy(self.wm['encoder'],self.rssm,self.ac,device=torch.device('cpu'))
        obs=dict(image=np.zeros((16,16,3),dtype=np.uint8),state=np.zeros(3,dtype=np.float32))
        first=policy(obs); self.assertFalse(torch.equal(policy._latent['control'],torch.zeros(1,3)))
        policy(obs);policy.reset();np.testing.assert_array_equal(first,policy(obs))
        with self.assertRaisesRegex(ValueError,'GT fallback'):
            LatentActorPolicy(self.wm['encoder'],RSSM(self.legacy,6,2),self.ac,device=torch.device('cpu'))

    def test_imagination_action_gradient_and_rl_remains_guarded(self):
        with torch.no_grad(): self.rssm.continuous.transition[-1].weight.normal_(0,.1)
        start=self.rssm.initial_state(2);start['control']=torch.full((2,3),float('nan'))
        with freeze_parameters(self.wm.values()):
            trajectory=imagine_rollout(start,self.wm,self.ac,horizon=3,gamma=.99,diagnostic_control=True)
            trajectory.control[-1].square().sum().backward()
        self.assertEqual(tuple(trajectory.control.shape),(4,2,3))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.ac.actor.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for m in self.wm.values() for p in m.parameters()))
        with self.assertRaisesRegex(ValueError,'no validated'): self.ac.require_imagination_ready()
        with self.assertRaisesRegex(ValueError,'explicit branch training'):
            train_world_model_step({},self.wm,None,{})

    def test_invalid_config_rejected(self):
        for extra in [dict(embed_start=6),dict(state_std=[0.]*3),dict(dim=0)]:
            with self.assertRaises(ValueError): ContinuousState(dict(self.cfg['rssm']['continuous_state'],**extra),6,2)

    def test_explicit_diagnostic_rl_requires_matching_control_critic(self):
        start=self.rssm.initial_state(2)
        opta=torch.optim.Adam(self.ac.actor.parameters(),lr=.001)
        optc=torch.optim.Adam(self.ac.critic.parameters(),lr=.001)
        with self.assertRaisesRegex(ValueError,'matching model-state Critic'):
            train_actor_critic_step(start,self.wm,self.ac,opta,optc,{},diagnostic_control=True)
        cfg=dict(self.ac.cfg,critic_control_state_dim=3)
        ac=ActorCritic(cfg,4,4,2)
        opta=torch.optim.Adam(ac.actor.parameters(),lr=.001);optc=torch.optim.Adam(ac.critic.parameters(),lr=.001)
        frozen={k:copy.deepcopy(m.state_dict()) for k,m in self.wm.items()}
        start['control']=torch.full((2,3),float('nan'))
        with patch.object(ac.critic,'forward',wraps=ac.critic.forward) as call:
            result=train_actor_critic_step(start,self.wm,ac,opta,optc,dict(imagination_horizon=3),diagnostic_control=True)
            self.assertEqual(tuple(call.call_args.kwargs['control_state'].shape),(6,3))
            self.assertTrue(torch.isfinite(call.call_args.kwargs['control_state']).all())
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        for name,m in self.wm.items():
            for k,v in m.state_dict().items():torch.testing.assert_close(v,frozen[name][k],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'no validated'):
            train_actor_critic_step(start,self.wm,ac,opta,optc,{})
        with self.assertRaisesRegex(ValueError,'finite model control'):
            ac.value(start)

    def test_learned_encoder_branch_preserves_legacy_embedding_and_rssm(self):
        old=build_world_model(self.legacy,(16,16,3),3,2,torch.device('cpu'),include_decoder=False)
        cfg=copy.deepcopy(self.cfg)
        cfg['encoder']['control_embedding']=dict(dim=5)
        cfg['rssm']['categorical_embed_dim']=6
        cfg['rssm']['continuous_state'].update(embed_start=6,embed_stop=11)
        wm=build_world_model(cfg,(16,16,3),3,2,torch.device('cpu'),include_decoder=False)
        wm['encoder'].warm_start_base(old['encoder'].state_dict())
        wm['rssm'].warm_start_categorical(old['rssm'].state_dict())
        image=self.batch['image'][:,0]; state=self.batch['state'][:,0]
        before=old['encoder'](image,state); after=wm['encoder'](image,state)
        torch.testing.assert_close(before,after[:,:6],rtol=0,atol=0)
        a=torch.randn(2,2)
        os,_=old['rssm'].observe_step(old['rssm'].initial_state(2),a,before,deterministic=True)
        ns,_=wm['rssm'].observe_step(wm['rssm'].initial_state(2),a,after,deterministic=True)
        for k in ['h','z']:torch.testing.assert_close(os[k],ns[k],rtol=0,atol=0)
        with self.assertRaises(ValueError): wm['encoder'].warm_start_base(wm['encoder'].state_dict())
        with self.assertRaises(RuntimeError): wm['encoder'].load_state_dict(old['encoder'].state_dict())
        for p in wm['encoder'].parameters():p.requires_grad_(False)
        wm['encoder'].control_embedding.requires_grad_(True)
        for p in wm['rssm'].parameters():p.requires_grad_(False)
        wm['rssm'].continuous.requires_grad_(True)
        loss=wm['rssm'].continuous.observe(wm['encoder'](image,state)).square().mean();loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in wm['encoder'].control_embedding.parameters()),0)
        self.assertTrue(all(p.grad is None for p in wm['encoder'].state_mlp.parameters()))


if __name__ == '__main__': unittest.main()
