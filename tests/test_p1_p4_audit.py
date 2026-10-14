import copy
import io
import unittest

import torch

from evaluation.p1_p4_diagnostics import oracle_start, boundary_predictions, discounted_reward_return,frozen_policy_for_audit
from models.continuous_state import ContinuousState
from models.actor_critic import ActorCritic
from models.world_model import build_world_model,load_world_model_state
from scripts.audit_p1_p4 import sample_starts, episode,verify_collection_provenance,run,sha
from argparse import Namespace
import json
import numpy as np
import tempfile
from pathlib import Path


class P1P4AuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = dict(dim=19, posterior_hidden=[], prior_hidden=[8],
                        state_mean=[.2]*19, state_std=[.4]*19)

    def branch(self, enabled=False):
        return ContinuousState(dict(self.cfg, exact_action_memory=enabled), 8, 2)

    def test_p1_oracle_is_explicit_detached_and_does_not_mutate_posterior(self):
        branch = self.branch()
        s = dict(h=torch.ones(2,3), z=torch.ones(2,2), c=torch.zeros(2,19))
        gt = torch.ones(2,19,requires_grad=True)
        oracle = oracle_start(s,gt,branch)
        torch.testing.assert_close(oracle['c'], branch.normalize(gt))
        self.assertFalse(oracle['c'].requires_grad)
        self.assertEqual(s['c'].abs().sum().item(), 0)
        self.assertIs(s['h'], oracle['h'])
        with self.assertRaises(ValueError): oracle_start(s,gt[None],branch)

    def test_p2_default_legacy_weights_and_output_unchanged(self):
        old = ContinuousState(self.cfg,8,2)
        explicit = self.branch(False); explicit.load_state_dict(old.state_dict(),strict=True)
        c=torch.randn(4,19);a=torch.rand(4,2)*2-1
        torch.testing.assert_close(old.imagine(c,a),explicit.imagine(c,a),rtol=0,atol=0)
        self.assertEqual(set(old.state_dict()),set(self.branch(True).state_dict()))

    def test_p2_memory_timing_normalization_and_gradients(self):
        branch=self.branch(True)
        c=torch.randn(2,19,requires_grad=True);a=torch.tensor([[-1.,1.],[.2,-.4]],requires_grad=True)
        out=branch.imagine(c,a)
        torch.testing.assert_close(branch.decode(out)[:,5:7],(a+1)/2)
        out[:,5:7].sum().backward()
        torch.testing.assert_close(a.grad,torch.full_like(a,1.25))
        self.assertEqual(c.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or p.grad.abs().sum()==0 for p in branch.transition.parameters()))
        a2=-a.detach()
        torch.testing.assert_close(branch.decode(branch.imagine(out.detach(),a2))[:,5:7],(a2+1)/2)

    def test_p2_other_coordinates_keep_action_gradient(self):
        branch=self.branch(True)
        with torch.no_grad():
            for p in branch.transition.parameters(): p.fill_(.1)
        a=torch.zeros(1,2,requires_grad=True)
        branch.imagine(torch.zeros(1,19),a)[:,0].sum().backward()
        self.assertGreater(a.grad.abs().sum().item(),0)

    def test_p2_config_checkpoint_round_trip_requires_explicit_flag(self):
        cfg=dict(self.cfg,exact_action_memory=True);branch=ContinuousState(cfg,8,2)
        buf=io.BytesIO();torch.save(dict(cfg=cfg,weights=branch.state_dict()),buf);buf.seek(0)
        saved=torch.load(buf,weights_only=True)
        restored=ContinuousState(saved['cfg'],8,2);restored.load_state_dict(saved['weights'],strict=True)
        self.assertTrue(restored.exact_action_memory)
        c=torch.randn(2,19);a=torch.ones(2,2)
        torch.testing.assert_close(restored.imagine(c,a),branch.imagine(c,a),rtol=0,atol=0)
        legacy=self.branch(False);legacy.load_state_dict(saved['weights'],strict=True)
        self.assertFalse(torch.equal(legacy.imagine(c,a),restored.imagine(c,a)))

    def test_p2_full_world_model_reload_and_live_posterior_parity(self):
        cfg=dict(encoder=dict(cnn_channels=[4],mlp_hidden=[4],embed_dim=8),
                 rssm=dict(deter_dim=4,stoch_dim=2,stoch_classes=2,hidden_dim=8,continuous_state=self.cfg))
        old=build_world_model(cfg,(16,16,3),19,2,torch.device('cpu'),include_decoder=False)
        opt=copy.deepcopy(cfg);opt['rssm']['continuous_state']['exact_action_memory']=True
        new=build_world_model(opt,(16,16,3),19,2,torch.device('cpu'),include_decoder=False)
        load_world_model_state(new,{'models':{k:v.state_dict() for k,v in old.items()}})
        e=torch.randn(2,8);a=torch.rand(2,2)*2-1
        x,_=old['rssm'].observe_step(old['rssm'].initial_state(2),a,e,deterministic=True)
        y,_=new['rssm'].observe_step(new['rssm'].initial_state(2),a,e,deterministic=True)
        for k in ['h','z','c']:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)

    def test_p2_rejects_wrong_contract_and_unexecuted_action(self):
        for cfg in [dict(self.cfg,dim=3),dict(self.cfg,exact_action_memory='true')]:
            with self.assertRaises(ValueError): ContinuousState(dict(cfg,exact_action_memory=cfg.get('exact_action_memory',True)),8,2)
        b=self.branch(True)
        for x in [1.01,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): b.imagine(torch.zeros(1,19),torch.full((1,2),x))

    def test_p3_shape_units_threshold_and_stationary_blind_spot(self):
        s=torch.ones(1,19);s[:,0:2]=.01;s[:,2]=.5
        r=torch.ones(1,1);cont=torch.full((1,1),8.)
        out=boundary_predictions(s,r,cont)
        self.assertEqual(out['reward'].shape,(1,1))
        self.assertAlmostEqual(out['reward'].item(),-.6,places=6)
        self.assertAlmostEqual(-torch.expm1(torch.tensor(.6)).item(),-.8221188,places=6)
        self.assertLess(out['continue'].item(),0)
        s[:,3]=0
        # Preserve this behavior only as a diagnostic: stationary does NOT prove safe.
        self.assertEqual(boundary_predictions(s,r,cont)['continue'].item(),8.)
        with self.assertRaises(ValueError): boundary_predictions(s,r[:,0],cont)

    def test_p3_preserves_input_gradient_not_head_parameter_gradient(self):
        head=torch.nn.Linear(3,1).requires_grad_(False)
        h=torch.ones(2,3,requires_grad=True)
        s=torch.ones(2,19);s[:,2]=.5
        out=boundary_predictions(s,head(h),head(h))
        out['continue'].sum().backward()
        self.assertGreater(h.grad.abs().sum().item(),0)
        self.assertIsNone(head.weight.grad)

    def test_p4_reward_only_return_includes_terminal_reward(self):
        r=torch.tensor([[1.],[2.],[100.]])
        c=torch.tensor([[1.],[0.],[1.]])
        torch.testing.assert_close(discounted_reward_return(r,c,.5),torch.tensor([2.]))
        with self.assertRaises(ValueError): discounted_reward_return(r,c+2)

    def test_rl_guard_not_removed_by_any_diagnostic(self):
        ac=ActorCritic(dict(actor_hidden=[8],critic_hidden=[8],control_state_dim=19,
                           control_state_source='continuous',control_state_mean=[0.]*19,
                           control_state_std=[1.]*19),4,4,2)
        with self.assertRaisesRegex(ValueError,'RL is disabled'): ac.require_imagination_ready()

    def test_short_episode_indices_stay_integer_and_keep_terminal(self):
        for n,h in [(1,1),(15,15),(23,5),(100,15)]:
            idx=sample_starts(n,h,8)
            self.assertEqual(idx.dtype,np.int64)
            self.assertEqual(idx[-1]+h,n)
            self.assertTrue((idx>=0).all())
        with self.assertRaises(ValueError):sample_starts(2,5,8)

    def test_dataset_action_successor_timing_and_no_reset_crossing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ep.npz';s=np.zeros((3,19));s[1:,5:7]=.5
            d=dict(state=s,c=s,hard=np.zeros((3,8)),action=np.zeros((2,2)),reward=np.ones(2),
                   terminal=np.array([False,True]),out=np.array([False,True]),crash=np.zeros(2),finish=np.zeros(2),truncated=np.zeros(2))
            np.savez(p,**d);self.assertEqual(len(episode(p,{'n':2})['state']),3)
            d['terminal'][0]=True;np.savez(p,**d)
            with self.assertRaises(ValueError):episode(p,{'n':2})

    def test_reject_cached_posterior_from_different_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text(json.dumps(dict(bc_sha256='expected')))
            self.assertEqual(verify_collection_provenance(p.parent/'collection_complete.json','expected')['baseline_sha256'],'expected')
            with self.assertRaises(ValueError):verify_collection_provenance(p.parent/'collection_complete.json','wrong')

    def test_offline_audit_smoke_persists_completion_and_stays_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'dataset';data.mkdir()
            cfg=dict(encoder=dict(cnn_channels=[4],mlp_hidden=[4],embed_dim=8),
                     rssm=dict(deter_dim=4,stoch_dim=2,stoch_classes=2,hidden_dim=8,continuous_state=self.cfg))
            wm=build_world_model(cfg,(16,16,3),19,2,torch.device('cpu'),include_decoder=False)
            cp=root/'baseline.pt';torch.save(dict(wm_cfg=cfg,image_shape=(16,16,3),models={k:v.state_dict() for k,v in wm.items()}),cp)
            digest=sha(cp);(root/'manifest.json').write_text(json.dumps(dict(bc_sha256=digest)))
            s=np.zeros((3,19),np.float32);s[1:,5:7]=.5
            d=dict(state=s,c=s,hard=np.zeros((3,8),np.float32),action=np.zeros((2,2),np.float32),reward=np.array([1.,-20.]),
                   terminal=np.array([False,True]),out=np.array([False,True]),crash=np.zeros(2),finish=np.zeros(2),truncated=np.zeros(2))
            meta=[]
            for split in ['train','test']:
                p=data/f'{split}_1_clean.npz';np.savez(p,**d)
                meta.append(dict(split=split,map=1,mode='clean',n=2,sha256=sha(p)))
            manifest=root/'collection_complete.json';manifest.write_text(json.dumps(dict(episodes=meta)))
            args=Namespace(output=str(root/'out'),checkpoint=str(cp),baseline_sha256=digest,device='cpu',prior_checkpoint=None,
                           manifest=[str(manifest)],splits=['test'],seeds=[19],stride=8)
            run(args)
            self.assertEqual(json.loads((root/'out/completed.json').read_text())['episodes'],1)
            self.assertTrue(json.loads((root/'out/gate.json').read_text())['rl_locked'])
            with np.load(root/'out/episodes/test_1_clean.npz') as evidence:
                pre='P1_oracle_hz_h1_seed19__'
                np.testing.assert_array_equal(evidence[pre+'hold'],evidence[pre+'gt_start'])
            self.assertEqual(sha(cp),digest)
            with self.assertRaises(ValueError):run(args)

    def test_policy_only_audit_is_strict_frozen_and_not_a_wm_resume(self):
        cfg=dict(encoder=dict(cnn_channels=[4],mlp_hidden=[4],embed_dim=8),
                 rssm=dict(deter_dim=4,stoch_dim=2,stoch_classes=2,hidden_dim=8,continuous_state=self.cfg))
        wm=build_world_model(cfg,(16,16,3),19,2,torch.device('cpu'),include_decoder=False)
        ac_cfg=dict(actor_hidden=[8],critic_hidden=[8],control_state_dim=19,control_state_source='continuous',
                    control_state_mean=[0.]*19,control_state_std=[1.]*19)
        ac=ActorCritic(ac_cfg,4,4,2)
        ck=dict(wm_cfg=cfg,actor_critic_cfg=ac_cfg,actor_critic=ac.state_dict(),image_shape=(16,16,3),
                transition_contract=2,state_dim=19,action_dim=2,models={k:v.state_dict() for k,v in wm.items()})
        ck['models']['reward']={'wrapper.unloadable':torch.zeros(1)}
        policy=frozen_policy_for_audit(ck,torch.device('cpu'))
        action=policy(dict(image=np.zeros((16,16,3),np.uint8),state=np.zeros(19,np.float32)))
        self.assertEqual(action.shape,(2,))
        self.assertTrue(all(not p.requires_grad for m in [policy.encoder,policy.rssm,policy.actor_critic] for p in m.parameters()))
        with self.assertRaises(ValueError):policy.actor_critic.require_imagination_ready()
        with self.assertRaises(RuntimeError):load_world_model_state(wm,ck)
        bad=copy.deepcopy(ck);bad['models']['encoder'].pop(next(iter(bad['models']['encoder'])))
        with self.assertRaises(RuntimeError):frozen_policy_for_audit(bad,torch.device('cpu'))
        bad=copy.deepcopy(ck);bad['actor_critic_cfg']['control_state_source']='observed'
        with self.assertRaises(ValueError):frozen_policy_for_audit(bad,torch.device('cpu'))


if __name__=='__main__': unittest.main()
