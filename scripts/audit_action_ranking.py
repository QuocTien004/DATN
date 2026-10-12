"""Paired recorded action plans against real rewards, not a Critic success test.

Only pairs with identical pre-intervention histories are accepted. Both plans
receive the same categorical RNG seed. No imagined Actor sampling, fitting or RL.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from models.world_model import build_world_model,load_world_model_state
from evaluation.p1_p4_diagnostics import oracle_start,boundary_predictions,discounted_reward_return
from scripts.audit_p1_p4 import sha,write,episode,verify_collection_provenance
from training.train_agent import symexp


@torch.no_grad()
def run(args):
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    assert sha(args.checkpoint)==args.baseline_sha256
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    wm=build_world_model(ck['wm_cfg'],ck['image_shape'],19,2,torch.device(args.device),include_decoder=False)
    load_world_model_state(wm,ck)
    for m in wm.values():m.eval().requires_grad_(False)
    rssm=wm['rssm'];branch=rssm.continuous
    if args.prior_checkpoint:
        prior=torch.load(args.prior_checkpoint,map_location='cpu',weights_only=False)['branch']
        for k,v in branch.state_dict().items():
            if not k.startswith('transition.'):torch.testing.assert_close(v.cpu(),prior[k].cpu(),rtol=0,atol=0)
        branch.load_state_dict(prior)
    exact=copy.deepcopy(branch);exact.exact_action_memory=True
    provenance=[verify_collection_provenance(p,args.baseline_sha256) for p in args.manifest]
    write(out/'config.json',dict(vars(args),baseline_sha256=sha(args.checkpoint),script_sha256=sha(__file__),collection_provenance=provenance,
          prior_sha256=sha(args.prior_checkpoint) if args.prior_checkpoint else None,
          manifests={p:sha(p) for p in args.manifest},rl_updates=0,real_tie_threshold=.1,
          comparison='fixed recorded plans, identical histories, common RNG, min(15, remaining lengths)'))
    rows=[];skips=[]
    for mp in args.manifest:
        p=Path(mp);metas=json.loads(p.read_text())['episodes']
        lookup={(m['split'],m['map'],m['mode']):m for m in metas if m['split'] in args.splits}
        for split,mapid,mode in sorted(lookup):
            if mode not in ['noise','swerve','fast']:continue
            cm=lookup.get((split,mapid,'clean'))
            if cm is None:continue
            pair=[]
            for meta in [cm,lookup[(split,mapid,mode)]]:
                path=p.parent/'dataset'/f"{split}_{mapid}_{meta['mode']}.npz"
                assert sha(path)==meta['sha256'];pair.append(episode(path,meta))
            K=min(15,*[len(a['action'])-30 for a in pair])
            if K<1:
                skips.append(dict(split=split,map=mapid,mode=mode,reason='ended before t30'));continue
            diff=max(float(np.max(np.abs(pair[0][k][:31]-pair[1][k][:31]))) for k in ['state','hard','c'])
            adiff=float(np.max(np.abs(pair[0]['action'][:30]-pair[1]['action'][:30])))
            if max(diff,adiff)>1e-5:
                skips.append(dict(split=split,map=mapid,mode=mode,reason='unpaired prefix',max_diff=max(diff,adiff)));continue
            real=[float(discounted_reward_return(torch.tensor(a['reward'][30:30+K,None]),
                       1-torch.tensor(a['terminal'][30:30+K,None],dtype=torch.float32),args.gamma)) for a in pair]
            for seed in args.seeds:
                plans=[]
                for a in pair:
                    torch.manual_seed(seed)
                    tensor=lambda x:torch.as_tensor(x,device=args.device,dtype=torch.float32)
                    s=dict(h=tensor(a['hard'][30,:rssm.deter_dim])[None],
                           z=tensor(a['hard'][30,rssm.deter_dim:]).reshape(1,rssm.stoch_dim,rssm.stoch_classes),
                           c=tensor(a['c'][30])[None])
                    c1=oracle_start(s,tensor(a['state'][30])[None],branch)['c'];c2=s['c'];c12=c1
                    traces={}
                    for j in range(K):
                        act=tensor(a['action'][30+j])[None]
                        s,_=rssm.imagine_step(s,act,deterministic=False)
                        c1=branch.imagine(c1,act);c2=exact.imagine(c2,act);c12=exact.imagine(c12,act)
                        rew=wm['reward'](s['h'],s['z']);cont=wm['continue'](s['h'],s['z'])
                        for label,c in [('P0',s['c']),('P1_oracle',c1),('P2',c2),('P1P2_oracle',c12)]:
                            for head,pred in [('hz',dict(reward=rew,**{'continue':cont})),('P3',boundary_predictions(branch.decode(c),rew,cont))]:
                                r,co=traces.setdefault(label+'_'+head,([],[]))
                                r.append(symexp(pred['reward'][:,0],3.26));co.append(pred['continue'][:,0].sigmoid())
                    plans.append({k:dict(reward=torch.stack(v[0]).cpu().tolist(),continuation=torch.stack(v[1]).cpu().tolist(),
                                         total=float(discounted_reward_return(torch.stack(v[0]),torch.stack(v[1]),args.gamma))) for k,v in traces.items()})
                rows.append(dict(split=split,map=mapid,mode=mode,horizon=K,seed=seed,prefix_error=max(diff,adiff),
                                 real_returns=real,real_delta=real[0]-real[1],plans=plans,
                                 predicted_delta={k:plans[0][k]['total']-plans[1][k]['total'] for k in plans[0]}))
            write(out/'progress.json',dict(rows=rows,skipped=skips,rl_updates=0))
    summary=[]
    for split in args.splits:
        for seed in args.seeds:
            rr=[r for r in rows if r['split']==split and r['seed']==seed and abs(r['real_delta'])>.1]
            for label in rows[0]['predicted_delta'] if rows else []:
                for mode in ['all','noise','swerve','fast']:
                    rrr=[r for r in rr if mode=='all' or r['mode']==mode]
                    summary.append(dict(split=split,seed=seed,model=label,mode=mode,n=len(rrr),
                       agreement=float(np.mean([np.sign(r['real_delta'])==np.sign(r['predicted_delta'][label]) for r in rrr])) if rrr else None))
    write(out/'results.json',dict(rows=rows,skipped=skips,summary=summary,critic_evaluated=False,rl_updates=0))
    write(out/'completed.json',dict(paired_rows=len(rows),skipped_pairs=len(skips),results_sha256=sha(out/'results.json'),rl_locked=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-sha256',required=True)
    p.add_argument('--prior-checkpoint');p.add_argument('--manifest',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--seeds',nargs='+',type=int,default=[19,20,21]);p.add_argument('--splits',nargs='+',default=['test','confirm'])
    p.add_argument('--gamma',type=float,default=.997)
    run(p.parse_args())
