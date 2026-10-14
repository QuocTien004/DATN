"""Frozen replay audit of P1 oracle starts, P2 memory, P3 heuristic heads.

No fitting, Actor updates, readiness bypass or RL. Cached posterior features must
come from the exact supplied baseline; manifests and checkpoint hashes are saved.
Run --help for the dataset contract. Episode boundaries are never concatenated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from evaluation.p1_p4_diagnostics import boundary_predictions, oracle_start
from models.world_model import build_world_model, load_world_model_state


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def sample_starts(n,horizon,stride):
    """Integer indices even when there is no t30; include final successor."""
    if n < horizon or horizon < 1 or stride < 1:
        raise ValueError('Invalid horizon/stride or episode too short')
    values=list(range(0,n-horizon+1,stride))+list(range(max(0,n-horizon-14),n-horizon+1))
    if 30<=n-horizon: values.append(30)
    return np.asarray(sorted(set(values)),dtype=np.int64)


def verify_collection_provenance(manifest,baseline_sha256):
    p=Path(manifest).parent/'manifest.json'
    data=json.loads(p.read_text())
    found=data.get('bc_sha256',data.get('baseline_sha256'))
    if found!=baseline_sha256:
        raise ValueError('Collection posterior cache must come from the supplied baseline')
    return dict(path=str(p),sha256=sha(p),baseline_sha256=found)


def episode(path,meta):
    with np.load(path,allow_pickle=False) as f:
        a={k:f[k].copy() for k in ['state','c','hard','action','reward','terminal','out','crash','finish','truncated']}
    n=len(a['action'])
    if n!=meta['n'] or a['state'].shape!=(n+1,19) or a['c'].shape!=(n+1,19):
        raise ValueError('Expected N executed actions and N+1 state/c nodes')
    if a['action'].shape!=(n,2) or any(len(a[k])!=n for k in ['reward','terminal','out','crash','finish','truncated']):
        raise ValueError('Transition labels must have length N')
    if a['hard'].shape[0]!=n+1 or a['terminal'][:-1].any() or a['truncated'][:-1].any():
        raise ValueError('Invalid nodes or reset within episode')
    if not all(np.isfinite(v).all() for v in a.values()) or np.abs(a['action']).max()>1:
        raise ValueError('Nonfinite data or unbounded action')
    np.testing.assert_allclose(a['state'][1:,5:7],(a['action']+1)/2,rtol=0,atol=1e-6)
    return a


def summarize(a,mean_reward):
    pred=a['prediction'].astype(float);gt=a['gt'].astype(float)
    hold=a['hold'].astype(float);start=a['gt_start'].astype(float)
    mse=((pred-gt)**2).mean(0); hm=((hold-gt)**2).mean(0)
    dm=(((pred-hold)-(gt-start))**2).mean(0); zm=((gt-start)**2).mean(0)
    y=np.sign(a['reward'])*np.log1p(np.abs(a['reward']))
    term=a['terminal'].astype(bool);negative=term&(a['reward']<0)
    r=a['pred_reward'];logit=a['pred_continue'];tp=(logit<=0)
    def rate(v,m): return float(v[m].mean()) if m.any() else None
    denom=float(((y-mean_reward)**2).mean())
    return dict(n=len(gt),rmse=np.sqrt(mse).tolist(),hold_ratio=(mse/np.maximum(hm,1e-12)).tolist(),
                delta_zero_ratio=(dm/np.maximum(zm,1e-12)).tolist(),
                reward_mse=float(((r-y)**2).mean()),reward_mean_ratio=float(((r-y)**2).mean()/max(denom,1e-12)),
                terminals=int(term.sum()),negative_failures=int(negative.sum()),
                terminal_recall=rate(tp,term),false_positive_rate=rate(tp,~term),
                negative_failure_recall=rate(r<0,negative),
                out_recall=rate(tp,a['out'].astype(bool)),crash_recall=rate(tp,a['crash'].astype(bool)),
                finish_recall=rate(tp,a['finish'].astype(bool)),
                truncated_false_terminal=rate(tp,a['truncated'].astype(bool)&~term))


@torch.no_grad()
def run(args):
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Output must be empty; do not overwrite evidence')
    out.mkdir(parents=True,exist_ok=True);(out/'episodes').mkdir()
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if sha(args.checkpoint)!=args.baseline_sha256: raise ValueError('Baseline hash mismatch')
    wm=build_world_model(ck['wm_cfg'],ck['image_shape'],19,2,torch.device(args.device),include_decoder=False)
    load_world_model_state(wm,ck)
    for m in wm.values():m.eval().requires_grad_(False)
    rssm=wm['rssm'];branch=rssm.continuous
    if branch is None or branch.dim!=19: raise ValueError('Expected continuous BC baseline')
    if getattr(wm['reward'],'continuous_state_dim',0) or getattr(wm['continue'],'continuous_state_dim',0):
        raise ValueError('P0 comparison requires legacy h/z heads')
    if args.prior_checkpoint:
        prior=torch.load(args.prior_checkpoint,map_location='cpu',weights_only=False)['branch']
        for k,v in branch.state_dict().items():
            if not k.startswith('transition.'):
                torch.testing.assert_close(v.cpu(),prior[k].cpu(),rtol=0,atol=0)
        branch.load_state_dict(prior,strict=True)
    exact=copy.deepcopy(branch);exact.exact_action_memory=True
    datasets=[]
    provenance=[verify_collection_provenance(p,args.baseline_sha256) for p in args.manifest]
    for manifest in args.manifest:
        p=Path(manifest);d=json.loads(p.read_text())
        for meta in d['episodes']:
            if meta['split'] in args.splits:
                name=f"{meta['split']}_{meta['map']}_{meta['mode']}"
                datasets.append((p.parent/'dataset'/(name+'.npz'),meta,name))
    if not datasets: raise ValueError('No episodes selected')
    config=dict(vars(args),checkpoint_sha256=sha(args.checkpoint),script_sha256=sha(__file__),
                manifests={p:sha(p) for p in args.manifest},collection_provenance=provenance,torch=torch.__version__,
                numpy=np.__version__,episodes=len(datasets),rl_updates=0,
                prior_sha256=sha(args.prior_checkpoint) if args.prior_checkpoint else None,
                p1='oracle observed start only, NOT a learned posterior',
                persistence_reference='learned start for P0/P2; GT start for oracle variants',
                seed_meaning='categorical prior RNG; c prior deterministic, not independent model training seeds',
                sampling='stride plus t30 plus final15 transitions; each horizon includes terminal successor',
                p3='Tien fixed .0725, .16, 120, .05, -.6 symlog; no fitted hzc reward supplied')
    write(out/'config.json',config)
    # Reward mean fitted on TRAIN only; no calibration or tuning on valid/test.
    train=[]
    for manifest in args.manifest:
        p=Path(manifest)
        for meta in json.loads(p.read_text())['episodes']:
            if meta['split']=='train':
                with np.load(p.parent/'dataset'/f"train_{meta['map']}_{meta['mode']}.npz") as f:r=f['reward']
                train.extend((np.sign(r)*np.log1p(np.abs(r))).tolist())
    if not train: raise ValueError('A train manifest is required for the mean-reward control')
    mean=float(np.mean(train));write(out/'train_constants.json',dict(mean_reward_symlog=mean))
    aggregate={};checks=[]
    for ei,(path,meta,name) in enumerate(datasets):
        if sha(path)!=meta['sha256']: raise ValueError('Dataset checksum mismatch: '+name)
        a=episode(path,meta);n=len(a['action'])
        t={k:torch.as_tensor(a[k],device=args.device,dtype=torch.float32) for k in ['state','c','hard','action']}
        h0=t['hard'][:,:rssm.deter_dim];z0=t['hard'][:,rssm.deter_dim:].reshape(n+1,rssm.stoch_dim,rssm.stoch_classes)
        evidence={}
        for H in [0,1,5,15]:
            # H0 means successor posterior vs the actual transition label.
            if H==0: idx=np.arange(n);end=idx+1
            else:
                if n<H:continue
                idx=sample_starts(n,H,args.stride)
                end=idx+H
            ti=torch.as_tensor(idx,device=args.device);te=torch.as_tensor(end,device=args.device)
            gt=t['state'][te];hold=branch.decode(t['c'][ti])
            for seed in (args.seeds if H else [args.seeds[0]]):
                torch.manual_seed(seed)
                if H:
                    s={'h':h0[ti],'z':z0[ti],'c':t['c'][ti]}
                    c0=s['c'];c1=oracle_start(s,t['state'][ti],branch)['c'];c2=c0;c12=c1
                    for j in range(H):
                        act=t['action'][ti+j]
                        s,_=rssm.imagine_step(s,act,deterministic=False)
                        c0=s['c'];c1=branch.imagine(c1,act);c2=exact.imagine(c2,act);c12=exact.imagine(c12,act)
                    variants={'P0':c0,'P1_oracle':c1,'P2':c2,'P1P2_oracle':c12}
                else:
                    s={'h':h0[te],'z':z0[te]};variants={'posterior':t['c'][te],'GT_oracle':branch.normalize(gt)}
                rew=wm['reward'](s['h'],s['z']);cont=wm['continue'](s['h'],s['z'])
                for label,c in variants.items():
                    physical=branch.decode(c)
                    heads={'hz':{'reward':rew,'continue':cont},'P3':boundary_predictions(physical,rew,cont)}
                    for head,pred in heads.items():
                        reference=t['state'][ti] if 'oracle' in label else hold
                        rec=dict(prediction=physical.cpu().numpy(),gt=gt.cpu().numpy(),hold=reference.cpu().numpy(),gt_start=a['state'][idx],
                                 start=idx,end=end,mode=np.full(len(idx),meta['mode']),map=np.full(len(idx),meta['map']),
                                 pred_reward=pred['reward'][:,0].cpu().numpy(),pred_continue=pred['continue'][:,0].cpu().numpy(),
                                 **{k:a[k][end-1] for k in ['reward','terminal','out','crash','finish','truncated']})
                        prefix=f'{label}_{head}_h{H}_seed{seed}'
                        evidence.update({prefix+'__'+k:v for k,v in rec.items()})
                        key=(meta['split'],label,head,H,seed)
                        aggregate.setdefault(key,[]).append(rec)
        np.savez_compressed(out/'episodes'/(name+'.npz'),**evidence)
        checks.append(dict(name=name,sha256=meta['sha256'],n=n,memory_max_error=float(np.max(np.abs(a['state'][1:,5:7]-(a['action']+1)/2)))))
        write(out/'progress.json',dict(completed=ei+1,total=len(datasets),last=name,rl_updates=0))
        print('EPISODE',ei+1,len(datasets),name,flush=True)
    rows=[]
    for key,parts in aggregate.items():
        data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        groups={'all':np.ones(len(data['start']),bool),'moving_offbc':np.isin(data['mode'],['noise','swerve','fast'])}
        groups.update({m:data['mode']==m for m in ['clean','noise','swerve','fast','timeout']})
        groups.update({f'map{m}':data['map']==m for m in np.unique(data['map'])})
        for group,mask in groups.items():
            if mask.any():rows.append(dict(split=key[0],model=key[1],head=key[2],horizon=key[3],seed=key[4],group=group,
                                          **summarize({k:v[mask] for k,v in data.items()},mean)))
    write(out/'metrics.json',rows);write(out/'data_checks.json',checks)
    # Conservative fail-closed: diagnostics never authorize a pilot themselves.
    write(out/'gate.json',dict(rl_locked=True,pilot_authorized=False,required='c gate + real failure heads + paired real action ranking + Critic audit',
                               oracle_not_deployable=True))
    write(out/'completed.json',dict(episodes=len(datasets),metrics_rows=len(rows),rl_updates=0,config_sha256=sha(out/'config.json'),metrics_sha256=sha(out/'metrics.json')))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-sha256',required=True)
    p.add_argument('--prior-checkpoint',help='Optional branch refit; non-transition weights must exactly match baseline')
    p.add_argument('--manifest',action='append',required=True,help='collection_complete.json with episodes; sibling dataset/{split}_{map}_{mode}.npz contains N+1 state,c,hard and N actions/labels')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--seeds',nargs='+',type=int,default=[19,20,21]);p.add_argument('--stride',type=int,default=8)
    p.add_argument('--splits',nargs='+',default=['valid','test','confirm'])
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.stride<=0: raise ValueError('stride must be positive')
    run(args)
