#!/usr/bin/env python
"""Check the actual model/camera/top-down on the presentation machine."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo_ui.controller import Controller
from demo_ui.recording import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',default='checkpoints')
    parser.add_argument('--checkpoint',required=True,help='Path relative to checkpoint-dir')
    parser.add_argument('--wm-checkpoint',default=None)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--rng-seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--safety-shield',action='store_true')
    parser.add_argument('--output-dir',default='demo_runs/render_check')
    args=parser.parse_args()
    if not 1<=args.steps<=30:parser.error('--steps must be 1..30 (render smoke only)')
    ctx=Controller(args.checkpoint_dir,args.output_dir)
    try:
        meta=ctx.load('check',args.checkpoint,args.wm_checkpoint,args.device)
        ctx.open('check',args.seed,args.rng_seed,args.steps,args.safety_shield)
        for _ in range(args.steps):
            result=ctx.step('check')
            if result['done']:break
        rgb=result['rgb'];top=result['topdown']
        summary=dict(checkpoint=meta,rgb_shape=list(rgb.shape),rgb_std=float(rgb.std()),
                     topdown_shape=list(top.shape) if top is not None else None,
                     topdown_std=float(top.std()) if top is not None else None,
                     record_folder=ctx.get('check').folder,steps=result['record']['step']+1,
                     topdown_warning=result['record'].get('topdown_warning'),training_updates=0)
        if float(rgb.std())<1:raise RuntimeError('Camera ít biến thiên/đen. Kiểm tra render trước khi trình diễn.')
        write_json(Path(args.output_dir)/'runtime_check.json',summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:ctx.close_all()


if __name__=='__main__':main()
