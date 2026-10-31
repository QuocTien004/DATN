import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.world_model import build_world_model, load_world_model_state
from models.actor_critic import ActorCritic
from evaluation.evaluate import LatentActorPolicy
from envs.metadrive_wrapper import make_env

ckpt0 = torch.load('checkpoints/critic_warmup/continuous_seed0_refit_warmup_c19.pt', map_location='cuda')
ckpt25 = torch.load('logs/rl_pilot_safe_250/pilot_checkpoint_update_25.pt', map_location='cuda')

wm = build_world_model(ckpt0['wm_cfg'], tuple(ckpt0['image_shape']), ckpt0['state_dim'], ckpt0['action_dim'], 'cuda', include_decoder=False)
load_world_model_state(wm, ckpt0)

ac0 = ActorCritic(ckpt0['actor_critic_cfg'], wm['rssm'].deter_dim, wm['rssm'].stoch_size, 2).cuda()
ac0.load_state_dict(ckpt0['actor_critic'])

ac25 = ActorCritic(ckpt25['actor_critic_cfg'], wm['rssm'].deter_dim, wm['rssm'].stoch_size, 2).cuda()
ac25.load_state_dict(ckpt25['actor_critic'])

pol0 = LatentActorPolicy(wm['encoder'], wm['rssm'], ac0, device='cuda', deterministic=True)
pol25 = LatentActorPolicy(wm['encoder'], wm['rssm'], ac25, device='cuda', deterministic=True)

def run_ep(pol, name):
    cfg = dict(ckpt0['env_cfg'])
    cfg['start_seed'] = 44200 + 1234
    cfg['num_scenarios'] = 1
    cfg['horizon'] = 1000
    env = make_env(cfg)
    obs, _ = env.reset()
    pol.reset()
    actions = []
    done = False
    while not done:
        a = pol(obs)
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    env.close()
    acts = torch.tensor(actions)
    print(f'=== {name} ===')
    print(f"Steps: {len(acts)}, Finish: {info.get('arrive_dest')}, Out: {info.get('out_of_road')}, Crash: {info.get('crash')}, Route: {info.get('route_completion', 0.0)*100:.2f}%")
    print(f"Steering: mean={acts[:, 0].mean():.3f}, min={acts[:, 0].min():.3f}, max={acts[:, 0].max():.3f}")
    print(f"Throttle: mean={acts[:, 1].mean():.3f}, min={acts[:, 1].min():.3f}, max={acts[:, 1].max():.3f}")

run_ep(pol0, 'POLICY 0 (Pre-RL)')
run_ep(pol25, 'POLICY 25 (After 25 updates)')
