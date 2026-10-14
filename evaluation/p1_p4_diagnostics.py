"""Read-only P1--P4 ablations. These helpers do NOT grant RL readiness.

P1 uses observed ground truth at the start only: it is an oracle diagnostic,
not the learned posterior used by BC/live evaluation. P3 reproduces a heuristic
whose threshold and penalty are hypotheses, not validated terminal physics.
"""
from __future__ import annotations

import math
import torch


def frozen_policy_for_audit(checkpoint, device):
    """Strictly load only encoder/RSSM/Actor-Critic for real-environment eval.

    This explicitly does not load reward/continue/decoder or optimizers, and is
    NOT a repaired World Model or a resume path. Useful for auditing a checkpoint
    whose experimental head serialization cannot be read by the standard WM
    builder. The production readiness guard remains unchanged.
    """
    from models.encoder import Encoder
    from models.rssm import RSSM
    from models.actor_critic import ActorCritic
    from evaluation.evaluate import LatentActorPolicy

    if checkpoint.get('transition_contract') != 2:
        raise ValueError('Policy audit requires transition contract 2')
    if (checkpoint.get('state_dim'), checkpoint.get('action_dim')) != (19, 2):
        raise ValueError('Policy audit requires MetaDrive state19/action2')
    cfg = checkpoint['wm_cfg']
    ac_cfg = checkpoint['actor_critic_cfg']
    if ac_cfg.get('control_state_source') != 'continuous':
        raise ValueError('Policy audit requires learned continuous control, not GT')
    encoder = Encoder(cfg, state_dim=19, image_shape=tuple(checkpoint['image_shape'])).to(device)
    rssm = RSSM(cfg, embed_dim=encoder.embed_dim, action_dim=2).to(device)
    ac = ActorCritic(ac_cfg, rssm.deter_dim, rssm.stoch_size, 2).to(device)
    encoder.load_state_dict(checkpoint['models']['encoder'], strict=True)
    rssm.load_state_dict(checkpoint['models']['rssm'], strict=True)
    ac.load_state_dict(checkpoint['actor_critic'], strict=True)
    for module in [encoder, rssm, ac]:
        module.eval().requires_grad_(False)
    return LatentActorPolicy(encoder, rssm, ac, device=device, deterministic=True)


def oracle_start(state, observed_start, continuous):
    """Replace only c at the current time. Never accept a future sequence."""
    if observed_start.shape != state['c'].shape or observed_start.ndim != 2:
        raise ValueError('Oracle start must match current c (B, D), not future targets')
    if not torch.isfinite(observed_start).all():
        raise ValueError('Oracle start must be finite')
    return dict(state, c=continuous.normalize(observed_start).detach())


def boundary_predictions(physical_state, reward_symlog, continue_logit, *,
                         threshold=.0725, alpha=.16, sharpness=120.,
                         speed_threshold=.05, penalty_symlog=-.6):
    """Tien P3 formula on decoded state; scalar heads retain (B, 1).

    Inputs retain gradients; freezing head weights must not use no_grad around
    their forwards if testing action gradients. This diagnostic installs no head
    in a checkpoint and makes no guarantee about crash/finish detection.
    """
    if physical_state.ndim != 2 or physical_state.shape[1] != 19:
        raise ValueError('Boundary diagnostic requires physical MetaDrive state19')
    if reward_symlog.shape != (len(physical_state), 1) or continue_logit.shape != reward_symlog.shape:
        raise ValueError('Scalar predictors must have shape (B, 1)')
    if not all(math.isfinite(x) for x in [threshold, alpha, sharpness, speed_threshold, penalty_symlog]):
        raise ValueError('Boundary parameters must be finite')
    if alpha < 0 or sharpness <= 0 or penalty_symlog >= 0:
        raise ValueError('Invalid boundary parameters')
    if not all(torch.isfinite(x).all() for x in [physical_state, reward_symlog, continue_logit]):
        raise ValueError('Boundary inputs must be finite')
    s = physical_state
    clearance = torch.minimum(s[:, 0:1], s[:, 1:2]) - alpha * (s[:, 2:3] - .5).abs()
    boundary = torch.where(s[:, 3:4] > speed_threshold,
                           sharpness * (clearance - threshold), torch.full_like(clearance, 10))
    terminal = torch.minimum(continue_logit, boundary)
    failure = (boundary <= 0) | ((continue_logit <= 0) & (reward_symlog < 0))
    reward = torch.where(failure, torch.minimum(reward_symlog, reward_symlog.new_tensor(penalty_symlog)), reward_symlog)
    return {'reward': reward, 'continue': terminal, 'clearance': clearance,
            'boundary_continue': boundary}


def discounted_reward_return(reward, continuation, gamma=.99):
    """Finite-horizon reward-only return; NOT a Critic/lambda return.

    reward[t] belongs to the transition into successor[t]. continuation[t]
    discounts *later* rewards, so a terminal transition's own reward is counted.
    """
    if reward.ndim != 2 or continuation.shape != reward.shape or not len(reward):
        raise ValueError('Expected nonempty (H, B) rewards and continuation')
    if not 0 <= gamma <= 1 or not torch.isfinite(reward).all() or not torch.isfinite(continuation).all():
        raise ValueError('Invalid reward/discount')
    if ((continuation < 0) | (continuation > 1)).any():
        raise ValueError('Continuation must be probabilities in [0, 1]')
    weights = torch.cumprod(torch.cat([torch.ones_like(continuation[:1]), gamma * continuation[:-1]], 0), 0)
    return (weights * reward).sum(0)
