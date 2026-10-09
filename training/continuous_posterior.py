"""Opt-in posterior-only ablation with clean-BC replay anchors.

No prior, Actor, coordinate definition, normalization, or RL changes. Current
observation enters through the learned control embedding, not a GT fallback.
"""
from __future__ import annotations

import math
import torch


def control_posterior(encoder, branch, observation_state):
    """Equivalent to filtering the control slice of a full image/state embedding."""
    if (encoder.control_embedding is None or branch.embed_start != encoder.base_embed_dim
            or branch.embed_stop != encoder.embed_dim):
        raise ValueError('Posterior must read exactly the separate control embedding')
    if observation_state.ndim != 2 or observation_state.shape[1] != encoder.state_dim:
        raise ValueError('Expected current observation state (B,state_dim)')
    control = encoder.encode_control(observation_state)
    # The branch never reads this prefix; no image/categorical weights are used.
    prefix = control.new_zeros((len(control), encoder.base_embed_dim))
    return branch.observe(torch.cat([prefix, control], dim=-1))


def posterior_replay_loss(branch, prediction, target_state, reference_c, replay_mask, *, anchor_weight=1.):
    """Physical-state supervision plus normalized old posterior on clean replay.

    All targets are detached. The anchor preserves the existing control input
    distribution on clean BC; it is not proof of closed-loop non-regression.
    """
    if prediction.ndim != 2 or prediction.shape != target_state.shape or reference_c.shape != prediction.shape:
        raise ValueError('Posterior, target and reference must have identical (B,D) shape')
    if replay_mask.shape != (len(prediction),) or replay_mask.dtype != torch.bool or not replay_mask.any():
        raise ValueError('A boolean mask with clean replay samples is required')
    if not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError('Invalid anchor weight')
    if not torch.isfinite(target_state).all() or not torch.isfinite(reference_c[replay_mask]).all():
        raise ValueError('Real supervision/anchor targets must be finite')
    supervised = (prediction - branch.normalize(target_state.detach())).square().mean()
    anchor = (prediction[replay_mask] - reference_c[replay_mask].detach()).square().mean()
    return supervised + anchor_weight * anchor, {'state_mse': supervised.detach(), 'replay_anchor_mse': anchor.detach()}


def frozen_actor_replay_loss(actor, branch, prediction, reference_c, h, z, replay_mask):
    """Preserve deterministic actions ONLY on clean replay, through a frozen Actor.

    Teacher c and cached h/z are detached. Student c must retain its graph: using
    no_grad around the student Actor would silently disable this objective.
    This is not an off-BC action target and never invokes a Critic or prior.
    """
    if any(p.requires_grad for p in actor.parameters()) or actor.training:
        raise ValueError('Action replay requires an eval-mode frozen Actor')
    if prediction.shape != reference_c.shape or prediction.ndim != 2:
        raise ValueError('Matching posterior/reference matrices required')
    if replay_mask.shape != (len(prediction),) or replay_mask.dtype != torch.bool or not replay_mask.any():
        raise ValueError('Clean replay mask required for action preservation')
    if h.shape[0] != len(prediction) or z.shape[0] != len(prediction):
        raise ValueError('Cached h/z must align with posterior nodes')
    q = replay_mask
    hd, zd = h[q].detach(), z[q].detach()
    with torch.no_grad():
        teacher = actor(hd, zd, control_state=branch.decode(reference_c[q].detach()), deterministic=True).action
    student = actor(hd, zd, control_state=branch.decode(prediction[q]), deterministic=True).action
    return (student - teacher).square().mean()


def train_continuous_posterior_step(world_model, optimizer, observation_state, reference_c, replay_mask,
                                    *, anchor_weight=1., grad_clip=10., frozen_actor=None,
                                    latent_h=None, latent_z=None, action_weight=0.):
    """Only encoder.control_embedding and branch.posterior may enter optimizer."""
    encoder = world_model['encoder']; branch = world_model['rssm'].continuous
    if branch is None or encoder.control_embedding is None:
        raise ValueError('Requires the learned continuous control branch')
    allowed = list(encoder.control_embedding.parameters()) + list(branch.posterior.parameters())
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(allowed) or {id(p) for p in actual} != {id(p) for p in allowed}:
        raise ValueError('Optimizer must contain exactly control embedding and posterior; prior is frozen')
    if not math.isfinite(grad_clip) or grad_clip <= 0 or not torch.isfinite(observation_state).all():
        raise ValueError('Invalid observations or gradient clip')
    if not math.isfinite(action_weight) or action_weight < 0:
        raise ValueError('Invalid action preservation weight')
    if action_weight and (frozen_actor is None or latent_h is None or latent_z is None):
        raise ValueError('Action preservation requires frozen Actor and cached h/z')
    roots = list(world_model.values()) + ([frozen_actor] if action_weight else [])
    params = [(p, p.requires_grad) for m in roots for p in m.parameters()]
    modes = {m: m.training for root in roots for m in root.modules()}
    try:
        for m in roots: m.eval().requires_grad_(False)
        for p in allowed: p.requires_grad_(True)
        prediction = control_posterior(encoder, branch, observation_state.detach())
        loss, metrics = posterior_replay_loss(branch, prediction, observation_state, reference_c, replay_mask,
                                              anchor_weight=anchor_weight)
        if action_weight:
            action_loss = frozen_actor_replay_loss(frozen_actor, branch, prediction, reference_c,
                                                   latent_h, latent_z, replay_mask)
            loss = loss + action_weight * action_loss
            metrics['clean_action_mse'] = action_loss.detach()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(allowed, grad_clip, error_if_nonfinite=True)
        optimizer.step()
        return dict(loss=float(loss.detach()), gradient_norm=float(norm),
                    **{k: float(v) for k, v in metrics.items()})
    finally:
        for p, flag in params: p.requires_grad_(flag)
        for m, flag in modes.items(): m.training = flag
