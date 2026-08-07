from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _kl_cat(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """KL(post || prior) for categorical groups. logits: (B, G, C) -> (B,)."""
    post = F.softmax(post_logits, dim=-1)
    prior = F.softmax(prior_logits, dim=-1)
    log_post = F.log_softmax(post_logits, dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    kl = (post * (log_post - log_prior)).sum(dim=-1)  # (B, G)
    return kl.sum(dim=-1)


def _flatten_time(x: torch.Tensor) -> torch.Tensor:
    """(B, T, ...) -> (B*T, ...)."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def train_world_model_step(
    batch: dict[str, torch.Tensor],
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """
    One World Model update on a sequence batch.

    Expected batch tensors (already on device):
      image: (B, T, C, H, W) float in [0, 1]
      state: (B, T, state_dim)
      action: (B, T, action_dim)
      reward: (B, T)
      done: (B, T) bool/float
    """
    enc = models["encoder"]
    rssm = models["rssm"]
    dec = models["decoder"]
    reward_pred = models["reward"]
    continue_pred = models["continue"]

    train_cfg = cfg.get("training", {})
    free_nats = float(train_cfg.get("free_nats", 1.0))
    kl_balance = float(train_cfg.get("kl_balance", 0.8))
    grad_clip = float(train_cfg.get("grad_clip", 1000.0))

    image = batch["image"]
    state = batch["state"]
    action = batch["action"]
    reward = batch["reward"]
    done = batch["done"].float()

    B, T = image.shape[:2]
    device = image.device

    # One shot encode (B*T); with batch=8,seq=32 this is 256 frames — fine on 4GB after decoder fix
    e = enc(_flatten_time(image), _flatten_time(state)).reshape(B, T, -1)

    prev = rssm.initial_state(B, device)
    posts_h = []
    posts_z = []
    prior_logits_seq = []
    post_logits_seq = []

    zero_action = torch.zeros(B, action.shape[-1], device=device)
    for t in range(T):
        a_tm1 = zero_action if t == 0 else action[:, t - 1]
        prev, stats = rssm.observe_step(prev, a_tm1, e[:, t])
        posts_h.append(prev["h"])
        posts_z.append(prev["z"])
        prior_logits_seq.append(stats["prior_logits"])
        post_logits_seq.append(stats["posterior_logits"])

    h = torch.stack(posts_h, dim=1)  # (B, T, deter)
    z = torch.stack(posts_z, dim=1)  # (B, T, G, C)
    prior_logits = torch.stack(prior_logits_seq, dim=1)
    post_logits = torch.stack(post_logits_seq, dim=1)

    h_flat = _flatten_time(h)
    z_flat = _flatten_time(z)

    recon = dec(h_flat, z_flat)
    recon_image = recon["image"].reshape(B, T, *image.shape[2:])
    recon_state = recon["state"].reshape(B, T, state.shape[-1])

    pred_reward = reward_pred(h_flat, z_flat).reshape(B, T)
    target_cont = 1.0 - done
    pred_cont_logit = continue_pred(h_flat, z_flat).reshape(B, T)

    loss_recon_img = F.mse_loss(recon_image, image)
    loss_recon_state = F.mse_loss(recon_state, state)
    loss_reward = F.mse_loss(pred_reward, reward)
    loss_continue = F.binary_cross_entropy_with_logits(pred_cont_logit, target_cont)

    kl_post_prior = _kl_cat(
        _flatten_time(post_logits), _flatten_time(prior_logits).detach()
    )
    kl_prior_post = _kl_cat(
        _flatten_time(post_logits).detach(), _flatten_time(prior_logits)
    )
    kl_post_prior = torch.clamp(kl_post_prior, min=free_nats).mean()
    kl_prior_post = torch.clamp(kl_prior_post, min=free_nats).mean()
    loss_kl = kl_balance * kl_post_prior + (1.0 - kl_balance) * kl_prior_post

    loss = (
        loss_recon_img
        + loss_recon_state
        + loss_reward
        + loss_continue
        + loss_kl
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for m in models.values() for p in m.parameters()],
        grad_clip,
    )
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "loss_recon_img": float(loss_recon_img.detach()),
        "loss_recon_state": float(loss_recon_state.detach()),
        "loss_reward": float(loss_reward.detach()),
        "loss_continue": float(loss_continue.detach()),
        "loss_kl": float(loss_kl.detach()),
    }
