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
    free_nats = float(train_cfg.get("free_nats", 0.1))
    kl_balance = float(train_cfg.get("kl_balance", 0.8))
    grad_clip = float(train_cfg.get("grad_clip", 1000.0))

    image = batch["image"]
    state = batch["state"]
    action = batch["action"]
    reward = batch["reward"]
    terminal = batch.get("terminated", batch["done"]).float()
    is_first = batch["is_first"].float() if "is_first" in batch else None
    if "next_image" not in batch or "next_state" not in batch:
        raise ValueError("WM training requires replay.sample(..., include_next=True) for transition-aligned targets")

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
        first_t = is_first[:, t] if is_first is not None else None
        a_tm1 = zero_action if t == 0 else action[:, t - 1]
        prev, stats = rssm.observe_step(prev, a_tm1, e[:, t], is_first=first_t)
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

    # Stored reward/done belong to action_t -> obs_(t+1), while reconstruction
    # above belongs to obs_t. Imagination also queries the heads on next latent.
    next_e = enc(_flatten_time(batch["next_image"]), _flatten_time(batch["next_state"]))
    next_post, next_stats = rssm.observe_step(
        {"h": h_flat, "z": z_flat}, _flatten_time(action), next_e
    )
    pred_reward = reward_pred(next_post["h"], next_post["z"]).reshape(B, T)
    target_cont = 1.0 - terminal
    pred_cont_logit = continue_pred(next_post["h"], next_post["z"]).reshape(B, T)

    loss_recon_img = F.mse_loss(recon_image, image)
    loss_recon_state = F.mse_loss(recon_state, state)
    sym_reward = torch.sign(reward) * torch.log1p(torch.abs(reward))
    # Weight negative rewards (crashes) by 5.0x to prevent smoothing away penalties
    rew_weight = torch.where(sym_reward < 0.0, float(train_cfg.get("negative_reward_weight", 5.0)), 1.0)
    loss_reward = ((pred_reward - sym_reward).square() * rew_weight).mean()
    
    # Weight terminal transitions (target_cont=0.0) by 15.0x to counteract 31:1 imbalance
    cont_weight = torch.where(target_cont == 0.0, float(train_cfg.get("terminal_weight", 15.0)), 1.0)
    bce_continue = F.binary_cross_entropy_with_logits(
        pred_cont_logit, target_cont, reduction="none"
    )
    loss_continue = (bce_continue * cont_weight).mean()

    # Train both current and successor priors, including terminal successors.
    all_post = torch.cat([_flatten_time(post_logits), next_stats["posterior_logits"]])
    all_prior = torch.cat([_flatten_time(prior_logits), next_stats["prior_logits"]])
    kl_post_prior_raw = _kl_cat(all_post, all_prior.detach())
    kl_prior_post_raw = _kl_cat(all_post.detach(), all_prior)
    kl_post_prior = torch.clamp(kl_post_prior_raw, min=free_nats).mean()
    kl_prior_post = torch.clamp(kl_prior_post_raw, min=free_nats).mean()
    # kl_balance weights dynamics (prior learning), not the representation term.
    loss_kl = kl_balance * kl_prior_post + (1.0 - kl_balance) * kl_post_prior

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
        error_if_nonfinite=True,
    )
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "loss_recon_img": float(loss_recon_img.detach()),
        "loss_recon_state": float(loss_recon_state.detach()),
        "loss_reward": float(loss_reward.detach()),
        "loss_continue": float(loss_continue.detach()),
        "loss_kl": float(loss_kl.detach()),
        "loss_kl_raw": float(kl_post_prior_raw.mean().detach()),
        "terminal_fraction": float(terminal.mean()),
        "continue_on_terminal": float(torch.sigmoid(pred_cont_logit.detach())[terminal.bool()].mean()) if terminal.any() else 0.0,
    }
