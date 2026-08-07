#!/usr/bin/env python
"""
Smoke-test A2–A4 World Model architecture (shapes only, no training).

  python scripts/test_world_model_arch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "metadrive")):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "metadrive"))

import torch

from models.decoder import Decoder
from models.encoder import Encoder
from models.predictors import ContinuePredictor, RewardPredictor
from models.rssm import RSSM
from utils.config import load_config


def main() -> None:
    cfg = load_config(_ROOT / "configs" / "world_model.yaml")
    device = torch.device("cpu")
    B = 4
    H = W = 256
    C = 3
    state_dim = 19
    action_dim = 2

    enc = Encoder(cfg, state_dim=state_dim, image_shape=(H, W, C)).to(device)
    rssm = RSSM(cfg, embed_dim=enc.embed_dim, action_dim=action_dim).to(device)
    dec = Decoder(
        cfg,
        deter_dim=rssm.deter_dim,
        stoch_size=rssm.stoch_size,
        image_shape=(H, W, C),
        state_dim=state_dim,
    ).to(device)
    reward_h = RewardPredictor(cfg, rssm.deter_dim, rssm.stoch_size).to(device)
    cont_h = ContinuePredictor(cfg, rssm.deter_dim, rssm.stoch_size).to(device)

    image = torch.rand(B, C, H, W, device=device)
    state = torch.rand(B, state_dim, device=device)
    action = torch.rand(B, action_dim, device=device) * 2 - 1

    e = enc(image, state)
    assert e.shape == (B, enc.embed_dim), e.shape

    prev = rssm.initial_state(B, device)
    assert prev["h"].shape == (B, rssm.deter_dim)
    assert prev["z"].shape == (B, rssm.stoch_dim, rssm.stoch_classes)

    post_state, post_stats = rssm.observe_step(prev, action, e)
    assert post_state["h"].shape == (B, rssm.deter_dim)
    assert post_state["z"].shape == (B, rssm.stoch_dim, rssm.stoch_classes)
    assert post_stats["prior_logits"].shape == (B, rssm.stoch_dim, rssm.stoch_classes)
    assert post_stats["posterior_logits"].shape == (B, rssm.stoch_dim, rssm.stoch_classes)

    img_state, img_stats = rssm.imagine_step(post_state, action)
    assert img_state["z"].shape == (B, rssm.stoch_dim, rssm.stoch_classes)
    assert img_stats["prior_logits"].shape == (B, rssm.stoch_dim, rssm.stoch_classes)

    recon = dec(post_state["h"], post_state["z"])
    assert recon["image"].shape == (B, C, H, W), recon["image"].shape
    assert recon["state"].shape == (B, state_dim), recon["state"].shape

    r = reward_h(post_state["h"], post_state["z"])
    c = cont_h(post_state["h"], post_state["z"])
    assert r.shape == (B, 1), r.shape
    assert c.shape == (B, 1), c.shape

    print("OK — A2–A4 architecture shapes")
    print(f"  e_t:              {tuple(e.shape)}")
    print(f"  h_t:              {tuple(post_state['h'].shape)}")
    print(f"  z_t / z_hat:      {tuple(post_state['z'].shape)}  (groups x classes)")
    print(f"  prior_logits:     {tuple(post_stats['prior_logits'].shape)}")
    print(f"  recon image:      {tuple(recon['image'].shape)}")
    print(f"  recon state:      {tuple(recon['state'].shape)}")
    print(f"  reward / continue:{tuple(r.shape)} / {tuple(c.shape)}")


if __name__ == "__main__":
    main()
