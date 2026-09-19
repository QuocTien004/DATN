import unittest
from pathlib import Path
import torch

from models.decoder import Decoder
from models.encoder import Encoder
from models.predictors import ContinuePredictor, RewardPredictor
from models.rssm import RSSM
from utils.config import load_config

_ROOT = Path(__file__).resolve().parents[1]


class TestWorldModelArchitecture(unittest.TestCase):
    def test_shapes(self) -> None:
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
        self.assertEqual(e.shape, (B, enc.embed_dim))

        prev = rssm.initial_state(B, device)
        self.assertEqual(prev["h"].shape, (B, rssm.deter_dim))
        self.assertEqual(prev["z"].shape, (B, rssm.stoch_dim, rssm.stoch_classes))

        post_state, post_stats = rssm.observe_step(prev, action, e)
        self.assertEqual(post_state["h"].shape, (B, rssm.deter_dim))
        self.assertEqual(post_state["z"].shape, (B, rssm.stoch_dim, rssm.stoch_classes))
        self.assertEqual(post_stats["prior_logits"].shape, (B, rssm.stoch_dim, rssm.stoch_classes))
        self.assertEqual(post_stats["posterior_logits"].shape, (B, rssm.stoch_dim, rssm.stoch_classes))

        img_state, img_stats = rssm.imagine_step(post_state, action)
        self.assertEqual(img_state["z"].shape, (B, rssm.stoch_dim, rssm.stoch_classes))
        self.assertEqual(img_stats["prior_logits"].shape, (B, rssm.stoch_dim, rssm.stoch_classes))

        recon = dec(post_state["h"], post_state["z"])
        self.assertEqual(recon["image"].shape, (B, C, H, W))
        self.assertEqual(recon["state"].shape, (B, state_dim))

        r = reward_h(post_state["h"], post_state["z"])
        c = cont_h(post_state["h"], post_state["z"])
        self.assertEqual(r.shape, (B, 1))
        self.assertEqual(c.shape, (B, 1))


if __name__ == "__main__":
    unittest.main()
