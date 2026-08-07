from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cat_state_action(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """z: (B, stoch_dim, stoch_classes) → flat; action: (B, A)."""
    return torch.cat([z.reshape(z.shape[0], -1), action], dim=-1)


def sample_categorical(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample one-hot from logits (B, G, C) with straight-through gradient.
    """
    probs = F.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1).reshape(probs.shape[0], probs.shape[1])
    hard = F.one_hot(idx, num_classes=probs.shape[-1]).to(dtype=logits.dtype)
    # Straight-through: forward hard, backward as soft probs
    return hard + probs - probs.detach()


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (DreamerV3-style categorical latent).

    State dict:
      h: (B, deter_dim)
      z: (B, stoch_dim, stoch_classes)   # one-hot / ST sample
    """

    def __init__(self, cfg: dict, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        rssm_cfg = cfg.get("rssm", {})
        self.deter_dim = int(rssm_cfg.get("deter_dim", 512))
        self.stoch_dim = int(rssm_cfg.get("stoch_dim", 32))
        self.stoch_classes = int(rssm_cfg.get("stoch_classes", 32))
        hidden = int(rssm_cfg.get("hidden_dim", 512))
        self.stoch_size = self.stoch_dim * self.stoch_classes

        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_size + self.action_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.cell = nn.GRUCell(hidden, self.deter_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.stoch_size),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.stoch_size),
        )

    def initial_state(self, batch_size: int, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        device = torch.device(device) if device is not None else next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        # Uniform one-hot start (first class)
        z = torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        z[..., 0] = 1.0
        return {"h": h, "z": z}

    def _logits_to_grouped(self, logits_flat: torch.Tensor) -> torch.Tensor:
        return logits_flat.reshape(-1, self.stoch_dim, self.stoch_classes)

    def _deter_step(self, prev_state: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        x = self.img_in(_cat_state_action(prev_state["z"], action))
        return self.cell(x, prev_state["h"])

    def observe_step(
        self,
        prev_state: dict[str, torch.Tensor],
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Posterior step (has observation embedding).

        Returns
        -------
        state : {h, z}
        stats : {prior_logits, posterior_logits}  each (B, G, C)
        """
        h = self._deter_step(prev_state, action)
        prior_logits = self._logits_to_grouped(self.prior_net(h))
        post_logits = self._logits_to_grouped(
            self.posterior_net(torch.cat([h, embed], dim=-1))
        )
        z = sample_categorical(post_logits)
        state = {"h": h, "z": z}
        stats = {"prior_logits": prior_logits, "posterior_logits": post_logits}
        return state, stats

    def imagine_step(
        self,
        prev_state: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Prior step (no observation) — used for imagination / ẑ_t.
        """
        h = self._deter_step(prev_state, action)
        prior_logits = self._logits_to_grouped(self.prior_net(h))
        z_hat = sample_categorical(prior_logits)
        state = {"h": h, "z": z_hat}
        stats = {"prior_logits": prior_logits}
        return state, stats

    def get_feature(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concat (h, z_flat) for heads: (B, deter_dim + stoch_size)."""
        return torch.cat([state["h"], state["z"].reshape(state["z"].shape[0], -1)], dim=-1)
