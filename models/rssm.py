from __future__ import annotations

import torch
import torch.nn as nn


class RSSM(nn.Module):
    """
    Recurrent State-Space Model: sequence model + prior + posterior.

    State = (h_t deterministic, z_t stochastic categorical).
    Phase A skeleton — implement in Phase B.
    """

    def __init__(self, cfg: dict, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        rssm_cfg = cfg.get("rssm", {})
        self.deter_dim = int(rssm_cfg.get("deter_dim", 512))
        self.stoch_dim = int(rssm_cfg.get("stoch_dim", 32))
        self.stoch_classes = int(rssm_cfg.get("stoch_classes", 32))
        self._dummy = nn.Parameter(torch.zeros(1))

    def initial_state(self, batch_size: int, device: torch.device):
        """Return initial (h_0, z_0)."""
        raise NotImplementedError("Implement RSSM.initial_state in Phase B.")

    def observe_step(self, prev_state, action, embed):
        """Posterior step when observation is available."""
        raise NotImplementedError("Implement RSSM.observe_step in Phase B.")

    def imagine_step(self, prev_state, action):
        """Prior step for imagination rollouts (no observation)."""
        raise NotImplementedError("Implement RSSM.imagine_step in Phase B.")
