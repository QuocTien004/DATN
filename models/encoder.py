from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Dual-branch encoder: CNN(image) + MLP(state) → embedding e_t.

    image: (B, C, H, W) float in [0, 1] (or roughly normalized)
    state: (B, state_dim)
    returns e_t: (B, embed_dim)
    """

    def __init__(self, cfg: dict, state_dim: int, image_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.image_shape = tuple(image_shape)  # (H, W, C)
        enc_cfg = cfg.get("encoder", {})
        self.embed_dim = int(enc_cfg.get("embed_dim", 512))
        channels = list(enc_cfg.get("cnn_channels", [32, 64, 128, 256]))
        mlp_hidden = list(enc_cfg.get("mlp_hidden", [256, 256]))

        h, w, c = self.image_shape
        layers: list[nn.Module] = []
        in_ch = c
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.cnn = nn.Sequential(*layers)
        cnn_flat = in_ch * 4 * 4

        half = self.embed_dim // 2
        self.image_proj = nn.Sequential(
            nn.Linear(cnn_flat, half),
            nn.ReLU(inplace=True),
            nn.Linear(half, half),
        )

        mlp: list[nn.Module] = []
        in_dim = self.state_dim
        for hidden in mlp_hidden:
            mlp.extend([nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)])
            in_dim = hidden
        mlp.append(nn.Linear(in_dim, self.embed_dim - half))
        self.state_mlp = nn.Sequential(*mlp)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"image must be (B,C,H,W), got {tuple(image.shape)}")
        if state.dim() != 2:
            raise ValueError(f"state must be (B,state_dim), got {tuple(state.shape)}")

        feat = self.cnn(image)
        feat = feat.reshape(feat.shape[0], -1)
        img_e = self.image_proj(feat)
        state_e = self.state_mlp(state)
        return torch.cat([img_e, state_e], dim=-1)
