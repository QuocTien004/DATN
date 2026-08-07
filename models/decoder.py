from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Reconstruct image (+ optional state) from (h_t, z_t).

    Returns dict:
      image: (B, C, H, W) in roughly [0, 1] via sigmoid
      state: (B, state_dim) if state_dim > 0
    """

    def __init__(
        self,
        cfg: dict,
        deter_dim: int,
        stoch_size: int,
        image_shape: tuple,
        state_dim: int = 19,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_shape = tuple(image_shape)  # (H, W, C)
        self.state_dim = int(state_dim)
        h, w, c = self.image_shape
        self.out_h = h
        self.out_w = w
        self.out_c = c

        dec_cfg = cfg.get("decoder", {})
        channels = list(dec_cfg.get("cnn_channels", [256, 128, 64, 32]))
        in_dim = int(deter_dim) + int(stoch_size)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, channels[0] * 4 * 4),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers.extend(
                [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        # Continue upsampling until near target size (4 -> 8 -> 16 -> 32 ... with remaining)
        # After len(channels)-1 upsamples starting 4x4: spatial = 4 * 2^(n-1)
        # For 4 channels: 4->8->16->32, then interpolate to HxW and 1x1 conv to C
        self.deconv = nn.Sequential(*layers)
        self.image_head = nn.Conv2d(in_ch, c, kernel_size=1)

        self.state_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.state_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.dim() == 3:
            z_flat = z.reshape(z.shape[0], -1)
        else:
            z_flat = z
        feat = torch.cat([h, z_flat], dim=-1)

        x = self.fc(feat)
        x = x.reshape(h.shape[0], -1, 4, 4)
        x = self.deconv(x)
        # Project to RGB *before* upsampling to save a lot of RAM (avoid Cx256x256 with C=32)
        x = self.image_head(x)
        x = nn.functional.interpolate(
            x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False
        )
        image = torch.sigmoid(x)
        state = self.state_head(feat)
        return {"image": image, "state": state}
