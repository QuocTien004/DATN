from __future__ import annotations

import numpy as np
import torch


def replay_batch_to_torch(
    batch: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert replay HWC uint8 sequences to batch-major Torch tensors."""
    image_np = batch["image"]
    if image_np.ndim != 5:
        raise ValueError(f"Expected image (B,T,H,W,C), got {image_np.shape}")
    image = torch.from_numpy(image_np).to(device=device, dtype=torch.float32) / 255.0
    image = image.permute(0, 1, 4, 2, 3).contiguous()
    return {
        "image": image,
        "state": torch.from_numpy(batch["state"]).to(device=device, dtype=torch.float32),
        "action": torch.from_numpy(batch["action"]).to(device=device, dtype=torch.float32),
        "reward": torch.from_numpy(batch["reward"]).to(device=device, dtype=torch.float32),
        "done": torch.from_numpy(batch["done"].astype(np.bool_)).to(device=device),
    }
