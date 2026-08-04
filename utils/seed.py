from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fix Python / NumPy / (optional) Torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic kernels can be slower; enable when debugging.
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
