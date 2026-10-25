"""Central random-seed management for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a required dependency
        pass


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) % (2**32 - 1))
