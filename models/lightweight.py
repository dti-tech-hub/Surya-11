"""PROPOSED lightweight model: shallow MLP over block means + engineered features.

Architecture (default hidden sizes [64, 32]):

```
Input (n_blocks × window_days + n_engineered)
  → Dense(64) → ReLU → Dropout
  → Dense(32) → ReLU → Dropout
  → Dense(1)  → logit
```

Design intent: far fewer parameters and FLOPs than the CNN-LSTM baseline while
retaining the engineered theft signatures (daily drops, peak/off-peak shift,
low-day fraction). Actual parameter counts and timings are *measured* by the
evaluation module — no claims are hard-coded.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.base import TorchBinaryClassifier
from utilities.config import get


class LightweightMLP(nn.Module):
    def __init__(self, n_inputs: int, hidden: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        hidden = list(hidden or [64, 32])
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.n_inputs = n_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_lightweight(cfg: dict, n_features: int, seed: int = 42) -> TorchBinaryClassifier:
    torch.manual_seed(seed)
    hidden = list(get(cfg, "model.hidden", [64, 32]))
    dropout = float(get(cfg, "model.dropout", 0.2))
    model = LightweightMLP(n_features, hidden=hidden, dropout=dropout)
    return TorchBinaryClassifier(
        model,
        name="powershield_lightweight_mlp",
        learning_rate=float(get(cfg, "federated.learning_rate", 1e-3)),
        weight_decay=float(get(cfg, "federated.weight_decay", 1e-4)),
    )
