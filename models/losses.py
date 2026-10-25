"""Loss functions with class-imbalance handling (weighted BCE / focal loss)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FocalBCELoss(nn.Module):
    """Focal loss on logits: FL(p_t) = (1 - p_t)^gamma · BCE(p_t).

    Down-weights easy negatives so the rare theft class dominates the gradient.
    Class weights are applied multiplicatively.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        modulating = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        loss = modulating * bce
        if self.weight is not None:
            w = self.weight[1] * targets + self.weight[0] * (1.0 - targets)
            loss = loss * w
        return loss.mean()


def build_loss(
    class_weights: dict[str, Any] | None = None,
    focal: bool = False,
    gamma: float = 2.0,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build the configured criterion.

    ``class_weights`` is the report dict from
    :func:`preprocessing.imbalance.compute_class_weights` (expects key
    ``weights`` mapping {0: w0, 1: w1}) or a plain {0: w0, 1: w1} mapping.
    """
    weights: dict | None = None
    if class_weights:
        weights = class_weights.get("weights", class_weights) if isinstance(class_weights, dict) else None
    if weights is None:
        return nn.BCEWithLogitsLoss()
    w0, w1 = float(weights.get(0, weights.get("0", 1.0))), float(weights.get(1, weights.get("1", 1.0)))
    if focal:
        return FocalBCELoss(gamma=gamma, weight=torch.tensor([w0, w1], dtype=torch.float32, device=device))
    # weighted BCE: pos_weight is the relative weight of the positive class
    pos_weight = torch.tensor([w1 / max(w0, 1e-12)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
