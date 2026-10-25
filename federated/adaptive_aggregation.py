"""PROPOSED adaptive aggregation for PowerShield-FL.

FedAvg weights updates only by client sample count, which lets large but noisy
or degenerate clients dominate. The adaptive scheme multiplies three
client-specific factors (all coefficients configurable in
``config/aggregation.yaml``):

```
AdaptiveWeight_i ∝
      n_i^alpha                      # sample contribution
    × quality_i^beta                 # local validation quality (F1-based)
    × reliability_i^reliability_power# update-norm reliability
    × balance_i                      # class-balance factor (λ strength)

quality_i      = max(0.5*val_F1_i + 0.5*val_AUC_i, quality_floor)  (if both val classes exist)
                 max(0.5, quality_floor)                           (single-class val client)
reliability_i  = exp(-| ‖Δ_i‖ / median‖Δ‖ − 1 |)   # penalises outlier updates
balance_i      = (1 − λ) + λ · 2·√(p_i·(1 − p_i))  # p_i = local theft ratio
```

A client with no theft windows cannot have a meaningful local F1 (the
zero_division convention would report 0). Treating that as zero quality would
discard a legitimate all-normal client, so such clients receive neutral quality
(0.5). The quality floor prevents one noisy client from collapsing to the
minimum weight.

Weights are normalised to sum to 1, floored at ``min_weight`` and (optionally)
robust-clipped at 3× the mean weight.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from utilities.config import get
from utilities.types import ClientUpdate


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _quality_from_update(update: ClientUpdate, floor: float = 0.1) -> float:
    """Client quality from local validation only (never from test data).

    F1 alone is unstable for a client whose local validation split has no
    theft windows (F1 is then 0 by convention). AUC is only present when both
    validation classes exist, so its presence is used as the signal that F1 is
    meaningful. Otherwise we fall back to the client's validation loss.
    """
    metrics = update.val_metrics or {}
    f1 = _finite(metrics.get("f1"))
    auc = _finite(metrics.get("roc_auc"))
    if f1 is not None and auc is not None:
        quality = 0.5 * f1 + 0.5 * auc
    else:
        # A single-class local validation split cannot rank theft detection
        # quality. Use a neutral value rather than rewarding a model that
        # predicts the only class present.
        quality = 0.5
    return float(np.clip(quality, floor, 1.0))


def adaptive_weights(
    updates: Sequence[ClientUpdate],
    cfg: dict,
    norms: Sequence[float] | None = None,
) -> np.ndarray:
    """Compute normalised adaptive aggregation weights."""
    alpha = float(get(cfg, "aggregation.alpha", 1.0))
    beta = float(get(cfg, "aggregation.beta", 0.5))
    rel_power = float(get(cfg, "aggregation.reliability_power", 0.5))
    balance_lambda = float(get(cfg, "aggregation.balance_lambda", 0.2))
    min_weight = float(get(cfg, "aggregation.min_weight", 0.01))
    quality_floor = float(get(cfg, "aggregation.quality_floor", 0.1))
    normalize = bool(get(cfg, "aggregation.normalize", True))
    robust_clip = bool(get(cfg, "aggregation.robust_clip", True))

    n = np.array([u.num_samples for u in updates], dtype=np.float64)
    quality = np.array([_quality_from_update(u, quality_floor) for u in updates], dtype=np.float64)
    if norms is None:
        norms = [u.update_norm for u in updates]
    norm_arr = np.array([float(x) for x in norms], dtype=np.float64)
    median_norm = float(np.median(norm_arr)) if norm_arr.size else 0.0
    if median_norm > 0:
        reliability = np.exp(-np.abs(norm_arr / median_norm - 1.0))
    else:
        reliability = np.ones_like(norm_arr)

    total = float(n.sum()) or 1.0
    p = np.array([u.num_positive for u in updates], dtype=np.float64) / np.maximum(n, 1.0)
    balance = (1.0 - balance_lambda) + balance_lambda * 2.0 * np.sqrt(np.clip(p * (1.0 - p), 0.0, 0.25))

    weights = (n / total) ** alpha * (quality**beta) * (reliability**rel_power) * balance
    weights = np.clip(weights, 0.0, None)

    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    if robust_clip:
        ceiling = 3.0 * max(weights.mean(), 1e-12)
        weights = np.minimum(weights, ceiling)
        if normalize and weights.sum() > 0:
            weights = weights / weights.sum()
    weights = np.maximum(weights, min_weight if normalize else 0.0)
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()
    return weights


def weight_diagnostics(
    updates: Sequence[ClientUpdate], weights: np.ndarray, cfg: dict
) -> dict[str, float | list]:
    """Per-client factor breakdown for logging and the dashboard."""
    n = np.array([u.num_samples for u in updates], dtype=np.float64)
    quality = np.array(
        [_quality_from_update(u, float(get(cfg, "aggregation.quality_floor", 0.1))) for u in updates]
    )
    norm_arr = np.array([u.update_norm for u in updates], dtype=np.float64)
    median_norm = float(np.median(norm_arr)) if norm_arr.size else 0.0
    reliability = np.exp(-np.abs(norm_arr / max(median_norm, 1e-12) - 1.0))
    p = np.array([u.num_positive for u in updates]) / np.maximum(n, 1.0)
    lam = float(get(cfg, "aggregation.balance_lambda", 0.2))
    balance = (1.0 - lam) + lam * 2.0 * np.sqrt(np.clip(p * (1.0 - p), 0.0, 0.25))
    fedavg_w = n / max(n.sum(), 1.0)
    return {
        "client_ids": [u.client_id for u in updates],
        "adaptive": weights.tolist(),
        "fedavg": fedavg_w.tolist(),
        "quality": quality.tolist(),
        "reliability": reliability.tolist(),
        "balance": balance.tolist(),
        "update_norms": norm_arr.tolist(),
    }
