"""Standard FedAvg aggregation (BASELINE).

Weighted average of client deltas, weights ∝ client sample counts.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from utilities.types import ClientUpdate


def fedavg_weights(updates: Sequence[ClientUpdate]) -> np.ndarray:
    """Sample-count weights w_i = n_i / Σ n_j."""
    counts = np.array([u.num_samples for u in updates], dtype=np.float64)
    return counts / max(counts.sum(), 1.0)


def aggregate_deltas(
    deltas: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[float] | np.ndarray,
) -> dict[str, torch.Tensor]:
    """Weighted mean of client delta dictionaries."""
    if len(deltas) == 0:
        raise ValueError("No deltas to aggregate")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != len(deltas):
        raise ValueError("weights and deltas length mismatch")
    w = w / max(w.sum(), 1e-12)
    keys = deltas[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(deltas[0][key], dtype=torch.float32)
        for delta, wi in zip(deltas, w):
            if key not in delta:
                raise KeyError(f"Delta missing key {key}")
            acc += delta[key].float() * float(wi)
        out[key] = acc
    return out


def apply_delta(global_state: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """new_global = global + delta."""
    return {k: global_state[k].float() + delta[k].float() for k in global_state}
