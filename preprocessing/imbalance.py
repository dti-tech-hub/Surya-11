"""Class-imbalance analysis and weighting utilities.

Electricity-theft datasets are typically heavily imbalanced (few theft
consumers). This module quantifies the imbalance and computes configurable
weights; the actual weighted loss is applied in :mod:`models`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from utilities.config import get


def class_distribution(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    total = int(y.size)
    return {
        "normal_samples": n_neg,
        "theft_samples": n_pos,
        "total": total,
        "theft_ratio": float(n_pos / total) if total else 0.0,
        "imbalance_ratio": float(n_neg / n_pos) if n_pos else float("inf"),
    }


def compute_class_weights(y: np.ndarray, cfg: dict) -> dict[str, Any]:
    """Return configured class weights (``balanced`` or none) + report.

    Strategy options (``preprocessing.class_imbalance.strategy``):

    - ``class_weight`` — inverse-frequency weights (sklearn 'balanced' rule)
    - ``focal``        — weights plus focal-loss gamma for the focal loss path
    - ``none``         — unweighted
    """
    strategy = str(get(cfg, "preprocessing.class_imbalance.strategy", "class_weight"))
    gamma = float(get(cfg, "preprocessing.class_imbalance.focal_gamma", 2.0))
    dist = class_distribution(y)
    report: dict[str, Any] = {"strategy": strategy, "distribution": dist, "focal_gamma": gamma}

    if strategy == "none":
        report["weights"] = {0: 1.0, 1: 1.0}
        return report

    n = dist["total"]
    n_neg, n_pos = dist["normal_samples"], dist["theft_samples"]
    if n_pos == 0 or n_neg == 0:
        # degenerate split: fall back to uniform weights
        report["weights"] = {0: 1.0, 1: 1.0}
        report["note"] = "single-class training subset; weights set to 1.0"
        return report
    w_neg = n / (2.0 * n_neg)
    w_pos = n / (2.0 * n_pos)
    report["weights"] = {0: float(w_neg), 1: float(w_pos)}
    if strategy == "focal":
        report["applies_focal_loss"] = True
    elif strategy != "class_weight":
        raise ValueError(f"Unknown imbalance strategy '{strategy}' (class_weight|focal|none)")
    return report


def imbalance_report(splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-split class distribution summary."""
    return {name: class_distribution(y) for name, y in splits.items()}
