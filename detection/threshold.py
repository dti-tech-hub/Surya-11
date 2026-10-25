"""Decision-threshold selection for theft detection.

The default 0.5 threshold is often suboptimal under class imbalance; we tune
the threshold on the *validation* split (maximise F1) and freeze it for test
evaluation — no test-set peeking.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.metrics import tune_threshold


def select_threshold(y_val: np.ndarray, prob_val: np.ndarray, enabled: bool = True) -> dict[str, Any]:
    if not enabled or len(y_val) == 0 or len(np.unique(y_val)) < 2:
        return {
            "threshold": 0.5,
            "tuned": False,
            "note": "threshold tuning skipped (disabled or single-class validation)",
        }
    best = tune_threshold(y_val, prob_val)
    return {"threshold": best["threshold"], "tuned": True, "val_f1": best["f1"]}


def uncertain_band(threshold: float, margin: float = 0.15) -> tuple[float, float]:
    """Probability band around the decision threshold for UNCERTAIN verdicts."""
    low = max(0.0, threshold - margin)
    high = min(1.0, threshold + margin)
    return low, high
