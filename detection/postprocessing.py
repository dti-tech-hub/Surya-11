"""Post-processing: aggregate window-level verdicts into consumer verdicts.

A consumer is flagged only when multiple of its windows are suspicious — a
single UNCERTAIN window should not create an accusation. Aggregation is a
deterministic rule (fraction of positive windows above a configurable ratio),
applied only for reporting; window-level metrics remain the primary
evaluation unit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def aggregate_consumer_verdicts(
    meta: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    positive_window_ratio: float = 0.4,
) -> pd.DataFrame:
    """Per-consumer fraud score = mean P(theft) across its windows."""
    df = meta.copy()
    df["probability"] = np.asarray(probabilities, dtype=np.float64)
    agg = df.groupby("consumer_id", sort=False).agg(
        mean_probability=("probability", "mean"),
        max_probability=("probability", "max"),
        n_windows=("probability", "size"),
        positive_windows=("probability", lambda s: float((s >= threshold).mean())),
        true_label=("label", "max"),
    ).reset_index()
    agg["flagged"] = (agg["positive_windows"] >= positive_window_ratio).astype(int)
    return agg


def consumer_metrics(agg: pd.DataFrame) -> dict[str, Any]:
    from evaluation.metrics import compute_classification_metrics

    y_true = agg["true_label"].to_numpy(dtype=int)
    y_prob = agg["mean_probability"].to_numpy(dtype=float)
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    metrics["rule"] = "consumer flagged if >= ratio of its windows exceed the window threshold"
    metrics["n_consumers"] = int(len(agg))
    return metrics
