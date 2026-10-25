"""Detection evaluation helpers shared by experiment drivers.

Wraps a fitted model, tunes the decision threshold on validation data only,
evaluates the untouched test split, and returns per-window predictions plus
per-consumer verdicts. Raw consumption sequences are never written to disk.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from detection.postprocessing import aggregate_consumer_verdicts, consumer_metrics
from detection.threshold import select_threshold
from evaluation.metrics import (
    compute_classification_metrics,
    confusion_matrix_rows,
    pr_curve_points,
    roc_curve_points,
)


def evaluate_detection(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cfg: dict,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    base_threshold: float = 0.5,
    meta: pd.DataFrame | None = None,
    with_curves: bool = True,
) -> dict[str, Any]:
    """Evaluate one fitted model on hold-out test windows.

    Threshold selection uses ``X_val``/``y_val`` when provided; otherwise the
    supplied ``base_threshold`` is used. Test labels are never used for tuning.
    """
    threshold = float(base_threshold)
    threshold_info: dict[str, Any] = {"threshold": threshold, "tuned": False}
    if X_val is not None and y_val is not None and len(y_val):
        prob_val = np.asarray(model.predict_proba(X_val), dtype=np.float64)
        threshold_info = select_threshold(
            np.asarray(y_val), prob_val, enabled=bool(cfg.get("evaluation", {}).get("tune_threshold", True))
        )
        threshold = float(threshold_info["threshold"])

    probs = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    metrics = compute_classification_metrics(y_test, probs, threshold=threshold)
    result: dict[str, Any] = {
        "metrics": metrics,
        "threshold_info": threshold_info,
        "confusion_matrix": confusion_matrix_rows(y_test, probs, threshold),
        "probabilities": probs,
        "targets": np.asarray(y_test).astype(int),
    }
    if with_curves:
        result["roc"] = roc_curve_points(y_test, probs)
        result["pr"] = pr_curve_points(y_test, probs)
    if meta is not None and len(meta) == len(probs):
        result["consumer"] = aggregate_consumer_verdicts(meta, probs, threshold=threshold)
        result["consumer_metrics"] = consumer_metrics(result["consumer"])
    return result


def predictions_frame(meta: pd.DataFrame | None, probs: np.ndarray, y_true: np.ndarray, threshold: float) -> pd.DataFrame:
    """Prediction table without raw consumption values."""
    n = len(probs)
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "y_true": np.asarray(y_true).astype(int),
            "probability_theft": np.asarray(probs, dtype=float),
            "y_pred": (np.asarray(probs) >= threshold).astype(int),
        }
    )
    if meta is not None and len(meta) == n:
        for col in ("consumer_id", "start_date", "end_date", "label"):
            if col in meta.columns:
                frame[col] = meta[col].reset_index(drop=True)
    return frame
