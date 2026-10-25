"""Classification metrics for imbalanced theft detection.

Emphasis metrics: precision, recall, F1, PR-AUC (accuracy alone is misleading
under class imbalance). Everything is computed from real predictions.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        }
    )
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    metrics["n_samples"] = int(y_true.size)
    metrics["positive_rate"] = float(y_true.mean())
    return metrics


def confusion_matrix_rows(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> list[list[int]]:
    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])
    return cm.astype(int).tolist()


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: Iterable[float] | None = None) -> dict[str, Any]:
    """Pick the decision threshold that maximises F1 on validation data."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    candidates = np.asarray(list(grid)) if grid else np.arange(0.05, 0.951, 0.05)
    best = {"threshold": 0.5, "f1": -1.0}
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    best["metrics_at_threshold"] = compute_classification_metrics(y_true, y_prob, best["threshold"])
    return best


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, list[float]]:
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, list[float]]:
    from sklearn.metrics import precision_recall_curve

    if len(np.unique(y_true)) < 2:
        return {"precision": [], "recall": [], "thresholds": []}
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thr.tolist()}
