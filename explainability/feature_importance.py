"""Model-agnostic permutation importance (SHAP-free explainability path).

Implemented directly rather than through ``sklearn.inspection`` because that
helper validates estimator protocol details that our lightweight inference
adapter intentionally does not implement. The permutation procedure is the
standard one: shuffle one feature column at a time and measure the drop in
hold-out ROC-AUC. Everything is computed from the fitted model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from utilities.config import get


def permutation_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cfg: dict,
    seed: int = 42,
    max_rows: int = 2000,
) -> dict[str, Any]:
    """Permutation importance on a validation subset (real model, real data)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    if len(y) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    if len(np.unique(y)) < 2:
        raise ValueError("permutation importance needs both classes in the validation subset")

    n_repeats = int(get(cfg, "explainability.permutation_repeats", 5))
    rng = np.random.default_rng(seed)
    baseline_probs = np.asarray(model.predict_proba(X), dtype=np.float64).reshape(-1)
    baseline = float(roc_auc_score(y, baseline_probs))

    importances = np.zeros(X.shape[1], dtype=np.float64)
    stds = np.zeros(X.shape[1], dtype=np.float64)
    for feature_idx in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, feature_idx])
            probs = np.asarray(model.predict_proba(X_perm), dtype=np.float64).reshape(-1)
            scores.append(float(roc_auc_score(y, probs)))
        drops = baseline - np.asarray(scores, dtype=np.float64)
        importances[feature_idx] = float(np.nanmean(drops))
        stds[feature_idx] = float(np.nanstd(drops, ddof=1)) if len(drops) > 1 else 0.0

    order = np.argsort(-importances)
    top = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "importance_mean": float(importances[i]),
            "importance_std": float(stds[i]),
        }
        for i in order[:20]
    ]
    return {
        "method": "permutation_importance",
        "implementation": "manual feature-shuffle, ROC-AUC drop",
        "n_repeats": n_repeats,
        "n_rows": int(len(y)),
        "baseline_roc_auc": baseline,
        "top_features": top,
    }
