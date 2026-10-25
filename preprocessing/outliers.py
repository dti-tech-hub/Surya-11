"""Outlier handling with train-fitted bounds (IQR / z-score), action = clip.

Electricity-theft anomalies are *legitimate signal*, so records are never
deleted: out-of-range readings are clipped into the train-fitted bounds.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from utilities.config import get


class OutlierClipper:
    """Fit bounds on training consumption; clip any split into those bounds."""

    def __init__(self, cfg: dict):
        self.enabled: bool = bool(get(cfg, "preprocessing.outliers.enabled", True))
        self.method: str = str(get(cfg, "preprocessing.outliers.method", "iqr"))
        self.iqr_multiplier: float = float(get(cfg, "preprocessing.outliers.iqr_multiplier", 3.0))
        self.zscore_threshold: float = float(get(cfg, "preprocessing.outliers.zscore_threshold", 5.0))
        self.action: str = str(get(cfg, "preprocessing.outliers.action", "clip"))
        self.lower: float | None = None
        self.upper: float | None = None
        self.stats: dict[str, Any] = {}

    def fit(self, train_values: np.ndarray) -> "OutlierClipper":
        values = np.asarray(train_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("OutlierClipper.fit received no finite values")
        if self.method == "iqr":
            q1, q3 = np.percentile(values, [25.0, 75.0])
            iqr = q3 - q1
            self.lower = float(q1 - self.iqr_multiplier * iqr)
            self.upper = float(q3 + self.iqr_multiplier * iqr)
        elif self.method == "zscore":
            mean, std = float(values.mean()), float(values.std())
            std = std if std > 0 else 1.0
            self.lower = float(mean - self.zscore_threshold * std)
            self.upper = float(mean + self.zscore_threshold * std)
        elif self.method == "none":
            self.lower, self.upper = float(values.min()), float(values.max())
        else:
            raise ValueError(f"Unknown outlier method '{self.method}' (expected iqr|zscore|none)")
        # consumption cannot be negative
        self.lower = max(self.lower, 0.0)
        self.stats = {
            "method": self.method,
            "lower": self.lower,
            "upper": self.upper,
            "train_mean": float(values.mean()),
            "train_std": float(values.std()),
        }
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.enabled or self.lower is None or self.upper is None:
            return np.asarray(values, dtype=np.float32)
        if self.action == "clip":
            out = np.clip(np.asarray(values, dtype=np.float64), self.lower, self.upper)
            return out.astype(np.float32)
        if self.action == "none":
            return np.asarray(values, dtype=np.float32)
        raise ValueError(f"Unknown outlier action '{self.action}' (expected clip|none)")

    def clip_fraction(self, values: np.ndarray) -> float:
        """Fraction of finite values outside the fitted bounds (before clipping)."""
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any() or self.lower is None or self.upper is None:
            return 0.0
        outside = (arr < self.lower) | (arr > self.upper)
        return float(outside[finite].mean())
