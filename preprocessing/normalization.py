"""Scaling fitted on training data only (leakage prevention)."""
from __future__ import annotations

from typing import Any

import numpy as np

from utilities.config import get


class ValueScaler:
    """Scalar mean/std (or min/max) scaler over raw consumption readings."""

    def __init__(self, method: str = "standard"):
        self.method = method
        self.center: float = 0.0
        self.scale: float = 1.0
        self.minimum: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "ValueScaler":
        return cls(method=str(get(cfg, "preprocessing.scaling.method", "standard")))

    def fit(self, values: np.ndarray) -> "ValueScaler":
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("ValueScaler.fit received no finite values")
        if self.method == "minmax":
            self.minimum = float(arr.min())
            rng = float(arr.max() - arr.min())
            self.scale = rng if rng > 0 else 1.0
        else:  # standard
            self.center = float(arr.mean())
            std = float(arr.std())
            self.scale = std if std > 0 else 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self.method == "minmax":
            return ((arr - self.minimum) / self.scale).astype(np.float32)
        return ((arr - self.center) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self.method == "minmax":
            return (arr * self.scale + self.minimum).astype(np.float32)
        return (arr * self.scale + self.center).astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {"method": self.method, "center": self.center, "scale": self.scale, "minimum": self.minimum}


class FeatureScaler:
    """Per-column standardization for engineered feature matrices."""

    def __init__(self, method: str = "standard"):
        self.method = method
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.rng: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureScaler":
        return cls(method=str(get(cfg, "preprocessing.scaling.method", "standard")))

    def fit(self, matrix: np.ndarray) -> "FeatureScaler":
        matrix = np.asarray(matrix, dtype=np.float64)
        if self.method == "minmax":
            self.min = np.nanmin(matrix, axis=0)
            self.rng = np.nanmax(matrix, axis=0) - self.min
            self.rng = np.where(self.rng > 0, self.rng, 1.0)
        else:
            self.mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
            self.std = np.where(std > 0, std, 1.0)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if self.method == "minmax":
            assert self.min is not None and self.rng is not None
            return ((matrix - self.min) / self.rng).astype(np.float32)
        assert self.mean is not None and self.std is not None
        return ((matrix - self.mean) / self.std).astype(np.float32)
