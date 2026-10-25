"""BASELINE centralized models (conventional ML) for the comparison study.

- Logistic Regression (class_weight='balanced')
- Random Forest (class_weight='balanced')
- MLP (torch, heavier hidden layers than the proposed lightweight model)

All expose fit(X, y) / predict_proba(X) so the evaluation harness treats them
uniformly. They are trained *centrally* on pooled training data — the privacy
contrast to federated training.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models.base import TorchBinaryClassifier
from models.losses import build_loss
from utilities.config import get


class SklearnClassifier:
    """Uniform wrapper around an sklearn binary classifier."""

    def __init__(self, estimator, name: str, class_weight: str | dict | None = "balanced"):
        self.estimator = estimator
        self.name = name
        self.class_weight = class_weight
        self.history: list[dict[str, Any]] = []
        self.train_seconds: float = 0.0

    @property
    def n_params(self) -> int:
        # sklearn models have no tensors; count leaf/coefficient payload
        est = self.estimator
        if hasattr(est, "coef_"):
            return int(np.asarray(est.coef_).size + (np.asarray(est.intercept_).size if hasattr(est, "intercept_") else 0))
        if hasattr(est, "estimators_"):
            return int(sum(e.tree_.node_count for e in est.estimators_))
        return 0

    def size_bytes(self) -> int:
        return int(self.n_params * 8)  # float64 node/coef payload estimate

    def fit(self, X: np.ndarray, y: np.ndarray, **_: Any) -> list[dict[str, Any]]:
        if self.class_weight is not None and "class_weight" in self.estimator.get_params():
            self.estimator.set_params(class_weight=self.class_weight)
        t0 = time.perf_counter()
        self.estimator.fit(np.asarray(X), np.asarray(y))
        self.train_seconds = time.perf_counter() - t0
        self.history = [{"epoch": 1, "train_loss": float("nan"), "duration_s": self.train_seconds}]
        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.estimator.predict_proba(np.asarray(X))
        # column for class 1
        classes = list(self.estimator.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X), dtype=np.float64)


class CentralMLP(TorchBinaryClassifier):
    """Heavier centralized MLP baseline (bigger than the proposed model)."""

    def __init__(self, n_inputs: int, hidden: list[int], dropout: float = 0.2, lr: float = 1e-3, wd: float = 1e-4):
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        super().__init__(nn.Sequential(*layers), name="centralized_mlp", learning_rate=lr, weight_decay=wd)


def build_central_models(cfg: dict, n_features: int, seed: int = 42) -> dict[str, Any]:
    """Build the configured centralized baseline zoo."""
    requested = list(get(cfg, "experiments.central_models", ["logistic_regression", "random_forest", "mlp"]))
    max_iter = int(get(cfg, "model.central.max_iter", 500) if get(cfg, "model.central") else 500)
    models: dict[str, Any] = {}
    for name in requested:
        if name == "logistic_regression":
            models[name] = SklearnClassifier(
                LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed),
                name="logistic_regression",
            )
        elif name == "random_forest":
            models[name] = SklearnClassifier(
                RandomForestClassifier(
                    n_estimators=int(get(cfg, "model.random_forest.n_estimators", 200) or 200),
                    max_depth=int(get(cfg, "model.random_forest.max_depth", 12) or 12),
                    n_jobs=-1,
                    random_state=seed,
                ),
                name="random_forest",
            )
        elif name == "mlp":
            torch.manual_seed(seed)
            models[name] = CentralMLP(
                n_features,
                hidden=list(get(cfg, "model.central_mlp_hidden", [128, 64])),
                dropout=float(get(cfg, "model.central_mlp_dropout", 0.2)),
                lr=1e-3,
                wd=1e-4,
            )
        else:
            raise ValueError(f"Unknown central model '{name}'")
    return models


def fit_central_model(model: Any, X: np.ndarray, y: np.ndarray, cfg: dict, class_weights: dict | None, seed: int) -> list[dict[str, Any]]:
    """Fit with the right calling convention for either wrapper family."""
    if isinstance(model, TorchBinaryClassifier):
        strategy = str(get(cfg, "preprocessing.class_imbalance.strategy", "class_weight"))
        return model.fit(
            X,
            y,
            epochs=int(get(cfg, "model.central_mlp_epochs", 8) or 8),
            batch_size=int(get(cfg, "federated.batch_size", 128)),
            class_weights=class_weights,
            focal=(strategy == "focal"),
            focal_gamma=float(get(cfg, "preprocessing.class_imbalance.focal_gamma", 2.0)),
            seed=seed,
        )
    return model.fit(X, y)
