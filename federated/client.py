"""Simulated federated client (one electricity retailer).

The client owns private training windows and never exports them: the only
outbound object is a :class:`utilities.types.ClientUpdate` containing the
*model delta* plus aggregate statistics (sample counts, losses, local metrics).
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from models.base import TorchBinaryClassifier
from preprocessing.imbalance import compute_class_weights
from utilities.config import get
from utilities.seed import make_rng
from utilities.types import ClientUpdate

import torch


def state_delta(local: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: local[k].detach().float() - global_state[k].detach().float() for k in global_state}


def delta_norm(delta: dict[str, torch.Tensor]) -> float:
    sq = sum(float(v.pow(2).sum().item()) for v in delta.values())
    return float(np.sqrt(sq))


class FederatedClient:
    """Holds one client's private data and trains local model updates."""

    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        cfg: dict,
        seed: int = 42,
        local_val_fraction: float | None = None,
        class_weights: dict[str, Any] | None = None,
    ):
        self.client_id = client_id
        self.cfg = cfg
        self.seed = seed
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y length mismatch")

        frac = local_val_fraction if local_val_fraction is not None else float(get(cfg, "clients.local_val_fraction", 0.15))
        rng = make_rng(seed)
        idx = np.arange(self.y.shape[0])
        rng.shuffle(idx)
        n_val = int(round(frac * self.y.shape[0]))
        # keep at least one positive and one negative in train when possible
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        if train_idx.size == 0:
            train_idx, val_idx = idx, np.empty(0, dtype=int)
        self.train_idx, self.val_idx = train_idx, val_idx
        self.class_weights = class_weights or compute_class_weights(self.y[train_idx], cfg)

    # ------------------------------------------------------------------ #
    @property
    def n_samples(self) -> int:
        return int(self.train_idx.shape[0])

    def summary(self) -> dict[str, Any]:
        y_tr = self.y[self.train_idx]
        return {
            "client_id": self.client_id,
            "n_samples": self.n_samples,
            "n_val": int(self.val_idx.size),
            "theft_ratio": float(y_tr.mean()) if y_tr.size else 0.0,
            "n_positive": int((y_tr == 1).sum()),
            "n_negative": int((y_tr == 0).sum()),
        }

    # ------------------------------------------------------------------ #
    def train(
        self,
        global_state: dict[str, torch.Tensor],
        model_factory: Callable[[], TorchBinaryClassifier],
        epochs: int | None = None,
    ) -> tuple[ClientUpdate, TorchBinaryClassifier]:
        """Local SGD from the broadcast global model; returns the delta update."""
        cfg = self.cfg
        epochs = int(epochs if epochs is not None else get(cfg, "federated.local_epochs", 2))
        strategy = str(get(cfg, "preprocessing.class_imbalance.strategy", "class_weight"))
        model = model_factory()
        model.set_weights(global_state)

        t0 = time.perf_counter()
        history = model.fit(
            self.X[self.train_idx],
            self.y[self.train_idx],
            epochs=epochs,
            batch_size=int(get(cfg, "federated.batch_size", 128)),
            class_weights=self.class_weights,
            focal=(strategy == "focal"),
            focal_gamma=float(get(cfg, "preprocessing.class_imbalance.focal_gamma", 2.0)),
            seed=self.seed,
        )
        train_seconds = time.perf_counter() - t0
        train_loss = float(history[-1]["train_loss"]) if history else float("nan")

        val_metrics: dict[str, float] = {}
        val_loss = float("nan")
        if self.val_idx.size:
            probs = model.predict_proba(self.X[self.val_idx])
            yv = self.y[self.val_idx]
            val_metrics = {
                "f1": float(f1_score(yv, probs >= 0.5, zero_division=0)),
                "precision": float(precision_score(yv, probs >= 0.5, zero_division=0)),
                "recall": float(recall_score(yv, probs >= 0.5, zero_division=0)),
            }
            if len(np.unique(yv)) > 1:
                val_metrics["roc_auc"] = float(roc_auc_score(yv, probs))
            # Unweighted validation BCE — used by adaptive aggregation as a
            # client-quality signal. Computed from the model's own logits.
            import torch as _torch

            xv = _torch.from_numpy(self.X[self.val_idx].astype(np.float32))
            yv_t = _torch.from_numpy(yv.astype(np.float32))
            with _torch.no_grad():
                logits = model.model(xv).squeeze(-1)
                val_loss = float(_torch.nn.functional.binary_cross_entropy_with_logits(logits, yv_t).item())
            val_metrics["val_loss"] = val_loss

        delta = state_delta(model.get_weights(), global_state)
        y_tr = self.y[self.train_idx]
        update = ClientUpdate(
            client_id=self.client_id,
            num_samples=self.n_samples,
            state_dict=delta,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            num_positive=int((y_tr == 1).sum()),
            num_negative=int((y_tr == 0).sum()),
            update_norm=delta_norm(delta),
            metadata={"train_seconds": train_seconds, "local_epochs": epochs},
        )
        update.validate()  # raises if anything forbidden would leak
        return update, model
