"""Shared torch classifier wrapper used by all neural models.

Provides a common fit / predict interface, training-history recording,
parameter/size accounting, and flat-vector weight access for federated
aggregation and communication-compression experiments.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.losses import build_loss
from utilities.seed import make_rng

DEVICE = torch.device("cpu")  # project is CPU-first by design (student laptop)


def get_flat_params(model: nn.Module) -> np.ndarray:
    """Single float32 vector of all parameters (order is deterministic)."""
    with torch.no_grad():
        return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    offset = 0
    flat = np.asarray(flat, dtype=np.float32)
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(flat[offset : offset + n]).view_as(p).to(p.device))
            offset += n
    if offset != flat.size:
        raise ValueError(f"Flat vector size {flat.size} does not match model size {offset}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class TorchBinaryClassifier:
    """Thin training/inference wrapper around a binary-logit ``nn.Module``."""

    def __init__(
        self,
        model: nn.Module,
        name: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = DEVICE,
    ):
        self.model = model.to(device)
        self.name = name
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return num_params(self.model)

    def size_bytes(self) -> int:
        return get_flat_params(self.model).nbytes

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 5,
        batch_size: int = 128,
        class_weights: dict[str, Any] | None = None,
        focal: bool = False,
        focal_gamma: float = 2.0,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        optimizer_name: str = "adam",
        seed: int = 42,
        log: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        torch.manual_seed(seed)
        criterion = build_loss(class_weights, focal=focal, gamma=focal_gamma, device=self.device)
        params = self.model.parameters()
        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(xt, yt),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        has_val = X_val is not None and y_val is not None and len(np.asarray(y_val)) > 0
        if has_val:
            xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(self.device)
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(self.device)

        self.model.train()
        self.history = []
        for epoch in range(int(epochs)):
            t0 = time.perf_counter()
            running, n_seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * yb.shape[0]
                n_seen += yb.shape[0]
            train_loss = running / max(n_seen, 1)
            entry: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "duration_s": time.perf_counter() - t0,
            }
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.model(xv).squeeze(-1), yv).item())
                self.model.train()
                entry["val_loss"] = val_loss
            self.history.append(entry)
            if log:
                extra = f" val_loss={entry['val_loss']:.4f}" if has_val else ""
                log(f"[{self.name}] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f}{extra}")
        self.model.eval()
        return self.history

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Probability of class 1 (theft) for each sample."""
        self.model.eval()
        out: list[np.ndarray] = []
        xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        for start in range(0, xt.shape[0], batch_size):
            logits = self.model(xt[start : start + batch_size].to(self.device)).squeeze(-1)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        self.model.eval()
        out: list[np.ndarray] = []
        xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        for start in range(0, xt.shape[0], batch_size):
            logits = self.model(xt[start : start + batch_size].to(self.device)).squeeze(-1)
            out.append(logits.cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def get_flat(self) -> np.ndarray:
        return get_flat_params(self.model)

    def set_flat(self, flat: np.ndarray) -> None:
        set_flat_params(self.model, flat)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": self.name, "state_dict": self.get_weights(), "metadata": metadata or {}}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, model_factory: Callable[[], nn.Module]) -> "TorchBinaryClassifier":
        payload = torch.load(Path(path), map_location=DEVICE, weights_only=False)
        wrapper = cls(model_factory(), name=payload.get("model_name", "loaded"))
        wrapper.set_weights(payload["state_dict"])
        return wrapper
