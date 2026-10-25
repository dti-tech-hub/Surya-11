"""Strict data contracts passed between federated clients and the server."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


# The only fields a client is permitted to transmit. Raw samples, timestamps,
# customer identifiers, and consumption sequences are deliberately absent.
CLIENT_UPDATE_FIELDS = frozenset(
    {
        "client_id",
        "num_samples",
        "state_dict",
        "train_loss",
        "val_loss",
        "val_metrics",
        "num_positive",
        "num_negative",
        "update_norm",
        "metadata",
    }
)


@dataclass
class ClientUpdate:
    """A model update plus aggregate statistics; never raw client data."""

    client_id: str
    num_samples: int
    state_dict: dict[str, torch.Tensor]
    train_loss: float = 0.0
    val_loss: float = float("nan")
    val_metrics: dict[str, float] = field(default_factory=dict)
    num_positive: int = 0
    num_negative: int = 0
    update_norm: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        extra = set(vars(self)) - CLIENT_UPDATE_FIELDS
        if extra:
            raise ValueError(f"ClientUpdate contains forbidden fields: {sorted(extra)}")
        if not isinstance(self.state_dict, dict):
            raise TypeError("state_dict must be a mapping")
        for key, value in self.state_dict.items():
            if not torch.is_tensor(value):
                raise TypeError(f"state_dict['{key}'] must be a tensor")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

    def detached_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict.items()}


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    probabilities: list[float]
    targets: list[int]
    threshold: float
    confusion_matrix: list[list[int]]
    sample_ids: list[int] = field(default_factory=list)
    consumer_ids: list[str] = field(default_factory=list)
