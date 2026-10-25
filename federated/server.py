"""Federated server: holds the global model and performs aggregation.

The server never receives raw samples — only client deltas (after the privacy
and communication layers). Aggregation *weights* are decided by the caller
(FedAvg sample counts vs PowerShield-FL adaptive weights).
"""
from __future__ import annotations

from typing import Sequence

import torch

from federated.aggregation import aggregate_deltas, apply_delta
from utilities.types import ClientUpdate


class FederatedServer:
    def __init__(self, global_state: dict[str, torch.Tensor]):
        self.global_state = {k: v.detach().float().clone() for k, v in global_state.items()}
        self.round = 0

    def get_global_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.global_state.items()}

    def aggregate(
        self,
        deltas: Sequence[dict[str, torch.Tensor]],
        weights: Sequence[float],
        scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Weighted-mean the deltas and add them to the global model.

        ``scale`` corrects for partial participation: a round that samples only
        a fraction of clients applies the correspondingly scaled aggregate, as
        in standard FedAvg.
        """
        averaged = aggregate_deltas(deltas, weights)
        averaged = {k: v * float(scale) for k, v in averaged.items()}
        self.global_state = apply_delta(self.global_state, averaged)
        self.round += 1
        return self.get_global_state()

    def validate_updates(self, updates: Sequence[ClientUpdate]) -> None:
        """Reject anything outside the strict ClientUpdate contract."""
        for update in updates:
            update.validate()
