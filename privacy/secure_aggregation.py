"""Simulated secure aggregation (additive pairwise masking).

Demonstration-grade simulation of the secure-aggregation *interface*:

- Each pair of clients (i, j) shares a seeded pseudo-random mask M_ij.
- Client i adds +M_ij to its update, client j adds −M_ij.
- Any individual transmitted update is masked (looks random to the server).
- The sum over all clients cancels the masks exactly.

This is NOT production cryptography: masks are generated from a shared seed via
a PRNG instead of Diffie-Hellman-derived one-time pads, there is no dropout
recovery protocol, and no malicious-party guarantees. It demonstrates the
communication/storage overhead and the architectural pattern only.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from utilities.config import get
from utilities.seed import make_rng

SIMULATION_WARNING = "SIMULATED secure aggregation — additive masking demo, not production crypto"


def _pair_mask(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(size=shape).astype(np.float32)


def mask_updates(
    states: Sequence[dict[str, torch.Tensor]],
    seed: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Apply cancelling pairwise masks to each client's update."""
    n = len(states)
    if n <= 1:
        return [{k: v.detach().clone() for k, v in s.items()} for s in states], {
            "clients": n,
            "pairs": 0,
            "note": "single client — nothing to mask against",
            "simulation": True,
        }
    rng = make_rng(seed)
    keys = list(states[0].keys())
    masked: list[dict[str, torch.Tensor]] = [
        {k: v.detach().clone().float() for k, v in s.items()} for s in states
    ]
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = 1.0 if (i + j) % 2 == 0 else -1.0
            for key in keys:
                shape = tuple(states[i][key].shape)
                m = _pair_mask(rng, shape)
                masked[i][key] = masked[i][key] + sign * torch.from_numpy(m)
                masked[j][key] = masked[j][key] - sign * torch.from_numpy(m)
            pairs += 1
    return masked, {"clients": n, "pairs": pairs, "simulation": True, "note": SIMULATION_WARNING}


def unmask_individual(
    masked: dict[str, torch.Tensor],
    true_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Server-side decode of one client's update (requires pairwise secrets).

    In a real protocol the server could NOT decode a single client's update;
    only the *sum* is unmasked. This helper exists solely so tests can verify
    that masks cancel in aggregate while individual transmissions stay masked.
    """
    return {k: masked[k].float() - true_state[k].float() for k in masked}


class SecureAggregationSimulator:
    """Config wrapper used by the federated simulator."""

    def __init__(self, cfg: dict, seed: int = 2026):
        self.enabled = bool(get(cfg, "privacy.secure_aggregation", False))
        self.seed = int(get(cfg, "privacy.secure_aggregation_seed", seed))

    def protect(self, states: Sequence[dict[str, torch.Tensor]]) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
        if not self.enabled:
            return list(states), {"enabled": False}
        masked, info = mask_updates(states, self.seed)
        info["enabled"] = True
        return masked, info

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "type": "simulated_additive_masking",
            "warning": SIMULATION_WARNING,
            "seed": self.seed,
        }
