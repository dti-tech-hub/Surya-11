"""Privacy layer interface.

Pipeline (PowerShield-FL, config ``privacy.yaml``):

```
Client Update (delta)
      ↓  clip_update      — L2-norm clipping to C
      ↓  add_noise        — Gaussian noise σ·C per coordinate
      ↓  secure_aggregate  — simulated additive masking (optional)
      ↓  compress          — top-k / int8 (communication layer)
Server
```

Honesty note: secure aggregation here is a *simulation* of additive masking to
demonstrate the interface and its overhead. It is NOT a production cryptographic
protocol (no secret sharing, no malicious-party guarantees).
"""
from __future__ import annotations

from typing import Any, Protocol

import torch

from utilities.config import get


class PrivacyMechanism(Protocol):
    def apply(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def describe(self) -> dict[str, Any]: ...


class NoPrivacy:
    """Pass-through baseline (no clipping, no noise)."""

    def apply(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in state.items()}

    def describe(self) -> dict[str, Any]:
        return {"enabled": False, "mechanism": "none", "clipping_norm": None, "noise_multiplier": None}


def build_privacy(cfg: dict, seed: int = 42) -> Any:
    """Factory: returns the configured privacy mechanism."""
    if not bool(get(cfg, "privacy.enabled", False)):
        return NoPrivacy()
    mechanism = str(get(cfg, "privacy.mechanism", "dp")).lower()
    if mechanism in ("none", "off"):
        return NoPrivacy()
    if mechanism == "dp":
        from privacy.differential_privacy import GaussianDPMechanism

        return GaussianDPMechanism(
            clipping_norm=float(get(cfg, "privacy.clipping_norm", 1.0)),
            noise_multiplier=float(get(cfg, "privacy.noise_multiplier", 0.1)),
            delta=float(get(cfg, "privacy.delta", 1e-5)),
            seed=seed,
        )
    raise ValueError(f"Unknown privacy mechanism '{mechanism}' (dp|none)")
