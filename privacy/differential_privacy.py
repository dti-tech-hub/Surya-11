"""Differential-privacy-style client update perturbation.

Mechanism: per-client update (delta) is L2-clipped to norm C, then Gaussian
noise N(0, (σC)²) is added element-wise before transmission.

Scope of the guarantee — stated honestly:

- The clipping bounds each client's *update* sensitivity to C, and the noise
  scale is σC. Under the standard Gaussian-mechanism analysis this satisfies
  (ε, δ)-DP for a single release with ε ≈ σ√(2 log(1.25/δ)) **only if** the
  sampling/amplification and composition across rounds are also accounted for.
- This implementation reports the *single-round* ε estimate via
  :func:`estimate_epsilon` and clearly labels it an ESTIMATE. It does not
  implement RDP composition accounting across rounds; reports must not claim a
  formal end-to-end DP guarantee.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from utilities.seed import make_rng
import numpy as np


def clip_state_l2(state: dict[str, torch.Tensor], clip_norm: float) -> tuple[dict[str, torch.Tensor], float]:
    """Scale the whole update so its global L2 norm equals ``clip_norm`` (or less)."""
    sq = 0.0
    for v in state.values():
        sq += float(v.detach().float().pow(2).sum().item())
    norm = math.sqrt(sq)
    if norm <= clip_norm or norm == 0.0:
        return {k: v.detach().clone() for k, v in state.items()}, norm
    scale = clip_norm / norm
    return {k: (v.detach().float() * scale) for k, v in state.items()}, norm


def state_norm(state: dict[str, torch.Tensor]) -> float:
    sq = 0.0
    for v in state.values():
        sq += float(v.detach().float().pow(2).sum().item())
    return math.sqrt(sq)


class GaussianDPMechanism:
    """Clip-then-noise client update perturbation (Gaussian mechanism)."""

    def __init__(self, clipping_norm: float = 1.0, noise_multiplier: float = 0.1, delta: float = 1e-5, seed: int = 42):
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive")
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier must be >= 0")
        self.clipping_norm = float(clipping_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed % (2**32 - 1))
        self.applied_noise_norm: float = 0.0
        self.last_original_norm: float = 0.0
        self.last_clipped: bool = False

    # ------------------------------------------------------------------ #
    def apply(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clipped, original_norm = clip_state_l2(state, self.clipping_norm)
        self.last_original_norm = original_norm
        self.last_clipped = original_norm > self.clipping_norm
        if self.noise_multiplier == 0.0:
            self.applied_noise_norm = 0.0
            return clipped
        noise_std = self.noise_multiplier * self.clipping_norm
        total_noise_sq = 0.0
        out: dict[str, torch.Tensor] = {}
        for key, value in clipped.items():
            noise = torch.from_numpy(
                self.rng.normal(0.0, noise_std, size=tuple(value.shape)).astype(np.float32)
            )
            total_noise_sq += float(noise.pow(2).sum().item())
            out[key] = value.float() + noise
        self.applied_noise_norm = math.sqrt(total_noise_sq)
        return out

    # ------------------------------------------------------------------ #
    @property
    def sigma(self) -> float:
        """Noise multiplier relative to clipping norm."""
        return self.noise_multiplier

    def estimate_epsilon(self) -> float | None:
        """Single-round (ε, δ) estimate for the Gaussian mechanism.

        For a Gaussian mechanism with L2 sensitivity Δ and noise σΔ, the
        classical (non-tight) single-release bound is

            ε = sqrt(2 ln(1.25/δ)) / σ.

        This is an ESTIMATE for one release of *one client's* clipped update.
        Composition across federation rounds and the fact that secure
        aggregation is only simulated are NOT accounted for. Reports must not
        claim an end-to-end formal DP guarantee.
        """
        if self.noise_multiplier == 0.0:
            return None
        if self.delta <= 0 or self.delta >= 1:
            return None
        return float(math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.noise_multiplier)

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mechanism": "gaussian_clip_noise",
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
            "epsilon_single_round_estimate": self.estimate_epsilon(),
            "guarantee_scope": (
                "single-round (eps, delta) estimate only; no RDP composition "
                "across rounds — do NOT claim end-to-end formal DP"
            ),
            "last_original_norm": self.last_original_norm,
            "last_clipped": self.last_clipped,
            "last_noise_norm": self.applied_noise_norm,
        }
