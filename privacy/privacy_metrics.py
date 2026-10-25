"""Privacy measurement: what the privacy layer actually did this round."""
from __future__ import annotations

from typing import Any

import numpy as np


def privacy_round_report(mechanism: Any, per_client: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-client clipping/noise observations into a round report."""
    described = mechanism.describe() if hasattr(mechanism, "describe") else {}
    orig_norms = [c.get("original_norm", float("nan")) for c in per_client]
    noise_norms = [c.get("noise_norm", float("nan")) for c in per_client]
    clipped_flags = [bool(c.get("clipped", False)) for c in per_client]
    return {
        "mechanism": described,
        "clients": len(per_client),
        "clipped_clients": int(sum(clipped_flags)),
        "mean_original_norm": float(np.nanmean(orig_norms)) if orig_norms else float("nan"),
        "mean_noise_norm": float(np.nanmean(noise_norms)) if noise_norms else float("nan"),
        "noise_to_signal_ratio": (
            float(np.nanmean(noise_norms) / max(np.nanmean(orig_norms), 1e-12))
            if orig_norms and np.isfinite(np.nanmean(orig_norms))
            else float("nan")
        ),
    }


def utility_impact_report(no_privacy_f1: float, with_privacy_f1: float) -> dict[str, Any]:
    """Compare F1 with and without the privacy layer (honest deltas only)."""
    delta = float(with_privacy_f1) - float(no_privacy_f1)
    return {
        "f1_no_privacy": float(no_privacy_f1),
        "f1_with_privacy": float(with_privacy_f1),
        "f1_delta": delta,
        "relative_change_pct": 100.0 * delta / max(float(no_privacy_f1), 1e-12),
    }
