"""Scalability experiment summarisation (client count sweeps)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def scalability_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_configurations": 0, "note": "scalability experiment NOT RUN"}
    ordered = sorted(rows, key=lambda r: int(r.get("clients", 0)))
    clients = np.array([int(r["clients"]) for r in ordered], dtype=float)
    wall = np.array([float(r.get("wall_time_s", np.nan)) for r in ordered], dtype=float)
    finite = np.isfinite(wall)
    slope = float(np.polyfit(clients[finite], wall[finite], 1)[0]) if finite.sum() >= 2 else float("nan")
    return {
        "n_configurations": len(ordered),
        "configurations": [int(c) for c in clients],
        "wall_time_s": [float(v) for v in wall],
        "wall_time_slope_s_per_client": slope,
        "max_clients": int(clients.max()),
        "note": "measured from the same QUICK/FULL configuration; wall time is machine-specific",
    }
