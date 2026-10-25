"""Communication measurement and comparison, derived from recorded round bytes."""
from __future__ import annotations

from typing import Any, Sequence


def round_frame(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in history:
        rows.append(
            {
                "round": int(entry["round"]),
                "clients": int(entry.get("n_clients", len(entry.get("clients", [])))),
                "bytes_download": int(entry.get("bytes_download", 0)),
                "bytes_upload": int(entry.get("bytes_upload", 0)),
                "bytes_per_round": int(entry.get("bytes_download", 0)) + int(entry.get("bytes_upload", 0)),
                "round_duration_s": float(entry.get("round_duration_s", 0.0)),
            }
        )
    return rows


def summarize(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = round_frame(history)
    if not rows:
        return {"rounds": 0, "total_bytes": 0}
    total = sum(r["bytes_per_round"] for r in rows)
    down = sum(r["bytes_download"] for r in rows)
    up = sum(r["bytes_upload"] for r in rows)
    clients = sum(r["clients"] for r in rows)
    return {
        "rounds": len(rows),
        "total_bytes": int(total),
        "total_download_bytes": int(down),
        "total_upload_bytes": int(up),
        "bytes_per_round_mean": float(total / len(rows)),
        "bytes_per_client_mean": float(total / clients) if clients else 0.0,
        "mean_round_duration_s": float(sum(r["round_duration_s"] for r in rows) / len(rows)),
        "per_round": rows,
    }


def reduction_percentage(reference_bytes: float, candidate_bytes: float) -> float:
    """Measured reduction of candidate vs reference (positive = less traffic)."""
    reference_bytes = float(reference_bytes)
    if reference_bytes <= 0:
        return 0.0
    return 100.0 * (1.0 - float(candidate_bytes) / reference_bytes)


def compare_communication(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref = float(reference.get("total_bytes", 0) or 0)
    cand = float(candidate.get("total_bytes", 0) or 0)
    return {
        "reference_total_bytes": int(ref),
        "candidate_total_bytes": int(cand),
        "absolute_saving_bytes": int(ref - cand),
        "reduction_pct": reduction_percentage(ref, cand),
        "measured": ref > 0,
    }
