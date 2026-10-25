"""Statistical analysis across repeated runs (multiple seeds).

Reports mean, standard deviation, 95% CI, and paired comparisons
(paired t-test + Wilcoxon signed-rank). Significance claims are limited to
what the tests actually support; with n=3 seeds the power is low and the
report says so.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats


def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"n": 1, "mean": mean, "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(arr.size)
    critical = float(stats.t.ppf((1 + confidence) / 2, df=arr.size - 1))
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci_low": float(mean - critical * sem),
        "ci_high": float(mean + critical * sem),
        "values": arr.tolist(),
    }


def paired_comparison(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """Paired tests between two methods over matched runs (same seeds)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = min(a_arr.size, b_arr.size)
    a_arr, b_arr = a_arr[:n], b_arr[:n]
    if n < 2:
        return {"n": int(n), "note": "fewer than 2 paired runs — statistical tests NOT RUN"}
    diff = a_arr - b_arr
    result: dict[str, Any] = {
        "n": int(n),
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)),
    }
    if np.allclose(diff, 0):
        result.update({"test": "degenerate", "significant": False, "note": "zero difference across all pairs"})
        return result
    try:
        t_stat, t_p = stats.ttest_rel(a_arr, b_arr)
        result["paired_ttest"] = {"statistic": float(t_stat), "p_value": float(t_p)}
    except ValueError as exc:  # pragma: no cover
        result["paired_ttest"] = {"error": str(exc)}
    try:
        w_stat, w_p = stats.wilcoxon(a_arr, b_arr)
        result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p)}
    except ValueError as exc:  # pragma: no cover
        result["wilcoxon"] = {"error": str(exc)}
    p_values = [v.get("p_value") for k, v in result.items() if isinstance(v, dict) and "p_value" in v]
    significant = any(p is not None and p < alpha for p in p_values)
    result["significant_at_alpha"] = bool(significant)
    result["alpha"] = alpha
    result["caveat"] = (
        "low statistical power with few seeds; treat non-significance as "
        "'inconclusive', not 'no difference'"
    )
    return result


def effect_size_cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(a_arr.size, b_arr.size)
    if n < 2:
        return float("nan")
    diff = a_arr[:n] - b_arr[:n]
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")
