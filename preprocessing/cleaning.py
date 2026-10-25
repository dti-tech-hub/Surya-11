"""Cleaning: duplicates, invalid values, missing values, Hampel-style noise.

All operations are per-consumer (a consumer's series is never mixed with
another's), so cleaning can be applied before the consumer-level split without
label leakage. Supervised-fitted steps (outlier bounds, scalers) live in
:mod:`preprocessing.pipeline` and are fitted on training consumers only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utilities.config import get


def _equal_length_matrix(df: pd.DataFrame) -> np.ndarray | None:
    """Return (n_consumers, T) matrix if every consumer has the same length."""
    counts = df.groupby("consumer_id", sort=False)["consumption"].size()
    if counts.nunique() == 1 and len(df) % len(counts) == 0:
        return df["consumption"].to_numpy(dtype=np.float64).reshape(len(counts), counts.iloc[0])
    return None


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(subset=["consumer_id", "timestamp"], keep="first")
    return df.reset_index(drop=True), before - len(df)


def handle_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mark physically impossible readings as missing (negative consumption)."""
    negatives = int((df["consumption"] < 0).sum())
    df = df.copy()
    df.loc[df["consumption"] < 0, "consumption"] = np.nan
    inf_mask = np.isinf(df["consumption"].to_numpy(dtype=np.float64, copy=True))
    df.loc[inf_mask, "consumption"] = np.nan
    return df, {"negative_values": negatives, "infinite_values": int(inf_mask.sum())}


def handle_missing(df: pd.DataFrame, cfg: dict, global_median: float | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing readings per consumer: interpolate | median | drop."""
    strategy = get(cfg, "preprocessing.missing.strategy", "interpolate")
    max_gap = int(get(cfg, "preprocessing.missing.max_interpolate_gap", 6))
    n_missing = int(df["consumption"].isna().sum())
    report: dict[str, Any] = {"missing_before": n_missing, "strategy": strategy}

    if strategy == "drop":
        return df.dropna(subset=["consumption"]).reset_index(drop=True), report

    if strategy == "median":
        fill = df["consumption"].median() if global_median is None else global_median
        df = df.copy()
        df["consumption"] = df["consumption"].fillna(fill)
        report["median_fill_value"] = float(fill)
        return df, report

    # interpolate (default): per-consumer linear interpolation with gap limit
    df = df.sort_values(["consumer_id", "timestamp"], kind="stable").reset_index(drop=True)
    mat = _equal_length_matrix(df)
    if mat is not None:
        frame = pd.DataFrame(mat.T)
        frame = frame.interpolate(limit=max_gap, method="linear", axis=0, limit_direction="both")
        df = df.copy()
        df["consumption"] = frame.to_numpy(dtype=np.float32).T.reshape(-1)
    else:
        def _interp(series: pd.Series) -> pd.Series:
            return series.interpolate(limit=max_gap, method="linear", limit_direction="both")

        df = df.copy()
        df["consumption"] = df.groupby("consumer_id", sort=False)["consumption"].transform(_interp)

    # trailing gaps too long for interpolation → consumer median, then global
    still_missing = int(df["consumption"].isna().sum())
    if still_missing:
        per_consumer = df.groupby("consumer_id")["consumption"].transform("median")
        df["consumption"] = df["consumption"].fillna(per_consumer)
        fill = df["consumption"].median() if global_median is None else global_median
        df["consumption"] = df["consumption"].fillna(fill)
    report["missing_after"] = int(df["consumption"].isna().sum())
    report["interpolated_or_filled"] = n_missing - report["missing_after"]
    return df, report


def denoise_hampel(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace spike outliers with the local rolling median (Hampel filter).

    Spike detection threshold is a robust z-score (deviation vs median absolute
    deviation of the per-consumer deviation distribution).
    """
    if not get(cfg, "preprocessing.noise.enabled", True):
        return df, {"enabled": False}
    window = int(get(cfg, "preprocessing.noise.rolling_window", 3))
    threshold = float(get(cfg, "preprocessing.noise.spike_threshold", 6.0))
    # A robust z-score alone flags smooth load curvature as "spikes" (a
    # sinusoidal daily profile is not a Hampel outlier). Require an additional
    # absolute effect-size floor relative to the consumer's typical load.
    min_frac = float(get(cfg, "preprocessing.noise.min_deviation_frac", 0.5))

    df = df.sort_values(["consumer_id", "timestamp"], kind="stable").reset_index(drop=True)
    values = df["consumption"].to_numpy(dtype=np.float64, copy=True)
    mat = _equal_length_matrix(df)
    total_replaced = 0

    if mat is not None:
        frame = pd.DataFrame(mat.T)
        rolling_med = frame.rolling(window, center=True, min_periods=1).median().to_numpy().T
        dev = np.abs(mat - rolling_med)
        med_dev = np.nanmedian(dev, axis=1, keepdims=True)
        typical_load = np.nanmedian(mat, axis=1, keepdims=True)
        robust_cut = threshold * 1.4826 * med_dev
        floor = min_frac * np.abs(typical_load)
        mask = dev > np.maximum(robust_cut, floor) + 1e-12
        mask &= np.isfinite(mat)
        replaced = mat.copy()
        replaced[mask] = rolling_med[mask]
        total_replaced = int(mask.sum())
        df = df.copy()
        df["consumption"] = replaced.reshape(-1).astype(np.float32)
    else:
        flat = values.copy()
        groups = df.groupby("consumer_id", sort=False).indices  # positions within frame
        for positions in groups.values():
            pos = np.asarray(positions)
            series = pd.Series(values[pos])
            rolling_med = series.rolling(window, center=True, min_periods=1).median().to_numpy()
            dev = np.abs(values[pos] - rolling_med)
            finite_dev = dev[np.isfinite(dev)]
            med_dev = np.median(finite_dev) if finite_dev.size else 0.0
            typical_load = np.nanmedian(values[pos]) if np.isfinite(values[pos]).any() else 0.0
            cut = max(threshold * 1.4826 * med_dev, min_frac * abs(float(typical_load)))
            mask = dev > (cut + 1e-12)
            mask &= np.isfinite(values[pos])
            updated = values[pos].copy()
            updated[mask] = rolling_med[mask]
            flat[pos] = updated
            total_replaced += int(mask.sum())
        df = df.copy()
        df["consumption"] = flat.astype(np.float32)
    return df, {
        "enabled": True,
        "window": window,
        "spike_threshold": threshold,
        "min_deviation_frac": min_frac,
        "values_replaced": total_replaced,
    }


def clean_dataset(
    df: pd.DataFrame,
    cfg: dict,
    global_median: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full cleaning chain: duplicates -> invalid -> noise -> missing.

    ``global_median`` must be fitted on training consumers only; the pipeline
    passes that value to avoid using validation/test values for imputation.
    """
    report: dict[str, Any] = {}
    df, report["duplicates_dropped"] = drop_duplicates(df)
    df, invalid = handle_invalid(df)
    report.update(invalid)
    df, noise = denoise_hampel(df, cfg)
    report["noise"] = noise
    df, missing = handle_missing(df, cfg, global_median=global_median)
    report["missing"] = missing
    report["global_median_source"] = "train consumers only" if global_median is not None else "full dataframe (caller did not supply train median)"
    report["rows_after_cleaning"] = len(df)
    return df, report
