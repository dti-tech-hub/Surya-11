"""Dataset ingestion for PowerShield-FL.

Supported sources (config ``data.source``):

- ``synthetic`` — clearly-labelled SYNTHETIC DEMO DATA generator (always available)
- ``sgcc``      — wide CSV layout of the SGCC electricity-theft dataset
- ``long_csv``  — tidy long-format CSV (timestamp, consumer_id, consumption)
- ``auto``      — use a real dataset when found under data/raw, else synthetic

All sources are normalised to the same contract (:class:`DatasetBundle`):

- ``long_df``   columns: consumer_id (str), timestamp (datetime64), consumption (float32)
- ``labels``    Series indexed by consumer_id with values in {0, 1} (1 = theft)
- ``consumer_meta`` DataFrame indexed by consumer_id; synthetic data records the
  theft activation day so window-level labels can avoid pre-theft leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utilities.config import get, resolve_path
from utilities.seed import make_rng

SYNTHETIC_LABEL = "SYNTHETIC DEMO DATA"


@dataclass
class DatasetBundle:
    """Normalised dataset container shared by every downstream module."""

    long_df: pd.DataFrame
    labels: pd.Series
    source: str
    is_synthetic: bool
    consumer_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def consumer_ids(self) -> list[str]:
        return list(self.labels.index.astype(str))

    @property
    def theft_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0

    def subset_consumers(self, ids: list[str]) -> "DatasetBundle":
        ids = [str(i) for i in ids]
        mask = self.long_df["consumer_id"].isin(ids)
        meta = self.consumer_meta.loc[self.consumer_meta.index.isin(ids)] if len(self.consumer_meta) else pd.DataFrame()
        return DatasetBundle(
            long_df=self.long_df.loc[mask].reset_index(drop=True),
            labels=self.labels.loc[ids],
            source=self.source,
            is_synthetic=self.is_synthetic,
            consumer_meta=meta,
            stats=dict(self.stats),
        )


def _dataset_stats(long_df: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    values = long_df["consumption"].to_numpy(dtype=np.float64, copy=False)
    finite = values[np.isfinite(values)]
    return {
        "n_consumers": int(labels.shape[0]),
        "n_rows": int(len(long_df)),
        "normal_samples": int((labels == 0).sum()),
        "theft_samples": int((labels == 1).sum()),
        "class_ratio_theft": float(labels.mean()) if len(labels) else 0.0,
        "missing_values": int(np.isnan(values).sum()),
        "duplicate_rows": int(long_df.duplicated(["consumer_id", "timestamp"]).sum()),
        "mean_consumption": float(np.nanmean(finite)) if finite.size else float("nan"),
        "std_consumption": float(np.nanstd(finite)) if finite.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Synthetic demo generator
# --------------------------------------------------------------------------- #
def generate_synthetic_dataset(cfg: dict, seed: int = 42) -> DatasetBundle:
    """Generate a labelled synthetic smart-meter panel (SYNTHETIC DEMO DATA).

    Model: per-consumer base load × intraday profile × seasonal/weekend factors
    + noise, with theft consumers experiencing metered-consumption *reduction*
    (scale factor < 1) from a random activation day, sometimes intermittent.
    """
    ds = get(cfg, "data.synthetic", {}) or {}
    n_consumers = int(ds.get("consumers", 240))
    n_days = int(ds.get("days", 180))
    interval = int(ds.get("interval_minutes", 30))
    steps = int(1440 // interval)
    theft_rate = float(ds.get("theft_rate", 0.15))
    rng = make_rng(seed)

    # Calendar index -------------------------------------------------------
    start = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(start, periods=n_days * steps, freq=f"{interval}min")
    dow_daily = pd.date_range(start, periods=n_days).dayofweek.to_numpy()  # (D,)
    tod = np.arange(steps) / steps                            # time-of-day in [0,1) for one day

    # Intraday profile: morning bump + evening peak ------------------------
    profile = (
        1.0
        + 0.35 * np.exp(-0.5 * ((tod - 0.33) / 0.06) ** 2)     # morning
        + 0.75 * np.exp(-0.5 * ((tod - 0.80) / 0.07) ** 2)     # evening peak
        + 0.10 * np.sin(2 * np.pi * tod)
    )
    profile = profile / profile.mean()

    # Per-consumer attributes ----------------------------------------------
    base = rng.uniform(float(ds.get("base_load_min", 0.25)), float(ds.get("base_load_max", 1.80)), n_consumers)
    evening_skew = rng.uniform(0.6, 1.5, n_consumers)          # peak sharpness heterogeneity
    seasonal_amp = float(ds.get("seasonal_amplitude", 0.22)) * rng.uniform(0.5, 1.5, n_consumers)
    weekend_factor = float(ds.get("weekend_factor", 1.08)) * rng.uniform(0.95, 1.10, n_consumers)

    days = np.arange(n_days)
    seasonal = 1.0 + seasonal_amp[:, None] * np.sin(2 * np.pi * (days - 20) / 365.0)   # (n, D)
    weekend = np.where(dow_daily[None, :] >= 5, weekend_factor[:, None], 1.0)           # (n, D)
    daily = base[:, None] * seasonal * weekend                                           # (n, D)

    # Build consumption tensor (n, D, steps) --------------------------------
    prof = profile[None, None, :] * (1.0 + (evening_skew[:, None, None] - 1.0) * (profile[None, None, :] - 1.0))
    prof = prof / prof.mean(axis=2, keepdims=True)
    mean_level = daily[:, :, None] * prof                                    # (n, D, steps)
    noise_sigma = float(ds.get("noise_sigma", 0.04))
    noise = rng.normal(0.0, 1.0, mean_level.shape) * (noise_sigma * daily[:, :, None])
    values = np.clip(mean_level + noise, 0.05, None)

    # Theft injection --------------------------------------------------------
    n_theft = int(round(theft_rate * n_consumers))
    theft_ids = rng.choice(n_consumers, size=min(n_theft, n_consumers), replace=False)
    theft_start_day = np.full(n_consumers, -1, dtype=int)
    theft_scale = np.ones(n_consumers)
    intermittent = np.zeros((n_consumers, n_days), dtype=bool)
    frac_min = float(ds.get("theft_start_frac_min", 0.25))
    for c in theft_ids:
        start_day = int(rng.integers(int(frac_min * n_days), max(int(frac_min * n_days) + 1, int(0.85 * n_days))))
        theft_start_day[c] = start_day
        theft_scale[c] = rng.uniform(float(ds.get("theft_scale_min", 0.35)), float(ds.get("theft_scale_max", 0.85)))
        active = rng.random(n_days) >= float(ds.get("theft_intermittent_prob", 0.25))
        active[:start_day] = False
        intermittent[c] = active
    if n_theft:
        mask = intermittent[theft_ids][:, :, None]                              # (k, D, 1)
        sub = values[theft_ids]
        sub = np.where(mask, sub * theft_scale[theft_ids][:, None, None], sub)
        values[theft_ids] = sub

    # Missing values & spikes -------------------------------------------------
    missing_rate = float(ds.get("missing_rate", 0.002))
    if missing_rate > 0:
        values[rng.random(values.shape) < missing_rate] = np.nan
    spike_rate = float(ds.get("spike_rate", 0.001))
    spike_mask = rng.random(values.shape) < spike_rate
    if spike_mask.any():
        values[spike_mask] = np.nan_to_num(values[spike_mask], nan=0.0) * rng.uniform(6.0, 15.0, int(spike_mask.sum()))

    # Long format -------------------------------------------------------------
    consumer_ids = np.array([f"C{idx:04d}" for idx in range(n_consumers)])
    long_df = pd.DataFrame(
        {
            "consumer_id": np.repeat(consumer_ids, n_days * steps),
            "timestamp": np.tile(timestamps, n_consumers),
            "consumption": values.reshape(-1).astype(np.float32),
        }
    )
    labels = pd.Series((theft_start_day >= 0).astype(int), index=pd.Index(consumer_ids, name="consumer_id"), name="label")
    meta = pd.DataFrame(
        {
            "theft_start_day": theft_start_day,
            "theft_scale": theft_scale,
            "base_load": base,
        },
        index=pd.Index(consumer_ids, name="consumer_id"),
    )
    bundle = DatasetBundle(
        long_df=long_df,
        labels=labels,
        source=SYNTHETIC_LABEL,
        is_synthetic=True,
        consumer_meta=meta,
    )
    bundle.stats = _dataset_stats(long_df, labels)
    bundle.stats.update(
        {
            "source": SYNTHETIC_LABEL,
            "days": n_days,
            "interval_minutes": interval,
            "readings_per_day": steps,
            "seed": int(seed),
            "label_note": "label=1 iff consumer has a theft episode; window labels respect theft_start_day",
        }
    )
    return bundle


# --------------------------------------------------------------------------- #
# Real dataset loaders
# --------------------------------------------------------------------------- #
def _load_sgcc(cfg: dict) -> DatasetBundle:
    raw_dir = resolve_path(cfg, get(cfg, "data.raw_dir", "data/raw"))
    data_file = raw_dir / get(cfg, "data.sgcc.consumption_file", "data.csv")
    label_file = raw_dir / get(cfg, "data.sgcc.label_file", "label.csv")
    if not data_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"SGCC dataset not found: expected {data_file} and {label_file}. See data/README.md."
        )
    wide = pd.read_csv(data_file)
    id_col = get(cfg, "data.sgcc.timestamp_column", "CONS_NO")
    wide[id_col] = wide[id_col].astype(str)
    long_df = wide.melt(id_vars=id_col, var_name="day", value_name="consumption")
    long_df = long_df.rename(columns={id_col: "consumer_id"})
    long_df["timestamp"] = pd.to_datetime(long_df["day"], errors="coerce")
    long_df = long_df.drop(columns=["day"]).dropna(subset=["timestamp"])
    long_df["consumption"] = pd.to_numeric(long_df["consumption"], errors="coerce").astype(np.float32)

    lab = pd.read_csv(label_file)
    lab[id_col] = lab[id_col].astype(str)
    label_col = "FLAG" if "FLAG" in lab.columns else [c for c in lab.columns if c != id_col][0]
    labels = lab.set_index(id_col)[label_col].astype(int)
    labels = labels.reindex(long_df["consumer_id"].unique())
    labels = labels.dropna().astype(int)
    labels.index.name = "consumer_id"
    long_df = long_df[long_df["consumer_id"].isin(labels.index)].reset_index(drop=True)

    bundle = DatasetBundle(long_df=long_df, labels=labels, source="sgcc", is_synthetic=False)
    bundle.stats = _dataset_stats(long_df, labels)
    bundle.stats["source"] = "sgcc"
    return bundle


def _load_long_csv(cfg: dict) -> DatasetBundle:
    raw_dir = resolve_path(cfg, get(cfg, "data.raw_dir", "data/raw"))
    filename = get(cfg, "data.long_csv.filename", "readings.csv")
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Long-format dataset not found: {path}. See data/README.md.")
    df = pd.read_csv(path)
    tcol = get(cfg, "data.long_csv.timestamp_col", "timestamp")
    ccol = get(cfg, "data.long_csv.consumer_col", "consumer_id")
    vcol = get(cfg, "data.long_csv.consumption_col", "consumption")
    df = df.rename(columns={tcol: "timestamp", ccol: "consumer_id", vcol: "consumption"})
    df["consumer_id"] = df["consumer_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    if "label" in df.columns:
        labels = df.groupby("consumer_id", sort=False)["label"].max().astype(int)
    else:
        raise ValueError("long_csv dataset requires a 'label' column (0 = normal, 1 = theft)")
    labels.index.name = "consumer_id"
    df = df.drop(columns=["label"])
    bundle = DatasetBundle(long_df=df[["consumer_id", "timestamp", "consumption"]], labels=labels, source="long_csv", is_synthetic=False)
    bundle.stats = _dataset_stats(bundle.long_df, labels)
    bundle.stats["source"] = "long_csv"
    return bundle


def detect_real_dataset(cfg: dict) -> str | None:
    """Return which real source is available under data/raw, or None."""
    raw_dir = resolve_path(cfg, get(cfg, "data.raw_dir", "data/raw"))
    if (raw_dir / get(cfg, "data.sgcc.consumption_file", "data.csv")).exists() and (
        raw_dir / get(cfg, "data.sgcc.label_file", "label.csv")
    ).exists():
        return "sgcc"
    if (raw_dir / get(cfg, "data.long_csv.filename", "readings.csv")).exists():
        return "long_csv"
    return None


def load_dataset(cfg: dict, seed: int = 42) -> DatasetBundle:
    """Load the configured dataset; ``auto`` prefers real data over synthetic."""
    source = get(cfg, "data.source", "auto")
    if source == "auto":
        source = detect_real_dataset(cfg) or "synthetic"
    if source == "synthetic":
        return generate_synthetic_dataset(cfg, seed=seed)
    if source == "sgcc":
        return _load_sgcc(cfg)
    if source == "long_csv":
        return _load_long_csv(cfg)
    raise ValueError(f"Unknown data source '{source}'")
