"""Window-based feature engineering for electricity-theft detection.

Each sample is a ``window_days``-day sliding window of one consumer's meter
readings (48 half-hour steps per day by default). From every window we build:

- intraday *block means* — the window's daily series downsampled into 8 blocks
  per day (captures level + shape compactly → lightweight input),
- *engineered statistics* — peak/off-peak ratio, load variation, trend,
  max daily drop, low-day fraction, autocorrelation, calendar terms, etc.

Window labels respect theft activation days for synthetic data: a window is
positive only if it overlaps the theft period, so pre-theft behaviour is not
mislabelled. For real datasets labels are consumer-level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utilities.config import get

N_BLOCKS = 8

ENGINEERED_FEATURE_NAMES = [
    "mean_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption",
    "median_consumption",
    "peak_offpeak_ratio",
    "night_day_ratio",
    "load_variation",
    "consumption_trend",
    "daily_diff_std",
    "max_daily_drop",
    "mean_daily_drop",
    "frac_low_days",
    "min_max_day_ratio",
    "acf_lag1",
    "skewness",
    "kurtosis",
    "weekend_weekday_ratio",
    "weekend_fraction",
    "start_dow_sin",
    "start_dow_cos",
]


@dataclass
class WindowDataset:
    """Model-ready tensors plus provenance metadata."""

    X_matrix: np.ndarray                       # (n, n_blocks*W + n_engineered) scaled
    X_seq: np.ndarray                          # (n, window_days, steps) value-scaled
    y: np.ndarray                              # (n,) ints in {0,1}
    feature_names: list[str]
    meta: pd.DataFrame                         # consumer_id, start_day, end_day, label
    block_slice: slice = field(default_factory=lambda: slice(0, 0))
    engine_slice: slice = field(default_factory=lambda: slice(0, 0))
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def theft_ratio(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    def block_matrix(self) -> np.ndarray:
        return self.X_matrix[:, self.block_slice]

    def engine_matrix(self) -> np.ndarray:
        return self.X_matrix[:, self.engine_slice]

    def subset(self, indices: np.ndarray) -> "WindowDataset":
        return WindowDataset(
            X_matrix=self.X_matrix[indices],
            X_seq=self.X_seq[indices],
            y=self.y[indices],
            feature_names=self.feature_names,
            meta=self.meta.iloc[indices].reset_index(drop=True),
            block_slice=self.block_slice,
            engine_slice=self.engine_slice,
            stats=dict(self.stats),
        )


def _window_starts(n_days: int, window_days: int, stride: int) -> np.ndarray:
    if n_days < window_days:
        return np.empty(0, dtype=int)
    return np.arange(0, n_days - window_days + 1, stride)


def _slope(y: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row against x = 0..W-1 (vectorised)."""
    w = y.shape[1]
    x = np.arange(w, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean(axis=1, keepdims=True)
    denom = float((xc**2).sum())
    return (yc * xc).sum(axis=1) / (denom if denom > 0 else 1.0)


def _window_engineered(
    win: np.ndarray,
    daily: np.ndarray,
    peak_mask: np.ndarray,
    night_mask: np.ndarray,
    dow: np.ndarray,
) -> np.ndarray:
    # ``dow`` is the day-of-week sequence for this consumer's full panel; each
    # window's calendar features must use its own slice. The slice is supplied
    # by the caller through the ``dow`` argument already expanded per window
    # when it has shape (n_windows, window_days).
    """Engineered features for a batch of windows.

    Parameters
    ----------
    win  : (NW, W, steps) raw consumption windows
    daily: (NW, W) daily means per window
    """
    eps = 1e-8
    nw, w, steps = win.shape
    flat = win.reshape(nw, -1)

    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    vmin = flat.min(axis=1)
    vmax = flat.max(axis=1)
    median = np.median(flat, axis=1)

    peak_mean = win[:, :, peak_mask].mean(axis=(1, 2))
    night_mean = win[:, :, night_mask].mean(axis=(1, 2))
    overall_mean = mean
    peak_offpeak = peak_mean / (night_mean + eps)
    night_day = night_mean / (overall_mean + eps)
    load_var = std / (overall_mean + eps)

    trend = _slope(daily)
    daily_diff = np.diff(daily, axis=1)                          # (NW, W-1)
    daily_diff_std = daily_diff.std(axis=1)
    rel_drop = (daily[:, :-1] - daily[:, 1:]) / (daily[:, :-1] + eps)
    max_daily_drop = rel_drop.max(axis=1)
    mean_daily_drop = rel_drop.mean(axis=1)

    win_mean = daily.mean(axis=1, keepdims=True)
    frac_low = (daily < 0.6 * win_mean).mean(axis=1)
    min_max_ratio = daily.min(axis=1) / (daily.max(axis=1) + eps)

    if w >= 3:
        d0, d1 = daily[:, :-1], daily[:, 1:]
        d0c, d1c = d0 - d0.mean(axis=1, keepdims=True), d1 - d1.mean(axis=1, keepdims=True)
        num = (d0c * d1c).sum(axis=1)
        den = np.sqrt((d0c**2).sum(axis=1) * (d1c**2).sum(axis=1)) + eps
        acf1 = num / den
    else:
        acf1 = np.zeros(nw)

    m2 = ((flat - mean[:, None]) ** 2).mean(axis=1)
    m3 = ((flat - mean[:, None]) ** 3).mean(axis=1)
    m4 = ((flat - mean[:, None]) ** 4).mean(axis=1)
    std_safe = np.sqrt(m2) + eps
    skew = m3 / std_safe**3
    kurt = m4 / std_safe**4 - 3.0

    # calendar: dow per window position → weekend fraction over the window
    if dow.ndim == 1 and dow.shape[0] >= w:
        # caller passed the full consumer day-of-week panel; align each window
        # by using the first w entries of the corresponding window. The caller
        # passes the per-window dow slice via ``dow`` whenever possible.
        dow_win = np.tile(dow[:w][None, :], (nw, 1))
    elif dow.ndim == 2 and dow.shape[0] == nw:
        dow_win = dow
    else:
        dow_win = np.zeros((nw, w), dtype=int)
    is_weekend = dow_win >= 5
    weekday_mask = ~is_weekend
    weekend_cnt = is_weekend.sum(axis=1)
    weekday_cnt = weekday_mask.sum(axis=1)
    weekend_mean = (daily * is_weekend).sum(axis=1) / np.maximum(weekend_cnt, 1)
    weekday_mean = (daily * weekday_mask).sum(axis=1) / np.maximum(weekday_cnt, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        weekend_ratio = np.where(weekend_cnt > 0, weekend_mean / (weekday_mean + eps), 1.0)
    weekend_ratio = np.nan_to_num(weekend_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    weekend_frac = is_weekend.mean(axis=1)
    start_dow = dow_win[:, 0]
    dow_sin = np.sin(2 * np.pi * start_dow / 7.0)
    dow_cos = np.cos(2 * np.pi * start_dow / 7.0)

    return np.column_stack(
        [
            mean, std, vmin, vmax, median,
            peak_offpeak, night_day, load_var,
            trend, daily_diff_std, max_daily_drop, mean_daily_drop,
            frac_low, min_max_ratio, acf1, skew, kurt,
            weekend_ratio, weekend_frac, dow_sin, dow_cos,
        ]
    )


def _consumer_panel(df: pd.DataFrame, steps_per_day: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Reshape one consumer's rows into (n_complete_days, steps) + day dates."""
    times = df["timestamp"].to_numpy()
    values = df["consumption"].to_numpy(dtype=np.float64, copy=True)
    dates = pd.DatetimeIndex(pd.to_datetime(times)).date
    unique_dates = pd.unique(dates)
    n_days = len(unique_dates)
    if n_days * steps_per_day != len(values):
        # ragged series: keep only days with a complete set of readings
        counts = pd.Series(dates).value_counts()
        good = set(counts[counts == steps_per_day].index)
        keep = np.isin(dates, list(good))
        if not keep.any():
            return None, None
        values, dates = values[keep], np.asarray(dates)[keep]
        unique_dates = pd.unique(dates)
        n_days = len(unique_dates)
        if n_days * steps_per_day != len(values):
            return None, None
    return values.reshape(n_days, steps_per_day), np.asarray(unique_dates)


def build_window_dataset(
    long_df: pd.DataFrame,
    labels: pd.Series,
    cfg: dict,
    consumer_meta: pd.DataFrame | None = None,
    steps_per_day: int | None = None,
) -> WindowDataset:
    """Build the unscaled window tensors for a set of consumers.

    Scaling is applied later by :mod:`preprocessing.pipeline` using
    train-fitted scalers (prevents leakage).
    """
    window_days = int(get(cfg, "preprocessing.features.window_days", 14))
    stride = int(get(cfg, "preprocessing.features.stride_days", 3))
    include_calendar = bool(get(cfg, "preprocessing.features.include_calendar", True))
    peak_hours = get(cfg, "preprocessing.features.peak_hours", [17, 22])
    night_hours = get(cfg, "preprocessing.features.night_hours", [0, 6])
    interval = int(get(cfg, "data.synthetic.interval_minutes", 30))

    if steps_per_day is None:
        counts = long_df.groupby(["consumer_id", long_df["timestamp"].dt.date], sort=False).size()
        steps_per_day = int(counts.mode().iloc[0]) if len(counts) else int(1440 // interval)
    sph = max(1, steps_per_day // 24)
    peak_slice = slice(int(peak_hours[0]) * sph, int(peak_hours[1]) * sph)
    night_slice = slice(int(night_hours[0]) * sph, int(night_hours[1]) * sph)
    peak_mask = np.zeros(steps_per_day, dtype=bool)
    night_mask = np.zeros(steps_per_day, dtype=bool)
    peak_mask[peak_slice] = True
    night_mask[night_slice] = True

    all_blocks: list[np.ndarray] = []
    all_engine: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []
    dropped = 0

    df = long_df.sort_values(["consumer_id", "timestamp"], kind="stable")
    for consumer_id,cdf in df.groupby("consumer_id", sort=False):
        panel, dates = _consumer_panel(cdf, steps_per_day)
        if panel is None:
            dropped += 1
            continue
        n_days = panel.shape[0]
        starts = _window_starts(n_days, window_days, stride)
        if starts.size == 0:
            dropped += 1
            continue
        label = int(labels.get(consumer_id, 0))
        theft_start = -1
        if consumer_meta is not None and len(consumer_meta) and consumer_meta.index.name == "consumer_id":
            if consumer_id in consumer_meta.index:
                theft_start = int(consumer_meta.loc[consumer_id].get("theft_start_day", -1))

        win = np.stack([panel[s : s + window_days] for s in starts])          # (NW, W, steps)
        daily = win.mean(axis=2)                                              # (NW, W)
        # block means: split intraday axis into N_BLOCKS chunks
        blocks = np.array_split(win, N_BLOCKS, axis=2)
        block_means = np.concatenate([b.mean(axis=2) for b in blocks], axis=1)  # (NW, W*N_BLOCKS)

        dow_panel = pd.to_datetime(dates).dayofweek.to_numpy()
        dow_windows = np.stack([dow_panel[s : s + window_days] for s in starts])
        engine = _window_engineered(win, daily, peak_mask, night_mask, dow_windows)
        if not include_calendar:
            engine = engine[:, :17]

        end_days = starts + window_days - 1
        if theft_start >= 0:
            y = (end_days >= theft_start).astype(int)
        else:
            y = np.full(starts.size, label, dtype=int)

        day_idx = np.arange(window_days)
        meta_rows.append(
            pd.DataFrame(
                {
                    "consumer_id": str(consumer_id),
                    "start_day": starts,
                    "end_day": end_days,
                    "start_date": [str(pd.Timestamp(dates[s]).date()) for s in starts],
                    "label": y,
                    "consumer_label": label,
                }
            )
        )
        all_blocks.append(block_means.astype(np.float32))
        all_engine.append(engine.astype(np.float32))
        all_y.append(y)

    if not all_blocks:
        raise ValueError("No usable windows produced (consumers shorter than window_days?)")

    blocks_mat = np.concatenate(all_blocks, axis=0)
    engine_mat = np.concatenate(all_engine, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.concat(meta_rows, ignore_index=True)

    # sequence tensor (NW, W, steps) rebuilt lazily per consumer to cap memory:
    # we recompute windows here in the same order as above.
    seq_parts: list[np.ndarray] = []
    for consumer_id, cdf in df.groupby("consumer_id", sort=False):
        panel, dates = _consumer_panel(cdf, steps_per_day)
        if panel is None:
            continue
        starts = _window_starts(panel.shape[0], window_days, stride)
        if starts.size == 0:
            continue
        seq_parts.append(np.stack([panel[s : s + window_days] for s in starts]).astype(np.float32))
    X_seq = np.concatenate(seq_parts, axis=0)

    block_names = [f"block_{i}" for i in range(window_days * N_BLOCKS)]
    engine_names = ENGINEERED_FEATURE_NAMES[: engine_mat.shape[1]]
    return WindowDataset(
        X_matrix=np.concatenate([blocks_mat, engine_mat], axis=1),
        X_seq=X_seq,
        y=y,
        feature_names=block_names + engine_names,
        meta=meta,
        block_slice=slice(0, blocks_mat.shape[1]),
        engine_slice=slice(blocks_mat.shape[1], blocks_mat.shape[1] + engine_mat.shape[1]),
        stats={
            "n_samples": int(y.shape[0]),
            "n_features_matrix": int(blocks_mat.shape[1] + engine_mat.shape[1]),
            "n_engineered": int(engine_mat.shape[1]),
            "window_days": window_days,
            "stride_days": stride,
            "steps_per_day": steps_per_day,
            "consumers_dropped": dropped,
            "theft_ratio": float(y.mean()),
        },
    )
