"""Leakage-safe data preparation pipeline.

Workflow (see AGENTS.md / docs/methodology.md):

```
Raw Dataset
   ↓  cleaning (per-consumer, unsupervised)
Consumer-level stratified split  (train / val / test consumers)
   ↓  fit outlier bounds + scalers on TRAIN consumers only
Transform all splits with the train-fitted objects
   ↓  window feature engineering
WindowDataset: train / val / test
```

Consumers — not windows — are the split unit, so a consumer's windows never
appear in two splits (no identity leakage across federated clients and test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.cleaning import clean_dataset
from preprocessing.features import WindowDataset, build_window_dataset
from preprocessing.imbalance import compute_class_weights, imbalance_report
from preprocessing.loader import DatasetBundle, load_dataset
from preprocessing.normalization import FeatureScaler, ValueScaler
from preprocessing.outliers import OutlierClipper
from utilities.config import get, resolve_path
from utilities.seed import make_rng


@dataclass
class DataSplits:
    """Everything the experiment drivers need after preprocessing."""

    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    labels: pd.Series
    train_consumers: list[str]
    val_consumers: list[str]
    test_consumers: list[str]
    source: str
    is_synthetic: bool
    bundle_stats: dict[str, Any] = field(default_factory=dict)
    cleaning_report: dict[str, Any] = field(default_factory=dict)
    imbalance: dict[str, Any] = field(default_factory=dict)
    preprocessing_report: dict[str, Any] = field(default_factory=dict)
    value_scaler: ValueScaler | None = None
    feature_scaler: FeatureScaler | None = None
    outlier_clipper: OutlierClipper | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "is_synthetic": self.is_synthetic,
            "n_consumers": {
                "train": len(self.train_consumers),
                "val": len(self.val_consumers),
                "test": len(self.test_consumers),
            },
            "n_windows": {"train": len(self.train), "val": len(self.val), "test": len(self.test)},
            "theft_ratio": {
                "train": self.train.theft_ratio,
                "val": self.val.theft_ratio,
                "test": self.test.theft_ratio,
            },
            "n_features": int(self.train.X_matrix.shape[1]),
            "n_engineered": int(self.train.engine_slice.stop - self.train.engine_slice.start),
            "imbalance": self.imbalance,
            "outlier_bounds": (self.outlier_clipper.stats if self.outlier_clipper else None),
        }


def split_consumers(labels: pd.Series, cfg: dict, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Stratified consumer-level split into train / val / test id lists."""
    test_size = float(get(cfg, "split.test_size", 0.20))
    val_size = float(get(cfg, "split.val_size", 0.15))
    stratify = bool(get(cfg, "split.stratify", True))
    ids = labels.index.to_numpy()
    y = labels.to_numpy()
    strat = y if stratify and np.unique(y).size > 1 else None
    train_val_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed, stratify=strat)
    strat_tv = (labels.loc[train_val_ids].to_numpy() if stratify else None)
    if strat_tv is not None and np.unique(strat_tv).size > 1:
        rel_val = val_size / max(1e-9, (1.0 - test_size))
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=rel_val, random_state=seed, stratify=strat_tv
        )
    else:
        train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, random_state=seed, stratify=None)
    return sorted(map(str, train_ids)), sorted(map(str, val_ids)), sorted(map(str, test_ids))


def prepare_data(
    cfg: dict,
    seed: int = 42,
    log: Callable[[str], None] | None = None,
    bundle: DatasetBundle | None = None,
) -> DataSplits:
    """Run the full pipeline and return train/val/test window datasets."""
    say = log or (lambda _msg: None)

    if bundle is None:
        say("Loading dataset...")
        bundle = load_dataset(cfg, seed=seed)
    say(f"Dataset: {bundle.source} ({len(bundle.labels)} consumers, {len(bundle.long_df)} rows)")

    # Split consumers BEFORE fitting any global preprocessing statistic. This
    # is the leakage-prevention boundary: nothing derived from validation or
    # test consumers may influence cleaning/scaling used by training.
    say("Splitting consumers (stratified train/val/test)...")
    train_ids, val_ids, test_ids = split_consumers(bundle.labels, cfg, seed)
    report: dict[str, Any] = {
        "split": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_unit": "consumer",
    }

    raw_train = bundle.long_df.loc[
        bundle.long_df["consumer_id"].isin(train_ids), "consumption"
    ].to_numpy(dtype=np.float64)
    raw_train = raw_train[np.isfinite(raw_train) & (raw_train >= 0)]
    train_median = float(np.median(raw_train)) if raw_train.size else None

    say("Cleaning (duplicates / invalid / noise / missing; train-fitted imputation)...")
    cleaned_df, cleaning_report = clean_dataset(bundle.long_df, cfg, global_median=train_median)

    train_mask = cleaned_df["consumer_id"].isin(train_ids)
    train_rows = cleaned_df.loc[train_mask]

    say("Fitting outlier bounds and scalers on TRAIN consumers...")
    clipper = OutlierClipper(cfg)
    train_values = train_rows["consumption"].to_numpy(dtype=np.float64)
    finite_train = train_values[np.isfinite(train_values)]
    rng = make_rng(seed + 1)
    if finite_train.size > 500_000:  # cap for speed; still representative
        finite_train = rng.choice(finite_train, size=500_000, replace=False)
    clipper.fit(finite_train)

    value_scaler = ValueScaler.from_config(cfg).fit(finite_train)
    feature_scaler: FeatureScaler | None = None

    def _transform_split(ids: list[str], is_train: bool) -> WindowDataset:
        nonlocal feature_scaler
        sub = cleaned_df[cleaned_df["consumer_id"].isin(ids)]
        values = clipper.transform(sub["consumption"].to_numpy())
        sub = sub.copy()
        sub["consumption"] = values
        ds = build_window_dataset(sub, bundle.labels, cfg, consumer_meta=bundle.consumer_meta)
        # scale: blocks + sequences use the value scaler; engineered columns get
        # their own scaler (fitted on train only)
        blocks = value_scaler.transform(ds.X_matrix[:, ds.block_slice])
        seq = value_scaler.transform(ds.X_seq)
        engine = ds.X_matrix[:, ds.engine_slice]
        if is_train:
            feature_scaler = FeatureScaler.from_config(cfg).fit(engine)
        assert feature_scaler is not None
        engine = feature_scaler.transform(engine)
        ds.X_matrix = np.concatenate([blocks, engine], axis=1).astype(np.float32)
        ds.X_seq = seq
        return ds

    say("Building train windows...")
    train_ds = _transform_split(train_ids, is_train=True)
    say("Building validation windows...")
    val_ds = _transform_split(val_ids, is_train=False)
    say("Building test windows...")
    test_ds = _transform_split(test_ids, is_train=False)

    imbalance = {
        "report": compute_class_weights(train_ds.y, cfg),
        "splits": imbalance_report({"train": train_ds.y, "val": val_ds.y, "test": test_ds.y}),
    }
    report["outliers"] = dict(clipper.stats)
    report["outlier_clip_fraction_train"] = clipper.clip_fraction(train_values)
    report["value_scaler"] = value_scaler.state()
    report["cleaning"] = cleaning_report
    report["label_rule"] = (
        "window-level labels respect theft_start_day (synthetic)"
        if bundle.is_synthetic
        else "consumer-level labels (real dataset)"
    )

    say(
        f"Windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | "
        f"features={train_ds.X_matrix.shape[1]}"
    )
    return DataSplits(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        labels=bundle.labels,
        train_consumers=train_ids,
        val_consumers=val_ids,
        test_consumers=test_ids,
        source=bundle.source,
        is_synthetic=bundle.is_synthetic,
        bundle_stats=dict(bundle.stats),
        cleaning_report=cleaning_report,
        imbalance=imbalance,
        preprocessing_report=report,
        value_scaler=value_scaler,
        feature_scaler=feature_scaler,
        outlier_clipper=clipper,
    )


def save_processed(splits: DataSplits, cfg: dict) -> "Path":  # noqa: F821
    """Cache processed arrays under data/processed for inspection/reuse."""
    out_dir = resolve_path(cfg, "data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "windows.npz"
    np.savez_compressed(
        path,
        X_train=splits.train.X_matrix,
        y_train=splits.train.y,
        X_val=splits.val.X_matrix,
        y_val=splits.val.y,
        X_test=splits.test.X_matrix,
        y_test=splits.test.y,
        feature_names=np.array(splits.train.feature_names, dtype=object),
    )
    return path
