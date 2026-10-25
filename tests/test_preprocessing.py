"""Tests for preprocessing pipeline, missing-value handling, outlier clipping, and data leakage prevention."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.cleaning import clean_dataset
from preprocessing.features import build_window_dataset
from preprocessing.imbalance import class_distribution, compute_class_weights
from preprocessing.loader import generate_synthetic_dataset
from preprocessing.normalization import FeatureScaler, ValueScaler
from preprocessing.outliers import OutlierClipper
from preprocessing.pipeline import prepare_data
from utilities.config import load_config


def test_synthetic_data_generation():
    cfg = {"data": {"synthetic": {"consumers": 10, "days": 30, "theft_rate": 0.2}}}
    bundle = generate_synthetic_dataset(cfg, seed=42)
    assert not bundle.long_df.empty
    assert "consumer_id" in bundle.long_df.columns
    assert "timestamp" in bundle.long_df.columns
    assert "consumption" in bundle.long_df.columns
    assert bundle.theft_rate > 0.0


def test_cleaning_no_nan_or_inf():
    cfg = {"data": {"synthetic": {"consumers": 5, "days": 10, "theft_rate": 0.2}}}
    bundle = generate_synthetic_dataset(cfg, seed=42)
    df = bundle.long_df.copy()
    # Insert deliberate NaNs and Infs
    df.loc[0, "consumption"] = np.nan
    df.loc[1, "consumption"] = np.inf

    cleaned, report = clean_dataset(df, cfg={"preprocessing": {"missing": {"strategy": "interpolate"}, "noise": {"enabled": False}}})
    assert not cleaned["consumption"].isna().any()
    assert not np.isinf(cleaned["consumption"].to_numpy(dtype=float)).any()


def test_outlier_clipping_does_not_drop_theft_anomalies():
    # Outliers must clip, NEVER drop legitimate theft records
    handler = OutlierClipper(cfg={"preprocessing": {"outliers": {"method": "iqr", "iqr_multiplier": 1.5, "action": "clip"}}})
    X = np.array([1.0, 2.0, 1.5, 1.8, 100.0], dtype=np.float32)
    handler.fit(X)
    X_trans = handler.transform(X)
    assert len(X_trans) == len(X), "Outlier handler dropped rows instead of clipping"
    assert X_trans[-1] < 100.0, "Extreme value was not clipped"


def test_data_leakage_prevention():
    cfg = load_config("quick")
    cfg["data"]["synthetic"]["consumers"] = 30
    cfg["data"]["synthetic"]["days"] = 30
    splits = prepare_data(cfg, seed=42)

    # Train, val, test consumers must be strictly disjoint
    train_cons = set(splits.train.meta["consumer_id"])
    val_cons = set(splits.val.meta["consumer_id"])
    test_cons = set(splits.test.meta["consumer_id"])

    assert len(train_cons.intersection(val_cons)) == 0, "Train and Val share consumers!"
    assert len(train_cons.intersection(test_cons)) == 0, "Train and Test share consumers!"
    assert len(val_cons.intersection(test_cons)) == 0, "Val and Test share consumers!"


def test_class_imbalance():
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1])
    imb = class_distribution(y)
    assert imb["total"] == 8
    assert imb["theft_samples"] == 2
    assert imb["normal_samples"] == 6

    weights = compute_class_weights(y, cfg={})
    assert weights["weights"][1] > weights["weights"][0]
