"""Tests for client simulation: IID and Non-IID partitioning."""
from __future__ import annotations

import numpy as np
import pytest

from federated.partition import build_partition, heterogeneity_report
from preprocessing.pipeline import prepare_data
from utilities.config import load_config


@pytest.fixture(scope="module")
def sample_splits():
    cfg = load_config("quick")
    cfg["data"]["synthetic"]["consumers"] = 40
    cfg["data"]["synthetic"]["days"] = 30
    return prepare_data(cfg, seed=42)


def test_iid_partition(sample_splits):
    cfg = {"clients": {"num_clients": 4, "partition_strategy": "iid"}}
    partitions = build_partition(sample_splits.train, cfg, seed=42)
    assert len(partitions) == 4

    total_samples = sum(len(p.indices) for p in partitions)
    assert total_samples == len(sample_splits.train.y)

    # Disjoint indices
    all_indices = set()
    for p in partitions:
        assert len(all_indices.intersection(p.indices)) == 0
        all_indices.update(p.indices)


def test_non_iid_partition(sample_splits):
    cfg = {
        "clients": {
            "num_clients": 4,
            "partition_strategy": "non_iid",
            "dirichlet_alpha": 0.5,
            "quantity_skew": 0.6,
            "min_samples_per_client": 2,
        }
    }
    partitions = build_partition(sample_splits.train, cfg, seed=42)
    assert len(partitions) == 4

    # Every training window must appear in exactly one client. Quantity skew
    # moves consumers; it must never silently discard training data.
    all_indices = np.concatenate([p.indices for p in partitions])
    assert len(all_indices) == len(sample_splits.train.y)
    assert len(np.unique(all_indices)) == len(all_indices)
    assigned_consumers = set()
    for p in partitions:
        assert assigned_consumers.isdisjoint(p.consumer_ids)
        assigned_consumers.update(p.consumer_ids)
    assert assigned_consumers == set(map(str, sample_splits.train_consumers))

    # Calculate heterogeneity
    report = heterogeneity_report(partitions, sample_splits.train)
    assert "theft_ratio_std" in report
    assert "size_cv" in report


def test_no_raw_consumption_in_partition():
    # Partition summary should only hold metadata and statistics, not raw consumption
    cfg = load_config("quick")
    cfg["data"]["synthetic"]["consumers"] = 20
    cfg["data"]["synthetic"]["days"] = 20
    splits = prepare_data(cfg, seed=42)
    partitions = build_partition(splits.train, cfg, seed=42)

    for p in partitions:
        summary = p.summary(splits.train)
        assert "raw_consumption" not in summary
        assert "X" not in summary
