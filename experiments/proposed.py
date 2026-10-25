"""Experiment C — PowerShield-FL (proposed).

PowerShield-FL = lightweight MLP + DP-style update protection + communication
compression + adaptive aggregation + explainability. The federated runner is
shared with FedAvg so both models see identical data partitions and rounds.
"""
from __future__ import annotations

from typing import Any

from experiments.common import ExperimentWriter
from experiments.fedavg import run_federated_experiment
from preprocessing.pipeline import DataSplits


def run_proposed(
    cfg: dict,
    splits: DataSplits,
    writer: ExperimentWriter,
    seed: int = 42,
    rounds: int | None = None,
    partitions=None,
    save_dashboard_samples: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_federated_experiment(
        cfg,
        splits,
        writer,
        method="proposed",
        seed=seed,
        rounds=rounds,
        partitions=partitions,
        save_dashboard_samples=save_dashboard_samples,
    )
