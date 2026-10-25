"""Experiment: privacy / utility trade-off sweep.

Each run holds everything else constant (same data, partitions, adaptive
aggregation, compression, rounds) and varies the Gaussian noise multiplier.
An additional ``privacy off`` run provides the utility reference.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from experiments.common import ExperimentWriter
from experiments.fedavg import run_federated_experiment
from preprocessing.pipeline import DataSplits
from utilities.config import get


def run_privacy_sweep(
    cfg: dict,
    splits: DataSplits,
    writer_factory: Callable[[str], ExperimentWriter],
    seed: int = 42,
    noise_values: list[float] | None = None,
    rounds: int | None = None,
    partitions=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    noise_values = noise_values or list(get(cfg, "experiments.privacy_sweep", [0.0, 0.001, 0.003, 0.006]))
    rounds = int(rounds if rounds is not None else get(cfg, "experiments.privacy_sweep_rounds", 4))
    rows: list[dict[str, Any]] = []

    # Reference: same proposed pipeline without clipping/noise.
    off_cfg = copy.deepcopy(cfg)
    off_cfg["privacy"]["enabled"] = False
    off_cfg["privacy"]["secure_aggregation"] = False
    writer = writer_factory("privacy_off")
    row, _ = run_federated_experiment(
        off_cfg, splits, writer, method="proposed", seed=seed, rounds=rounds, partitions=partitions
    )
    row["privacy_level"] = "off"
    row["privacy_off"] = True
    row["noise_multiplier"] = 0.0
    row["epsilon_estimate"] = None
    rows.append(row)
    writer.save_json(row, "sweep_row.json")

    for sigma in noise_values:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["privacy"]["enabled"] = True
        run_cfg["privacy"]["mechanism"] = "dp"
        run_cfg["privacy"]["noise_multiplier"] = float(sigma)
        run_cfg["privacy"]["secure_aggregation"] = False
        writer = writer_factory(f"privacy_noise_{sigma:g}")
        row, artifacts = run_federated_experiment(
            run_cfg, splits, writer, method="proposed", seed=seed, rounds=rounds, partitions=partitions
        )
        row["privacy_level"] = f"noise_{sigma:g}"
        row["privacy_off"] = False
        row["epsilon_estimate"] = artifacts["result"].get("privacy", {}).get("epsilon_single_round_estimate")
        rows.append(row)
        writer.save_json(row, "sweep_row.json")

    reference = rows[0]["f1"] if rows else float("nan")
    for row in rows[1:]:
        row["f1_change_vs_privacy_off"] = float(row["f1"] - reference)

    return rows, {
        "reference": "privacy off (no clipping, no noise)",
        "noise_multipliers": noise_values,
        "rounds": rounds,
        "caveat": (
            "epsilon values are single-release classical Gaussian-mechanism estimates; "
            "they do not account for composition across rounds and secure aggregation "
            "is simulated, so no end-to-end DP guarantee is claimed."
        ),
    }
