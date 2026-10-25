"""Experiment: scalability across simulated client counts."""
from __future__ import annotations

import copy
from typing import Any, Callable

from evaluation.scalability import scalability_summary
from experiments.common import ExperimentWriter
from experiments.fedavg import run_federated_experiment
from preprocessing.pipeline import DataSplits
from utilities.config import get


def run_scalability_sweep(
    cfg: dict,
    splits: DataSplits,
    writer_factory: Callable[[str], ExperimentWriter],
    seed: int = 42,
    client_counts: list[int] | None = None,
    rounds: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client_counts = client_counts or list(get(cfg, "experiments.scalability_clients", [2, 5, 10]))
    rounds = int(rounds if rounds is not None else get(cfg, "experiments.scalability_rounds", 4))
    rows: list[dict[str, Any]] = []
    for n_clients in client_counts:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["clients"]["num_clients"] = int(n_clients)
        writer = writer_factory(f"scalability_{n_clients}clients")
        row, _ = run_federated_experiment(
            run_cfg, splits, writer, method="proposed", seed=seed, rounds=rounds, partitions=None
        )
        row["clients"] = int(n_clients)
        rows.append(row)
        writer.save_json(row, "sweep_row.json")
    return rows, {
        "client_counts": client_counts,
        "rounds": rounds,
        "summary": scalability_summary(rows),
        "caveat": "wall time and memory are machine-specific and measured on this CPU-only host",
    }
