"""Experiment: communication / accuracy trade-off under update compression."""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from evaluation.communication import reduction_percentage
from experiments.common import ExperimentWriter
from experiments.fedavg import run_federated_experiment
from preprocessing.pipeline import DataSplits
from utilities.config import get


def run_communication_sweep(
    cfg: dict,
    splits: DataSplits,
    writer_factory: Callable[[str], ExperimentWriter],
    seed: int = 42,
    sweep: list[dict[str, Any]] | None = None,
    rounds: int | None = None,
    partitions=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sweep = sweep or list(
        get(
            cfg,
            "experiments.communication_sweep",
            [
                {"name": "none", "compression": False, "method": "none"},
                {"name": "quantize_8bit", "compression": True, "method": "quantize"},
                {"name": "topk_20pct", "compression": True, "method": "topk"},
                {"name": "topk_quantize", "compression": True, "method": "topk_quantize"},
            ],
        )
    )
    rounds = int(rounds if rounds is not None else get(cfg, "experiments.communication_sweep_rounds", 4))
    rows: list[dict[str, Any]] = []

    for spec in sweep:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["communication"]["compression"] = bool(spec.get("compression", True))
        run_cfg["communication"]["method"] = str(spec.get("method", "none"))
        run_cfg["communication"]["compress_broadcast"] = bool(spec.get("compress_broadcast", spec.get("compression", True)))
        writer = writer_factory(f"communication_{spec['name']}")
        row, _ = run_federated_experiment(
            run_cfg, splits, writer, method="proposed", seed=seed, rounds=rounds, partitions=partitions
        )
        row["compression_label"] = str(spec["name"])
        rows.append(row)
        writer.save_json(row, "sweep_row.json")

    reference = float(rows[0]["total_bytes"]) if rows else float("nan")
    for row in rows:
        row["communication_reduction_pct"] = reduction_percentage(reference, float(row["total_bytes"]))
    return rows, {
        "reference": rows[0]["compression_label"] if rows else "NOT RUN",
        "reference_total_bytes": int(reference) if np.isfinite(reference) else None,
        "rounds": rounds,
        "method": "upload top-k / int8 quantization; server broadcast quantized when compression enabled",
        "note": "measured payload bytes include per-tensor headers and indices",
    }
