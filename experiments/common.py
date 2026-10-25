"""Shared experiment orchestration: run directories, metadata, evaluation.

Every experiment run gets a unique ID, saves its config + metrics + logs under
``results/raw/<run_id>/``, and contributes rows to the aggregate tables under
``results/``.
"""
from __future__ import annotations

import json
import platform
import sys
import time

# Windows consoles often default to a legacy code page (e.g. cp1252). The
# project logs occasional UTF-8 symbols; never let console encoding crash an
# experiment. Files are always written explicitly as UTF-8.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from evaluation.metrics import (
    compute_classification_metrics,
    confusion_matrix_rows,
    pr_curve_points,
    roc_curve_points,
)
from preprocessing.pipeline import DataSplits, prepare_data
from utilities.config import get, resolve_path
from utilities.seed import set_seed
from utilities.system import environment_info


@dataclass
class ExperimentWriter:
    """Writes artifacts for a single experiment run."""

    run_dir: Path
    figures_dir: Path
    log_path: Path
    _log_handle: Any = field(default=None, repr=False)

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save_json(self, obj: Any, name: str = "metrics.json") -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
        return path

    def save_figure(self, fig_path: Path) -> Path:
        """Copy a figure into the run's figures dir (keeps results/figures canonical)."""
        import shutil

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        target = self.figures_dir / fig_path.name
        if fig_path.resolve() != target.resolve():
            shutil.copyfile(fig_path, target)
        return target


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def new_run_id(name: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return f"experiment_{stamp}_{safe_name}_s{seed}"


def create_writer(cfg: dict, name: str, seed: int) -> ExperimentWriter:
    root = resolve_path(cfg, get(cfg, "experiment.output_dir", "results"))
    run_id = new_run_id(name, seed)
    run_dir = root / "raw" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = ExperimentWriter(
        run_dir=run_dir,
        figures_dir=run_dir / "figures",
        log_path=run_dir / "logs.txt",
    )
    writer.log(f"run {run_id} created")
    return writer


def save_metadata(cfg: dict, writer: ExperimentWriter, dataset_stats: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    meta = {
        "run_dir": str(writer.run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": get(cfg, "experiment.seed"),
        "mode": get(cfg, "experiment.mode"),
        "environment": environment_info(),
        "dataset": dataset_stats,
        "config": cfg,
    }
    if extra:
        meta.update(extra)
    return writer.save_json(meta, "experiment_metadata.json")


def prepare_splits(cfg: dict, seed: int, log: Callable[[str], None] | None = None) -> DataSplits:
    set_seed(seed)
    return prepare_data(cfg, seed=seed, log=log)


def evaluate_model_on(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    with_curves: bool = True,
) -> dict[str, Any]:
    """Full metric bundle for one model on one split."""
    probs = np.asarray(model.predict_proba(X), dtype=np.float64)
    result: dict[str, Any] = {"metrics": compute_classification_metrics(y, probs, threshold)}
    result["confusion_matrix"] = confusion_matrix_rows(y, probs, threshold)
    if with_curves:
        result["roc"] = roc_curve_points(y, probs)
        result["pr"] = pr_curve_points(y, probs)
    return result


def training_time_of(history: list[dict[str, Any]]) -> float:
    return float(sum(h.get("duration_s", 0.0) for h in history))


def system_snapshot() -> dict[str, Any]:
    info = environment_info()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ram_gb": info.get("ram_gb"),
        "gpu": info.get("gpu"),
        "logical_cores": info.get("logical_cores"),
    }
