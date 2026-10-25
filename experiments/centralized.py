"""Experiment A — centralized ML baselines (pooled data, no federation)."""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from evaluation.metrics import tune_threshold
from evaluation.performance import measure_inference_time, model_resource_report
from experiments.common import ExperimentWriter, evaluate_model_on, training_time_of
from models.baselines import build_central_models, fit_central_model
from models.cnn_lstm import build_cnn_lstm
from models.lightweight import build_lightweight
from preprocessing.pipeline import DataSplits
from utilities.config import get


def _row(name: str, kind: str, evaluation: dict[str, Any], train_time_s: float, resources: dict[str, Any], seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    m = evaluation["metrics"]
    row = {
        "method": name,
        "kind": kind,
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "roc_auc": m.get("roc_auc", float("nan")),
        "pr_auc": m.get("pr_auc", float("nan")),
        "threshold": m["threshold"],
        "train_time_s": float(train_time_s),
        "n_params": resources.get("n_parameters", 0),
        "model_size_kb": resources.get("model_size_kb", float("nan")),
        "inference_per_sample_ms": (resources.get("inference") or {}).get("per_sample_ms", float("nan")),
        "inference_peak_memory_mb": float(resources.get("inference_peak_memory_mb", float("nan"))),
        "rounds": 0,
        "clients": 0,
        "total_bytes": 0,
        "seed": seed,
    }
    if extra:
        row.update(extra)
    return row


def run_centralized_baselines(
    cfg: dict,
    splits: DataSplits,
    writer: ExperimentWriter,
    seed: int = 42,
    include_lightweight_central: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train sklearn/torch baselines centrally; returns (rows, artifacts)."""
    tr, va, te = splits.train, splits.val, splits.test
    class_weights = splits.imbalance["report"]
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"curves": {}, "thresholds": {}}

    writer.log("Centralized baselines: training on pooled data...")
    models = build_central_models(cfg, tr.X_matrix.shape[1], seed=seed)
    for name, model in models.items():
        writer.log(f"  training {name}...")
        fit_start = time.perf_counter()
        history = fit_central_model(model, tr.X_matrix, tr.y, cfg, class_weights, seed=seed)
        train_time = time.perf_counter() - fit_start

        prob_val = model.predict_proba(va.X_matrix)
        thr = tune_threshold(va.y, prob_val) if get(cfg, "evaluation.tune_threshold", True) else {"threshold": 0.5}
        threshold = float(thr["threshold"])
        evaluation = evaluate_model_on(model, te.X_matrix, te.y, threshold=threshold)
        resources = model_resource_report(model, X=te.X_matrix[:512])
        row = _row(f"centralized_{name}", "centralized", evaluation, train_time, resources, seed, extra={"tuned_threshold": threshold})
        rows.append(row)
        artifacts["curves"][f"centralized_{name}"] = evaluation
        artifacts["thresholds"][name] = threshold
        artifacts[f"model_{name}"] = model
        writer.log(f"  {name}: f1={row['f1']:.4f} auc={row['roc_auc']:.4f} train={train_time:.1f}s")

    # lightweight model trained centrally (reference for federated variants)
    if include_lightweight_central:
        lw = build_lightweight(cfg, tr.X_matrix.shape[1], seed=seed)
        fit_start = time.perf_counter()
        history = lw.fit(
            tr.X_matrix,
            tr.y,
            epochs=int(get(cfg, "model.central_mlp_epochs", 8) or 8),
            batch_size=int(get(cfg, "federated.batch_size", 128)),
            class_weights=class_weights,
            seed=seed,
        )
        train_time = time.perf_counter() - fit_start
        prob_val = lw.predict_proba(va.X_matrix)
        thr = tune_threshold(va.y, prob_val) if get(cfg, "evaluation.tune_threshold", True) else {"threshold": 0.5}
        threshold = float(thr["threshold"])
        evaluation = evaluate_model_on(lw, te.X_matrix, te.y, threshold=threshold)
        resources = model_resource_report(lw, X=te.X_matrix[:512])
        row = _row("centralized_lightweight_mlp", "centralized", evaluation, train_time, resources, seed, extra={"tuned_threshold": threshold})
        rows.append(row)
        artifacts["curves"]["centralized_lightweight_mlp"] = evaluation
        artifacts["thresholds"]["lightweight"] = threshold
        artifacts["model_lightweight"] = lw
        writer.log(f"  lightweight(central): f1={row['f1']:.4f} train={train_time:.1f}s")

    # CNN-LSTM baseline (heavier model) on the sequence tensor
    if bool(get(cfg, "experiments.cnn_lstm_enabled", True)):
        writer.log("  training cnn_lstm baseline (sequence input)...")
        cnn = build_cnn_lstm(cfg, window_days=tr.X_seq.shape[1], steps_per_day=tr.X_seq.shape[2], seed=seed)
        epochs = int(get(cfg, "model.cnn_lstm.epochs", 6))
        fit_start = time.perf_counter()
        cnn.fit(
            tr.X_seq,
            tr.y,
            epochs=epochs,
            batch_size=int(get(cfg, "model.cnn_lstm.batch_size", 128)),
            class_weights=class_weights,
            seed=seed,
        )
        train_time = time.perf_counter() - fit_start
        prob_val = cnn.predict_proba(va.X_seq)
        thr = tune_threshold(va.y, prob_val) if get(cfg, "evaluation.tune_threshold", True) else {"threshold": 0.5}
        threshold = float(thr["threshold"])
        evaluation = evaluate_model_on(cnn, te.X_seq, te.y, threshold=threshold)
        resources = model_resource_report(cnn, X=te.X_seq[:256])
        row = _row("cnn_lstm_baseline", "centralized", evaluation, train_time, resources, seed, extra={"tuned_threshold": threshold, "epochs": epochs})
        rows.append(row)
        artifacts["curves"]["cnn_lstm_baseline"] = evaluation
        artifacts["thresholds"]["cnn_lstm"] = threshold
        artifacts["model_cnn_lstm"] = cnn
        writer.log(f"  cnn_lstm: f1={row['f1']:.4f} train={train_time:.1f}s params={row['n_params']}")

    return rows, artifacts
