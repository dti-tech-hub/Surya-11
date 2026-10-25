"""Experiment B — standard FedAvg baseline (and the shared federated runner)."""
from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from detection.threshold import select_threshold
from evaluation.communication import summarize as summarize_communication
from evaluation.detection import evaluate_detection, predictions_frame
from evaluation.performance import model_resource_report, rss_memory_mb
from evaluation.metrics import compute_classification_metrics
from experiments.common import ExperimentWriter, save_metadata
from federated.partition import build_partition, heterogeneity_report
from federated.simulator import FederatedSimulator
from models.base import TorchBinaryClassifier
from models.lightweight import build_lightweight
from privacy.base import NoPrivacy, build_privacy
from preprocessing.pipeline import DataSplits
from utilities.config import get, resolve_path
from utilities.seed import set_seed


def configure_federated_method(cfg: dict, method: str) -> dict:
    """Return a copy of cfg configured as FedAvg baseline or PowerShield-FL."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("communication", {})
    run_cfg.setdefault("privacy", {})
    if method == "fedavg":
        run_cfg["aggregation"]["method"] = "fedavg"
        run_cfg["privacy"]["enabled"] = False
        run_cfg["privacy"]["secure_aggregation"] = False
        run_cfg["communication"]["compression"] = False
        run_cfg["communication"]["method"] = "none"
        run_cfg["communication"]["compress_broadcast"] = False
    elif method in ("proposed", "powershield"):
        # Use exactly the configured adaptive/privacy/compression settings.
        run_cfg["aggregation"]["method"] = "adaptive"
    else:
        raise ValueError(f"Unknown federated method '{method}'")
    return run_cfg


def _model_factory(cfg: dict, n_features: int, seed: int):
    def factory() -> TorchBinaryClassifier:
        return build_lightweight(cfg, n_features, seed=seed)

    return factory


def _save_checkpoint(
    model: TorchBinaryClassifier,
    cfg: dict,
    writer: ExperimentWriter,
    row: dict[str, Any],
    splits: DataSplits,
    threshold: float,
) -> dict[str, str]:
    """Save the run checkpoint plus a stable "<method>_latest" copy."""
    method = row["method"]
    run_model_dir = writer.run_dir / "model"
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_model_dir / "checkpoint.pt"
    metadata = {
        "run_id": writer.run_dir.name,
        "method": method,
        "architecture": "lightweight_mlp",
        "n_inputs": int(splits.train.X_matrix.shape[1]),
        "hidden": list(get(cfg, "model.hidden", [64, 32])),
        "dropout": float(get(cfg, "model.dropout", 0.2)),
        "feature_names": list(splits.train.feature_names),
        "threshold": float(threshold),
        "test_metrics": dict(row),
        "dataset_source": splits.source,
        "is_synthetic": bool(splits.is_synthetic),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    model.save(run_path, metadata)
    (run_model_dir / "checkpoint.meta.json").write_text(
        __import__("json").dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    checkpoints = resolve_path(cfg, "models/checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    stable_name = "fedavg_model" if method == "fedavg_baseline" else "powershield_model"
    stable_path = checkpoints / f"{stable_name}.pt"
    shutil.copyfile(run_path, stable_path)
    latest_meta = checkpoints / f"{stable_name}.meta.json"
    latest_meta.write_text(__import__("json").dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {"checkpoint": str(run_path), "latest_checkpoint": str(stable_path), "metadata": str(latest_meta)}


def run_federated_experiment(
    cfg: dict,
    splits: DataSplits,
    writer: ExperimentWriter,
    method: str,
    seed: int = 42,
    rounds: int | None = None,
    partitions=None,
    save_dashboard_samples: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one federated method, evaluate it, and persist all artifacts."""
    run_cfg = configure_federated_method(cfg, method)
    method_label = "fedavg_baseline" if method == "fedavg" else "powershield_fl"
    rounds = int(rounds if rounds is not None else get(run_cfg, "federated.rounds", 8))
    set_seed(seed)
    writer.save_json(run_cfg, "config.json")

    if partitions is None:
        partitions = build_partition(splits.train, run_cfg, seed)
    heterogeneity = heterogeneity_report(partitions, splits.train)
    writer.save_json(heterogeneity, "client_heterogeneity.json")

    n_features = int(splits.train.X_matrix.shape[1])
    factory = _model_factory(run_cfg, n_features, seed)
    probe = factory()
    privacy = build_privacy(run_cfg, seed=seed)
    writer.log(
        f"{method_label}: clients={len(partitions)} rounds={rounds} "
        f"aggregation={get(run_cfg, 'aggregation.method')} privacy={privacy.describe().get('mechanism')} "
        f"compression={get(run_cfg, 'communication.method')}"
    )

    rss_before = rss_memory_mb()
    t0 = time.perf_counter()
    sim = FederatedSimulator(
        cfg=run_cfg,
        train_X=splits.train.X_matrix,
        train_y=splits.train.y,
        partitions=partitions,
        model_factory=factory,
        privacy_mechanism=privacy,
        val_X=splits.val.X_matrix,
        val_y=splits.val.y,
        seed=seed,
        log=writer.log,
    )
    result = sim.run(rounds=rounds)
    wall = time.perf_counter() - t0
    rss_after = rss_memory_mb()

    # Evaluate the best-validation checkpoint on the untouched test consumers.
    model = factory()
    model.set_weights(result["best_state"])
    evaluation = evaluate_detection(
        model,
        splits.test.X_matrix,
        splits.test.y,
        cfg=run_cfg,
        X_val=splits.val.X_matrix,
        y_val=splits.val.y,
        base_threshold=0.5,
        meta=splits.test.meta,
    )
    test_metrics = evaluation["metrics"]
    threshold = float(test_metrics["threshold"])
    resources = model_resource_report(model, X=splits.test.X_matrix[:512])
    comm = summarize_communication(result["history"])

    row: dict[str, Any] = {
        "method": method_label,
        "kind": "federated",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics.get("roc_auc", float("nan")),
        "pr_auc": test_metrics.get("pr_auc", float("nan")),
        "threshold": threshold,
        "train_time_s": float(wall),
        "wall_time_s": float(wall),
        "n_params": int(probe.n_params),
        "model_size_kb": float(resources.get("model_size_kb", float("nan"))),
        "inference_per_sample_ms": (resources.get("inference") or {}).get("per_sample_ms", float("nan")),
        "inference_peak_memory_mb": float(resources.get("inference_peak_memory_mb", float("nan"))),
        "rss_delta_mb": float(rss_after - rss_before),
        "rounds": int(rounds),
        "clients": len(partitions),
        "total_bytes": int(comm.get("total_bytes", 0)),
        "bytes_per_round": float(comm.get("bytes_per_round_mean", 0.0)),
        "best_val_f1": float(result.get("best_val_f1", float("nan"))),
        "best_round": int(result.get("best_round", 0)),
        "seed": int(seed),
        "run_id": writer.run_dir.name,
        "aggregation": str(get(run_cfg, "aggregation.method")),
        "compression": str(get(run_cfg, "communication.method")),
        "privacy": str(privacy.describe().get("mechanism", "none")),
        "noise_multiplier": float(get(run_cfg, "privacy.noise_multiplier", 0.0)),
        "dataset_source": splits.source,
        "is_synthetic": bool(splits.is_synthetic),
        "per_consumer": evaluation.get("consumer_metrics", {}),
    }

    writer.save_json(
        {
            "row": row,
            "metrics": test_metrics,
            "threshold_info": evaluation["threshold_info"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "roc": evaluation.get("roc"),
            "pr": evaluation.get("pr"),
            "communication": comm,
            "privacy": result.get("privacy", {}),
            "secure_aggregation": result.get("secure_aggregation", {}),
            "heterogeneity": heterogeneity,
            "clients": result.get("clients", []),
            "history": result["history"],
            "wall_time_s": wall,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "best_round": result.get("best_round"),
            "best_val_f1": result.get("best_val_f1"),
        },
        "metrics.json",
    )

    history_rows = []
    for entry in result["history"]:
        gm = entry.get("global_metrics", {})
        history_rows.append(
            {
                "round": entry["round"],
                "n_clients": entry.get("n_clients"),
                "train_loss": entry.get("train_loss"),
                "val_loss": gm.get("val_loss"),
                "val_accuracy": gm.get("accuracy"),
                "val_precision": gm.get("precision"),
                "val_recall": gm.get("recall"),
                "val_f1": gm.get("f1"),
                "val_roc_auc": gm.get("roc_auc"),
                "bytes_upload": entry.get("bytes_upload"),
                "bytes_download": entry.get("bytes_download"),
                "round_duration_s": entry.get("round_duration_s"),
                "mean_kept_fraction": entry.get("mean_kept_fraction"),
            }
        )
    pd.DataFrame(history_rows).to_csv(writer.run_dir / "metrics.csv", index=False)

    preds = predictions_frame(splits.test.meta, evaluation["probabilities"], evaluation["targets"], threshold)
    preds.to_csv(writer.run_dir / "test_predictions.csv", index=False)
    if "consumer" in evaluation:
        evaluation["consumer"].to_csv(writer.run_dir / "consumer_verdicts.csv", index=False)

    checkpoint_info = _save_checkpoint(model, run_cfg, writer, row, splits, threshold)

    if save_dashboard_samples:
        np.savez_compressed(
            writer.run_dir / "test_samples.npz",
            X=splits.test.X_matrix.astype(np.float32),
            y=splits.test.y.astype(np.int64),
            feature_names=np.asarray(splits.train.feature_names, dtype=object),
            consumer_ids=splits.test.meta["consumer_id"].astype(str).to_numpy(),
            start_date=splits.test.meta["start_date"].astype(str).to_numpy(),
        )

    save_metadata(
        run_cfg,
        writer,
        {
            **splits.bundle_stats,
            "split": splits.preprocessing_report.get("split"),
            "windows": splits.summary()["n_windows"],
            "feature_count": n_features,
        },
        extra={"row": row, "checkpoint": checkpoint_info, "privacy": result.get("privacy", {})},
    )
    writer.log(
        f"{method_label}: test_f1={row['f1']:.4f} auc={row['roc_auc']:.4f} "
        f"bytes={row['total_bytes']} wall={wall:.1f}s"
    )
    return row, {
        "model": model,
        "result": result,
        "evaluation": evaluation,
        "partitions": partitions,
        "heterogeneity": heterogeneity,
        "checkpoint": checkpoint_info,
        "model_factory": factory,
    }


def run_fedavg(
    cfg: dict,
    splits: DataSplits,
    writer: ExperimentWriter,
    seed: int = 42,
    rounds: int | None = None,
    partitions=None,
    save_dashboard_samples: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_federated_experiment(
        cfg, splits, writer, method="fedavg", seed=seed, rounds=rounds,
        partitions=partitions, save_dashboard_samples=save_dashboard_samples,
    )
