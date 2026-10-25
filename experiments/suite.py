"""End-to-end experiment suite used by ``main.py`` and the scripts.

Stages: dataset → centralized baselines → FedAvg → PowerShield-FL → sweeps →
explainability → figures/tables → report. Every number comes from a real run
recorded under ``results/raw/``.
"""
from __future__ import annotations

import copy
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from evaluation.comparison import build_table, save_table
from evaluation.figures import (
    plot_bar_compare,
    plot_class_distribution,
    plot_client_distribution,
    plot_communication_accuracy,
    plot_confusion_matrix,
    plot_convergence,
    plot_feature_importance,
    plot_metric_comparison,
    plot_privacy_utility,
    plot_roc_pr,
    plot_scalability,
    plot_shap_summary,
)
from evaluation.metrics import compute_classification_metrics
from evaluation.reporting import generate_report
from evaluation.statistical_tests import mean_std_ci, paired_comparison
from experiments.centralized import run_centralized_baselines
from experiments.common import (
    ExperimentWriter,
    _json_default,
    create_writer,
    new_run_id,
    save_metadata,
)
from experiments.communication import run_communication_sweep
from experiments.fedavg import run_fedavg
from experiments.privacy import run_privacy_sweep
from experiments.proposed import run_proposed
from experiments.scalability import run_scalability_sweep
from explainability.explanations import explain_prediction
from explainability.feature_importance import permutation_feature_importance
from explainability.shap_explainer import SHAP_AVAILABLE, ModelExplainer
from federated.partition import build_partition, heterogeneity_report
from models.baselines import SklearnClassifier
from models.base import TorchBinaryClassifier
from preprocessing.pipeline import DataSplits, prepare_data
from utilities.config import get, resolve_path
from utilities.seed import set_seed
from utilities.system import environment_info

FEDERATED_FIGURES = "federated"
COMPARISON_FIGURES = "comparison"


def _stamp(summary: dict[str, Any]) -> str:
    source = summary.get("dataset", {}).get("source", "unknown")
    return f"{source}" + (" (SYNTHETIC DEMO DATA)" if summary.get("dataset", {}).get("is_synthetic") else "")


def _copy_into_run(path: Path, writer: ExperimentWriter) -> None:
    try:
        writer.save_figure(path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Dataset / partition artifacts
# --------------------------------------------------------------------------- #
def _dataset_table(splits: DataSplits) -> pd.DataFrame:
    stats = splits.bundle_stats
    s = splits.summary()
    rows = [
        ("dataset_source", splits.source),
        ("synthetic_demo_data", str(bool(splits.is_synthetic))),
        ("consumers_total", stats.get("n_consumers", len(splits.labels))),
        ("consumers_train", s["n_consumers"]["train"]),
        ("consumers_validation", s["n_consumers"]["val"]),
        ("consumers_test", s["n_consumers"]["test"]),
        ("raw_rows", stats.get("n_rows")),
        ("raw_missing_values", stats.get("missing_values")),
        ("raw_duplicate_rows", stats.get("duplicate_rows")),
        ("window_samples_train", s["n_windows"]["train"]),
        ("window_samples_validation", s["n_windows"]["val"]),
        ("window_samples_test", s["n_windows"]["test"]),
        ("feature_count", s["n_features"]),
        ("theft_ratio_train_windows", s["theft_ratio"]["train"]),
        ("theft_ratio_test_windows", s["theft_ratio"]["test"]),
        ("split_unit", "consumer"),
        ("label_rule", splits.preprocessing_report.get("label_rule")),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def _save_partition_artifacts(cfg: dict, splits: DataSplits, partitions, seed: int) -> dict[str, Any]:
    out_dir = resolve_path(cfg, "data/partitions")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"partitions_seed{seed}_n{len(partitions)}.npz"
    np.savez_compressed(path, **{p.client_id: p.indices.astype(np.int64) for p in partitions})
    summary = {
        "seed": int(seed),
        "strategy": get(cfg, "clients.partition_strategy", "non_iid"),
        "n_clients": len(partitions),
        "clients": [p.summary(splits.train) for p in partitions],
        "heterogeneity": heterogeneity_report(partitions, splits.train),
        "partition_unit": "consumer",
        "artifact": str(path),
        "note": "only consumer IDs and window indices are stored; raw consumption stays out",
    }
    (out_dir / f"partitions_seed{seed}_n{len(partitions)}.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# Centralized stage
# --------------------------------------------------------------------------- #
def _model_row(rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def _save_central_artifacts(cfg: dict, splits: DataSplits, writer: ExperimentWriter, rows, artifacts) -> dict[str, Any]:
    model_dir = writer.run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = resolve_path(cfg, "models/checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for key, model in artifacts.items():
        if not key.startswith("model_"):
            continue
        short = key[len("model_") :]
        method = f"centralized_{short}"
        row = _model_row(rows, method)
        metadata = {
            "run_id": writer.run_dir.name,
            "method": method,
            "feature_names": list(splits.train.feature_names),
            "test_metrics": row or {},
            "dataset_source": splits.source,
            "is_synthetic": bool(splits.is_synthetic),
        }
        if isinstance(model, SklearnClassifier):
            path = model_dir / f"{short}.joblib"
            joblib.dump(model.estimator, path)
            (model_dir / f"{short}.meta.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
            shutil.copyfile(path, checkpoints / f"centralized_{short}.joblib")
        elif isinstance(model, TorchBinaryClassifier):
            path = model_dir / f"{short}.pt"
            model.save(path, metadata)
            (model_dir / f"{short}.meta.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
            shutil.copyfile(path, checkpoints / f"centralized_{short}.pt")
        else:
            continue
        saved[short] = str(path)
    return saved


def run_centralized_stage(cfg: dict, splits: DataSplits, seed: int, log: Callable[[str], None]) -> dict[str, Any]:
    writer = create_writer(cfg, "centralized", seed)
    writer.save_json(cfg, "config.json")
    log(f"Centralized stage -> {writer.run_dir}")
    import psutil

    rss0 = psutil.Process().memory_info().rss / (1024**2)
    rows, artifacts = run_centralized_baselines(cfg, splits, writer, seed=seed)
    rss1 = psutil.Process().memory_info().rss / (1024**2)
    for row in rows:
        row["run_id"] = writer.run_dir.name
        row["rss_delta_mb"] = float(rss1 - rss0)
        row["memory_note"] = "centralized-stage RSS delta (all models), machine-specific"
    saved_models = _save_central_artifacts(cfg, splits, writer, rows, artifacts)

    metrics_payload = {
        "rows": rows,
        "thresholds": artifacts.get("thresholds", {}),
        "curves": {
            name: {
                "metrics": ev.get("metrics", {}),
                "roc": ev.get("roc"),
                "pr": ev.get("pr"),
                "confusion_matrix": ev.get("confusion_matrix"),
            }
            for name, ev in artifacts.get("curves", {}).items()
        },
        "runtime": {"rss_before_mb": rss0, "rss_after_mb": rss1, "rss_delta_mb": rss1 - rss0},
        "checkpoints": saved_models,
        "note": "sklearn n_parameters counts coefficients/tree nodes; model_size_kb is an estimate for sklearn",
    }
    writer.save_json(metrics_payload, "metrics.json")
    pd.DataFrame(rows).to_csv(writer.run_dir / "metrics.csv", index=False)
    save_metadata(
        cfg,
        writer,
        {"source": splits.source, "is_synthetic": splits.is_synthetic, "split": splits.preprocessing_report.get("split"), "features": len(splits.train.feature_names)},
        extra={"stage": "centralized", "rows": rows, "checkpoints": saved_models},
    )
    return {"rows": rows, "artifacts": artifacts, "writer": writer, "run_id": writer.run_dir.name}


# --------------------------------------------------------------------------- #
# Explainability stage
# --------------------------------------------------------------------------- #
def run_explainability_stage(
    cfg: dict,
    splits: DataSplits,
    run_dir: Path,
    model: TorchBinaryClassifier,
    threshold: float,
    seed: int,
    log: Callable[[str], None],
) -> dict[str, Any]:
    out_dir = Path(run_dir) / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = resolve_path(cfg, "results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "PASS", "run_dir": str(run_dir)}

    log("Explainability: permutation importance...")
    try:
        importance = permutation_feature_importance(
            model,
            splits.val.X_matrix,
            splits.val.y,
            splits.train.feature_names,
            cfg,
            seed=seed,
            max_rows=int(get(cfg, "explainability.permutation_max_rows", 1500)),
        )
        (out_dir / "permutation_importance.json").write_text(json.dumps(importance, indent=2, default=_json_default), encoding="utf-8")
        fig = plot_feature_importance(importance["top_features"], figures_dir / "feature_importance.png", _stamp({"dataset": {"source": splits.source, "is_synthetic": splits.is_synthetic}}))
        _copy_into_run(fig, ExperimentWriter(run_dir=Path(run_dir), figures_dir=Path(run_dir) / "figures", log_path=Path(run_dir) / "logs.txt"))
        result["permutation_importance"] = importance
    except Exception as exc:  # real failure, reported honestly
        result["permutation_importance"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
        log(f"WARNING: permutation importance failed: {exc!r}")

    log("Explainability: SHAP...")
    try:
        if not SHAP_AVAILABLE:
            raise RuntimeError("shap is not installed")
        explainer = ModelExplainer(model, splits.val.X_matrix, cfg, feature_names=splits.train.feature_names)
        shap_summary = explainer.summary_dict(splits.test.X_matrix)
        (out_dir / "shap_summary.json").write_text(json.dumps(shap_summary, indent=2, default=_json_default), encoding="utf-8")
        fig = plot_shap_summary(
            shap_summary["mean_abs_shap"],
            figures_dir / "shap_summary.png",
            _stamp({"dataset": {"source": splits.source, "is_synthetic": splits.is_synthetic}}),
        )
        _copy_into_run(fig, ExperimentWriter(run_dir=Path(run_dir), figures_dir=Path(run_dir) / "figures", log_path=Path(run_dir) / "logs.txt"))
        result["shap"] = {"status": "PASS", **{k: v for k, v in shap_summary.items() if k != "mean_abs_shap"}}
    except Exception as exc:
        result["shap"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
        log(f"WARNING: SHAP failed: {exc!r}")

    log("Explainability: local model-derived explanations...")
    try:
        probs = model.predict_proba(splits.test.X_matrix)
        n_local = int(get(cfg, "explainability.local_explanations", 10))
        order = np.argsort(-probs)
        chosen = list(order[:n_local])
        # include a couple of confident-normal examples
        chosen += list(order[-min(3, len(order)) :])
        records = []
        for idx in dict.fromkeys(int(i) for i in chosen):
            record = explain_prediction(model, splits.test.X_matrix[idx], splits.train.feature_names, threshold)
            meta = splits.test.meta.iloc[idx]
            record.update(
                {
                    "sample_index": int(idx),
                    "consumer_id": str(meta.get("consumer_id", "")),
                    "start_date": str(meta.get("start_date", "")),
                    "y_true": int(splits.test.y[idx]),
                    "prediction_matches": int(record["verdict"] == "POTENTIAL_THEFT") == int(splits.test.y[idx]),
                }
            )
            records.append(record)
        payload = {"status": "PASS", "threshold": float(threshold), "records": records, "attribution": "gradient x input (torch model)"}
        (out_dir / "local_explanations.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        result["local_explanations"] = payload
    except Exception as exc:
        result["local_explanations"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
        log(f"WARNING: local explanations failed: {exc!r}")

    component_statuses = [
        str(result.get("permutation_importance", {}).get("status", "PASS" if "top_features" in result.get("permutation_importance", {}) else "FAILED")),
        str(result.get("shap", {}).get("status", "FAILED")),
        str(result.get("local_explanations", {}).get("status", "FAILED")),
    ]
    passed = sum(1 for s in component_statuses if s == "PASS")
    result["status"] = "PASS" if passed == 3 else ("PARTIAL" if passed > 0 else "EXPERIMENT FAILED")
    result["components"] = {
        "permutation_importance": component_statuses[0],
        "shap": component_statuses[1],
        "local_explanations": component_statuses[2],
    }
    (out_dir / "explainability_status.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Tables / figures
# --------------------------------------------------------------------------- #
def _avg_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = [k for k in keys if k in frame.columns]
    agg = frame.groupby(group_cols, dropna=False)[numeric].mean(numeric_only=True).reset_index()
    return agg.to_dict(orient="records")


def build_and_save_tables(summary: dict[str, Any], cfg: dict) -> dict[str, Any]:
    out = resolve_path(cfg, "results/tables")
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    dataset_table = pd.DataFrame(summary.get("dataset", {}).get("table", []))
    if not dataset_table.empty:
        paths["table1_dataset_statistics"] = save_table(dataset_table, out, "table1_dataset_statistics")

    central = summary.get("main", {}).get("centralized", [])
    if central:
        df = build_table(
            central,
            preferred_order=["method", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_time_s", "n_params", "model_size_kb", "seed"],
        )
        paths["table2_model_comparison"] = save_table(df, out, "table2_model_comparison")

    fed = summary.get("main", {}).get("federated", [])
    if fed:
        df = build_table(
            fed,
            preferred_order=["method", "clients", "rounds", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s", "total_bytes", "bytes_per_round", "seed"],
        )
        paths["table3_federated_comparison"] = save_table(df, out, "table3_federated_comparison")

    privacy_rows = summary.get("privacy_sweep", {}).get("rows", [])
    if privacy_rows:
        df = build_table(
            privacy_rows,
            preferred_order=["privacy_level", "noise_multiplier", "epsilon_estimate", "f1", "precision", "recall", "train_time_s", "total_bytes", "f1_change_vs_privacy_off"],
        )
        paths["table4_privacy_comparison"] = save_table(df, out, "table4_privacy_comparison")

    scalability_rows = summary.get("scalability", {}).get("rows", [])
    if scalability_rows:
        df = build_table(
            scalability_rows,
            preferred_order=["clients", "rounds", "train_time_s", "total_bytes", "rss_delta_mb", "f1", "accuracy", "seed"],
        )
        paths["table5_scalability"] = save_table(df, out, "table5_scalability")

    return paths


def build_and_save_figures(summary: dict[str, Any], cfg: dict, artifacts: dict[str, Any]) -> dict[str, Any]:
    out = resolve_path(cfg, "results/figures")
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    source = _stamp(summary)

    def attempt(name: str, fn: Callable[[], Path]) -> None:
        try:
            paths[name] = str(fn())
        except Exception as exc:
            paths[name] = f"NOT RUN: {exc!r}"

    dataset = summary.get("dataset", {})
    labels = dataset.get("labels", [])
    if labels:
        attempt("class_distribution", lambda: plot_class_distribution(labels, out / "class_distribution.png", source))

    clients = summary.get("partitions", {}).get("clients", [])
    if clients:
        attempt("client_distribution", lambda: plot_client_distribution(clients, out / "client_distribution.png", source))

    fed = summary.get("main", {}).get("federated", [])
    central = summary.get("main", {}).get("centralized", [])
    all_rows = central + fed
    if all_rows:
        averaged = _avg_rows(all_rows, ["method"])
        attempt("f1_comparison", lambda: plot_metric_comparison(averaged, ["f1"], out / "f1_comparison.png", source))
        attempt("precision_recall_comparison", lambda: plot_metric_comparison(averaged, ["precision", "recall"], out / "precision_recall_comparison.png", source))
        attempt("training_time_comparison", lambda: plot_bar_compare([r["method"] for r in averaged], [r.get("train_time_s", np.nan) for r in averaged], out / "training_time_comparison.png", "Training time", "seconds", source))
        attempt("memory_comparison", lambda: plot_bar_compare([r["method"] for r in averaged], [r.get("inference_peak_memory_mb", np.nan) for r in averaged], out / "memory_comparison.png", "Inference peak Python memory", "MB", source))

    histories = {}
    for method in ("fedavg_baseline", "powershield_fl"):
        ev = artifacts.get("federated_artifacts", {}).get(method)
        if ev:
            histories[method] = ev["result"].get("history", [])
    if histories:
        attempt("training_convergence", lambda: plot_convergence(histories, out / "training_convergence.png", source))

    if fed:
        reference = next((r for r in fed if r["method"] == "fedavg_baseline"), None)
        candidate = next((r for r in fed if r["method"] == "powershield_fl"), None)
        if reference and candidate:
            attempt(
                "communication_comparison",
                lambda: plot_bar_compare(
                    ["FedAvg", "PowerShield-FL"],
                    [reference.get("total_bytes", np.nan), candidate.get("total_bytes", np.nan)],
                    out / "communication_comparison.png",
                    "Total federated communication",
                    "bytes",
                    source,
                ),
            )

    proposed = artifacts.get("federated_artifacts", {}).get("powershield_fl")
    if proposed:
        ev = proposed["evaluation"]
        attempt(
            "confusion_matrix",
            lambda: plot_confusion_matrix(ev["targets"], ev["probabilities"], ev["metrics"]["threshold"], out / "confusion_matrix.png", source),
        )
        curves = {}
        for name, item in (artifacts.get("central_artifacts", {}).get("curves", {}) or {}).items():
            curves[name] = {"fpr": (item.get("roc") or {}).get("fpr", []), "tpr": (item.get("roc") or {}).get("tpr", []), "precision": (item.get("pr") or {}).get("precision", []), "recall": (item.get("pr") or {}).get("recall", [])}
        for name in ("fedavg_baseline", "powershield_fl"):
            item = artifacts.get("federated_artifacts", {}).get(name)
            if item:
                ev2 = item["evaluation"]
                curves[name] = {"fpr": ev2.get("roc", {}).get("fpr", []), "tpr": ev2.get("roc", {}).get("tpr", []), "precision": ev2.get("pr", {}).get("precision", []), "recall": ev2.get("pr", {}).get("recall", [])}
        if curves:
            attempt("roc_curve", lambda: plot_roc_pr(curves, out / "roc_curve.png", source))

    privacy_rows = summary.get("privacy_sweep", {}).get("rows", [])
    if privacy_rows:
        attempt("privacy_utility_tradeoff", lambda: plot_privacy_utility(privacy_rows, out / "privacy_utility_tradeoff.png", source))
    communication_rows = summary.get("communication_sweep", {}).get("rows", [])
    if communication_rows:
        attempt("communication_accuracy_tradeoff", lambda: plot_communication_accuracy(communication_rows, out / "communication_accuracy_tradeoff.png", source))
    scalability_rows = summary.get("scalability", {}).get("rows", [])
    if scalability_rows:
        attempt("scalability", lambda: plot_scalability(scalability_rows, out / "scalability.png", source))
    return paths


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #
def run_suite(
    cfg: dict,
    demo: bool = False,
    stages: Sequence[str] | None = None,
    seed: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Execute the configured suite and return the aggregate summary."""
    stages = list(stages or ["centralized", "fedavg", "proposed", "sweeps", "explainability", "figures", "report"])
    seeds = [int(seed)] if seed is not None else [int(s) for s in get(cfg, "experiments.seeds", [42])]
    if demo and seed is None:
        seeds = seeds[:1]
    suite_id = new_run_id("suite", seeds[0])
    results_root = resolve_path(cfg, get(cfg, "experiment.output_dir", "results"))
    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log(f"Suite {suite_id} | mode={get(cfg, 'experiment.mode')} seeds={seeds} demo={demo}")

    summary: dict[str, Any] = {
        "suite_id": suite_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": get(cfg, "experiment.mode", "quick"),
        "demo": bool(demo),
        "seeds": seeds,
        "environment": environment_info(),
        "config": cfg,
        "stages": {},
        "main": {"centralized": [], "federated": [], "fedavg": [], "proposed": []},
        "privacy_sweep": {"rows": [], "meta": {}},
        "communication_sweep": {"rows": [], "meta": {}},
        "scalability": {"rows": [], "meta": {}},
        "partitions": {},
        "explainability": {},
        "figures": {},
        "tables": {},
    }

    central_rows: list[dict[str, Any]] = []
    fedavg_rows: list[dict[str, Any]] = []
    proposed_rows: list[dict[str, Any]] = []
    federated_artifacts: dict[str, Any] = {}
    central_artifacts: dict[str, Any] = {}
    first_seed: int | None = None
    first_partitions = None
    first_splits: DataSplits | None = None
    run_dirs: list[str] = []
    dashboard_run_dir: Path | None = None
    dashboard_threshold: float | None = None

    for seed_idx, current_seed in enumerate(seeds):
        set_seed(current_seed)
        log(f"Preparing data for seed {current_seed} ...")
        try:
            splits = prepare_data(cfg, seed=current_seed, log=log)
        except Exception as exc:
            summary["stages"]["dataset"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
            log(f"ERROR: dataset preparation failed: {exc!r}")
            break

        if first_seed is None:
            first_seed = current_seed
            first_splits = splits
            summary["stages"]["dataset"] = {"status": "PASS", "source": splits.source, "is_synthetic": splits.is_synthetic}
            summary["dataset"] = {
                "source": splits.source,
                "is_synthetic": bool(splits.is_synthetic),
                "stats": splits.bundle_stats,
                "summary": splits.summary(),
                "cleaning": splits.cleaning_report,
                "preprocessing": splits.preprocessing_report,
                "imbalance": splits.imbalance,
                "labels": splits.labels.astype(int).tolist() if len(splits.labels) <= 5000 else [],
                "table": _dataset_table(splits).to_dict(orient="records"),
            }
            dataset_json = metrics_dir / "dataset_statistics.json"
            dataset_json.write_text(json.dumps(summary["dataset"], indent=2, default=_json_default), encoding="utf-8")
            # dataset table + class figure
            save_table(_dataset_table(splits), resolve_path(cfg, "results/tables"), "table1_dataset_statistics")

        # Main partitions are shared by FedAvg and PowerShield-FL for fairness.
        try:
            partitions = build_partition(splits.train, cfg, current_seed)
            if first_partitions is None:
                first_partitions = partitions
                summary["partitions"] = _save_partition_artifacts(cfg, splits, partitions, current_seed)
                summary["stages"]["partitions"] = {"status": "PASS", "clients": len(partitions)}
        except Exception as exc:
            summary["stages"]["partitions"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
            log(f"ERROR: partition failed: {exc!r}")
            partitions = None

        if "centralized" in stages:
            try:
                result = run_centralized_stage(cfg, splits, current_seed, log)
                central_rows.extend(result["rows"])
                if not central_artifacts:
                    central_artifacts = {"curves": result["artifacts"].get("curves", {}), "run_id": result["run_id"]}
                summary["stages"][f"centralized_s{current_seed}"] = {"status": "PASS", "run_id": result["run_id"], "rows": len(result["rows"])}
            except Exception as exc:
                summary["stages"][f"centralized_s{current_seed}"] = {"status": "EXPERIMENT FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
                log(f"ERROR: centralized stage failed: {exc!r}")

        if partitions is not None and "fedavg" in stages:
            try:
                writer = create_writer(cfg, "fedavg", current_seed)
                row, art = run_fedavg(cfg, splits, writer, seed=current_seed, partitions=partitions)
                fedavg_rows.append(row)
                run_dirs.append(str(writer.run_dir))
                federated_artifacts.setdefault("fedavg_baseline", art)
                summary["stages"][f"fedavg_s{current_seed}"] = {"status": "PASS", "run_id": writer.run_dir.name, "f1": row["f1"], "total_bytes": row["total_bytes"]}
            except Exception as exc:
                summary["stages"][f"fedavg_s{current_seed}"] = {"status": "EXPERIMENT FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
                log(f"ERROR: FedAvg stage failed: {exc!r}")

        if partitions is not None and "proposed" in stages:
            try:
                writer = create_writer(cfg, "powershield", current_seed)
                save_samples = seed_idx == 0
                row, art = run_proposed(cfg, splits, writer, seed=current_seed, partitions=partitions, save_dashboard_samples=save_samples)
                proposed_rows.append(row)
                run_dirs.append(str(writer.run_dir))
                federated_artifacts["powershield_fl"] = art
                summary["stages"][f"proposed_s{current_seed}"] = {"status": "PASS", "run_id": writer.run_dir.name, "f1": row["f1"], "total_bytes": row["total_bytes"]}
                if save_samples:
                    dashboard_run_dir = writer.run_dir
                    dashboard_threshold = row["threshold"]
                    # Explainability uses this seeded model/data.
                    if "explainability" in stages:
                        try:
                            expl = run_explainability_stage(cfg, splits, writer.run_dir, art["model"], row["threshold"], current_seed, log)
                            summary["explainability"] = expl
                            summary["stages"][f"explainability_s{current_seed}"] = {"status": expl.get("status", "PASS")}
                        except Exception as exc:
                            summary["explainability"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
                            summary["stages"][f"explainability_s{current_seed}"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
                            log(f"ERROR: explainability stage failed: {exc!r}")
            except Exception as exc:
                summary["stages"][f"proposed_s{current_seed}"] = {"status": "EXPERIMENT FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
                log(f"ERROR: PowerShield-FL stage failed: {exc!r}")

    summary["main"]["centralized"] = central_rows
    summary["main"]["fedavg"] = fedavg_rows
    summary["main"]["proposed"] = proposed_rows
    summary["main"]["federated"] = fedavg_rows + proposed_rows

    # Communication reduction measured against FedAvg for the same seed(s).
    if fedavg_rows and proposed_rows:
        ref = float(np.mean([r["total_bytes"] for r in fedavg_rows]))
        cand = float(np.mean([r["total_bytes"] for r in proposed_rows]))
        summary["main"]["communication_comparison"] = {
            "fedavg_mean_total_bytes": ref,
            "proposed_mean_total_bytes": cand,
            "reduction_pct": 100.0 * (1.0 - cand / ref) if ref else 0.0,
            "measured": ref > 0 and cand > 0,
        }

    # Multi-seed statistics (only if >1 seed actually ran).
    stats: dict[str, Any] = {}
    for name, rows in (("fedavg_baseline", fedavg_rows), ("powershield_fl", proposed_rows)):
        if rows:
            stats[name] = {
                metric: mean_std_ci([float(r.get(metric, np.nan)) for r in rows])
                for metric in ("f1", "precision", "recall", "roc_auc", "train_time_s", "total_bytes")
            }
    if len(fedavg_rows) > 1 and len(proposed_rows) > 1:
        n = min(len(fedavg_rows), len(proposed_rows))
        stats["paired_f1_fedavg_vs_proposed"] = paired_comparison(
            [float(r["f1"]) for r in fedavg_rows[:n]],
            [float(r["f1"]) for r in proposed_rows[:n]],
            alpha=float(get(cfg, "experiments.statistical_alpha", 0.05)),
        )
    else:
        stats["paired_f1_fedavg_vs_proposed"] = {
            "n": min(len(fedavg_rows), len(proposed_rows)),
            "note": "STATISTICAL TEST NOT RUN — fewer than 2 paired seeds",
        }
    summary["statistics"] = stats

    # Sweeps use the first prepared seed and the same main partitions.
    if not demo and "sweeps" in stages and first_partitions is not None and first_seed is not None:
        # Sweeps reuse the first seed's already-prepared splits and partitions so
        # every privacy/compression/scalability comparison sees identical data.
        sweep_splits = first_splits

        def writer_factory(name: str) -> ExperimentWriter:
            return create_writer(cfg, name, first_seed)

        if sweep_splits is not None and get(cfg, "experiments.run_privacy_sweep", True):
            try:
                log("Sweep: privacy / utility...")
                rows, meta = run_privacy_sweep(cfg, sweep_splits, writer_factory, seed=first_seed, partitions=first_partitions)
                summary["privacy_sweep"] = {"rows": rows, "meta": meta}
                summary["stages"]["privacy_sweep"] = {"status": "PASS", "runs": len(rows)}
            except Exception as exc:
                summary["stages"]["privacy_sweep"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
                log(f"ERROR: privacy sweep failed: {exc!r}")

        if sweep_splits is not None and get(cfg, "experiments.run_communication_sweep", True):
            try:
                log("Sweep: communication / accuracy...")
                rows, meta = run_communication_sweep(cfg, sweep_splits, writer_factory, seed=first_seed, partitions=first_partitions)
                summary["communication_sweep"] = {"rows": rows, "meta": meta}
                summary["stages"]["communication_sweep"] = {"status": "PASS", "runs": len(rows)}
            except Exception as exc:
                summary["stages"]["communication_sweep"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
                log(f"ERROR: communication sweep failed: {exc!r}")

        if sweep_splits is not None and get(cfg, "experiments.run_scalability", True):
            try:
                log("Sweep: scalability...")
                rows, meta = run_scalability_sweep(cfg, sweep_splits, writer_factory, seed=first_seed)
                summary["scalability"] = {"rows": rows, "meta": meta}
                summary["stages"]["scalability"] = {"status": "PASS", "runs": len(rows)}
            except Exception as exc:
                summary["stages"]["scalability"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
                log(f"ERROR: scalability sweep failed: {exc!r}")

    # Artifacts for dashboard / report.
    summary["artifacts"] = {
        "dashboard_run_dir": str(dashboard_run_dir) if dashboard_run_dir else None,
        "proposed_checkpoint": str(Path(dashboard_run_dir) / "model" / "checkpoint.pt") if dashboard_run_dir else None,
        "proposed_threshold": dashboard_threshold,
        "run_dirs": run_dirs,
    }

    # Tables and figures derive only from recorded rows.
    try:
        summary["tables"] = build_and_save_tables(summary, cfg)
        summary["stages"]["tables"] = {"status": "PASS", "count": len(summary["tables"])}
    except Exception as exc:
        summary["stages"]["tables"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
        log(f"ERROR: table generation failed: {exc!r}")

    if "figures" in stages:
        try:
            summary["figures"] = build_and_save_figures(
                summary,
                cfg,
                {"federated_artifacts": federated_artifacts, "central_artifacts": central_artifacts},
            )
            summary["stages"]["figures"] = {"status": "PASS", "count": len(summary["figures"])}
        except Exception as exc:
            summary["stages"]["figures"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
            log(f"ERROR: figure generation failed: {exc!r}")

    # Copy canonical figures into the main run dirs for self-contained artifacts.
    for run_dir in [art.get("run_dir") for art in federated_artifacts.values() if art.get("run_dir")]:
        pass
    try:
        proposed_art = federated_artifacts.get("powershield_fl")
        if proposed_art and dashboard_run_dir:
            fig_dir = Path(dashboard_run_dir) / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for path_str in summary.get("figures", {}).values():
                if isinstance(path_str, str) and not path_str.startswith("NOT RUN") and Path(path_str).exists():
                    shutil.copyfile(path_str, fig_dir / Path(path_str).name)
    except Exception:
        pass

    summary_path = metrics_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (results_root / "raw").mkdir(parents=True, exist_ok=True)
    global_meta = {
        "suite_id": suite_id,
        "generated_at": summary["generated_at"],
        "mode": summary["mode"],
        "seeds": seeds,
        "environment": summary["environment"],
        "dataset": {k: v for k, v in summary.get("dataset", {}).items() if k not in ("labels",)},
        "config": cfg,
        "stages": summary["stages"],
    }
    (results_root / "raw" / "experiment_metadata.json").write_text(json.dumps(global_meta, indent=2, default=_json_default), encoding="utf-8")
    (results_root / "raw" / f"{suite_id}.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    if "report" in stages:
        try:
            report_path = generate_report(cfg, summary, results_root / "reports" / "final_experiment_report.md")
            summary["stages"]["report"] = {"status": "PASS", "path": str(report_path)}
        except Exception as exc:
            summary["stages"]["report"] = {"status": "EXPERIMENT FAILED", "error": repr(exc)}
            log(f"ERROR: report generation failed: {exc!r}")
        summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    log(f"Suite complete: {summary_path}")
    return summary
