"""Figure generation — every plot derives from actual executed results.

Figures refuse to plot empty inputs (they raise instead of fabricating), and
each is stamped with the data source label (e.g. SYNTHETIC DEMO DATA).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE_STAMP = ""


def _stamp(ax: Any, source: str) -> None:
    if source:
        ax.text(0.0, -0.12, f"data source: {source}", transform=ax.transAxes, fontsize=8, color="gray")


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(labels: Sequence[int], path: str | Path, source: str = "") -> Path:
    labels = np.asarray(labels)
    counts = [(labels == 0).sum(), (labels == 1).sum()]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Normal", "Theft"], counts, color=["#4c78a8", "#e45756"])
    ax.set_title("Consumer class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom")
    _stamp(ax, source)
    return save_fig(fig, path)


def plot_client_distribution(per_client: Sequence[dict[str, Any]], path: str | Path, source: str = "") -> Path:
    if not per_client:
        raise ValueError("No client data to plot")
    ids = [c["client_id"] for c in per_client]
    samples = [c["n_samples"] for c in per_client]
    theft = [c.get("theft_ratio", 0.0) for c in per_client]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(ids, samples, color="#72b7b2")
    axes[0].set_title("Windows per client")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(ids, theft, color="#f58518")
    axes[1].set_title("Theft ratio per client")
    axes[1].tick_params(axis="x", rotation=30)
    _stamp(axes[1], source)
    return save_fig(fig, path)


def plot_convergence(histories: dict[str, list[dict[str, Any]]], path: str | Path, source: str = "") -> Path:
    """Round-vs-loss and round-vs-F1 curves for one or more methods."""
    if not histories:
        raise ValueError("No training histories to plot")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, history in histories.items():
        rounds = [h["round"] for h in history]
        axes[0].plot(rounds, [h.get("train_loss", np.nan) for h in history], marker="o", label=f"{name} train")
        val_loss = [h.get("global_metrics", {}).get("val_loss", np.nan) for h in history]
        f1s = [h.get("global_metrics", {}).get("f1", np.nan) for h in history]
        if any(np.isfinite(v) for v in val_loss):
            axes[0].plot(rounds, val_loss, marker=".", linestyle="--", label=f"{name} val")
        axes[1].plot(rounds, f1s, marker="o", label=name)
    axes[0].set_xlabel("round"); axes[0].set_ylabel("loss"); axes[0].set_title("Training convergence"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("round"); axes[1].set_ylabel("F1 (validation)"); axes[1].set_title("F1 vs round"); axes[1].legend(fontsize=8)
    _stamp(axes[1], source)
    return save_fig(fig, path)


def plot_metric_comparison(rows: Sequence[dict[str, Any]], metrics: Sequence[str], path: str | Path, source: str = "") -> Path:
    if not rows:
        raise ValueError("No result rows to plot")
    names = [r["method"] for r in rows]
    x = np.arange(len(names))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4))
    for i, metric in enumerate(metrics):
        vals = [r.get(metric, np.nan) for r in rows]
        ax.bar(x + i * width, vals, width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.set_title("Detection metric comparison")
    _stamp(ax, source)
    return save_fig(fig, path)


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, path: str | Path, source: str = "") -> Path:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Theft"])
    ax.set_yticks([0, 1], ["Normal", "Theft"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (t={threshold:.2f})")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center", color="black")
    fig.colorbar(im, fraction=0.046)
    _stamp(ax, source)
    return save_fig(fig, path)


def plot_roc_pr(curves: dict[str, dict[str, list[float]]], path: str | Path, source: str = "") -> Path:
    """curves: method → {'fpr':..., 'tpr':..., 'precision':..., 'recall':...}"""
    if not curves:
        raise ValueError("No curves to plot")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, curve in curves.items():
        if curve.get("fpr"):
            axes[0].plot(curve["fpr"], curve["tpr"], label=name)
        if curve.get("recall"):
            axes[1].plot(curve["recall"][::-1], curve["precision"][::-1], label=name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision–Recall"); axes[1].legend(fontsize=8)
    _stamp(axes[1], source)
    return save_fig(fig, path)


def plot_bar_compare(labels: Sequence[str], values: Sequence[float], path: str | Path, title: str, ylabel: str = "", source: str = "") -> Path:
    if len(values) == 0:
        raise ValueError("No values to plot")
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(labels)), 3.8))
    ax.bar(list(labels), np.asarray(values, dtype=float), color="#54a24b")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    if ylabel:
        ax.set_ylabel(ylabel)
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v, f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    _stamp(ax, source)
    return save_fig(fig, path)


def plot_privacy_utility(sweep_rows: Sequence[dict[str, Any]], path: str | Path, source: str = "") -> Path:
    if not sweep_rows:
        raise ValueError("No privacy sweep rows")
    sigmas = [r["noise_multiplier"] for r in sweep_rows]
    f1s = [r.get("f1", np.nan) for r in sweep_rows]
    eps = [r.get("epsilon_estimate", np.nan) for r in sweep_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(sigmas, f1s, marker="o")
    axes[0].set_xlabel("noise multiplier (σ)"); axes[0].set_ylabel("test F1"); axes[0].set_title("Privacy → utility")
    axes[1].plot(sigmas, eps, marker="s", color="#e45756")
    axes[1].set_xlabel("noise multiplier (σ)"); axes[1].set_ylabel("ε estimate (single round)")
    axes[1].set_title("Privacy strength")
    _stamp(axes[1], source)
    return save_fig(fig, path)


def plot_communication_accuracy(sweep_rows: Sequence[dict[str, Any]], path: str | Path, source: str = "") -> Path:
    if not sweep_rows:
        raise ValueError("No communication sweep rows")
    names = [r["compression"] for r in sweep_rows]
    comm = [r.get("total_bytes", np.nan) for r in sweep_rows]
    f1 = [r.get("f1", np.nan) for r in sweep_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(names, comm, color="#4c78a8")
    axes[0].set_title("Total communication (bytes)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, f1, color="#f58518")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test F1 vs compression")
    axes[1].tick_params(axis="x", rotation=20)
    _stamp(axes[1], source)
    return save_fig(fig, path)


def plot_scalability(sweep_rows: Sequence[dict[str, Any]], path: str | Path, source: str = "") -> Path:
    if not sweep_rows:
        raise ValueError("No scalability rows")
    clients = [r["clients"] for r in sweep_rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(clients, [r.get("wall_time_s", np.nan) for r in sweep_rows], marker="o")
    axes[0].set_title("Training wall time vs clients"); axes[0].set_xlabel("clients")
    axes[1].plot(clients, [r.get("total_bytes", np.nan) for r in sweep_rows], marker="s", color="#72b7b2")
    axes[1].set_title("Communication vs clients"); axes[1].set_xlabel("clients")
    f1s = [r.get("f1", np.nan) for r in sweep_rows]
    axes[2].plot(clients, f1s, marker="^", color="#e45756")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Test F1 vs clients"); axes[2].set_xlabel("clients")
    _stamp(axes[2], source)
    return save_fig(fig, path)


def plot_shap_summary(mean_abs: dict[str, float], path: str | Path, source: str = "") -> Path:
    """Horizontal bar of mean |SHAP| (top features)."""
    if not mean_abs:
        raise ValueError("No SHAP values to plot")
    items = sorted(mean_abs.items(), key=lambda kv: kv[1])[-15:]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    from explainability.explanations import human_feature_name

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(names))))
    ax.barh([human_feature_name(n) for n in names], vals, color="#4c78a8")
    ax.set_title("SHAP feature importance (mean |value|)")
    _stamp(ax, source)
    return save_fig(fig, path)


def plot_feature_importance(top_features: Sequence[dict[str, Any]], path: str | Path, source: str = "") -> Path:
    if not top_features:
        raise ValueError("No feature importance to plot")
    from explainability.explanations import human_feature_name

    top = list(top_features)[:15][::-1]
    names = [human_feature_name(f["feature"]) for f in top]
    vals = [f["importance_mean"] for f in top]
    errs = [f.get("importance_std", 0.0) for f in top]
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(names))))
    ax.barh(names, vals, xerr=errs, color="#54a24b", alpha=0.9)
    ax.set_title("Permutation feature importance (ROC-AUC drop)")
    _stamp(ax, source)
    return save_fig(fig, path)
