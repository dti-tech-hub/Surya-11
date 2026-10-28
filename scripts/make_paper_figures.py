"""Generate clean, data-driven figures for the PowerShield-FL conference paper.

Every number plotted here is read from the authoritative suite artifacts:
  results/metrics/summary.json
  results/raw/<run_id>/metrics.json

The script adapts to the executed mode (quick/full), seed count, rounds, and
client count recorded in ``summary.json``. The architecture diagram is
conceptual except for configuration values read from the same artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SUMMARY = json.loads((RESULTS / "metrics" / "summary.json").read_text(encoding="utf-8"))
MODE = SUMMARY.get("mode", "unknown")
SEEDS = [int(s) for s in SUMMARY.get("seeds", [])]
FED_ROWS = SUMMARY["main"]["federated"]
CENTRAL_ROWS = SUMMARY["main"]["centralized"]
PRIV = SUMMARY["privacy_sweep"]["rows"]
COMM = SUMMARY["communication_sweep"]["rows"]
SCALE = SUMMARY["scalability"]["rows"]
PARTS = SUMMARY.get("partitions", {})
CONFIG = SUMMARY.get("config", {})
DATASET = SUMMARY.get("dataset", {})
STATS = DATASET.get("stats", {})


def _mean_rows(rows: list[dict]) -> dict[str, dict]:
    """Average numeric metrics per method; keep first row for non-numeric fields."""
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    mean = frame.groupby("method", dropna=False)[numeric].mean().reset_index()
    out: dict[str, dict] = {}
    for record in mean.to_dict(orient="records"):
        base = next(r for r in rows if r["method"] == record["method"])
        out[record["method"]] = {**base, **record}
    return out


FED = _mean_rows(FED_ROWS)
CENTRAL = _mean_rows(CENTRAL_ROWS)
CONSUMER = {
    method: {
        key: float(pd.Series([r["per_consumer"][key] for r in FED_ROWS if r["method"] == method]).mean())
        for key in ("precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn")
    }
    for method in ("fedavg_baseline", "powershield_fl")
}


def _row_for(method: str, seed: int | None = None) -> dict:
    if seed is None:
        seed = SEEDS[0] if SEEDS else None
    for row in FED_ROWS:
        if row["method"] == method and row.get("seed") == seed:
            return row
    return next(r for r in FED_ROWS if r["method"] == method)


def _read_run(row: dict) -> dict:
    path = RESULTS / "raw" / row["run_id"] / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    # Data provenance is stated in the paper caption and the document-wide
    # data-status callout.  Keeping it out of the image avoids overlapping
    # x-axis labels (a defect in the first render).
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / name}")


# --------------------------------------------------------------------------- #
# 1. Conceptual architecture (no measured values)
# --------------------------------------------------------------------------- #
def architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    n_consumers = STATS.get("n_consumers", "N/A")
    days = STATS.get("days", "N/A")
    readings = STATS.get("readings_per_day", 48)
    n_clients = PARTS.get("n_clients", "K")
    hidden = CONFIG.get("model", {}).get("hidden", [64, 32])
    n_params = CENTRAL.get("centralized_lightweight_mlp", {}).get("n_params", FED.get("powershield_fl", {}).get("n_params", "N/A"))
    rounds = int(FED.get("fedavg_baseline", {}).get("rounds", 0) or 0)
    local_epochs = CONFIG.get("federated", {}).get("local_epochs", "N/A")
    sigma = CONFIG.get("privacy", {}).get("noise_multiplier", "N/A")
    clip = CONFIG.get("privacy", {}).get("clipping_norm", "N/A")
    compression = CONFIG.get("communication", {}).get("method", "N/A")
    seed_text = f"seed {SEEDS[0]}" if len(SEEDS) == 1 else f"{len(SEEDS)} seeds"
    params_text = f"{int(n_params):,}" if isinstance(n_params, (int, float, np.integer)) else str(n_params)

    def box(x, y, w, h, title, body, face="#eef4fb", edge="#2f5d8a"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#17324d")
        ax.text(
            x + w / 2,
            y + h / 2 - 2.4,
            body,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#222222",
            linespacing=1.28,
        )

    # Left: local data and clients
    box(1, 71, 24, 20, "Raw smart-meter panels", f"{n_consumers} synthetic consumers\n{days} days × {readings} readings\nconsumer-level split\n14-day windows, 133 features")
    box(1, 36, 24, 24, f"K = {n_clients} simulated clients", "Dirichlet α = 0.5\nquantity skew = 0.6\nlocal LightweightMLP\n133–64–32–1\n" + f"{params_text} parameters")
    box(31, 43, 19, 20, "ClientUpdate only", "Δ model weights\nsample counts, losses\naggregate local metrics\nno raw windows / IDs / timestamps")
    box(55, 43, 18, 20, "Privacy layer", f"L2 clip C = {clip}\nGaussian σ = {sigma}\nsecure aggregation\nsimulated; disabled")
    box(55, 16, 18, 18, "Communication", f"{compression} compression\n(default)\nactual serialized bytes\nare measured")
    box(79, 35, 19, 27, "Server aggregation", "FedAvg: nₖ / N\nor adaptive:\n(nₖ/N)^α · qₖ^β ·\nrₖ^γ · bₖ")
    box(79, 70, 19, 18, "Global model", "validation-tuned\nthreshold, frozen\nwindow- and\nconsumer-level test")
    box(79, 8, 19, 16, "Explainability", "SHAP KernelExplainer\npermutation importance\ngradient × input")

    # Local boundary
    ax.plot([53.1, 53.1], [4, 96], linestyle=(0, (5, 4)), color="#b03a2e", linewidth=1.4)
    ax.text(53.1, 98, "privacy boundary: raw data never crosses", ha="center", va="top", fontsize=7.6, color="#b03a2e")

    def arrow(x1, y1, x2, y2, color="#456b8c"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.25,
                color=color,
                shrinkA=1,
                shrinkB=1,
            )
        )

    arrow(13, 71, 13, 60)
    arrow(25, 49, 31, 51)
    arrow(50, 53, 55, 53)
    arrow(64, 43, 64, 34)
    arrow(73, 25, 79, 43)
    arrow(88.5, 62, 88.5, 70)
    arrow(88.5, 35, 88.5, 24)
    arrow(73, 53, 79, 50)

    ax.text(50, 3.2, f"{MODE} mode: {rounds} rounds, {local_epochs} local epochs, full participation, {seed_text}.", ha="center", fontsize=7.8, color="#555555")
    fig.savefig(FIG / "paper_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / 'paper_architecture.png'}")


# --------------------------------------------------------------------------- #
# 2. Convergence
# --------------------------------------------------------------------------- #
def convergence() -> None:
    fed = _read_run(_row_for("fedavg_baseline"))
    prop = _read_run(_row_for("powershield_fl"))
    rounds = [h["round"] for h in fed["history"]]
    last_round = rounds[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(rounds, [h["train_loss"] for h in fed["history"]], marker="o", label="FedAvg — train", color="#3b6ea5")
    ax.plot(rounds, [h["mean_val_loss"] for h in fed["history"]], marker="s", linestyle="--", label="FedAvg — local val", color="#3b6ea5", alpha=0.55)
    ax.plot(rounds, [h["train_loss"] for h in prop["history"]], marker="o", label="PowerShield — train", color="#2e8b57")
    ax.plot(rounds, [h["mean_val_loss"] for h in prop["history"]], marker="s", linestyle="--", label="PowerShield — local val", color="#2e8b57", alpha=0.55)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training and local-validation loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.4, frameon=False)

    ax = axes[1]
    ax.plot(rounds, [h["global_metrics"]["f1"] for h in fed["history"]], marker="o", label="FedAvg", color="#3b6ea5")
    ax.plot(rounds, [h["global_metrics"]["f1"] for h in prop["history"]], marker="o", label="PowerShield-FL", color="#2e8b57")
    ax.set_xlim(0.8, last_round + 0.8)
    for rr, h in zip(rounds, fed["history"]):
        if rr in (1, last_round):
            ax.annotate(f'{h["global_metrics"]["f1"]:.4f}', (rr, h["global_metrics"]["f1"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="#3b6ea5")
    for rr, h in zip(rounds, prop["history"]):
        if rr in (1, last_round):
            ax.annotate(f'{h["global_metrics"]["f1"]:.4f}', (rr, h["global_metrics"]["f1"]), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7, color="#2e8b57")
    f1_values = [h["global_metrics"]["f1"] for h in fed["history"] + prop["history"]]
    ax.set_ylim(max(0.0, min(f1_values) - 0.05), min(1.01, max(f1_values) + 0.02))
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Server validation F1")
    ax.set_title("(b) Validation F1 by round")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    _save(fig, "paper_convergence.png")


# --------------------------------------------------------------------------- #
# 3. Communication
# --------------------------------------------------------------------------- #
def communication() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    ax = axes[0]
    labels = ["FedAvg", "PowerShield-FL"]
    vals = [FED["fedavg_baseline"]["total_bytes"], FED["powershield_fl"]["total_bytes"]]
    rounds = int(FED["fedavg_baseline"].get("rounds", 0) or 0)
    clients = int(FED["fedavg_baseline"].get("clients", 0) or 0)
    bars = ax.bar(labels, vals, color=["#3b6ea5", "#2e8b57"], width=0.55)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(vals) * 0.02, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"Total measured bytes ({rounds} rounds, {clients} clients)")
    ax.set_title("(a) Actual serialized traffic")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.text(0.5, 0.83, f"reduction = {SUMMARY['main']['communication_comparison']['reduction_pct']:.2f}%", transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold", color="#b03a2e")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    colors = ["#7f8c8d", "#2e8b57", "#d68910", "#b03a2e"]
    for row, color in zip(COMM, colors):
        ax.scatter(row["total_bytes"] / 1e6, row["f1"], s=70, color=color, zorder=3)
        ax.annotate(
            row["compression_label"],
            (row["total_bytes"] / 1e6, row["f1"]),
            textcoords="offset points",
            xytext=(5, 7),
            fontsize=7.5,
        )
    ax.set_xlabel(f"Total bytes (millions, {SUMMARY['communication_sweep']['meta'].get('rounds', 'N/A')}-round sweep)")
    ax.set_ylabel("Test F1")
    ax.set_title("(b) Utility vs measured bytes")
    f1_values = [row["f1"] for row in COMM]
    ax.set_ylim(min(f1_values) - 0.006, max(f1_values) + 0.006)
    ax.grid(alpha=0.25)

    _save(fig, "paper_communication.png")


# --------------------------------------------------------------------------- #
# 4. Privacy sweep
# --------------------------------------------------------------------------- #
def privacy() -> None:
    labels = [row["privacy_level"] for row in PRIV]
    f1 = [row["f1"] for row in PRIV]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    ax = axes[0]
    colors = ["#7f8c8d"] + ["#2e8b57"] * (len(labels) - 1)
    bars = ax.bar(labels, f1, color=colors, width=0.62)
    span = max(f1) - min(f1)
    offset = max(0.0004, span * 0.06)
    for bar, value in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.4f}", ha="center", fontsize=7.5)
    ax.set_ylim(min(f1) - span * 0.35 - 0.001, max(f1) + span * 0.45 + 0.001)
    ax.set_ylabel(f"Test F1 ({SUMMARY['privacy_sweep']['meta'].get('rounds', 'N/A')} rounds)")
    ax.set_title("(a) Utility across privacy settings")
    ax.tick_params(axis="x", labelsize=7.2, rotation=18)
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    sigmas = [row["noise_multiplier"] for row in PRIV if row["noise_multiplier"] > 0 and row.get("epsilon_estimate") is not None]
    eps_vals = [row["epsilon_estimate"] for row in PRIV if row["noise_multiplier"] > 0 and row.get("epsilon_estimate") is not None]
    ax.plot(sigmas, eps_vals, marker="o", color="#b03a2e", linewidth=1.6)
    for s, e in zip(sigmas, eps_vals):
        ax.annotate(f"{e:,.0f}", (s, e), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xlabel("Gaussian noise multiplier σ")
    ax.set_ylabel("Single-release ε estimate")
    ax.set_title("(b) Reported privacy estimate")
    ax.grid(alpha=0.25)

    _save(fig, "paper_privacy.png")


# --------------------------------------------------------------------------- #
# 5. Scalability
# --------------------------------------------------------------------------- #
def scalability() -> None:
    clients = [row["clients"] for row in SCALE]
    rounds = SUMMARY["scalability"]["meta"].get("rounds", "N/A")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(clients, [row["train_time_s"] for row in SCALE], marker="o", color="#3b6ea5")
    for c, row in zip(clients, SCALE):
        ax.annotate(f'{row["train_time_s"]:.2f}s', (c, row["train_time_s"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlabel("Clients")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("(a) Training time")
    ax.set_xticks(clients)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(clients, [row["total_bytes"] / 1e6 for row in SCALE], marker="s", color="#2e8b57")
    for c, row in zip(clients, SCALE):
        ax.annotate(f'{row["total_bytes"] / 1e6:.2f}M', (c, row["total_bytes"] / 1e6), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlabel("Clients")
    ax.set_ylabel("Total bytes (millions)")
    ax.set_title("(b) Communication")
    ax.set_xticks(clients)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(clients, [row["f1"] for row in SCALE], marker="^", color="#b03a2e")
    for c, row in zip(clients, SCALE):
        ax.annotate(f'{row["f1"]:.4f}', (c, row["f1"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    f1_values = [row["f1"] for row in SCALE]
    ax.set_ylim(max(0.0, min(f1_values) - 0.03), min(1.0, max(f1_values) + 0.02))
    ax.set_xlabel("Clients")
    ax.set_ylabel(f"Test F1 ({rounds} rounds)")
    ax.set_title("(c) Detection")
    ax.set_xticks(clients)
    ax.grid(alpha=0.25)

    _save(fig, "paper_scalability.png")


# --------------------------------------------------------------------------- #
# 6. Permutation importance
# --------------------------------------------------------------------------- #
def feature_importance() -> None:
    perm = SUMMARY["explainability"]["permutation_importance"]
    positive = [f for f in perm["top_features"] if f["importance_mean"] > 0]
    top = positive[:10] if positive else perm["top_features"][:10]
    human = {
        "min_max_day_ratio": "min/max day ratio",
        "max_daily_drop": "max daily drop",
        "daily_diff_std": "daily-difference std",
        "mean_daily_drop": "mean daily drop",
        "night_day_ratio": "night/day ratio",
        "peak_offpeak_ratio": "peak/off-peak ratio",
        "min_consumption": "minimum consumption",
        "weekend_weekday_ratio": "weekend/weekday ratio",
        "load_variation": "load variation",
    }
    names = [human.get(f["feature"], f["feature"].replace("_", " ")) for f in top][::-1]
    means = [f["importance_mean"] for f in top][::-1]
    stds = [f["importance_std"] for f in top][::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color="#2e8b57", alpha=0.85, error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, names, fontsize=8.5)
    n_rows = perm.get("n_rows", "N/A")
    rows_text = f"{int(n_rows):,}" if isinstance(n_rows, (int, float)) else str(n_rows)
    ax.set_xlabel(f"Mean ROC-AUC drop ({perm.get('n_repeats', 'N/A')} repeats, {rows_text} rows)")
    ax.set_title("Permutation importance of the final global model")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(np.array(means) + np.array(stds)) * 1.15)
    _save(fig, "paper_feature_importance.png")


# --------------------------------------------------------------------------- #
# 7. Confusion matrices and consumer-level gap
# --------------------------------------------------------------------------- #
def confusion_and_gap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for ax, name, title in zip(axes, ["fedavg_baseline", "powershield_fl"], ["FedAvg", "PowerShield-FL"]):
        m = _read_run(_row_for(name))["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.05)
        for (i, j), value in np.ndenumerate(cm):
            color = "white" if value > cm.max() * 0.55 else "#17324d"
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)
        ax.set_xticks([0, 1], ["Pred. normal", "Pred. theft"], fontsize=8)
        ax.set_yticks([0, 1], ["Actual normal", "Actual theft"], fontsize=8)
        seed = _row_for(name).get("seed", SEEDS[0] if SEEDS else "N/A")
        ax.set_title(f"{title}\nwindow-level test confusion matrix (seed {seed})")
        ax.set_xlabel(f'F1 = {_row_for(name)["f1"]:.4f}')
    _save(fig, "paper_confusion.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    metrics = ["Precision", "Recall", "F1"]
    win_fed = [FED["fedavg_baseline"]["precision"], FED["fedavg_baseline"]["recall"], FED["fedavg_baseline"]["f1"]]
    win_prop = [FED["powershield_fl"]["precision"], FED["powershield_fl"]["recall"], FED["powershield_fl"]["f1"]]
    con_fed = [CONSUMER["fedavg_baseline"]["precision"], CONSUMER["fedavg_baseline"]["recall"], CONSUMER["fedavg_baseline"]["f1"]]
    con_prop = [CONSUMER["powershield_fl"]["precision"], CONSUMER["powershield_fl"]["recall"], CONSUMER["powershield_fl"]["f1"]]
    x = np.arange(len(metrics))
    width = 0.19
    ax.bar(x - 1.5 * width, win_fed, width, label="FedAvg — window", color="#3b6ea5")
    ax.bar(x - 0.5 * width, win_prop, width, label="PowerShield — window", color="#2e8b57")
    ax.bar(x + 0.5 * width, con_fed, width, label="FedAvg — consumer", color="#9fc5e8")
    ax.bar(x + 1.5 * width, con_prop, width, label="PowerShield — consumer", color="#a9dfbf")
    for i, vals in enumerate([win_fed, win_prop, con_fed, con_prop]):
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1.5) * width, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Window-level vs consumer-level performance ({len(SEEDS)} seed(s))")
    ax.legend(fontsize=7.2, ncol=2, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.22)
    _save(fig, "paper_window_vs_consumer.png")


if __name__ == "__main__":
    architecture()
    convergence()
    communication()
    privacy()
    scalability()
    feature_importance()
    confusion_and_gap()
    print("done")
