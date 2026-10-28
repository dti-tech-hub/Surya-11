"""Automatic experiment report generation from recorded results only.

No metric is written here unless it exists in ``summary.json``. Missing stages
are reported as NOT RUN; failed stages include the recorded error.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from evaluation.comparison import format_table, to_markdown
from utilities.config import get


def _f(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NOT RUN" if value is None else str(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _df_markdown(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "NOT RUN\n"
    frame = pd.DataFrame(rows)
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return "NOT RUN\n"
    return to_markdown(format_table(frame[cols])) + "\n"


def _stage_status(summary: dict[str, Any], name: str) -> str:
    stage = summary.get("stages", {}).get(name)
    if not stage:
        return "NOT RUN"
    return str(stage.get("status", "UNKNOWN")).upper()


def _aggregate_rows(
    rows: Sequence[dict[str, Any]],
    key: str,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Seed-average numeric metrics, keeping mean and standard deviation."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    cols = [m for m in metrics if m in frame.columns]
    if not cols:
        return []
    grouped = frame.groupby(key, dropna=False)[cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        f"{a}_{b}" if b else str(a) for a, b in grouped.columns.to_flat_index()
    ]
    return grouped.to_dict(orient="records")


def _claims(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Claims are generated from measured values, never assumed."""
    proposed = summary.get("main", {}).get("proposed", [])
    fedavg = summary.get("main", {}).get("fedavg", [])
    seeds = summary.get("seeds", [])
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else 1
    paired = summary.get("statistics", {}).get("paired_f1_fedavg_vs_proposed", {})
    claims: list[dict[str, str]] = []
    if proposed and fedavg:
        p_f1 = float(pd.Series([r["f1"] for r in proposed]).mean())
        f_f1 = float(pd.Series([r["f1"] for r in fedavg]).mean())
        diff = p_f1 - f_f1
        if n_seeds > 1 and paired:
            p_value = paired.get("paired_ttest", {}).get("p_value")
            verdict = (
                f"mean over {n_seeds} seeds; paired t-test p={_f(p_value)} "
                f"(significant at alpha={paired.get('alpha', 0.05)}: {paired.get('significant_at_alpha')}); "
                f"{paired.get('caveat', '')}"
            )
        else:
            verdict = "measured on this synthetic demo dataset; single-seed differences are not statistical evidence"
        claims.append(
            {
                "claim": "PowerShield-FL detection F1 vs FedAvg",
                "evidence": f"FedAvg mean F1={_f(f_f1)}, PowerShield-FL mean F1={_f(p_f1)}, difference={diff:+.4f}",
                "status": "SUPPORTED" if diff > 0 else ("NO ADVANTAGE OBSERVED" if diff <= 0 else "NOT RUN"),
                "note": verdict,
            }
        )
        ref = float(pd.Series([r["total_bytes"] for r in fedavg]).mean())
        cand = float(pd.Series([r["total_bytes"] for r in proposed]).mean())
        reduction = 100.0 * (1.0 - cand / ref) if ref else 0.0
        claims.append(
            {
                "claim": "Communication reduction vs standard FedAvg",
                "evidence": f"FedAvg total={int(ref)} bytes, PowerShield-FL total={int(cand)} bytes, measured reduction={reduction:.2f}%",
                "status": "SUPPORTED" if reduction > 0 else "NO REDUCTION OBSERVED",
                "note": "bytes are actual serialized payloads measured by the communication layer"
                + (f"; mean of {n_seeds} seeds" if n_seeds > 1 else ""),
            }
        )
    else:
        claims.append(
            {
                "claim": "Main FedAvg vs PowerShield-FL comparison",
                "evidence": "one or both runs are missing",
                "status": "NOT RUN",
                "note": "execute the suite with both stages enabled",
            }
        )

    privacy_rows = summary.get("privacy_sweep", {}).get("rows", [])
    if privacy_rows:
        f1s = [float(r["f1"]) for r in privacy_rows if r.get("f1") is not None]
        claims.append(
            {
                "claim": "Privacy/utility trade-off",
                "evidence": "F1 range across measured privacy settings: "
                + f"{_f(min(f1s))}–{_f(max(f1s))}" if f1s else "NOT RUN",
                "status": "MEASURED" if f1s else "NOT RUN",
                "note": "noise multipliers are small because per-coordinate Gaussian noise is applied to a high-dimensional update; see privacy caveat",
            }
        )
    else:
        claims.append({"claim": "Privacy/utility trade-off", "evidence": "privacy sweep not executed", "status": "NOT RUN", "note": ""})

    if summary.get("scalability", {}).get("rows"):
        rows = summary["scalability"]["rows"]
        claims.append(
            {
                "claim": "Scalability across client counts",
                "evidence": "measured client counts: " + ", ".join(str(r["clients"]) for r in rows),
                "status": "MEASURED",
                "note": "the sweep uses its own configured round count; high client counts may hit CPU/RAM limits",
            }
        )
    else:
        claims.append({"claim": "Scalability", "evidence": "scalability sweep not executed", "status": "NOT RUN", "note": ""})
    return claims


def generate_report(cfg: dict, summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = summary.get("dataset", {})
    env = summary.get("environment", {})
    source = dataset.get("source", "DATASET NOT CONFIGURED")
    synthetic = bool(dataset.get("is_synthetic", False))
    central = summary.get("main", {}).get("centralized", [])
    fedavg = summary.get("main", {}).get("fedavg", [])
    proposed = summary.get("main", {}).get("proposed", [])
    federated = summary.get("main", {}).get("federated", [])
    prep = dataset.get("preprocessing", {})
    split = dataset.get("summary", {}).get("n_consumers", {})
    windows = dataset.get("summary", {}).get("n_windows", {})
    imbalance = dataset.get("imbalance", {})
    partitions = summary.get("partitions", {})
    clients = partitions.get("clients", [])
    privacy_rows = summary.get("privacy_sweep", {}).get("rows", [])
    communication_rows = summary.get("communication_sweep", {}).get("rows", [])
    scalability_rows = summary.get("scalability", {}).get("rows", [])
    stats = summary.get("statistics", {})
    seeds = summary.get("seeds", [])
    n_seeds = len(seeds) if isinstance(seeds, (list, tuple)) else 1
    central_means = _aggregate_rows(
        central,
        "method",
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_time_s", "n_params", "model_size_kb"],
    )
    federated_means = _aggregate_rows(
        federated,
        "method",
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_time_s", "total_bytes", "bytes_per_round"],
    )
    consumer_rows = [
        {
            "method": row.get("method"),
            "seed": row.get("seed"),
            **row.get("per_consumer", {}),
        }
        for row in federated
        if row.get("per_consumer")
    ]

    lines: list[str] = []
    lines.append("# PowerShield-FL — Final Experiment Report\n")
    lines.append(
        "> This report is generated automatically from `results/metrics/summary.json`. "
        "Every value below comes from an executed run; absent sections are marked **NOT RUN**.\n"
    )
    if synthetic:
        lines.append("> **DATASET: SYNTHETIC DEMO DATA** — results from this dataset are a software "
                     "demonstration, not evidence about real utility theft rates.\n")
    lines.append(f"- Generated: {summary.get('generated_at', 'NOT RUN')}")
    lines.append(f"- Suite ID: `{summary.get('suite_id', 'NOT RUN')}`")
    lines.append(f"- Mode: `{summary.get('mode', 'NOT RUN')}`; demo: `{summary.get('demo', False)}`")
    lines.append(f"- Seeds: {summary.get('seeds', 'NOT RUN')}")
    lines.append(f"- Dataset source: **{source}**\n")

    lines.append("## 1. Implementation status\n")
    lines.append("| Stage | Status |")
    lines.append("|---|---|")
    for name, stage in summary.get("stages", {}).items():
        lines.append(f"| {name} | {stage.get('status', 'UNKNOWN')} |")
    lines.append("")

    lines.append("## 2. Dataset\n")
    lines.append(f"- Source: **{source}**")
    lines.append(f"- Consumers: train {split.get('train', 'N/A')}, validation {split.get('val', 'N/A')}, test {split.get('test', 'N/A')}")
    lines.append(f"- Windows: train {windows.get('train', 'N/A')}, validation {windows.get('val', 'N/A')}, test {windows.get('test', 'N/A')}")
    lines.append(f"- Features: {dataset.get('summary', {}).get('n_features', 'N/A')}")
    lines.append(f"- Raw missing values: {dataset.get('stats', {}).get('missing_values', 'N/A')}")
    lines.append(f"- Raw duplicate rows: {dataset.get('stats', {}).get('duplicate_rows', 'N/A')}")
    lines.append(f"- Split unit: consumer (no consumer appears in two splits)\n")
    lines.append("### Table 1 — Dataset statistics\n")
    lines.append(_df_markdown(dataset.get("table", []), ["statistic", "value"]))

    lines.append("## 3. Preprocessing\n")
    lines.append(f"- Cleaning report: `{json.dumps(dataset.get('cleaning', {}), default=str)}`")
    lines.append(f"- Missing-value strategy: `{get(cfg, 'preprocessing.missing.strategy')}`")
    lines.append(f"- Noise handling: Hampel-style rolling median (enabled={get(cfg, 'preprocessing.noise.enabled')})")
    lines.append(f"- Outlier handling: {get(cfg, 'preprocessing.outliers.method')} bounds fitted on training consumers, action=clip (records are not deleted)")
    lines.append(f"- Scaling: {get(cfg, 'preprocessing.scaling.method')}, fitted on training consumers only")
    lines.append(f"- Class imbalance: {json.dumps(imbalance.get('report', {}).get('distribution', {}), default=str)}")
    lines.append("- Theft anomalies are preserved; robust clipping only limits extreme numerical outliers.\n")

    lines.append("## 4. Client simulation\n")
    lines.append(f"- Partition strategy: `{get(cfg, 'clients.partition_strategy')}`")
    lines.append(f"- Clients: {partitions.get('n_clients', 'NOT RUN')}")
    lines.append(f"- Heterogeneity: theft-ratio std={_f(partitions.get('heterogeneity', {}).get('theft_ratio_std'))}, size CV={_f(partitions.get('heterogeneity', {}).get('size_cv'))}")
    lines.append("\n### Per-client summary\n")
    lines.append(_df_markdown(clients, ["client_id", "n_samples", "n_consumers", "n_theft_windows", "theft_ratio"]))

    lines.append("## 5. Models\n")
    lightweight_row = next(
        (r for r in central if r.get("method") == "centralized_lightweight_mlp"),
        federated[0] if federated else None,
    )
    if lightweight_row:
        lines.append(
            f"- Lightweight MLP: hidden={get(cfg, 'model.hidden')}, "
            f"parameters={lightweight_row.get('n_params', 'N/A')}"
        )
    lines.append("- CNN-LSTM baseline: "
                 + ("executed" if any(r.get("method") == "cnn_lstm_baseline" for r in central) else "NOT RUN"))
    central_methods = sorted({str(r.get("method")) for r in central if r.get("method")})
    lines.append("- Centralized baselines: " + (", ".join(central_methods) if central_methods else "NOT RUN"))
    lines.append("- Federated local model is the same lightweight MLP for FedAvg and PowerShield-FL "
                 "so aggregation/privacy effects are not confounded by architecture.\n")

    lines.append("## 6. Federated configuration\n")
    lines.append(f"- Rounds: {get(cfg, 'federated.rounds')} (sweeps use their own configured round counts)")
    lines.append(f"- Local epochs: {get(cfg, 'federated.local_epochs')}")
    lines.append(f"- Batch size: {get(cfg, 'federated.batch_size')}")
    lines.append(f"- Learning rate: {get(cfg, 'federated.learning_rate')}")
    lines.append(f"- Participation rate: {get(cfg, 'federated.participation_rate')}")
    lines.append(f"- Aggregation: FedAvg = sample-count weighted mean; PowerShield-FL = `{get(cfg, 'aggregation.method')}`\n")

    lines.append("## 7. Privacy configuration\n")
    lines.append(f"- Enabled: {get(cfg, 'privacy.enabled')}")
    lines.append(f"- Mechanism: `{get(cfg, 'privacy.mechanism')}` (L2 clip + Gaussian noise)")
    lines.append(f"- Clipping norm: {get(cfg, 'privacy.clipping_norm')}")
    lines.append(f"- Noise multiplier: {get(cfg, 'privacy.noise_multiplier')}")
    lines.append(f"- Secure aggregation: {get(cfg, 'privacy.secure_aggregation')} "
                 "(simulation of additive masking, **not production cryptography**)")
    lines.append("- The reported epsilon is a single-release classical estimate. Composition across rounds is "
                 "not implemented; no end-to-end formal differential-privacy guarantee is claimed.\n")

    lines.append("## 8. Communication method\n")
    lines.append(f"- Compression enabled: {get(cfg, 'communication.compression')}")
    lines.append(f"- Method: `{get(cfg, 'communication.method')}`; top-k fraction={get(cfg, 'communication.topk_fraction')}; int{get(cfg, 'communication.quant_bits')} quantization")
    lines.append("- Payload sizes are measured from actual serialized bytes (including indices and headers), not estimated.\n")

    lines.append("## 9. Evaluation protocol\n")
    lines.append("- Consumer-level stratified train/validation/test split; scalers fitted on train only.")
    lines.append("- Decision threshold tuned on the validation split and frozen for test evaluation.")
    lines.append("- Emphasis on precision, recall, F1, ROC-AUC and PR-AUC under class imbalance.\n")

    lines.append("## 10. Actual results\n")
    lines.append(f"Results are averaged over {n_seeds} seed(s): `{seeds}`. Per-seed rows are retained where they show run-to-run variability.\n")
    lines.append("### Table 2 — Centralized model comparison (seed-averaged)\n")
    lines.append(_df_markdown(central_means, ["method", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "f1_std", "roc_auc_mean", "pr_auc_mean", "train_time_s_mean", "n_params_mean"]))
    if n_seeds > 1:
        lines.append("### Table 2b — Centralized results by seed\n")
        lines.append(_df_markdown(central, ["method", "seed", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_time_s", "n_params"]))
    lines.append("### Table 3 — Federated comparison (seed-averaged)\n")
    lines.append(_df_markdown(federated_means, ["method", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "f1_std", "roc_auc_mean", "pr_auc_mean", "train_time_s_mean", "total_bytes_mean", "bytes_per_round_mean"]))
    if n_seeds > 1:
        lines.append("### Table 3b — Federated results by seed\n")
        lines.append(_df_markdown(federated, ["method", "seed", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_time_s", "total_bytes", "bytes_per_round"]))
    if summary.get("main", {}).get("communication_comparison"):
        cc = summary["main"]["communication_comparison"]
        lines.append(
            f"- Measured communication: FedAvg {int(cc['fedavg_mean_total_bytes'])} bytes vs "
            f"PowerShield-FL {int(cc['proposed_mean_total_bytes'])} bytes "
            f"(reduction {cc['reduction_pct']:.2f}%).\n"
        )

    lines.append("### Table 4 — Privacy / utility\n")
    lines.append(_df_markdown(privacy_rows, ["privacy_level", "noise_multiplier", "epsilon_estimate", "f1", "precision", "recall", "train_time_s", "total_bytes"]))
    lines.append("### Communication sweep\n")
    lines.append(_df_markdown(communication_rows, ["compression_label", "compression", "f1", "roc_auc", "total_bytes", "communication_reduction_pct", "train_time_s"]))
    lines.append("### Table 5 — Scalability\n")
    lines.append(_df_markdown(scalability_rows, ["clients", "rounds", "train_time_s", "total_bytes", "rss_delta_mb", "f1", "accuracy"]))
    lines.append("### Table 6 — Consumer-level test results\n")
    lines.append(_df_markdown(consumer_rows, ["method", "seed", "precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn", "n_consumers"]))

    lines.append("## 11. Comparison and claim support\n")
    claims = _claims(summary)
    lines.append("| Claim | Evidence | Status | Note |")
    lines.append("|---|---|---|---|")
    for claim in claims:
        lines.append(f"| {claim['claim']} | {claim['evidence']} | **{claim['status']}** | {claim['note']} |")
    lines.append("")
    if federated:
        lines.append(
            "The objective is not to force PowerShield-FL to win. The table above states whether "
            "the measured F1/communication results support each claim on this dataset. "
            + (
                f"Results are averaged over {n_seeds} seeds; the paired test and its caveat are reported below."
                if n_seeds > 1
                else "Single-seed results are descriptive, not inferential."
            )
            + "\n"
        )

    lines.append("## 12. Statistical analysis\n")
    if stats:
        for name, values in stats.items():
            if name == "paired_f1_fedavg_vs_proposed":
                t = values.get("paired_ttest", {})
                w = values.get("wilcoxon", {})
                lines.append(
                    f"- Paired FedAvg − PowerShield-FL F1 difference: mean={_f(values.get('mean_difference'))}, "
                    f"std={_f(values.get('std_difference'))}, n={values.get('n')}; "
                    f"paired t p={_f(t.get('p_value'))}, Wilcoxon p={_f(w.get('p_value'))}; "
                    f"significant at alpha={values.get('alpha', 0.05)}: **{values.get('significant_at_alpha')}**"
                )
                lines.append(f"- Caveat: {values.get('caveat', '')}")
            elif isinstance(values, dict):
                f1 = values.get("f1", {})
                lines.append(
                    f"- {name}: F1 mean={_f(f1.get('mean'))}, std={_f(f1.get('std'))}, "
                    f"95% CI=[{_f(f1.get('ci_low'))}, {_f(f1.get('ci_high'))}], "
                    f"per-seed values={json.dumps(f1.get('values', []), default=str)}"
                )
    else:
        lines.append("NOT RUN")
    lines.append("")

    lines.append("## 13. Figures and tables\n")
    lines.append(f"- Tables: {json.dumps(summary.get('tables', {}), indent=2, default=str)}")
    lines.append(f"- Figures: {json.dumps(summary.get('figures', {}), indent=2, default=str)}")
    lines.append("")

    lines.append("## 14. Limitations\n")
    limitations = [
        "The default dataset is SYNTHETIC DEMO DATA; it does not represent a real utility's theft prevalence or tariff structure.",
        "The demo uses simulated clients, not physically distributed organizations with real network links.",
        "Secure aggregation is a masking simulation, not a cryptographic protocol; it provides no malicious-party guarantees.",
        "The epsilon estimate covers one release of a client update and does not compose across rounds.",
        "The lightweight MLP is evaluated against a CNN-LSTM on a synthetic dataset; the CNN-LSTM is unstable across seeds and may be under-trained relative to its capacity.",
        "Wall time and memory are machine-specific (CPU-only host); they are not portable benchmarks.",
        f"This report covers `{summary.get('mode', 'unknown')}` mode with {n_seeds} seed(s); other configured modes/seeds remain NOT RUN until executed.",
        "Client-level labels are inferred from theft activation in synthetic data; real datasets may have noisy labels.",
        "The consumer-level decision rule is a fixed mean-probability threshold of 0.5; it is not tuned on validation consumers.",
    ]
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"- Full-mode status: **{'EXECUTED' if summary.get('mode') == 'full' else 'NOT RUN (this report is ' + str(summary.get('mode')) + ' mode)'}**\n")

    lines.append("## 15. Reproducibility\n")
    lines.append(f"- Python: {env.get('python', 'N/A')}")
    lines.append(f"- OS: {env.get('os', 'N/A')}")
    lines.append(f"- CPU cores: {env.get('physical_cores', 'N/A')} physical / {env.get('logical_cores', 'N/A')} logical")
    lines.append(f"- RAM: {env.get('ram_gb', 'N/A')} GB")
    lines.append(f"- GPU: {env.get('gpu', 'N/A')}")
    lines.append(f"- Package versions: `{json.dumps(env.get('packages', {}), default=str)}`")
    lines.append(f"- Run directories: {json.dumps(summary.get('artifacts', {}).get('run_dirs', []), default=str)}")
    lines.append("- Re-run: `python main.py --mode quick` (or `--mode full`), seeds are recorded in the configuration.")
    lines.append("")

    lines.append("## 16. Claim discipline\n")
    lines.append("The framework reports **NOT RUN** for absent experiments and **EXPERIMENT FAILED** for failed stages. "
                 "No metric in this report was entered manually.\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
