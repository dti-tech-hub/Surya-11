"""PowerShield-FL Streamlit dashboard.

Reads only real artifacts under results/ and models/checkpoints/. When the
experiment suite has not been executed, every page shows:

    No experimental result available. Run the experiment first.

No demo metrics are hard-coded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainability.explanations import explain_prediction, human_feature_name  # noqa: E402
from models.base import TorchBinaryClassifier  # noqa: E402
from models.lightweight import LightweightMLP  # noqa: E402

SUMMARY_PATH = ROOT / "results" / "metrics" / "summary.json"
NO_RESULTS = "No experimental result available. Run the experiment first."
DEMO_BADGE = "SYNTHETIC DEMO DATA"

st.set_page_config(page_title="PowerShield-FL", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    return load_json(str(SUMMARY_PATH), SUMMARY_PATH.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def load_checkpoint(checkpoint: str, meta_path: str, mtime: float):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    hidden = list(meta.get("hidden", [64, 32]))
    n_inputs = int(meta["n_inputs"])
    model = TorchBinaryClassifier.load(
        checkpoint,
        lambda: LightweightMLP(n_inputs, hidden=hidden, dropout=0.0),
    )
    return model, meta


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NOT RUN"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def mean_metric(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(pd.Series([r.get(key) for r in rows], dtype=float).mean())


def metric_cards(rows: list[dict], label: str) -> None:
    if not rows:
        st.info(f"{label}: NOT RUN")
        return
    cols = st.columns(6)
    cols[0].metric("F1", fmt(mean_metric(rows, "f1")))
    cols[1].metric("Recall", fmt(mean_metric(rows, "recall")))
    cols[2].metric("Precision", fmt(mean_metric(rows, "precision")))
    cols[3].metric("ROC-AUC", fmt(mean_metric(rows, "roc_auc")))
    cols[4].metric("Training time (s)", fmt(mean_metric(rows, "train_time_s"), 1))
    cols[5].metric("Communication (bytes)", fmt(mean_metric(rows, "total_bytes"), 0))


def rows_frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame[[c for c in columns if c in frame.columns]]


def image_if_exists(path_value: str | None, caption: str = "") -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        st.image(str(path), caption=caption or path.name, width="stretch")


summary_data = summary()

st.title("⚡ PowerShield-FL")
st.caption("Privacy-preserving, communication-efficient federated electricity-theft detection")

if summary_data is None:
    st.warning(NO_RESULTS)
    st.code("python main.py --demo\n# or\npython main.py --mode quick", language="bash")
    st.stop()

dataset = summary_data.get("dataset", {})
synthetic = bool(dataset.get("is_synthetic", False))
if synthetic:
    st.warning(f"**{DEMO_BADGE}** — all metrics on this dashboard come from the synthetic demo generator, not a real utility dataset.")
else:
    st.info(f"Dataset source: {dataset.get('source', 'unknown')}")

with st.sidebar:
    st.header("Run information")
    st.write(f"**Suite:** `{summary_data.get('suite_id', 'N/A')}`")
    st.write(f"**Mode:** {summary_data.get('mode', 'N/A')}")
    st.write(f"**Seeds:** {summary_data.get('seeds', 'N/A')}")
    st.write(f"**Generated:** {summary_data.get('generated_at', 'N/A')}")
    stage_badges = []
    for name, stage in summary_data.get("stages", {}).items():
        status = str(stage.get("status", "UNKNOWN"))
        icon = "✅" if status == "PASS" else "❌" if "FAILED" in status else "⚠️"
        stage_badges.append(f"{icon} {name}: {status}")
    st.markdown("\n\n".join(stage_badges))
    st.divider()
    st.code("streamlit run dashboard/app.py", language="bash")

pages = st.tabs(
    [
        "Overview",
        "Dataset",
        "Federated Clients",
        "Training",
        "Model Comparison",
        "Privacy",
        "Communication",
        "Detection",
        "Explainability",
        "Report",
    ]
)

central = summary_data.get("main", {}).get("centralized", [])
fedavg = summary_data.get("main", {}).get("fedavg", [])
proposed = summary_data.get("main", {}).get("proposed", [])
federated = summary_data.get("main", {}).get("federated", [])
partitions = summary_data.get("partitions", {})

# --------------------------------------------------------------------------- #
with pages[0]:
    st.subheader("Overview")
    ds_summary = dataset.get("summary", {})
    cols = st.columns(4)
    cols[0].metric("Consumers", ds_summary.get("n_consumers", {}).get("train", 0) + ds_summary.get("n_consumers", {}).get("val", 0) + ds_summary.get("n_consumers", {}).get("test", 0))
    cols[1].metric("Federated clients", len(partitions.get("clients", [])))
    cols[2].metric("Federated rounds", summary_data.get("config", {}).get("federated", {}).get("rounds", "NOT RUN"))
    cols[3].metric("Features", ds_summary.get("n_features", "N/A"))
    st.markdown("### PowerShield-FL (proposed)")
    metric_cards(proposed, "PowerShield-FL")
    st.markdown("### FedAvg baseline")
    metric_cards(fedavg, "FedAvg")
    st.markdown("### Centralized baselines")
    metric_cards(central, "Centralized")

    cc = summary_data.get("main", {}).get("communication_comparison")
    if cc:
        st.metric("Measured communication reduction vs FedAvg", f"{cc.get('reduction_pct', 0):.2f}%")
    else:
        st.info("Communication comparison: NOT RUN")
    if summary_data.get("demo"):
        st.info("DEMO run: sweeps are intentionally skipped. Run `python main.py --mode quick` for the full quick suite.")
    st.markdown("**Data source:** " + str(dataset.get("source", "unknown")))

# --------------------------------------------------------------------------- #
with pages[1]:
    st.subheader("Dataset")
    table = pd.DataFrame(dataset.get("table", []))
    if table.empty:
        st.info(NO_RESULTS)
    else:
        # The summary stores a mixture of strings and numbers in `value`;
        # stringify for display so Arrow does not emit conversion warnings.
        if "value" in table.columns:
            table["value"] = table["value"].map(str)
        st.dataframe(table, width="stretch", hide_index=True)
    st.markdown("#### Class distribution (consumers)")
    image_if_exists(summary_data.get("figures", {}).get("class_distribution"))
    st.markdown("#### Class imbalance report")
    st.json(dataset.get("imbalance", {}))
    st.markdown("#### Cleaning report")
    st.json(dataset.get("cleaning", {}))

# --------------------------------------------------------------------------- #
with pages[2]:
    st.subheader("Federated clients")
    clients = partitions.get("clients", [])
    if clients:
        st.dataframe(pd.DataFrame(clients), width="stretch", hide_index=True)
        st.markdown(
            f"Partition strategy: **{partitions.get('strategy', 'N/A')}** · "
            f"theft-ratio std: **{fmt(partitions.get('heterogeneity', {}).get('theft_ratio_std'))}** · "
            f"size CV: **{fmt(partitions.get('heterogeneity', {}).get('size_cv'))}**"
        )
        image_if_exists(summary_data.get("figures", {}).get("client_distribution"))
    else:
        st.info(NO_RESULTS)
    st.caption("Only consumer IDs / aggregate statistics are shown; raw consumption stays with clients.")

# --------------------------------------------------------------------------- #
with pages[3]:
    st.subheader("Training")
    run_dir_value = summary_data.get("artifacts", {}).get("dashboard_run_dir")
    metrics_csv = Path(run_dir_value) / "metrics.csv" if run_dir_value else None
    if metrics_csv and metrics_csv.exists():
        history = pd.read_csv(metrics_csv)
        st.markdown("PowerShield-FL round history (actual run):")
        st.dataframe(history, width="stretch", hide_index=True)
        chart_cols = [c for c in ["train_loss", "val_loss", "val_f1", "val_accuracy"] if c in history.columns]
        if chart_cols:
            st.line_chart(history.set_index("round")[chart_cols])
        st.markdown("Round communication")
        if "bytes_upload" in history.columns:
            st.bar_chart(history.set_index("round")[["bytes_upload", "bytes_download"]])
    else:
        st.info(NO_RESULTS)
    image_if_exists(summary_data.get("figures", {}).get("training_convergence"))

# --------------------------------------------------------------------------- #
with pages[4]:
    st.subheader("Model comparison")
    comparison = central + federated
    if comparison:
        frame = pd.DataFrame(comparison)
        columns = [
            "method", "kind", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "train_time_s", "n_params", "model_size_kb", "total_bytes", "seed",
        ]
        st.dataframe(frame[[c for c in columns if c in frame.columns]], width="stretch", hide_index=True)
        image_if_exists(summary_data.get("figures", {}).get("f1_comparison"))
        c1, c2 = st.columns(2)
        with c1:
            image_if_exists(summary_data.get("figures", {}).get("precision_recall_comparison"))
        with c2:
            image_if_exists(summary_data.get("figures", {}).get("roc_curve"))
        c3, c4 = st.columns(2)
        with c3:
            image_if_exists(summary_data.get("figures", {}).get("confusion_matrix"))
        with c4:
            image_if_exists(summary_data.get("figures", {}).get("training_time_comparison"))
        st.markdown("#### Note")
        st.caption("Centralized models use pooled data. FedAvg and PowerShield-FL use the same lightweight model and client partitions. Single-seed results are descriptive, not significance tests.")
    else:
        st.info(NO_RESULTS)

# --------------------------------------------------------------------------- #
with pages[5]:
    st.subheader("Privacy / utility")
    rows = summary_data.get("privacy_sweep", {}).get("rows", [])
    if rows:
        st.dataframe(rows_frame(rows, ["privacy_level", "noise_multiplier", "epsilon_estimate", "f1", "precision", "recall", "train_time_s", "total_bytes"]), width="stretch", hide_index=True)
    else:
        st.info("Privacy sweep: NOT RUN. Run `python main.py --mode quick`.")
    meta = summary_data.get("privacy_sweep", {}).get("meta", {})
    if meta:
        st.json(meta)
    st.warning("Secure aggregation in this project is a simulation of additive masking, not production cryptography. The epsilon figure is a single-release estimate without composition; no end-to-end formal DP guarantee is claimed.")
    image_if_exists(summary_data.get("figures", {}).get("privacy_utility_tradeoff"))

# --------------------------------------------------------------------------- #
with pages[6]:
    st.subheader("Communication")
    rows = summary_data.get("communication_sweep", {}).get("rows", [])
    if rows:
        st.dataframe(rows_frame(rows, ["compression_label", "compression", "f1", "roc_auc", "total_bytes", "bytes_per_round", "communication_reduction_pct"]), width="stretch", hide_index=True)
    else:
        st.info("Communication sweep: NOT RUN. Run `python main.py --mode quick`.")
    cc = summary_data.get("main", {}).get("communication_comparison")
    if cc:
        st.json(cc)
    st.caption("All byte counts are measured from actual serialized payloads, not estimated from model size.")
    image_if_exists(summary_data.get("figures", {}).get("communication_comparison"))
    image_if_exists(summary_data.get("figures", {}).get("communication_accuracy_tradeoff"))

# --------------------------------------------------------------------------- #
with pages[7]:
    st.subheader("Theft detection")
    st.caption("Select a held-out test window. The model returns NORMAL, POTENTIAL_THEFT or UNCERTAIN; factors come from the model's gradient×input attribution.")
    run_dir_value = summary_data.get("artifacts", {}).get("dashboard_run_dir")
    checkpoint = summary_data.get("artifacts", {}).get("proposed_checkpoint")
    if not run_dir_value or not checkpoint:
        st.info(NO_RESULTS)
    else:
        run_dir = Path(run_dir_value)
        samples_path = run_dir / "test_samples.npz"
        meta_path = run_dir / "model" / "checkpoint.meta.json"
        if not (samples_path.exists() and Path(checkpoint).exists() and meta_path.exists()):
            st.info("Model artifacts are missing. Re-run `python main.py --demo` or the quick suite.")
        else:
            try:
                model, meta = load_checkpoint(checkpoint, str(meta_path), samples_path.stat().st_mtime)
                samples = np.load(samples_path, allow_pickle=True)
                X, y = samples["X"], samples["y"]
                probs = model.predict_proba(X)
                threshold = float(meta.get("threshold", 0.5))
                suspicious = int(np.argmax(probs))
                normal = int(np.argmin(probs))
                mode = st.radio("Sample source", ["Most suspicious test window", "Most confident normal window", "Random test window"], horizontal=True)
                if mode.startswith("Most suspicious"):
                    idx = suspicious
                elif mode.startswith("Most confident"):
                    idx = normal
                else:
                    idx = int(st.number_input("Test sample index", min_value=0, max_value=len(X) - 1, value=int(len(X) // 2), step=1))
                consumer_ids = samples["consumer_ids"] if "consumer_ids" in samples else np.array([""] * len(X))
                record = explain_prediction(model, X[idx], list(meta.get("feature_names", [])), threshold)
                cols = st.columns(4)
                cols[0].metric("Verdict", record["verdict"])
                cols[1].metric("P(theft)", f"{100 * record['probability_theft']:.1f}%")
                cols[2].metric("Confidence", f"{100 * record['confidence']:.1f}%")
                cols[3].metric("True test label", "THEFT" if int(y[idx]) == 1 else "NORMAL")
                st.write(f"Consumer: `{consumer_ids[idx]}` · test window index: `{idx}` · threshold: `{threshold:.2f}`")
                st.code(record["text"], language="text")
                if record.get("top_factors"):
                    factors = pd.DataFrame(record["top_factors"])
                    factors["human_name"] = factors["feature"].map(human_feature_name)
                    st.dataframe(factors[[ "human_name", "contribution", "value"]], width="stretch", hide_index=True)

                with st.expander("Upload model-ready feature row (CSV)"):
                    upload = st.file_uploader("CSV with the same engineered feature columns", type=["csv"])
                    if upload is not None:
                        try:
                            frame = pd.read_csv(upload)
                            feature_names = list(meta.get("feature_names", []))
                            missing = [c for c in feature_names if c not in frame.columns]
                            if missing:
                                st.error(f"Missing {len(missing)} feature columns; the first missing is {missing[0]!r}.")
                            elif len(frame) == 0:
                                st.error("Empty CSV.")
                            else:
                                row = frame.iloc[0][feature_names].to_numpy(dtype=np.float32)
                                rec = explain_prediction(model, row, feature_names, threshold)
                                st.metric("Uploaded sample verdict", rec["verdict"])
                                st.code(rec["text"], language="text")
                        except Exception as exc:
                            st.error(f"Could not parse/score the upload: {exc!r}")
            except Exception as exc:
                st.error(f"Could not load the model for detection: {exc!r}")
                st.caption("This dashboard refuses to display fabricated predictions if the checkpoint cannot be loaded.")

# --------------------------------------------------------------------------- #
with pages[8]:
    st.subheader("Explainability")
    run_dir_value = summary_data.get("artifacts", {}).get("dashboard_run_dir")
    if not run_dir_value:
        st.info(NO_RESULTS)
    else:
        expl_dir = Path(run_dir_value) / "explainability"
        perm = expl_dir / "permutation_importance.json"
        shap_file = expl_dir / "shap_summary.json"
        local = expl_dir / "local_explanations.json"
        if perm.exists():
            st.markdown("#### Permutation importance (actual model, validation data)")
            st.json(json.loads(perm.read_text(encoding="utf-8")))
        else:
            st.info("Permutation importance: NOT RUN")
        image_if_exists(summary_data.get("figures", {}).get("feature_importance"))
        if shap_file.exists():
            st.markdown("#### SHAP summary")
            st.json(json.loads(shap_file.read_text(encoding="utf-8")))
            image_if_exists(summary_data.get("figures", {}).get("shap_summary"))
        else:
            st.info("SHAP: NOT RUN / failed. See the run's explainability_status.json.")
        if local.exists():
            st.markdown("#### Local explanations for flagged windows")
            payload = json.loads(local.read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame(payload.get("records", [])), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
with pages[9]:
    st.subheader("Final experiment report")
    report_path = ROOT / "results" / "reports" / "final_experiment_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
        st.download_button("Download report", report_path.read_text(encoding="utf-8"), file_name="final_experiment_report.md")
    else:
        st.info("Report: NOT RUN. Run the experiment suite first.")
    tables = summary_data.get("tables", {})
    if tables:
        st.markdown("#### Generated tables")
        for name, paths in tables.items():
            st.write(f"**{name}**")
            csv_path = Path(paths.get("csv", ""))
            if csv_path.exists():
                st.download_button(f"Download {name}.csv", csv_path.read_bytes(), file_name=csv_path.name, key=name)
