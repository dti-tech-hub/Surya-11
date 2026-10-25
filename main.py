"""PowerShield-FL command-line entry point.

Examples:
    python main.py --demo
    python main.py --mode quick
    python main.py --mode full
    python main.py --experiment proposed
    python main.py --report-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

from evaluation.reporting import generate_report
from experiments.suite import run_suite
from utilities.config import load_config, load_dotenv, resolve_path
from utilities.system import environment_info, environment_ok
from utilities.seed import set_seed

ROOT = Path(__file__).resolve().parent
DEMO_OVERRIDES = [
    "data.synthetic.consumers=180",
    "data.synthetic.days=120",
    "data.synthetic.theft_rate=0.16",
    "clients.num_clients=4",
    "federated.rounds=3",
    "federated.local_epochs=2",
    "experiments.seeds=[42]",
    "experiments.central_models=[logistic_regression, mlp]",
    "experiments.cnn_lstm_enabled=false",
    "experiments.run_privacy_sweep=false",
    "experiments.run_communication_sweep=false",
    "experiments.run_scalability=false",
    "model.central_mlp_epochs=5",
    "explainability.shap_samples=15",
    "explainability.shap_background=15",
    "explainability.shap_nsamples=50",
    "explainability.permutation_max_rows=400",
    "explainability.permutation_repeats=3",
    "explainability.local_explanations=6",
]


def _stage_ok(summary: dict, name: str) -> bool:
    return str(summary.get("stages", {}).get(name, {}).get("status", "")).upper() == "PASS"


def _print_demo(summary: dict) -> None:
    dataset = summary.get("dataset", {})
    partitions = summary.get("partitions", {})
    cfg = summary.get("config", {})
    stages = summary.get("stages", {})
    def status(*names: str) -> str:
        present = [n for n in names if n in stages]
        return "OK" if present and all(_stage_ok(summary, n) for n in present) else "FAIL"

    def stage_key(prefix: str) -> str | None:
        return next((k for k in stages if k.startswith(prefix)), None)

    central_key = stage_key("centralized_s")
    fedavg_key = stage_key("fedavg_s")
    proposed_key = stage_key("proposed_s")
    explain_key = stage_key("explainability_s")
    print("\n" + "=" * 55)
    print("             POWERSHIELD-FL DEMO")
    print("=" * 55)
    print(f"Environment              : {'OK' if environment_ok()[0] else 'FAIL'}")
    print(f"Dataset                  : {dataset.get('source', 'NOT RUN')}")
    print(f"Clients                  : {len(partitions.get('clients', []))}")
    print(f"Federated Rounds         : {cfg.get('federated', {}).get('rounds', 'NOT RUN')}")
    print("")
    print(f"Preprocessing            : {status('dataset')}")
    print(f"Client Simulation        : {status('partitions')}")
    print(f"Centralized Baselines    : {status(central_key) if central_key else 'NOT RUN'}")
    print(f"Local Training / FedAvg  : {status(fedavg_key) if fedavg_key else 'NOT RUN'}")
    proposed_status = status(proposed_key) if proposed_key else 'NOT RUN'
    print(f"Privacy Layer            : {proposed_status}")
    print(f"Communication Layer      : {proposed_status}")
    print(f"Adaptive Aggregation     : {proposed_status}")
    print(f"Global Model             : {proposed_status}")
    print(f"Detection                : {'OK' if summary.get('main', {}).get('proposed') else 'FAIL'}")
    print(f"Explainability           : {status(explain_key) if explain_key else 'FAIL'}")
    print("")
    print("Results generated at:")
    print(f"  {resolve_path(cfg, 'results')}")
    print("")
    print("Dashboard:")
    print("  streamlit run dashboard/app.py")
    print("  http://localhost:8501")
    print("")
    print("=" * 55)
    print("             DEMO COMPLETED")
    print("=" * 55)


def _print_quick_summary(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("          POWERSHIELD-FL EXPERIMENT SUITE")
    print("=" * 55)
    print(f"Suite       : {summary.get('suite_id')}")
    print(f"Mode        : {summary.get('mode')}")
    print(f"Dataset     : {summary.get('dataset', {}).get('source', 'NOT RUN')}")
    print(f"Stages      : {json.dumps({k: v.get('status') for k, v in summary.get('stages', {}).items()}, default=str)}")
    for name in ("fedavg", "proposed"):
        rows = summary.get("main", {}).get(name, [])
        if rows:
            f1 = sum(r["f1"] for r in rows) / len(rows)
            print(f"{name:12s}: mean test F1={f1:.4f} over {len(rows)} seed(s)")
    cc = summary.get("main", {}).get("communication_comparison")
    if cc:
        print(f"Communication reduction vs FedAvg: {cc['reduction_pct']:.2f}% (measured)")
    print("")
    print("Results : results/")
    print("Report  : results/reports/final_experiment_report.md")
    print("Dashboard: streamlit run dashboard/app.py")
    print("=" * 55)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PowerShield-FL: privacy-preserving federated electricity-theft detection")
    parser.add_argument("--mode", choices=["quick", "full"], default=None, help="resource profile")
    parser.add_argument("--demo", action="store_true", help="fast end-to-end synthetic demo")
    parser.add_argument(
        "--experiment",
        choices=["all", "centralized", "fedavg", "proposed", "privacy", "communication", "scalability"],
        default="all",
        help="run only one stage/experiment",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    parser.add_argument("--report-only", action="store_true", help="regenerate the report from the existing summary")
    parser.add_argument("--serve-dashboard", action="store_true", help="launch the Streamlit dashboard")
    args = parser.parse_args(argv)

    load_dotenv()
    if args.serve_dashboard:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")])

    if args.report_only:
        mode = args.mode or "quick"
        cfg = load_config(mode, args.override)
        summary_path = resolve_path(cfg, "results/metrics/summary.json")
        if not summary_path.exists():
            print("No experimental result available. Run the experiment first.")
            return 1
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        out = generate_report(cfg, summary, resolve_path(cfg, "results/reports/final_experiment_report.md"))
        print(f"Report written to {out}")
        return 0

    mode = args.mode or "quick"
    overrides = list(args.override)
    if args.demo:
        mode = "quick"
        if args.seed is not None:
            DEMO_OVERRIDES = [o for o in globals()["DEMO_OVERRIDES"] if not o.startswith("experiments.seeds=")]
            overrides = DEMO_OVERRIDES + [f"experiments.seeds=[{int(args.seed)}]"] + overrides
        else:
            overrides = globals()["DEMO_OVERRIDES"] + overrides
    if args.seed is not None:
        overrides.append(f"experiment.seed={int(args.seed)}")
        if not args.demo:
            overrides.append(f"experiments.seeds=[{int(args.seed)}]")

    # Single-experiment mode: switch off the other sweeps / stages.
    if args.experiment in ("privacy", "communication", "scalability"):
        overrides += [
            "experiments.run_privacy_sweep=" + ("true" if args.experiment == "privacy" else "false"),
            "experiments.run_communication_sweep=" + ("true" if args.experiment == "communication" else "false"),
            "experiments.run_scalability=" + ("true" if args.experiment == "scalability" else "false"),
        ]

    cfg = load_config(mode, overrides)
    set_seed(int(args.seed or cfg.get("experiment", {}).get("seed", 42)))
    ok, issues = environment_ok()
    if not ok:
        print("Environment issues:")
        for issue in issues:
            print(f"  - {issue}")
        if any("Missing dependency" in i for i in issues):
            print("Install dependencies with: pip install -r requirements.txt")
            return 1

    if args.demo:
        stages = ["centralized", "fedavg", "proposed", "explainability", "figures", "report"]
        summary = run_suite(cfg, demo=True, stages=stages, seed=cfg.get("experiment", {}).get("seed", 42), log=print)
        _print_demo(summary)
        failures = [k for k, v in summary.get("stages", {}).items() if "FAILED" in str(v.get("status", "")).upper()]
        return 1 if failures else 0

    stage_map = {
        "all": None,
        "centralized": ["centralized", "figures", "report"],
        "fedavg": ["fedavg", "figures", "report"],
        "proposed": ["proposed", "explainability", "figures", "report"],
        "privacy": ["sweeps", "figures", "report"],
        "communication": ["sweeps", "figures", "report"],
        "scalability": ["sweeps", "figures", "report"],
    }
    summary = run_suite(cfg, demo=False, stages=stage_map[args.experiment], log=print)
    _print_quick_summary(summary)
    failures = [k for k, v in summary.get("stages", {}).items() if "FAILED" in str(v.get("status", "")).upper()]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
