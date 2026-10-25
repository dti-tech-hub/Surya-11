"""Regenerate the final Markdown report from the latest recorded summary."""
from __future__ import annotations

import argparse
import json

from evaluation.reporting import generate_report
from utilities.config import load_config, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    cfg = load_config(args.mode)
    path = resolve_path(cfg, args.summary or "results/metrics/summary.json")
    if not path.exists():
        print("No experimental result available. Run the experiment first.")
        return 1
    summary = json.loads(path.read_text(encoding="utf-8"))
    out = generate_report(cfg, summary, resolve_path(cfg, "results/reports/final_experiment_report.md"))
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
