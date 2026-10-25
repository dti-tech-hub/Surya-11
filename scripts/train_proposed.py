"""Train and evaluate PowerShield-FL only (plus explainability and report)."""
from __future__ import annotations

import argparse

from experiments.suite import run_suite
from utilities.config import load_config
from utilities.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    overrides = list(args.override)
    if args.rounds:
        overrides.append(f"federated.rounds={args.rounds}")
    cfg = load_config(args.mode, overrides)
    set_seed(args.seed)
    summary = run_suite(cfg, stages=["proposed", "explainability", "figures", "report"], seed=args.seed)
    failures = [k for k, v in summary["stages"].items() if "FAILED" in str(v.get("status", "")).upper()]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
