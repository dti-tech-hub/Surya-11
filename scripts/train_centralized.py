"""Train and evaluate the centralized baselines only."""
from __future__ import annotations

import argparse

from experiments.suite import run_suite
from utilities.config import load_config
from utilities.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.mode, args.override)
    set_seed(args.seed)
    summary = run_suite(cfg, stages=["centralized", "figures", "report"], seed=args.seed)
    failures = [k for k, v in summary["stages"].items() if "FAILED" in str(v.get("status", "")).upper()]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
