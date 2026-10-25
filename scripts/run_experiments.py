"""Run the complete configured experiment suite."""
from __future__ import annotations

import argparse

from experiments.suite import run_suite
from utilities.config import load_config
from utilities.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.mode, args.override)
    seed = args.seed if args.seed is not None else int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    summary = run_suite(cfg, demo=args.demo, seed=seed)
    failures = [k for k, v in summary["stages"].items() if "FAILED" in str(v.get("status", "")).upper()]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
