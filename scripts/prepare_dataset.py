"""Prepare data: load, clean, split, engineer windows, save cache + statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.figures import plot_class_distribution
from preprocessing.pipeline import prepare_data, save_processed
from utilities.config import load_config, resolve_path
from utilities.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.mode, args.override)
    set_seed(args.seed)
    splits = prepare_data(cfg, seed=args.seed, log=print)
    path = save_processed(splits, cfg)
    out = resolve_path(cfg, "results/metrics")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "source": splits.source,
                "is_synthetic": splits.is_synthetic,
                "bundle_stats": splits.bundle_stats,
                "summary": splits.summary(),
                "cleaning": splits.cleaning_report,
                "preprocessing": splits.preprocessing_report,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    figures = resolve_path(cfg, "results/figures")
    figures.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(splits.labels.astype(int).tolist(), figures / "class_distribution.png", splits.source)
    print(f"Processed windows cached at {path}")
    print(f"Dataset statistics written to {out / 'dataset_statistics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
