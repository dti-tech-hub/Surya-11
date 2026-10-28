"""Create simulated IID/Non-IID client partitions and save their metadata."""
from __future__ import annotations

import argparse
import json

import numpy as np

from evaluation.figures import plot_client_distribution
from federated.partition import build_partition, heterogeneity_report
from preprocessing.pipeline import prepare_data
from utilities.config import get, load_config, resolve_path
from utilities.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--strategy", choices=["iid", "non_iid"], default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    overrides = list(args.override)
    if args.clients:
        overrides.append(f"clients.num_clients={args.clients}")
    if args.strategy:
        overrides.append(f"clients.partition_strategy={args.strategy}")
    cfg = load_config(args.mode, overrides)
    set_seed(args.seed)
    splits = prepare_data(cfg, seed=args.seed, log=print)
    partitions = build_partition(splits.train, cfg, args.seed)
    summary = {
        "seed": args.seed,
        "strategy": cfg["clients"]["partition_strategy"],
        "n_clients": len(partitions),
        "clients": [p.summary(splits.train) for p in partitions],
        "heterogeneity": heterogeneity_report(partitions, splits.train),
        "note": "only consumer IDs and window indices are stored",
    }
    out = resolve_path(cfg, "data/partitions")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"partitions_seed{args.seed}_n{len(partitions)}"
    np.savez_compressed(out / f"{stem}.npz", **{p.client_id: p.indices for p in partitions})
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    figures = resolve_path(cfg, get(cfg, "experiment.output_dir", "results")) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_client_distribution(summary["clients"], figures / "client_distribution.png", splits.source)
    for client in summary["clients"]:
        print(f"{client['client_id']}: {client['n_samples']} windows, theft_ratio={client['theft_ratio']:.4f}, consumers={client['n_consumers']}")
    print(f"Partition metadata: {out / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
