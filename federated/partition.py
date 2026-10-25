"""Client simulation: partition the training set across simulated retailers.

Partitioning happens at the *consumer* level — every consumer's windows belong
to exactly one client (a utility company owns a customer base). The server only
ever sees index sets; raw samples stay with the client.

Strategies (config ``clients.partition_strategy``):

- ``iid``     — consumers shuffled and dealt round-robin → similar distributions
- ``non_iid`` — Dirichlet(label proportions, alpha) + quantity skew, so clients
  differ in class balance, sample count, and consumption distribution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.features import WindowDataset
from utilities.config import get
from utilities.seed import make_rng


@dataclass
class ClientPartition:
    """Index set of one simulated client's private training data."""

    client_id: str
    indices: np.ndarray                       # rows of the shared train WindowDataset
    consumer_ids: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.indices.shape[0])

    def summary(self, dataset: WindowDataset) -> dict[str, Any]:
        y = dataset.y[self.indices]
        return {
            "client_id": self.client_id,
            "n_samples": self.n_samples,
            "n_consumers": len(self.consumer_ids),
            "n_theft_windows": int((y == 1).sum()),
            "theft_ratio": float(y.mean()) if y.size else 0.0,
        }


def _consumer_index_map(dataset: WindowDataset) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for pos, consumer_id in enumerate(dataset.meta["consumer_id"].astype(str)):
        groups.setdefault(consumer_id, []).append(pos)
    return {k: np.asarray(v, dtype=int) for k, v in groups.items()}


def _assign_remainder(clients: list[list[str]], rng: np.random.Generator) -> None:
    """Give any client with too few consumers a few random ones from rich clients."""
    sizes = [len(c) for c in clients]
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    for i, members in enumerate(clients):
        while len(members) < max(1, int(0.2 * mean_size)):
            j = int(np.argmax([len(c) for c in clients]))
            if len(clients[j]) <= 1:
                break
            moved = clients[j].pop(int(rng.integers(0, len(clients[j]))))
            members.append(moved)


def _redistribute_quantity_skew(
    clients: list[list[str]],
    quantity_skew: float,
    rng: np.random.Generator,
) -> None:
    """Create size skew by *moving* consumers, never by dropping them.

    An earlier implementation randomly thinned each client's consumer list and
    discarded the rest. That silently removed up to half of the training data.
    Here every consumer stays in exactly one bucket; a bounded number of moves
    from smaller to larger buckets creates quantity skew while preserving full
    coverage.
    """
    if quantity_skew <= 0 or len(clients) < 2:
        return
    total = sum(len(bucket) for bucket in clients)
    n_moves = int(round(min(max(quantity_skew, 0.0), 1.0) * 0.5 * total))
    for _ in range(n_moves):
        eligible = [i for i, bucket in enumerate(clients) if len(bucket) > 1]
        if not eligible:
            break
        donor = int(rng.choice(eligible))
        weights = np.asarray([(len(bucket) + 1) ** 2 for bucket in clients], dtype=np.float64)
        weights[donor] = 0.0
        if weights.sum() <= 0:
            break
        receiver = int(rng.choice(len(clients), p=weights / weights.sum()))
        moved = clients[donor].pop(int(rng.integers(0, len(clients[donor]))))
        clients[receiver].append(moved)


def partition_iid(dataset: WindowDataset, n_clients: int, seed: int) -> list[ClientPartition]:
    rng = make_rng(seed)
    consumers = list(_consumer_index_map(dataset).keys())
    rng.shuffle(consumers)
    buckets: list[list[str]] = [[] for _ in range(n_clients)]
    for i, cid in enumerate(consumers):
        buckets[i % n_clients].append(cid)
    return _materialise(buckets, dataset)


def partition_non_iid(
    dataset: WindowDataset,
    n_clients: int,
    seed: int,
    dirichlet_alpha: float = 0.5,
    quantity_skew: float = 0.6,
    min_samples: int = 2,
) -> list[ClientPartition]:
    """Dirichlet label skew + quantity skew across clients."""
    rng = make_rng(seed)
    index_map = _consumer_index_map(dataset)
    buckets: list[list[str]] = [[] for _ in range(n_clients)]

    for label in (0, 1):
        # A consumer can have pre-theft windows labelled 0 and later windows
        # labelled 1. Use the consumer's maximum window label, not the first
        # window, so theft consumers are not mistaken for all-normal clients.
        members = [cid for cid, idx in index_map.items() if int(dataset.y[idx].max()) == label]
        rng.shuffle(members)
        if not members:
            continue
        proportions = rng.dirichlet(np.full(n_clients, max(dirichlet_alpha, 0.05)))
        cuts = (np.cumsum(proportions) * len(members)).astype(int)[:-1]
        for bucket, chunk in zip(buckets, np.split(np.asarray(members, dtype=object), cuts)):
            bucket.extend(map(str, chunk.tolist()))

    # Quantity skew is created by moving consumers between buckets. Do NOT
    # thin/drop consumers: every training consumer must appear in exactly one
    # client. `_assign_remainder` also fills any empty/small bucket by moving,
    # not copying, consumers from a rich bucket.
    _redistribute_quantity_skew(buckets, float(quantity_skew), rng)
    _assign_remainder(buckets, rng)
    partitions = _materialise(buckets, dataset)

    # enforce min samples by lending whole consumers from the richest client
    index_map = _consumer_index_map(dataset)
    for part in partitions:
        guard = 0
        while part.n_samples < min_samples and guard < 10:
            guard += 1
            donor = max(partitions, key=lambda p: p.n_samples)
            if donor.n_samples <= min_samples or not donor.consumer_ids:
                break
            moved = donor.consumer_ids.pop()
            pos = index_map[moved]
            donor.indices = np.setdiff1d(donor.indices, pos, assume_unique=False)
            part.indices = np.concatenate([part.indices, pos])
            part.consumer_ids.append(moved)
    return partitions


def _materialise(buckets: list[list[str]], dataset: WindowDataset) -> list[ClientPartition]:
    index_map = _consumer_index_map(dataset)
    partitions: list[ClientPartition] = []
    for i, members in enumerate(buckets):
        members = [m for m in members if m in index_map]
        indices = (
            np.concatenate([index_map[m] for m in members]) if members else np.empty(0, dtype=int)
        )
        partitions.append(ClientPartition(client_id=f"client_{i:02d}", indices=indices, consumer_ids=members))
    return partitions


def build_partition(dataset: WindowDataset, cfg: dict, seed: int) -> list[ClientPartition]:
    n_clients = int(get(cfg, "clients.num_clients", 5))
    strategy = str(get(cfg, "clients.partition_strategy", "non_iid"))
    if strategy == "iid":
        parts = partition_iid(dataset, n_clients, seed + int(get(cfg, "clients.seed_offset", 0)))
    elif strategy == "non_iid":
        parts = partition_non_iid(
            dataset,
            n_clients,
            seed + int(get(cfg, "clients.seed_offset", 0)),
            dirichlet_alpha=float(get(cfg, "clients.dirichlet_alpha", 0.5)),
            quantity_skew=float(get(cfg, "clients.quantity_skew", 0.6)),
            min_samples=int(get(cfg, "clients.min_samples_per_client", 2)),
        )
    else:
        raise ValueError(f"Unknown partition strategy '{strategy}' (iid|non_iid)")
    for part in parts:
        part.stats = part.summary(dataset)
    return parts


def heterogeneity_report(partitions: list[ClientPartition], dataset: WindowDataset) -> dict[str, Any]:
    """Quantify client differences (section 22: heterogeneity analysis)."""
    rows = []
    raw_values = []  # per-client mean consumption (unscaled proxy via block matrix)
    for part in partitions:
        summary = part.summary(dataset)
        blocks = dataset.block_matrix()[part.indices]
        summary["mean_consumption_scaled"] = float(blocks.mean()) if part.n_samples else 0.0
        summary["std_consumption_scaled"] = float(blocks.std()) if part.n_samples else 0.0
        rows.append(summary)
        raw_values.append(blocks)

    df = pd.DataFrame(rows)
    theft = df["theft_ratio"].to_numpy()
    sizes = df["n_samples"].to_numpy()
    report = {
        "per_client": rows,
        "strategy_note": "clients differ by design under non_iid; iid should be near-uniform",
        "theft_ratio_std": float(theft.std()),
        "size_cv": float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else 0.0,
        "theft_ratio_range": [float(theft.min()), float(theft.max())] if len(theft) else [0.0, 0.0],
    }
    return report
