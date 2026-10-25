"""Federated training simulator: the round loop of PowerShield-FL / FedAvg.

Per round:

```
select clients (participation_rate)
   → each selected client trains locally on its private data (ClientUpdate)
   → PRIVACY layer: clip + Gaussian noise (+ optional simulated secure agg)
   → COMMUNICATION layer: compress updates, measure real payload bytes
   → SERVER: decompress, weight (FedAvg or adaptive), aggregate, broadcast
   → evaluate the global model on the server-held validation split
```

Configuration decides which mode runs:

- FedAvg baseline   : aggregation.method=fedavg, privacy off, compression off
- PowerShield-FL    : aggregation.method=adaptive, privacy on, compression on
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import f1_score

from evaluation.metrics import compute_classification_metrics
from privacy.privacy_metrics import privacy_round_report
from federated.adaptive_aggregation import adaptive_weights, weight_diagnostics
from federated.client import FederatedClient
from federated.communication import (
    CommunicationMeter,
    compress_state,
    decompress_state,
    payload_bytes,
    uncompressed_bytes,
)
from federated.partition import ClientPartition
from federated.server import FederatedServer
from models.base import TorchBinaryClassifier
from privacy.secure_aggregation import SecureAggregationSimulator
from utilities.config import get
from utilities.seed import make_rng
from utilities.types import ClientUpdate


class FederatedSimulator:
    def __init__(
        self,
        cfg: dict,
        train_X: np.ndarray,
        train_y: np.ndarray,
        partitions: list[ClientPartition],
        model_factory: Callable[[], TorchBinaryClassifier],
        privacy_mechanism: Any,
        val_X: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
        seed: int = 42,
        log: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.seed = seed
        self.log = log or (lambda _m: None)
        self.model_factory = model_factory
        self.val_X = val_X
        self.val_y = val_y
        self.privacy = privacy_mechanism
        self.secure_agg = SecureAggregationSimulator(cfg, seed=int(get(cfg, "privacy.secure_aggregation_seed", 2026)))
        self.meter = CommunicationMeter()

        self.clients: list[FederatedClient] = [
            FederatedClient(
                client_id=p.client_id,
                X=train_X[p.indices],
                y=train_y[p.indices],
                cfg=cfg,
                seed=seed + i,
                class_weights=None,
            )
            for i, p in enumerate(partitions)
        ]
        probe = model_factory()
        self.global_state = probe.get_weights()
        self.server = FederatedServer(self.global_state)
        self.model_bytes = uncompressed_bytes(self.global_state)
        self.aggregation_method = str(get(cfg, "aggregation.method", "adaptive"))
        self.participation = float(get(cfg, "federated.participation_rate", 1.0))

    # ------------------------------------------------------------------ #
    def _evaluate_global(self) -> dict[str, float]:
        if self.val_X is None or self.val_y is None or len(self.val_y) == 0:
            return {}
        model = self.model_factory()
        model.set_weights(self.server.get_global_state())
        probs = model.predict_proba(self.val_X)
        metrics = compute_classification_metrics(self.val_y, probs, threshold=0.5)
        logits = model.predict_logits(self.val_X)
        metrics["val_loss"] = float(
            np.mean(np.logaddexp(0.0, logits) - self.val_y * logits)
        )
        return metrics

    # ------------------------------------------------------------------ #
    def run(self, rounds: int | None = None) -> dict[str, Any]:
        rounds = int(rounds if rounds is not None else get(self.cfg, "federated.rounds", 8))
        rng = make_rng(self.seed + 997)
        history: list[dict[str, Any]] = []
        best = {"f1": -1.0, "round": 0, "state": self.server.get_global_state()}
        compress_broadcast = bool(get(self.cfg, "communication.compress_broadcast", True)) and bool(
            get(self.cfg, "communication.compression", False)
        )
        # Simulated secure aggregation masks *dense* updates. Combining it with
        # top-k sparsification would corrupt the cancelling masks, so upload
        # compression is disabled (and reported) whenever masking is enabled.
        upload_cfg = self.cfg
        if self.secure_agg.enabled and bool(get(self.cfg, "communication.compression", False)):
            import copy

            upload_cfg = copy.deepcopy(self.cfg)
            upload_cfg.setdefault("communication", {})["method"] = "none"
            self.log(
                "WARNING: secure aggregation is simulated with dense updates; "
                "upload compression disabled for this run to preserve mask cancellation."
            )
        t_start = time.perf_counter()

        for rnd in range(1, rounds + 1):
            t0 = time.perf_counter()
            n_select = max(1, int(round(self.participation * len(self.clients))))
            selected = (
                self.clients
                if n_select >= len(self.clients)
                else [self.clients[i] for i in sorted(rng.choice(len(self.clients), size=n_select, replace=False))]
            )

            # ---- local training (privacy-free deltas) -----------------------
            updates: list[ClientUpdate] = []
            for client in selected:
                update, _model = client.train(self.server.get_global_state(), self.model_factory)
                updates.append(update)
            self.server.validate_updates(updates)  # strict contract check

            # ---- privacy layer ---------------------------------------------
            privacy_info: list[dict[str, Any]] = []
            protected: list[dict[str, torch.Tensor]] = []
            for update in updates:
                original_norm = update.update_norm
                state = self.privacy.apply(update.state_dict)
                protected.append(state)
                privacy_info.append(
                    {
                        "client_id": update.client_id,
                        "original_norm": original_norm,
                        "noise_norm": getattr(self.privacy, "applied_noise_norm", 0.0),
                        "clipped": bool(getattr(self.privacy, "last_clipped", False)),
                    }
                )

            # ---- simulated secure aggregation ------------------------------
            protected, sec_info = self.secure_agg.protect(protected)

            # ---- communication layer (compress + measure) ------------------
            deltas: list[dict[str, torch.Tensor]] = []
            bytes_up: list[int] = []
            kept_fracs: list[float] = []
            for state in protected:
                header, payload, stats = compress_state(state, upload_cfg)
                bytes_up.append(payload_bytes(header, payload))
                kept_fracs.append(float(stats.get("kept_fraction", 1.0)))
                deltas.append(decompress_state(header, payload))

            if compress_broadcast:
                header_d, payload_d, _ = compress_state(self.server.get_global_state(), self.cfg)
                bytes_down_client = payload_bytes(header_d, payload_d)
            else:
                # Even "uncompressed" communication is a serialized payload;
                # measure its real byte length (header + float32 arrays) rather
                # than an idealised parameter-count estimate.
                header_d, payload_d, _ = compress_state(
                    self.server.get_global_state(), {"communication": {"method": "none"}}
                )
                bytes_down_client = payload_bytes(header_d, payload_d)

            # ---- aggregation -----------------------------------------------
            if self.aggregation_method == "adaptive":
                weights = adaptive_weights(updates, self.cfg)
                diag = weight_diagnostics(updates, weights, self.cfg)
            else:
                counts = np.array([u.num_samples for u in updates], dtype=np.float64)
                weights = counts / max(counts.sum(), 1.0)
                diag = {"client_ids": [u.client_id for u in updates], "adaptive": weights.tolist(), "fedavg": weights.tolist()}
            participation_scale = len(selected) / max(1, len(self.clients))
            self.server.aggregate(deltas, weights, scale=participation_scale)

            # ---- evaluation + logging --------------------------------------
            metrics = self._evaluate_global()
            train_loss = float(np.mean([u.train_loss for u in updates]))
            duration = time.perf_counter() - t0
            self.meter.record_round(
                rnd,
                per_client_bytes_down=[bytes_down_client] * len(selected),
                per_client_bytes_up=bytes_up,
                duration_s=duration,
                extra={"compression_kept_fraction": float(np.mean(kept_fracs)) if kept_fracs else 1.0},
            )
            entry: dict[str, Any] = {
                "round": rnd,
                "clients": [c.client_id for c in selected],
                "n_clients": len(selected),
                "train_loss": train_loss,
                "mean_val_loss": float(np.mean([u.val_loss for u in updates])) if updates else float("nan"),
                "round_duration_s": duration,
                "bytes_upload": int(sum(bytes_up)),
                "bytes_download": int(bytes_down_client * len(selected)),
                "mean_kept_fraction": float(np.mean(kept_fracs)) if kept_fracs else 1.0,
                "aggregation_weights": diag.get("adaptive", []),
                "weight_diagnostics": diag,
                "secure_aggregation": sec_info,
                "privacy_round": privacy_round_report(self.privacy, privacy_info),
                "update_norms": [u.update_norm for u in updates],
                "global_metrics": metrics,
            }
            if metrics:
                f1 = metrics.get("f1", float("nan"))
                if np.isfinite(f1) and f1 > best["f1"]:
                    best = {"f1": float(f1), "round": rnd, "state": self.server.get_global_state()}
            history.append(entry)
            self.log(
                f"round {rnd}/{rounds} loss={train_loss:.4f} "
                f"f1={metrics.get('f1', float('nan')):.4f} bytes={entry['bytes_upload'] + entry['bytes_download']} "
                f"time={duration:.1f}s"
            )

        wall = time.perf_counter() - t_start
        if best["round"] == 0:  # no finite validation F1 was observed
            best = {"f1": float("nan"), "round": rounds, "state": self.server.get_global_state()}
        return {
            "history": history,
            "final_state": self.server.get_global_state(),
            "best_state": best["state"],
            "best_val_f1": best["f1"],
            "best_round": best["round"],
            "wall_time_s": wall,
            "rounds": rounds,
            "clients": [c.summary() for c in self.clients],
            "communication": self.meter.summary(uncompressed_reference_bytes=self.model_bytes),
            "privacy": self.privacy.describe() if hasattr(self.privacy, "describe") else {},
            "secure_aggregation": self.secure_agg.describe(),
            "aggregation_method": self.aggregation_method,
        }


def train_federated(
    cfg: dict,
    train_X: np.ndarray,
    train_y: np.ndarray,
    partitions: list[ClientPartition],
    model_factory: Callable[[], TorchBinaryClassifier],
    privacy_mechanism: Any,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    rounds: int | None = None,
    seed: int = 42,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    sim = FederatedSimulator(
        cfg=cfg,
        train_X=train_X,
        train_y=train_y,
        partitions=partitions,
        model_factory=model_factory,
        privacy_mechanism=privacy_mechanism,
        val_X=val_X,
        val_y=val_y,
        seed=seed,
        log=log,
    )
    return sim.run(rounds=rounds)
