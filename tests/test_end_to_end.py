"""End-to-end integration test: synthetic data -> preprocessing -> client simulation -> local training -> aggregation -> detection -> explainability."""
from __future__ import annotations

import numpy as np
import pytest

from detection.threshold import select_threshold
from evaluation.metrics import compute_classification_metrics
from explainability.explanations import explain_prediction
from federated.client import FederatedClient
from federated.partition import build_partition
from federated.server import FederatedServer
from federated.simulator import FederatedSimulator
from models.lightweight import build_lightweight
from privacy.base import build_privacy
from preprocessing.pipeline import prepare_data
from utilities.config import load_config


def test_complete_end_to_end_pipeline():
    cfg = load_config("quick")
    cfg["data"]["synthetic"]["consumers"] = 25
    cfg["data"]["synthetic"]["days"] = 20
    cfg["clients"]["num_clients"] = 3
    cfg["federated"]["rounds"] = 2
    cfg["federated"]["local_epochs"] = 1
    cfg["privacy"]["enabled"] = True
    cfg["communication"]["compression"] = True
    cfg["aggregation"]["method"] = "adaptive"

    # 1. Dataset & Preprocessing
    splits = prepare_data(cfg, seed=42)
    assert len(splits.train.y) > 0
    assert len(splits.val.y) > 0
    assert len(splits.test.y) > 0

    # 2. Client Partitioning
    partitions = build_partition(splits.train, cfg, seed=42)
    assert len(partitions) == 3

    # 3. Model factory & Privacy
    n_features = splits.train.X_matrix.shape[1]
    factory = lambda: build_lightweight(cfg, n_features, seed=42)
    privacy = build_privacy(cfg, seed=42)

    # 4. Federated Training (PowerShield-FL)
    sim = FederatedSimulator(
        cfg=cfg,
        train_X=splits.train.X_matrix,
        train_y=splits.train.y,
        partitions=partitions,
        model_factory=factory,
        privacy_mechanism=privacy,
        val_X=splits.val.X_matrix,
        val_y=splits.val.y,
        seed=42,
    )
    result = sim.run(rounds=2)

    assert len(result["history"]) == 2
    assert "best_state" in result
    assert result["communication"]["total_bytes"] > 0

    # 5. Global Evaluation
    eval_model = factory()
    eval_model.set_weights(result["best_state"])

    val_probs = eval_model.predict_proba(splits.val.X_matrix)
    thresh_info = select_threshold(splits.val.y, val_probs)
    thresh = thresh_info["threshold"]

    test_probs = eval_model.predict_proba(splits.test.X_matrix)
    metrics = compute_classification_metrics(splits.test.y, test_probs, threshold=thresh)

    assert "f1" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0

    # 6. Explainability on a test sample
    expl = explain_prediction(
        eval_model,
        splits.test.X_matrix[0],
        list(splits.train.feature_names),
        threshold=thresh,
    )
    assert expl["verdict"] in ("NORMAL", "POTENTIAL_THEFT", "UNCERTAIN")
    assert len(expl["top_factors"]) > 0
