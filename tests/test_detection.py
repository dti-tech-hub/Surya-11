"""Tests for detection postprocessing, threshold tuning, and prediction labels."""
from __future__ import annotations

import numpy as np
import pytest

from detection.predictor import TheftPredictor
from detection.threshold import select_threshold
from evaluation.metrics import compute_classification_metrics


def test_select_threshold_maximizes_f1():
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.55, 0.65, 0.45, 0.8, 0.15])

    info = select_threshold(y_true, y_prob)
    assert 0.0 < info["threshold"] < 1.0
    assert "val_f1" in info


def test_theft_predictor_labels():
    class DummyModel:
        def predict_proba(self, X):
            return np.array([0.9, 0.1, 0.6])

    predictor = TheftPredictor(
        model=DummyModel(),
        feature_names=[f"f{i}" for i in range(5)],
        threshold=0.6,
        uncertain_margin=0.15,  # [0.45, 0.75] = UNCERTAIN
    )

    preds = predictor.predict_batch(np.zeros((3, 5)))
    assert preds[0]["verdict"] == "POTENTIAL_THEFT"
    assert preds[1]["verdict"] == "NORMAL"
    assert preds[2]["verdict"] == "UNCERTAIN"


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
