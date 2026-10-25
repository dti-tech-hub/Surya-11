"""Tests for explainability: SHAP, permutation importance, and local explanations."""
from __future__ import annotations

import numpy as np
import pytest

from explainability.explanations import explain_prediction, human_feature_name
from explainability.feature_importance import permutation_feature_importance
from explainability.shap_explainer import ModelExplainer, SHAP_AVAILABLE
from models.base import TorchBinaryClassifier
from models.lightweight import LightweightMLP


@pytest.fixture(scope="module")
def trained_dummy_model():
    model = LightweightMLP(6, hidden=[16, 8])
    clf = TorchBinaryClassifier(model, name="expl_test")
    X = np.random.randn(40, 6).astype(np.float32)
    y = np.random.randint(0, 2, 40).astype(np.float32)
    clf.fit(X, y, epochs=1, batch_size=16)
    return clf, X, y


def test_permutation_importance(trained_dummy_model):
    clf, X, y = trained_dummy_model
    feature_names = ["mean_cons", "peak_cons", "std_cons", "min_cons", "max_cons", "trend"]
    res = permutation_feature_importance(clf, X, y, feature_names, cfg={}, max_rows=40)

    assert res["method"] == "permutation_importance"
    assert len(res["top_features"]) == len(feature_names)
    assert "importance_mean" in res["top_features"][0]


def test_explain_prediction_local(trained_dummy_model):
    clf, X, _ = trained_dummy_model
    feature_names = [f"feat_{i}" for i in range(6)]
    sample = X[0]
    rec = explain_prediction(clf, sample, feature_names, threshold=0.5)

    assert "verdict" in rec
    assert rec["verdict"] in ("NORMAL", "POTENTIAL_THEFT", "UNCERTAIN")
    assert "probability_theft" in rec
    assert "confidence" in rec
    assert "top_factors" in rec
    assert len(rec["top_factors"]) > 0


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap library not installed")
def test_shap_explainer(trained_dummy_model):
    clf, X, _ = trained_dummy_model
    cfg = {"explainability": {"shap_samples": 10, "shap_background": 15, "shap_nsamples": 20}}
    expl = ModelExplainer(clf, X, cfg=cfg)
    summary = expl.summary_dict(X[:5])

    assert "explainer" in summary
    assert "mean_abs_shap" in summary
    assert len(summary["mean_abs_shap"]) > 0


def test_human_feature_name():
    h = human_feature_name("daily_mean")
    assert isinstance(h, str) and len(h) > 0
