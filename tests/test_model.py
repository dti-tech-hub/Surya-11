"""Tests for model architectures: LightweightMLP, CNN-LSTM, baselines, and TorchBinaryClassifier."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sklearn.linear_model import LogisticRegression

from models.base import TorchBinaryClassifier
from models.baselines import SklearnClassifier
from models.cnn_lstm import CNNLSTM
from models.lightweight import LightweightMLP, build_lightweight
from utilities.config import load_config


def test_lightweight_mlp_dimensions():
    model = LightweightMLP(50, hidden=[64, 32], dropout=0.1)
    x = torch.randn(8, 50)
    out = model(x)
    assert out.shape == (8, 1)


def test_lightweight_mlp_parameter_count():
    cfg = load_config("quick")
    classifier = build_lightweight(cfg, n_features=30, seed=42)
    # Should be truly lightweight (< 15,000 parameters)
    assert classifier.n_params < 15000
    assert classifier.size_bytes() > 0


def test_torch_classifier_fit_and_predict():
    model = LightweightMLP(10, hidden=[16, 8])
    clf = TorchBinaryClassifier(model, name="test_mlp")

    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.float32)

    history = clf.fit(X, y, epochs=2, batch_size=16)
    assert len(history) == 2

    probs = clf.predict_proba(X)
    assert probs.shape == (50,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_weight_roundtrip():
    model = LightweightMLP(10, hidden=[16, 8])
    clf = TorchBinaryClassifier(model, name="test")

    weights = clf.get_weights()
    flat = clf.get_flat()

    clf.set_weights(weights)
    flat2 = clf.get_flat()
    np.testing.assert_allclose(flat, flat2)


def test_cnn_lstm_dimensions():
    model = CNNLSTM(window_days=14, steps_per_day=48, conv_channels=8, lstm_hidden=16)
    x = torch.randn(4, 14, 48)
    out = model(x)
    assert out.shape == (4, 1)


def test_sklearn_baseline():
    lr = SklearnClassifier(LogisticRegression(max_iter=100), name="logistic_regression")
    X = np.random.randn(40, 10).astype(np.float32)
    y = np.random.randint(0, 2, 40)
    lr.fit(X, y)
    probs = lr.predict_proba(X)
    assert probs.shape == (40,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
