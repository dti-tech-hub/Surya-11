"""Tests for aggregation algorithms: FedAvg, adaptive aggregation, and communication compression."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from federated.adaptive_aggregation import adaptive_weights
from federated.aggregation import aggregate_deltas, apply_delta
from federated.communication import compress_state, decompress_state, payload_bytes
from utilities.types import ClientUpdate


def test_fedavg_weighted_mean():
    delta1 = {"w": torch.tensor([1.0, 2.0])}
    delta2 = {"w": torch.tensor([3.0, 4.0])}
    weights = [0.25, 0.75]

    aggregated = aggregate_deltas([delta1, delta2], weights)
    expected = 0.25 * np.array([1.0, 2.0]) + 0.75 * np.array([3.0, 4.0])
    np.testing.assert_allclose(aggregated["w"].numpy(), expected)


def test_apply_delta():
    base = {"w": torch.tensor([10.0, 20.0])}
    delta = {"w": torch.tensor([1.0, -2.0])}
    new_state = apply_delta(base, delta)
    np.testing.assert_allclose(new_state["w"].numpy(), [11.0, 18.0])


def test_adaptive_aggregation_weights():
    # Construct mock client updates
    u1 = ClientUpdate(
        client_id="c1",
        state_dict={"w": torch.zeros(2)},
        num_samples=100,
        train_loss=0.5,
        val_loss=0.6,
        val_metrics={"f1": 0.8, "roc_auc": 0.9},
        num_positive=10,
        num_negative=90,
        update_norm=1.0,
    )
    u2 = ClientUpdate(
        client_id="c2",
        state_dict={"w": torch.zeros(2)},
        num_samples=100,
        train_loss=0.5,
        val_loss=0.6,
        val_metrics={"f1": 0.4, "roc_auc": 0.5},
        num_positive=10,
        num_negative=90,
        update_norm=1.0,
    )
    cfg = {"aggregation": {"alpha": 1.0, "beta": 1.0, "reliability_power": 1.0}}
    weights = adaptive_weights([u1, u2], cfg)

    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
    # u1 has higher val quality, so its adaptive weight must be strictly higher
    assert weights[0] > weights[1]


def test_zero_positive_client_is_not_collapsed_to_min_weight():
    """A legitimate all-normal client has no local F1, but should still contribute."""
    positive_client = ClientUpdate(
        client_id="theft_client",
        state_dict={"w": torch.zeros(2)},
        num_samples=500,
        train_loss=0.5,
        val_loss=0.4,
        val_metrics={"f1": 0.8, "roc_auc": 0.9, "val_loss": 0.4},
        num_positive=100,
        num_negative=400,
        update_norm=1.0,
    )
    normal_client = ClientUpdate(
        client_id="normal_client",
        state_dict={"w": torch.zeros(2)},
        num_samples=100,
        train_loss=0.5,
        val_loss=0.5,
        val_metrics={"f1": 0.0, "val_loss": 0.5},
        num_positive=0,
        num_negative=100,
        update_norm=1.0,
    )
    cfg = {"aggregation": {"alpha": 1.0, "beta": 0.5, "reliability_power": 0.5, "quality_floor": 0.1, "min_weight": 0.01}}
    weights = adaptive_weights([positive_client, normal_client], cfg)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[1] > 0.05, "all-normal client should retain a meaningful share"


def test_communication_compression_none():
    state = {"layer.weight": torch.randn(10, 5), "layer.bias": torch.randn(5)}
    cfg = {"communication": {"method": "none"}}
    header, payload, stats = compress_state(state, cfg)
    decompressed = decompress_state(header, payload)

    for k in state:
        np.testing.assert_allclose(decompressed[k].numpy(), state[k].numpy())


def test_communication_compression_topk_quantize():
    state = {"layer.weight": torch.randn(20, 20)}
    cfg = {"communication": {"method": "topk_quantize", "topk_fraction": 0.2, "quant_bits": 8}}
    header, payload, stats = compress_state(state, cfg)

    assert stats["kept_fraction"] <= 0.25
    assert stats["quantized"] is True

    # Check payload bytes is smaller than raw float32 bytes
    p_bytes = payload_bytes(header, payload)
    raw_bytes = state["layer.weight"].numel() * 4
    assert p_bytes < raw_bytes

    # Check decompression produces valid tensor
    decompressed = decompress_state(header, payload)
    assert decompressed["layer.weight"].shape == (20, 20)
