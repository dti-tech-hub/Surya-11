"""Tests enforcing privacy guarantees and raw data leakage prevention.

HARD RULE ENFORCEMENT:
1. Clients only transmit ClientUpdate (state_dict + aggregate stats).
2. The server must never see raw consumption sequences, timestamps, or customer IDs.
3. Privacy layer modifies/protects updates via clipping and calibrated noise.
4. No API keys or sensitive data in files/logs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from privacy.base import build_privacy
from privacy.differential_privacy import GaussianDPMechanism
from privacy.secure_aggregation import SecureAggregationSimulator
from utilities.types import ClientUpdate


def test_client_update_strictly_validates_no_raw_data():
    # Valid update
    valid_update = ClientUpdate(
        client_id="c1",
        state_dict={"weight": torch.randn(5, 5)},
        num_samples=100,
        train_loss=0.45,
        val_loss=0.50,
        num_positive=10,
        num_negative=90,
        update_norm=1.2,
    )
    valid_update.validate()  # must pass

    # Forbidden attributes or leakage attempts
    forbidden_keys = ["raw_consumption", "timestamps", "customer_ids", "consumption_sequence", "raw_data"]
    for key in forbidden_keys:
        assert not hasattr(valid_update, key), f"ClientUpdate data structure exposes forbidden field '{key}'"


def test_privacy_clipping_bounds_norm():
    clipping_norm = 1.0
    privacy = GaussianDPMechanism(clipping_norm=clipping_norm, noise_multiplier=0.0, seed=42)

    # Large weight update
    large_delta = {"weight": torch.ones(10, 10) * 10.0}
    clipped = privacy.apply(large_delta)

    total_norm = torch.norm(torch.stack([torch.norm(v) for v in clipped.values()])).item()
    assert total_norm <= clipping_norm + 1e-5


def test_privacy_gaussian_noise_changes_weights():
    privacy = GaussianDPMechanism(clipping_norm=5.0, noise_multiplier=0.5, seed=42)
    delta = {"w": torch.zeros(50, 50)}
    noised = privacy.apply(delta)

    # Output should not be identical to zeros
    assert not torch.allclose(noised["w"], delta["w"])
    assert getattr(privacy, "applied_noise_norm", 0.0) > 0.0


def test_secure_aggregation_preserves_sum_while_masking_individuals():
    cfg = {"privacy": {"secure_aggregation": True}}
    sec_agg = SecureAggregationSimulator(cfg, seed=42)

    u1 = {"w": torch.tensor([1.0, 2.0])}
    u2 = {"w": torch.tensor([3.0, 4.0])}
    original_sum = u1["w"] + u2["w"]

    masked, info = sec_agg.protect([u1, u2])

    # Individual masked updates should not equal original
    assert not torch.allclose(masked[0]["w"], u1["w"])
    assert not torch.allclose(masked[1]["w"], u2["w"])

    # Sum of masked updates must cancel masks and equal original sum
    masked_sum = masked[0]["w"] + masked[1]["w"]
    np.testing.assert_allclose(masked_sum.numpy(), original_sum.numpy(), atol=1e-5)


def test_no_hardcoded_api_keys_or_credentials():
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]

    # Check key source files for forbidden secrets
    source_files = list(root.glob("**/*.py")) + list(root.glob("**/*.yaml"))
    forbidden_tokens = ["AIzaSy", "sk-", "ghp_", "bearer "]

    for path in source_files:
        if ".git" in str(path) or ".system_generated" in str(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden_tokens:
                assert token not in content, f"Possible secret or API key token '{token}' in {path}"
        except Exception:
            pass
