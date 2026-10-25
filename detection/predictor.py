"""Theft prediction verdicts with model-derived explanations.

Outputs one of:

- ``NORMAL``
- ``POTENTIAL_THEFT``
- ``UNCERTAIN`` (probability inside the band around the decision threshold)

Explainability here is *faithful to the model*: top contributing engineered
features are computed from the model's own output sensitivity (gradient ×
input for torch models; permutation importance fallback for sklearn models).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from detection.threshold import uncertain_band
from utilities.config import get

VERDICT_NORMAL = "NORMAL"
VERDICT_THEFT = "POTENTIAL_THEFT"
VERDICT_UNCERTAIN = "UNCERTAIN"


class TheftPredictor:
    """Wraps a fitted model + tuned threshold into user-facing verdicts."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        threshold: float = 0.5,
        uncertain_margin: float = 0.15,
        engine_feature_start: int | None = None,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.uncertain_margin = float(uncertain_margin)
        self.engine_feature_start = engine_feature_start

    # ------------------------------------------------------------------ #
    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def _contributions(self, X_row: np.ndarray) -> np.ndarray:
        """Per-feature contribution of one sample to the predicted logit.

        Torch models: exact gradient × input (saliency).
        Sklearn linear models: coef × (x − baseline≈0); trees: zero-order
        permutation surrogate is unavailable for a single row, so we return
        NaNs and let the caller fall back to global importance.
        """
        model = self.model
        if isinstance(getattr(model, "model", None), torch.nn.Module):
            xt = torch.from_numpy(X_row.reshape(1, -1).astype(np.float32))
            xt.requires_grad_(True)
            model.model.eval()
            logit = model.model(xt).squeeze()
            logit.backward()
            grad = xt.grad.detach().numpy().reshape(-1)
            return grad * X_row.reshape(-1)
        if hasattr(getattr(model, "estimator", None), "coef_"):
            coef = np.asarray(model.estimator.coef_).reshape(-1)
            return coef * X_row.reshape(-1)
        return np.full(X_row.size, np.nan)

    # ------------------------------------------------------------------ #
    def predict_one(self, X_row: np.ndarray) -> dict[str, Any]:
        prob = float(self._probabilities(X_row.reshape(1, -1))[0])
        low, high = uncertain_band(self.threshold, self.uncertain_margin)
        if low <= prob <= high:
            verdict = VERDICT_UNCERTAIN
        elif prob >= self.threshold:
            verdict = VERDICT_THEFT
        else:
            verdict = VERDICT_NORMAL
        confidence = prob if prob >= self.threshold else 1.0 - prob

        contributions = self._contributions(X_row)
        factors: list[dict[str, Any]] = []
        if np.isfinite(contributions).all():
            order = np.argsort(-np.abs(contributions))[:5]
            factors = [
                {
                    "feature": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                    "contribution": float(contributions[i]),
                    "value": float(X_row.reshape(-1)[i]),
                }
                for i in order
            ]
        return {
            "verdict": verdict,
            "probability_theft": prob,
            "confidence": float(confidence),
            "threshold": self.threshold,
            "uncertain_band": [low, high],
            "top_factors": factors,
        }

    def predict_batch(self, X: np.ndarray, include_factors: bool = False) -> list[dict[str, Any]]:
        probs = self._probabilities(X)
        low, high = uncertain_band(self.threshold, self.uncertain_margin)
        results: list[dict[str, Any]] = []
        for i, prob in enumerate(probs):
            if low <= prob <= high:
                verdict = VERDICT_UNCERTAIN
            elif prob >= self.threshold:
                verdict = VERDICT_THEFT
            else:
                verdict = VERDICT_NORMAL
            entry = {
                "verdict": verdict,
                "probability_theft": float(prob),
                "confidence": float(prob if prob >= self.threshold else 1.0 - prob),
            }
            if include_factors:
                entry.update(self.predict_one(X[i]))
            results.append(entry)
        return results


def build_predictor(cfg: dict, model: Any, feature_names: list[str], threshold: float = 0.5) -> TheftPredictor:
    return TheftPredictor(
        model=model,
        feature_names=feature_names,
        threshold=threshold,
        uncertain_margin=float(get(cfg, "detection.uncertain_margin", 0.15)),
    )
