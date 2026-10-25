"""SHAP-based explainability, computed from the actual fitted model.

- Torch models → ``shap.DeepExplainer`` on the lightweight MLP input tensor
  (falls back to KernelExplainer if DeepExplainer is unavailable for the
  architecture).
- Sklearn models (LR/RF) → ``shap.TreeExplainer`` (RF) / ``shap.LinearExplainer``
  (LR), with KernelExplainer as a generic fallback.

SHAP runs are capped to a small sample count in QUICK mode to stay
laptop-friendly; the cap is recorded so reports can state it.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from utilities.config import get

try:
    import shap  # noqa: F401

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - shap is in requirements
    SHAP_AVAILABLE = False


class ModelExplainer:
    """Produce SHAP values for a fitted binary classifier."""

    def __init__(self, model: Any, X_background: np.ndarray, cfg: dict, feature_names: list[str] | None = None):
        if not SHAP_AVAILABLE:
            raise RuntimeError("shap is not installed; use permutation importance instead")
        self.model = model
        self.feature_names = feature_names or [f"f{i}" for i in range(X_background.shape[1])]
        self.max_samples = int(get(cfg, "explainability.shap_samples", 200))
        self.nsamples = int(get(cfg, "explainability.shap_nsamples", 100))
        self.explainer_name = "unknown"
        bg_size = int(get(cfg, "explainability.shap_background", 50))
        self._background = X_background[: min(len(X_background), bg_size)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explainer = self._build_explainer(X_background)

    # ------------------------------------------------------------------ #
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X), dtype=np.float64).reshape(-1, 1)

    def _build_explainer(self, X_background: np.ndarray):
        import shap

        estimator = getattr(self.model, "estimator", None)
        torch_model = getattr(self.model, "model", None)
        bg = self._background
        if torch_model is not None:
            # DeepExplainer is brittle with the current PyTorch/SHAP
            # combination (it can pass numpy arrays into the torch graph).
            # The model-agnostic KernelExplainer is slower but reliable and
            # produces genuine SHAP values from this model's predictions.
            self.explainer_name = "KernelExplainer"
            return shap.KernelExplainer(self._model_predict, bg[: min(len(bg), 30)])
        if estimator is not None and hasattr(estimator, "estimators_"):
            try:
                self.explainer_name = "TreeExplainer"
                return shap.TreeExplainer(estimator, data=bg)
            except Exception:
                pass
        if estimator is not None and hasattr(estimator, "coef_"):
            try:
                self.explainer_name = "LinearExplainer"
                return shap.LinearExplainer(estimator, bg)
            except Exception:
                pass
        # generic fallback — small background for runtime
        self.explainer_name = "KernelExplainer"
        return shap.KernelExplainer(self._model_predict, bg[:50])

    # ------------------------------------------------------------------ #
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """SHAP values for up to ``max_samples`` rows; shape (n, n_features)."""
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] > self.max_samples:
            X = X[: self.max_samples]
        def _call() -> Any:
            if self.explainer_name == "DeepExplainer":
                import torch

                try:
                    x_tensor = torch.from_numpy(X.astype(np.float32))
                    return self._explainer.shap_values(x_tensor)
                except Exception:
                    import shap

                    self.explainer_name = "KernelExplainer"
                    self._explainer = shap.KernelExplainer(self._model_predict, self._background[:30])
                    return self._explainer.shap_values(X, nsamples=self.nsamples)
            if self.explainer_name == "KernelExplainer":
                try:
                    return self._explainer.shap_values(X, nsamples=self.nsamples, silent=True)
                except TypeError:
                    # newer SHAP versions removed the ``silent`` argument
                    return self._explainer.shap_values(X, nsamples=self.nsamples)
            return self._explainer.shap_values(X)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = _call()
        if isinstance(values, list):  # legacy shap list output for binary models
            values = values[-1]
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 3:
            values = values[..., -1]
        return values

    def summary_dict(self, X: np.ndarray) -> dict[str, Any]:
        values = self.shap_values(X)
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(-mean_abs)[:15]
        return {
            "explainer": self.explainer_name,
            "n_samples_explained": int(values.shape[0]),
            "mean_abs_shap": {
                self.feature_names[i] if i < len(self.feature_names) else f"f{i}": float(mean_abs[i])
                for i in order
            },
        }
