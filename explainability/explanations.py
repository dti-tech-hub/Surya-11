"""Human-readable explanation records for theft predictions.

Combines the model's verdict with its own attribution (SHAP when available,
gradient/coefficient contributions otherwise) into the display format:

```
Prediction : POTENTIAL_THEFT
Confidence : 87.3%
Important Factors: feature A (+0.42), feature B (−0.31), ...
```
"""
from __future__ import annotations

from typing import Any

import numpy as np

ENGINEERING_LABELS = {
    "mean_consumption": "average consumption level",
    "std_consumption": "consumption variability",
    "min_consumption": "minimum consumption",
    "max_consumption": "maximum consumption",
    "median_consumption": "median consumption",
    "peak_offpeak_ratio": "evening-peak vs night usage ratio",
    "night_day_ratio": "night vs overall usage ratio",
    "load_variation": "load variation (std/mean)",
    "consumption_trend": "consumption trend over the window",
    "daily_diff_std": "day-to-day volatility",
    "max_daily_drop": "largest day-over-day drop",
    "mean_daily_drop": "mean day-over-day drop",
    "frac_low_days": "fraction of unusually low days",
    "min_max_day_ratio": "lowest/highest day ratio",
    "acf_lag1": "daily autocorrelation",
    "skewness": "consumption skewness",
    "kurtosis": "consumption kurtosis",
    "weekend_weekday_ratio": "weekend vs weekday usage",
    "weekend_fraction": "weekend share of window",
    "start_dow_sin": "calendar term (sine)",
    "start_dow_cos": "calendar term (cosine)",
}


def human_feature_name(name: str) -> str:
    if name in ENGINEERING_LABELS:
        return ENGINEERING_LABELS[name]
    if name.startswith("block_"):
        try:
            idx = int(name.split("_")[1])
            day, slot = divmod(idx, 8)
            hour_start = slot * 3
            return f"6h block {slot + 1} (day {day + 1}, {hour_start:02d}-{hour_start + 3:02d}h)"
        except (IndexError, ValueError):
            return name
    return name


def format_explanation(record: dict[str, Any]) -> str:
    lines = [
        f"Prediction : {record.get('verdict', 'UNKNOWN')}",
        f"Confidence : {100.0 * float(record.get('confidence', 0.0)):.1f}%",
        f"Probability: {100.0 * float(record.get('probability_theft', 0.0)):.1f}%",
        "Important Factors:",
    ]
    factors = record.get("top_factors", [])
    if not factors:
        lines.append("  (no per-feature attribution available for this model type)")
    for factor in factors:
        direction = "↑ theft-risk" if factor.get("contribution", 0) > 0 else "↓ normal"
        lines.append(
            f"  - {human_feature_name(factor['feature'])}: contribution {factor.get('contribution', 0.0):+.4f} ({direction})"
        )
    return "\n".join(lines)


def explain_prediction(
    model: Any,
    X_row: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.5,
    shap_values_row: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one explanation record (model-derived, nothing fabricated)."""
    prob = float(np.asarray(model.predict_proba(X_row.reshape(1, -1))).reshape(-1)[0])
    verdict = "POTENTIAL_THEFT" if prob >= threshold else "NORMAL"
    confidence = prob if prob >= threshold else 1.0 - prob

    contributions: np.ndarray | None = None
    if shap_values_row is not None:
        contributions = np.asarray(shap_values_row, dtype=np.float64).reshape(-1)
    else:
        import torch

        torch_model = getattr(model, "model", None)
        if isinstance(torch_model, torch.nn.Module):
            xt = torch.from_numpy(X_row.reshape(1, -1).astype(np.float32))
            xt.requires_grad_(True)
            torch_model.eval()
            torch_model(xt).squeeze().backward()
            contributions = xt.grad.detach().numpy().reshape(-1) * X_row.reshape(-1)
        elif hasattr(getattr(model, "estimator", None), "coef_"):
            coef = np.asarray(model.estimator.coef_).reshape(-1)
            contributions = coef * X_row.reshape(-1)

    factors: list[dict[str, Any]] = []
    if contributions is not None and np.isfinite(contributions).all():
        order = np.argsort(-np.abs(contributions))[:5]
        factors = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
                "human_name": human_feature_name(feature_names[i] if i < len(feature_names) else f"f{i}"),
                "contribution": float(contributions[i]),
                "value": float(X_row.reshape(-1)[i]),
            }
            for i in order
        ]
    record = {
        "verdict": verdict,
        "probability_theft": prob,
        "confidence": float(confidence),
        "threshold": float(threshold),
        "top_factors": factors,
        "attribution_source": "shap" if shap_values_row is not None else ("gradient" if isinstance(getattr(model, "model", None), torch.nn.Module) else "coefficient"),
    }
    record["text"] = format_explanation(record)
    return record
