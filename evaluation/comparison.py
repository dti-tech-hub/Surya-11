"""Comparison tables (Tables 2–5) built from real result rows."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def build_table(rows: Sequence[dict[str, Any]], preferred_order: Sequence[str] | None = None) -> pd.DataFrame:
    """Assemble a DataFrame from result rows, ordering columns sensibly."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    front = [c for c in (preferred_order or []) if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def format_table(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Round float columns for display without destroying the raw data."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(float).round(digits)
    return out


def to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # tabulate is optional; fall back to a plain fixed-width table
        return "```\n" + format_table(df).to_string(index=False) + "\n```"


def save_table(df: pd.DataFrame, path: str | Path, name: str | None = None) -> dict[str, str]:
    """Persist a table as CSV + Markdown; returns the written paths."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    stem = name or "table"
    csv_path = path / f"{stem}.csv"
    md_path = path / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    title = f"# {stem.replace('_', ' ').title()}\n\n" if name else ""
    md_path.write_text(title + to_markdown(format_table(df)) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path)}


TABLE_TITLES = {
    "table1_dataset_statistics": "Dataset Statistics",
    "table2_model_comparison": "Model Comparison",
    "table3_federated_comparison": "Federated Comparison",
    "table4_privacy_comparison": "Privacy / Utility Comparison",
    "table5_scalability": "Scalability",
}
