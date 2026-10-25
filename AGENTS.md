# AGENTS.md — PowerShield-FL

## What this project is

A B.Tech CSE research framework answering: *Can a lightweight, privacy-preserving,
communication-efficient Federated Learning framework detect electricity theft
effectively while reducing overhead and handling heterogeneous client data?*

Three approaches are compared experimentally:
- **A. Centralized ML** (Logistic Regression, Random Forest, MLP) — baseline
- **B. Standard FedAvg** — baseline
- **C. PowerShield-FL** (proposed) = lightweight model + privacy layer (clipping + Gaussian noise
  + optional simulated secure aggregation) + communication compression (top-k / quantization)
  + adaptive aggregation (size × quality × reliability weighting) + explainability

## Hard rules

1. **NEVER fabricate results.** Every number in reports/tables/figures must come from actual
   code execution saved under `results/`. Un-run experiments are reported as `NOT RUN`.
2. **NEVER leak raw client data.** Clients only transmit `utilities.types.ClientUpdate`
   (state_dict + aggregate stats). The server must never see raw consumption sequences,
   timestamps, or customer IDs. Tests in `tests/test_privacy.py` enforce this.
3. **Data leakage prevention:** split consumers into train/val/test FIRST; fit scalers and
   imputers on training data ONLY; transform val/test with the fitted objects.
4. **Never delete theft anomalies** during outlier handling — clip, don't drop.
5. Synthetic data must always be labelled `SYNTHETIC DEMO DATA` in outputs.
6. All randomness flows through `utilities.seed.set_seed` / `make_rng`.

## Architecture map

- `config/*.yaml` — single source of configuration; mode presets in `config/config.yaml`
  (`quick` default, `full`). Load via `utilities.config.load_config(mode, overrides)`.
- `utilities/` — config loading, seeding, env inspection, strict data contracts
  (`ClientUpdate`, `EvaluationResult`).
- `preprocessing/` — loader → cleaning → outliers → normalization → features → imbalance.
  A `PreprocessingBundle` holds train-fitted transformers and is applied to val/test.
- `models/` — `lightweight.py` (proposed MLP), `cnn_lstm.py` (baseline), `baselines.py`
  (sklearn LR/RF + torch MLP wrapper).
- `federated/` — client local training, server aggregation, simulator loop, IID/Non-IID
  partitioning, FedAvg + adaptive aggregation, communication measurement/compression.
- `privacy/` — update clipping, Gaussian noise, simulated secure aggregation, privacy metrics
  (noise level, epsilon estimate under (ε,δ)-style accounting — clearly documented as an
  estimate, not a formal guarantee).
- `detection/` — threshold selection, prediction labelling (NORMAL / POTENTIAL_THEFT / UNCERTAIN).
- `explainability/` — SHAP (Deep/Kernel) + permutation importance + per-prediction factor list.
- `evaluation/` — classification metrics, computational metrics, communication metrics,
  scalability, comparison tables, statistical tests (paired t-test / Wilcoxon across seeds).
- `experiments/` — runnable experiment drivers producing `results/` artifacts.
- `dashboard/app.py` — Streamlit; shows `NOT RUN` states honestly when results are absent.
- `scripts/` — per-step CLI scripts + `run_all_checks.py` + `run_experiments.py` + `generate_report.py`.
- `tests/` — pytest suite including privacy-leakage and end-to-end tests.

## Conventions

- Python 3.10+, type hints, `from __future__ import annotations`.
- No new heavy dependencies without checking `requirements.txt` first; everything must run
  on CPU in QUICK mode in a few minutes.
- Every experiment writes `results/raw/experiment_<timestamp>/` with `config.json`,
  `metrics.json`, `metrics.csv`, `logs.txt` and figures.
- Models checkpoint to `models/checkpoints/<name>.pt` with a sidecar `<name>.meta.json`.
- After changing core code, run: `python -m pytest tests -q` then `python main.py --demo`.

## Commands

- `python main.py --demo` — fast end-to-end synthetic demo
- `python main.py --mode quick` — full quick-mode experiment suite
- `python main.py --mode full` — final experiment suite
- `python scripts/run_all_checks.py` — environment + pipeline validation
- `streamlit run dashboard/app.py` — dashboard
