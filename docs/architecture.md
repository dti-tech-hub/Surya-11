# PowerShield-FL — System Architecture

## Overview

PowerShield-FL is a modular research framework for privacy-preserving federated electricity-theft detection. Every component is configuration-driven and independently testable. The executed results use **SYNTHETIC DEMO DATA**; real SGCC/long-CSV loaders are provided but no real dataset is bundled.

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  data/raw/ → loader → consumer-level split (train/val/test) │
│  raw rows cleaned per consumer (never mixed across consumers)│
│  train-fitted imputer -> outlier bounds -> scaler           │
│  window feature engineering -> 133 features                 │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 CLIENT SIMULATION                           │
│  IID or non-IID Dirichlet partition of TRAIN consumers      │
│  Raw windows stay on each simulated client                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              FEDERATED ROUND LOOP                           │
│  1. Server broadcasts global LightweightMLP state           │
│  2. Client trains locally and computes a parameter delta    │
│  3. ClientUpdate contains delta + aggregate statistics only │
│  4. Privacy: L2 clipping + Gaussian noise                   │
│     (optional simulated additive-mask secure aggregation)   │
│  5. Communication: compress and measure serialized bytes    │
│  6. Server: FedAvg or adaptive weighted aggregation         │
│  7. Global model evaluated on validation split              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION & EXPLAINABILITY                    │
│  test metrics: F1, precision, recall, ROC-AUC, PR-AUC, CM   │
│  actual serialized bytes, wall time, RSS, parameters        │
│  SHAP KernelExplainer + permutation + gradient×input        │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                 Streamlit dashboard
```

## Component Design

### `utilities/`
- `config.py` — YAML loading, quick/full mode presets, dotted-path overrides, minimal `.env` loader.
- `seed.py` — `set_seed()` and `make_rng()`; all stochastic steps receive seeded RNGs.
- `system.py` — environment/dependency inspection.
- `types.py` — `ClientUpdate` contract. Its fields are only `state_dict`, sample counts, aggregate losses/metrics, and metadata. Raw samples, timestamps, and customer IDs have no field.

### `preprocessing/`
The leakage-prevention boundary is the consumer split:

```text
Raw dataset
  ↓ loader (synthetic | SGCC | long CSV)
Consumer-level stratified split (train / val / test)  ← FIRST
  ↓ cleaning per consumer: duplicates, invalid values, Hampel denoise, missing values
     (global imputation median fitted on TRAIN consumers only)
  ↓ IQR/z-score bounds fitted on TRAIN consumers; clip, never delete theft anomalies
  ↓ StandardScaler/MinMaxScaler fitted on TRAIN consumers
  ↓ window features (14-day windows, 3-day stride)
  ↓ PreprocessingBundle applied to validation/test
```

Consumers, not windows, are split, so no consumer's windows appear in two splits. Cleaning is per-consumer; the only global statistic (missing-value median) is fitted on training consumers and passed explicitly.

### `models/`
- `lightweight.py` — 3-layer MLP with 133 inputs, [64, 32] hidden units; **10,689 parameters**.
- `cnn_lstm.py` — 1D CNN + LSTM sequence baseline (21,617 parameters).
- `baselines.py` — sklearn Logistic Regression/Random Forest wrappers and a torch MLP wrapper.
- `base.py` — `TorchBinaryClassifier`: fit/predict, flat parameter access, save/load, resource accounting.

### `federated/`
- `client.py` — local training, local validation metrics, delta computation; returns only `ClientUpdate`.
- `server.py` — holds global state; validates the update contract; aggregates deltas.
- `simulator.py` — round loop, client selection, privacy/compression, server aggregation, validation logging.
- `partition.py` — IID and non-IID Dirichlet consumer partitions with per-client statistics.
- `aggregation.py` — weighted delta aggregation and delta application.
- `adaptive_aggregation.py` — sample × quality × reliability × class-balance weighting.
- `communication.py` — `none`, `topk`, `quantize`, `topk_quantize`; actual serialized-byte measurement.

### `privacy/`
- `base.py` — `NoPrivacy`, `build_privacy()`.
- `differential_privacy.py` — L2 clipping, Gaussian noise, single-release epsilon estimate.
- `secure_aggregation.py` — simulated additive mask cancellation (not production cryptography).
- `privacy_metrics.py` — per-round noise/clipping diagnostics.

### `detection/`
- `threshold.py` — validation-only F1 threshold selection.
- `predictor.py` — NORMAL / POTENTIAL_THEFT / UNCERTAIN and gradient×input factors.
- `postprocessing.py` — consumer-level aggregation of window verdicts.

### `explainability/`
- `shap_explainer.py` — KernelExplainer for torch models, TreeExplainer for RF, LinearExplainer for LR. DeepExplainer is intentionally not used because it was brittle with the installed SHAP/PyTorch combination.
- `feature_importance.py` — manual permutation importance (ROC-AUC drop), because sklearn's helper expects a fuller estimator protocol than the inference adapter provides.
- `explanations.py` — human-readable local explanation from the model's own gradient×input attribution.

### `evaluation/`
Classification metrics, detection/threshold evaluation, communication summaries, performance/memory accounting, scalability rows, comparison tables, statistical tests, figures, and the automatic Markdown report.

### `experiments/`
- `suite.py` — orchestrates dataset → centralized → FedAvg → proposed → sweeps → explainability → tables/figures → report.
- `centralized.py` — LR, RF, MLP, LightweightMLP, CNN-LSTM.
- `fedavg.py` — shared federated runner and FedAvg baseline.
- `proposed.py` — PowerShield-FL configuration.
- `privacy.py`, `communication.py`, `scalability.py` — measured sweeps.

## Privacy Boundary

```text
CLIENT SIDE                          SERVER SIDE
────────────────────────────────────────────────
Raw consumption windows              NEVER received
Consumer identifiers                 NEVER received
Timestamps                           NEVER received
                                     ─────────────
ClientUpdate:                        Received:
  state_dict (model delta)           yes
  num_samples                        yes
  train_loss / val_loss              scalars
  val_metrics                        aggregate scalars
  num_positive / num_negative        counts
  update_norm                        scalar
```

`ClientUpdate.validate()` rejects unexpected fields and non-tensor state entries. `tests/test_privacy.py` checks this contract and scans logs for raw identifiers.

## Configuration

Quick mode (`config/config.yaml`):

```yaml
experiment:
  mode: quick
  seed: 42

data:
  synthetic:
    consumers: 240
    days: 180
    theft_rate: 0.15

clients:
  num_clients: 5
  partition_strategy: non_iid

federated:
  rounds: 8
  local_epochs: 2
  batch_size: 128
  learning_rate: 0.001
  participation_rate: 1.0

privacy:
  enabled: true
  clipping_norm: 1.0
  noise_multiplier: 0.001
  secure_aggregation: false

communication:
  compression: true
  method: quantize
  quant_bits: 8

aggregation:
  method: adaptive
```

Full mode is 600 consumers × 365 days, 10 clients, 25 rounds, 3 seeds. It is **configured but NOT RUN** on the recorded CPU host.

## Experiment Tracking

Run IDs: `experiment_YYYYMMDD_HHMMSS_microseconds_<name>_s<seed>`.

```text
results/raw/<run_id>/
├── config.json
├── metrics.json
├── metrics.csv
├── logs.txt
├── model/
│   ├── checkpoint.pt
│   └── checkpoint.meta.json
├── test_predictions.csv
└── explainability/
    ├── permutation_importance.json
    ├── shap_summary.json
    └── local_explanations.json
```

The suite writes `results/metrics/summary.json`, `results/raw/experiment_metadata.json`, five tables, fifteen figures, and the final report.

## Implemented vs Not Implemented

| Feature | Status |
|---|---|
| Custom FL simulator | Implemented and executed |
| Standard FedAvg | Implemented and executed |
| Adaptive aggregation | Implemented and executed |
| L2 clipping + Gaussian noise | Implemented and executed |
| Secure aggregation | **Simulated masking**, not cryptography |
| CKKS homomorphic encryption | **Not implemented** (not justified on this CPU budget) |
| Top-k/quantization communication | Implemented and measured |
| SHAP/permutation/local attribution | Implemented and executed |
| Dashboard | Implemented and headlessly tested |
| Full-mode 3-seed run | **NOT RUN** |
| Real SGCC/IRISE execution | **DATASET NOT CONFIGURED** |
