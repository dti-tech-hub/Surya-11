# PowerShield-FL — Methodology

## Research Question

> Can a lightweight, privacy-preserving, communication-efficient Federated Learning framework detect electricity theft effectively while reducing computational and communication overhead and handling heterogeneous client data?

## Why Federated Learning?

Centralized detection pools raw consumption records at one server. Federated Learning instead trains a shared model across simulated utilities/retailers; raw windows never leave a client. The server receives only a model delta and aggregate statistics. This reduces the data-centralisation risk but does not by itself guarantee privacy: updates can leak information, so the proposed layer adds clipping/noise and an optional masking simulation.

## Learning Model

`LightweightMLP` is a 3-layer MLP over 133 engineered/block features:

```text
133 inputs → Dense(64) → ReLU → Dropout(0.2)
           → Dense(32) → ReLU → Dropout(0.2)
           → Dense(1)  → sigmoid probability
```

It has **10,689 trainable parameters** and a checkpoint under 45 KB. The same architecture, partitions, seeds, and rounds are used for FedAvg and PowerShield-FL, so measured differences are attributable to aggregation/privacy/communication, not model capacity. The CNN-LSTM (21,617 parameters) is a separate centralized research baseline.

## Data Preprocessing and Leakage Prevention

The order is deliberate:

```text
Raw dataset
  ↓
Consumer-level stratified split (65% train / 15% validation / 20% test)  ← FIRST
  ↓
Clean per consumer: duplicates, invalid values, Hampel noise, missing values
  ↓
Fit IQR outlier bounds and scaler on TRAIN consumers only
  ↓
Transform train/validation/test with the train-fitted objects
  ↓
Build 14-day windows (3-day stride) and 133 features
```

- Splitting consumers first prevents a customer's windows from appearing in two splits.
- The global missing-value median is computed from training consumers only and passed into cleaning.
- IQR handling clips values; theft anomalies are never deleted.
- Class imbalance is measured and handled with class-weighted loss; validation threshold tuning is used instead of assuming 0.5.
- The synthetic generator inserts missing values, spikes, and theft patterns; all artifacts label the data `SYNTHETIC DEMO DATA`.

## Client Simulation

- Default: 5 clients, non-IID Dirichlet partition (`alpha=0.5`) with quantity skew.
- Quantity skew moves **whole consumers** between clients. It never thins a client's index list or drops windows; the final quick partition assigns all 8,736 training windows exactly once. Client sizes are `[2016, 3640, 336, 2408, 336]`, and one 43-consumer client contains zero theft windows.
- Each client receives only the training-window indices for its consumers.
- A client holds its own local training/validation split.
- The server receives `ClientUpdate`: delta tensors, sample counts, aggregate losses, local metrics, and update norm.
- No raw consumption, timestamp, or customer identifier is present in the contract.

## Privacy Layer

For each client update `Δ`:

1. Compute global L2 norm `||Δ||₂`.
2. If `||Δ||₂ > C`, scale the whole update to norm `C` (clipping).
3. Add independent Gaussian noise `N(0, (σC)²)` per coordinate.
4. Optionally apply simulated pairwise additive masks (disabled by default).

The reported epsilon is:

```text
ε ≈ sqrt(2 ln(1.25/δ)) / σ        (single release of one client update)
```

This is **not** an end-to-end DP guarantee: RDP composition over rounds, client sampling, and the simulated nature of secure aggregation are not accounted for. The default `σ=0.001` is deliberately small because the update has ~10,689 coordinates; the epsilon estimate is correspondingly large. Reports state this limitation explicitly.

Secure aggregation is a demonstration of additive mask cancellation with a shared PRNG seed. It is not cryptographic secure aggregation, has no dropout recovery, and provides no malicious-server guarantee. When masking is enabled, upload compression is disabled automatically because sparsifying masks would break cancellation.

## Communication Efficiency

Implemented methods (actual serialized bytes measured, including JSON headers and metadata):

| Method | What is sent |
|---|---|
| `none` | float32 values |
| `quantize` | int8 values + per-tensor scale (default) |
| `topk` | top 20% values by magnitude + int32 indices |
| `topk_quantize` | top 20% int8 values + int32 indices |

The final configuration uses `quantize`: in the 4-round communication sweep it preserved validation/test F1 (0.9814) while reducing measured bytes by **73.87%** versus uncompressed communication. Top-k reduced test F1 to 0.9272 in this setup. The project does not claim that top-k is universally harmful; it reports the measured result.

## Adaptive Aggregation

Standard FedAvg uses `n_i / Σn`. PowerShield-FL computes:

```text
w_i ∝ (n_i / Σn)^α
      × quality_i^β
      × reliability_i^reliability_power
      × balance_i
```

with:

```text
quality_i =
    0.5 · val_F1_i + 0.5 · val_AUC_i    if the local validation split has both classes
    0.5                                 if it has only one class
    clipped to [quality_floor, 1]

reliability_i = exp(-| ||Δ_i|| / median(||Δ||) - 1 |)

balance_i = (1 - λ) + λ · 2·sqrt(p_i·(1 - p_i)),   p_i = local theft ratio
```

Weights are normalized, optionally clipped at 3× the mean, and floored at `min_weight=0.01`. All coefficients are in `config/aggregation.yaml`.

**Why the single-class fallback matters:** one client in the quick partition has zero theft windows. Its local F1 is 0 by the `zero_division=0` convention, which previously collapsed its weight to the 0.01 floor. That is not evidence of poor data quality; the client is a legitimate all-normal utility. The fallback gives it neutral quality (0.5), and the unit test `test_zero_positive_client_is_not_collapsed_to_min_weight` enforces this.

## Detection

- Validation predictions select the F1-optimal threshold.
- The threshold is frozen and applied once to test predictions.
- Verdicts are `NORMAL`, `POTENTIAL_THEFT`, or `UNCERTAIN` inside a configurable probability band.
- Consumer verdicts aggregate window probabilities with a configurable positive-window ratio.

## Explainability

All explanations come from the fitted model:

- **SHAP KernelExplainer** on a background sample of validation rows (torch model). TreeExplainer/LinearExplainer are used for RF/LR. DeepExplainer was removed because it failed with the installed SHAP/PyTorch combination.
- **Permutation importance**: manual feature shuffling on validation, measuring ROC-AUC drop, repeated with a seeded RNG.
- **Local attribution**: gradient×input for the lightweight torch model; coefficient×input for linear models.

No explanation is hard-coded or sampled from a static table.

## Experimental Design

### Comparison hierarchy

```text
A. Centralized: LR, RF, MLP, LightweightMLP, CNN-LSTM on pooled training data
B. FedAvg: same LightweightMLP/partitions/rounds; no privacy, no compression
C. PowerShield-FL: same model; clip + Gaussian noise + int8 quantization
   + adaptive aggregation + explainability
```

### Quick-mode settings

240 synthetic consumers × 180 days × 48 readings; 5 clients; 8 rounds; 2 local epochs; batch 128; seed 42. The quick suite also runs privacy, communication, and scalability sweeps with 4 rounds.

### Sweeps

- Privacy: `noise_multiplier ∈ {0, 0.001, 0.003, 0.006}`.
- Communication: `{none, quantize, topk, topk_quantize}`.
- Scalability: `{2, 5, 10}` clients in quick mode; full mode adds 20.

### Statistics

The code supports multiple seeds, mean/std, 95% bootstrap CI, and paired t-test/Wilcoxon. The executed quick suite has **one seed**, so the report explicitly says `STATISTICAL TEST NOT RUN — fewer than 2 paired seeds`. No significance claim is made.

## Claim Discipline

- `PASS` / measured values come from `results/metrics/summary.json`.
- `NOT RUN` marks unexecuted experiments, including full mode.
- `DATASET NOT CONFIGURED` marks absent real data.
- `EXPERIMENT FAILED` includes the recorded exception.
- No metric is manually typed into the report generator; the report is regenerated from the summary.
