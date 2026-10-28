# PowerShield-FL — Final Project Status

**Date:** 2026-10-28  
**Project:** B.Tech CSE research framework for privacy-preserving, communication-efficient federated electricity-theft detection  
**Authoritative suite:** `experiment_20261028_072746_856067_suite_s42`  
**Dataset:** `SYNTHETIC DEMO DATA` (no real utility data is bundled or claimed)  
**Mode:** `full`; seeds `[42, 123, 2026]`; multi-seed results with paired statistical tests

> This status file records what was actually executed. Values are read from `results/metrics/summary.json` and the generated report. Unrun experiments are marked **NOT RUN**.

## 1. Executive status

| Item | Status |
|---|---|
| Core implementation | **COMPLETE** |
| Quick-mode experiment suite | **PASS** — all stages passed (archived under `archive/quick_20261025_014623/`) |
| Full-mode experiment suite (3 seeds) | **PASS** — all 18 stages passed |
| Unit/integration tests | **PASS** — 33 passed |
| End-to-end component validation | **PASS** — 12 PASS, 1 WARNING (synthetic dataset), 0 FAIL |
| Dashboard AppTest | **PASS** — 10 tabs, 0 exceptions |
| Demo smoke test | **PASS** — exit code 0 |
| Real SGCC/IRISE data | **DATASET NOT CONFIGURED** |
| Formal end-to-end differential privacy | **NOT CLAIMED** (single-release estimate only) |
| Production cryptographic secure aggregation | **NOT IMPLEMENTED** (simulation only) |

## 2. Full-mode results (authoritative)

Full mode: 600 synthetic consumers × 365 days × 48 half-hourly readings; 10 non-IID clients; 25 federated rounds; 2 local epochs; batch 128; seeds `[42, 123, 2026]`. Sweeps use 4 rounds.

### 2.1 Dataset and preprocessing

| Statistic | Value |
|---|---:|
| Consumers (train / validation / test) | 390 / 90 / 120 |
| Window samples (train / validation / test) | 46,020 / 10,620 / 14,160 |
| Features | 133 |
| Raw rows | 10,512,000 |
| Raw missing values | 20,802 |
| Raw duplicate rows | 0 |
| Train-window theft ratio | 6.00% |
| Validation-window theft ratio | 5.64% |
| Test-window theft ratio | 6.02% |
| Split unit | consumer |

The consumer split is performed before fitting preprocessing. Imputation, IQR bounds, and scaling are fitted on training consumers only. Theft anomalies are clipped, not deleted.

### 2.2 Non-IID partition (seed 42)

| Client | Samples | Consumers | Theft windows | Theft ratio |
|---|---:|---:|---:|---:|
| client_00 | 826 | 7 | 0 | 0.0000 |
| client_01 | 2,714 | 23 | 151 | 0.0556 |
| client_02 | 1,652 | 14 | 117 | 0.0708 |
| client_03 | 826 | 7 | 206 | 0.2494 |
| client_04 | 12,272 | 104 | 312 | 0.0254 |
| client_05 | 826 | 7 | 69 | 0.0835 |
| client_06 | 826 | 7 | 88 | 0.1065 |
| client_07 | 826 | 7 | 37 | 0.0448 |
| client_08 | 24,426 | 207 | 1,756 | 0.0719 |
| client_09 | 826 | 7 | 25 | 0.0303 |

Heterogeneity: theft-ratio std **0.0655**, size CV **1.6110**. Partition differs across seeds; the table above is seed 42.

### 2.3 Centralized baselines (mean over 3 seeds)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9986 | 0.9991 | 0.9739 | 0.9863 | 0.9950 | 2.80 | 134 |
| Random Forest | 0.9965 | 0.9863 | 0.9468 | 0.9660 | 0.9962 | 43.82 | 52,499 nodes |
| MLP | 0.9982 | 1.0000 | 0.9646 | 0.9819 | 0.9940 | 43.13 | 25,473 |
| Lightweight MLP (central) | 0.9984 | 0.9995 | 0.9688 | 0.9839 | 0.9949 | 49.75 | 10,689 |
| CNN-LSTM baseline | 0.9866 | 0.9529 | 0.7700 | 0.8476 | 0.9696 | 348.45 | 21,617 |

CNN-LSTM is unstable across seeds (F1 0.9410, 0.8294, 0.7724) and is not tuned in this mode.

### 2.4 Federated comparison (mean over 3 seeds)

| Method | Clients | Rounds | Precision | Recall | F1 | ROC-AUC | Time (s) | Total bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 10 | 25 | 1.0000 | 0.9653 | 0.9823 ± 0.0071 | 0.9930 | 284.24 | 21,584,500 |
| **PowerShield-FL** | 10 | 25 | 0.9989 | 0.9622 | 0.9802 ± 0.0090 | 0.9930 | 316.62 | 5,640,225 |

**Measured communication reduction: `73.87%`** from actual serialized payloads.

Per-seed window-level F1: FedAvg 0.9839 / 0.9745 / 0.9885; PowerShield-FL 0.9784 / 0.9722 / 0.9899. Paired difference (FedAvg − PowerShield) = +0.0021; paired t p = 0.4038; Wilcoxon p = 0.5; not significant at α = 0.05. The objective was not to force PowerShield-FL to win; on this synthetic dataset it is competitive on F1 and substantially cheaper in communication.

Per-seed window confusion (TP/FP/TN/FN): FedAvg 826/0/13,307/27 (seed 42), 631/0/13,496/33 (seed 123), 685/0/13,459/16 (seed 2026); PowerShield-FL 817/0/13,307/36, 630/2/13,494/34, 687/0/13,459/14.

### 2.5 Consumer-level test results (per seed)

| Method | Seed | Precision | Recall | F1 | TP / FP / TN / FN |
|---|---:|---:|---:|---:|---:|
| FedAvg | 42 | 1.0000 | 0.5714 | 0.7273 | 8 / 0 / 106 / 6 |
| FedAvg | 123 | 1.0000 | 0.2143 | 0.3529 | 3 / 0 / 106 / 11 |
| FedAvg | 2026 | 1.0000 | 0.4286 | 0.6000 | 6 / 0 / 106 / 8 |
| PowerShield-FL | 42 | 1.0000 | 0.5000 | 0.6667 | 7 / 0 / 106 / 7 |
| PowerShield-FL | 123 | 1.0000 | 0.2143 | 0.3529 | 3 / 0 / 106 / 11 |
| PowerShield-FL | 2026 | 1.0000 | 0.4286 | 0.6000 | 6 / 0 / 106 / 8 |

Consumer-level ROC-AUC and PR-AUC are 1.0000 for every run; the gap is a calibration/threshold problem, not a ranking failure. The rule used is the mean window probability thresholded at 0.5.

### 2.6 Privacy / utility sweep (4 rounds)

| Privacy level | Noise multiplier | Epsilon estimate (single release) | F1 | Total bytes |
|---|---:|---:|---:|---:|
| off | 0 | — | 0.9735 | 902,359 |
| noise_0 | 0 | — | 0.9692 | 902,405 |
| noise_0.001 | 0.001 | 4,844.81 | 0.9692 | 902,464 |
| noise_0.003 | 0.003 | 1,614.94 | 0.9692 | 902,483 |
| noise_0.006 | 0.006 | 807.47 | 0.9692 | 902,493 |

Clipping, not the small Gaussian noise, accounts for the F1 change. The project does **not** claim a monotonic privacy–utility trade-off. Epsilon is a classical single-release Gaussian-mechanism estimate; composition across rounds is not accounted for.

### 2.7 Communication sweep (4 rounds)

| Setting | F1 | ROC-AUC | Total bytes | Reduction vs none |
|---|---:|---:|---:|---:|
| none (float32) | 0.9679 | 0.9737 | 3,453,520 | 0.00% |
| int8 `quantize` (default) | 0.9692 | 0.9734 | 902,464 | 73.87% |
| top-k 20% | 0.9522 | 0.9729 | 1,400,880 | 59.44% |
| top-k + int8 | 0.9542 | 0.9735 | 902,486 | 73.87% |

Dense int8 quantization preserved the 4-round utility while reducing measured bytes. Top-k lost F1 in this setup; it remains implemented and measured, but is not the default.

### 2.8 Scalability (4 rounds, PowerShield-FL)

| Clients | F1 | Accuracy | Time (s) | Total bytes |
|---:|---:|---:|---:|---:|
| 2 | 0.9754 | 0.9971 | 36.96 | 180,495 |
| 5 | 0.9735 | 0.9969 | 32.62 | 451,256 |
| 10 | 0.9692 | 0.9964 | 33.64 | 902,464 |
| 20 | 0.9494 | 0.9940 | 48.00 | 1,804,808 |

The 10/20-client rows use only 4 rounds and are not directly comparable to the 25-round main comparison. Time and memory are machine-specific.

### 2.9 Explainability (seed 42 global model)

Permutation importance over 1,500 test windows and 5 repeats; baseline ROC-AUC 1.0000. Top features: `min_max_day_ratio` (0.0201 ± 0.0028), `daily_diff_std` (0.0090 ± 0.0010), `weekend_weekday_ratio` (0.0031 ± 0.0027), `max_daily_drop` (0.0018 ± 0.0002), `mean_daily_drop` (0.0004 ± 0.0002). SHAP used 40 test windows, 30-row background, 100 evaluation samples, and produced 10 local explanations. Feature importance is model-specific and correlational, not causal.

## 3. Implementation summary

| Component | Implementation |
|---|---|
| Local model | `LightweightMLP`: 133 → 64 → 32 → 1; **10,689** parameters; checkpoint **41.75 KB** |
| Privacy | L2 clipping (`C=1.0`) + Gaussian noise (`σ=0.001`); optional additive-mask secure-aggregation simulation (disabled) |
| Communication | Dense int8 quantization by default; top-k and top-k+int8 implemented and measured |
| Aggregation | FedAvg or adaptive size × quality × reliability × class-balance weighting |
| Detection | Validation-tuned threshold, frozen for test; NORMAL / POTENTIAL_THEFT / UNCERTAIN |
| Explainability | SHAP KernelExplainer, manual permutation importance, gradient×input local factors |
| Dashboard | 10 tabs; reads real artifacts only; honest `NOT RUN`/missing-checkpoint states |
| Tests | 33 tests, including privacy-contract, partition-coverage, zero-positive-client, and end-to-end checks |

## 4. Validation evidence

```text
python -m pytest tests -q
33 passed

python scripts/run_all_checks.py
12 PASS, 1 WARNING (synthetic dataset), 0 FAIL, 0 NOT CONFIGURED out of 13 checks

Streamlit AppTest
10 tabs, 0 exceptions

python main.py --demo
exit code 0
```

All five notebooks under `notebooks/` were executed headlessly; all code cells completed. Notebook 3 verifies partition coverage, and notebook 5 reads the final full suite.

## 5. Environment

| Item | Value |
|---|---|
| Python | 3.14.5 |
| OS | Windows 11 (`Windows-11-10.0.26200-SP0`) |
| CPU | 10 physical / 12 logical cores |
| RAM | 15.72 GB |
| GPU | CPU only |
| PyTorch | 2.14.1+cpu |
| NumPy / Pandas | 2.4.4 / 3.0.1 |
| scikit-learn | 1.8.0 |
| SHAP | 0.53.0 |
| Streamlit | 1.65.0 |

## 6. Commands

```bash
# Fast smoke test (overwrites results/; rerun full afterward for final tables)
python main.py --demo

# Authoritative full suite
python main.py --mode full

# Component validation + dashboard + pytest
python scripts/run_all_checks.py

# Tests only
python -m pytest tests -q

# Dashboard
streamlit run dashboard/app.py

# Regenerate report from the saved summary
python main.py --mode full --report-only
```

## 7. Artifacts

```text
results/metrics/summary.json                  authoritative full summary
results/reports/final_experiment_report.md    generated report
results/tables/table1..table5.{csv,md}        comparison tables
results/figures/*.png                         figures
results/raw/experiment_*/                     per-run config, metrics, logs, checkpoints
models/checkpoints/                           named model checkpoints + metadata
data/partitions/                              consumer/window partition artifacts
```

## 8. Limitations and honest non-claims

1. **Synthetic data only.** No real SGCC/IRISE/AMI dataset is bundled; real-data execution is `DATASET NOT CONFIGURED`.
2. **Three seeds.** The full suite ran seeds `[42, 123, 2026]`. The FedAvg vs PowerShield-FL F1 difference is not statistically significant (paired t p = 0.4038; Wilcoxon p = 0.5); low power with 3 seeds.
3. **No formal end-to-end DP.** Epsilon is a single-release estimate; composition over rounds is not implemented.
4. **Secure aggregation is a simulation.** It demonstrates additive mask cancellation with a seeded PRNG; it is not cryptographic secure aggregation and has no dropout recovery.
5. **Simulated clients.** The project does not claim a real distributed deployment or network measurements.
6. **Timing is machine-specific.** Wall time, RSS, and inference latency are measured on this CPU host and are not portable benchmarks.
7. **Top-k lost utility here.** The default is dense int8 quantization; this is a measured choice, not a claim that top-k is universally harmful.
8. **Consumer labels are synthetic.** Window labels respect synthetic theft activation; real labels may be noisier.
9. **Consumer threshold not tuned.** The reported consumer F1 uses a fixed 0.5 mean-probability threshold; a validation-tuned threshold or positive-window-ratio rule could differ.
10. **CNN-LSTM unstable.** Its F1 ranges from 0.7724 to 0.9410 across seeds; it is under-trained relative to capacity in this mode.
11. **Demo is not final evidence.** The demo overwrites `results/`; use full mode for the comparison tables.

## 9. Final claim

On the executed **synthetic full run** (600 consumers, 10 clients, 25 rounds, 3 seeds), PowerShield-FL reached window-level F1 **0.9802 ± 0.0090** versus FedAvg **0.9823 ± 0.0071**, with comparable precision (0.9989 vs 1.0000) and recall (0.9622 vs 0.9653), while reducing measured communication by **73.87%** (21,584,500 → 5,640,225 bytes). The F1 difference is not statistically significant. The lightweight model has **10,689** parameters. Consumer-level F1 ranges from 0.3529 to 0.6667 depending on seed, despite perfect consumer-level ranking (ROC-AUC = 1.0). These are descriptive synthetic results with the limitations above; no real-world or statistically significant superiority is claimed.

---

*Every numeric value in this file is traceable to `results/metrics/summary.json` or the generated report. Unrun work remains marked `NOT RUN` / `DATASET NOT CONFIGURED`.*