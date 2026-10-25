# PowerShield-FL — Final Project Status

**Date:** 2026-10-25  
**Project:** B.Tech CSE research framework for privacy-preserving, communication-efficient federated electricity-theft detection  
**Authoritative suite:** `experiment_20261025_014623_793809_suite_s42`  
**Dataset:** `SYNTHETIC DEMO DATA` (no real utility data is bundled or claimed)  
**Mode:** `quick`; seed `[42]`; single-seed descriptive results

> This status file records what was actually executed. Values are read from `results/metrics/summary.json` and the generated report. Unrun experiments are marked **NOT RUN**.

## 1. Executive status

| Item | Status |
|---|---|
| Core implementation | **COMPLETE** |
| Quick-mode experiment suite | **PASS** — all 12 stages passed |
| Unit/integration tests | **PASS** — 33 passed |
| End-to-end component validation | **PASS** — 12 PASS, 1 WARNING (synthetic dataset), 0 FAIL |
| Dashboard AppTest | **PASS** — 10 tabs, 0 exceptions |
| Live dashboard health check | **PASS** — HTTP 200 / `ok` on port 8511 |
| Demo smoke test with corrected partitioner | **PASS** — exit code 0 |
| Notebooks 1–5 headless execution | **PASS** — all code cells executed |
| Full mode (3 seeds) | **NOT RUN** on this CPU host |
| Real SGCC/IRISE data | **DATASET NOT CONFIGURED** |
| Formal end-to-end differential privacy | **NOT CLAIMED** (single-release estimate only) |
| Production cryptographic secure aggregation | **NOT IMPLEMENTED** (simulation only) |

## 2. Final quick-mode results

Quick mode: 240 synthetic consumers × 180 days × 48 half-hourly readings; 5 non-IID clients; 8 federated rounds; 2 local epochs; batch 128; seed 42. Sweeps use 4 rounds.

### 2.1 Dataset and preprocessing

| Statistic | Value |
|---|---:|
| Consumers (train / validation / test) | 156 / 36 / 48 |
| Window samples (train / validation / test) | 8,736 / 2,016 / 2,688 |
| Features | 133 |
| Raw rows | 2,073,600 |
| Raw missing values | 4,244 |
| Raw duplicate rows | 0 |
| Train-window theft ratio | 8.08% |
| Test-window theft ratio | 6.03% |
| Split unit | consumer |

The consumer split is performed before fitting preprocessing. Imputation, IQR bounds, and scaling are fitted on training consumers only. The theft anomalies are clipped, not deleted.

### 2.2 Corrected non-IID partition

The quantity-skew implementation now moves whole consumers between clients instead of thinning client index lists. The final partition assigns **all 8,736 training windows exactly once**.

| Client | Samples | Consumers | Theft windows | Theft ratio |
|---|---:|---:|---:|---:|
| client_00 | 2,016 | 36 | 346 | 0.1716 |
| client_01 | 3,640 | 65 | 329 | 0.0904 |
| client_02 | 336 | 6 | 18 | 0.0536 |
| client_03 | 2,408 | 43 | 0 | 0.0000 |
| client_04 | 336 | 6 | 13 | 0.0387 |

Heterogeneity: theft-ratio std **0.0581**, size CV **0.7273**. `client_03` has no theft windows; the adaptive-quality fallback gives a single-class local validation split neutral quality `0.5` rather than collapsing its weight to the floor.

### 2.3 Centralized baselines

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9993 | 0.9938 | 0.9938 | 0.9938 | 0.9986 | 0.50 | 134 |
| Random Forest | 0.9996 | 1.0000 | 0.9938 | **0.9969** | 0.9950 | 5.17 | 21,862 |
| MLP | 0.9829 | 0.7816 | 0.9938 | 0.8750 | 0.9996 | 15.45 | 25,473 |
| Lightweight MLP (central) | 0.9933 | 0.9045 | 0.9938 | 0.9471 | 0.9991 | 7.52 | 10,689 |
| CNN-LSTM baseline | 0.9769 | 1.0000 | 0.6173 | 0.7634 | 0.9444 | 36.16 | 21,617 |

### 2.4 Federated comparison

| Method | Clients | Rounds | Precision | Recall | F1 | ROC-AUC | Time (s) | Total bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 5 | 8 | 1.0000 | 0.9877 | 0.9938 | 0.9973 | 15.51 | 3,453,520 |
| **PowerShield-FL** | 5 | 8 | **1.0000** | **0.9938** | **0.9969** | **0.9969** | **15.97** | **902,449** |

**Measured communication reduction: `73.87%`** from actual serialized payloads, including headers and quantization metadata. This is not a theoretical `parameter_count × dtype_size` estimate.

The F1 difference is `+0.0031` in one synthetic seed. It is descriptive, not statistically significant, and the proposed row also differs by privacy and quantization, not only aggregation.

### 2.5 Privacy / utility sweep (4 rounds)

| Privacy level | Noise multiplier | Epsilon estimate (single release) | F1 | Validation F1 | Total bytes |
|---|---:|---:|---:|---:|---:|
| off | 0 | — | 0.9907 | 0.9914 | 451,241 |
| noise_0 | 0 | — | 0.9814 | 0.9885 | 451,235 |
| noise_0.001 | 0.001 | 4,844.81 | 0.9814 | 0.9885 | 451,222 |
| noise_0.003 | 0.003 | 1,614.94 | 0.9814 | 0.9885 | 451,243 |
| noise_0.006 | 0.006 | 807.47 | 0.9814 | 0.9885 | 451,242 |

The F1 is flat across nonzero noise settings; the project does **not** claim a monotonic privacy–utility trade-off. Epsilon is a classical single-release Gaussian-mechanism estimate. Composition across rounds, client sampling, and the simulated secure-aggregation layer are not accounted for, so no end-to-end DP guarantee is claimed.

### 2.6 Communication sweep (4 rounds)

| Setting | F1 | Validation F1 | Total bytes | Reduction vs none |
|---|---:|---:|---:|---:|
| none (float32) | 0.9814 | 0.9885 | 1,726,760 | 0.00% |
| int8 `quantize` (default) | 0.9814 | 0.9885 | 451,222 | 73.87% |
| top-k 20% | 0.9272 | 0.9767 | 700,440 | 59.44% |
| top-k + int8 | 0.9236 | 0.9738 | 451,227 | 73.87% |

Dense int8 quantization preserved the 4-round validation/test utility while reducing measured bytes. Top-k lost F1 in this setup; it remains implemented and measured, but is not the default.

### 2.7 Scalability (4 rounds)

| Clients | F1 | Validation F1 | Time (s) | Total bytes |
|---:|---:|---:|---:|---:|
| 2 | 0.9846 | 0.9943 | 6.56 | 180,497 |
| 5 | 0.9814 | 0.9885 | 7.60 | 451,222 |
| 10 | 0.8380 | 0.9290 | 8.95 | 902,445 |

The 5-client sweep is a 4-round run and is not directly comparable to the 8-round main comparison. Time and memory are machine-specific.

## 3. Implementation summary

| Component | Implementation |
|---|---|
| Local model | `LightweightMLP`: 133 → 64 → 32 → 1; **10,689** parameters; checkpoint **41.75 KB** |
| Privacy | L2 clipping (`C=1.0`) + Gaussian noise (`σ=0.001`); optional additive-mask secure-aggregation simulation |
| Communication | Dense int8 quantization by default; top-k and top-k+int8 implemented and measured |
| Aggregation | FedAvg or adaptive size × quality × reliability × class-balance weighting |
| Detection | Validation-tuned threshold, frozen for test; NORMAL / POTENTIAL_THEFT / UNCERTAIN |
| Explainability | SHAP KernelExplainer, manual permutation importance, gradient×input local factors |
| Dashboard | 10 tabs; reads real artifacts only; honest `NOT RUN`/missing-checkpoint states |
| Tests | 33 tests, including privacy-contract, partition-coverage, zero-positive-client, and end-to-end checks |

## 4. Validation evidence

```text
python -m pytest tests -q
33 passed in 29.26s

python scripts/run_all_checks.py
12 PASS, 1 WARNING (synthetic dataset), 0 FAIL, 0 NOT CONFIGURED out of 13 checks

Streamlit AppTest
10 tabs, 0 exceptions

Live Streamlit health
GET http://127.0.0.1:8511/_stcore/health -> HTTP 200, body "ok"

python main.py --demo
exit code 0 (smoke test after final partition correction; demo artifacts were not retained)
```

All five notebooks under `notebooks/` were executed headlessly; all code cells completed. Notebook 3 verifies partition coverage, and notebook 5 reads the final quick suite.

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
# Fast smoke test (overwrites results/; rerun quick afterward for final tables)
python main.py --demo

# Authoritative quick suite
python main.py --mode quick --seed 42

# Component validation + dashboard + pytest
python scripts/run_all_checks.py

# Tests only
python -m pytest tests -q

# Dashboard
streamlit run dashboard/app.py

# Regenerate report from the saved summary
python main.py --report-only
```

## 7. Artifacts

```text
results/metrics/summary.json                  authoritative quick summary
results/reports/final_experiment_report.md    generated report
results/tables/table1..table5.{csv,md}        comparison tables
results/figures/*.png                         15 figures
results/raw/experiment_*/                     per-run config, metrics, logs, checkpoints
models/checkpoints/                           named model checkpoints + metadata
data/partitions/                              consumer/window partition artifacts
```

The report generator reads `summary.json`; it does not accept hand-entered metrics.

## 8. Limitations and honest non-claims

1. **Synthetic data only.** No real SGCC/IRISE/AMI dataset is bundled; real-data execution is `DATASET NOT CONFIGURED`.
2. **One seed.** The quick suite ran seed 42 only. The difference between FedAvg and PowerShield-FL is descriptive; no significance test was run.
3. **Full mode not run.** 600 consumers, 10 clients, 25 rounds, seeds `[42, 123, 2026]`, and 20-client scalability are configured but `NOT RUN`.
4. **No formal end-to-end DP.** Epsilon is a single-release estimate; composition over rounds is not implemented.
5. **Secure aggregation is a simulation.** It demonstrates additive mask cancellation with a seeded PRNG; it is not cryptographic secure aggregation and has no dropout recovery.
6. **Simulated clients.** The project does not claim a real distributed deployment or network measurements.
7. **Timing is machine-specific.** Wall time, RSS, and inference latency are measured on this CPU host and are not portable benchmarks.
8. **Top-k lost utility here.** The default is dense int8 quantization; this is a measured choice, not a claim that top-k is universally harmful.
9. **Consumer labels are synthetic.** Window labels respect synthetic theft activation; real labels may be noisier.
10. **Demo is not final evidence.** The demo overwrites `results/`; use quick mode for the comparison tables.

## 9. Final claim

On the executed **synthetic, single-seed quick run**, PowerShield-FL reached test F1 **0.9969** versus FedAvg **0.9938**, with the same precision (1.0000) and higher recall (0.9938 vs 0.9877), while reducing measured communication by **73.87%**. The lightweight model has **10,689** parameters. These are descriptive synthetic results with the limitations above; no real-world or statistically significant superiority is claimed.

---

*Every numeric value in this file is traceable to `results/metrics/summary.json` or the generated report. Unrun work remains marked `NOT RUN` / `DATASET NOT CONFIGURED`.*
