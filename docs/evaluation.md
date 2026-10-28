# PowerShield-FL — Evaluation Framework

## Protocol

1. **Consumer-level stratified split**: 65% train / 15% validation / 20% test. A consumer's windows never cross splits.
2. **Train-only fitting**: imputation median, IQR bounds, and scaler statistics come from training consumers only.
3. **Threshold tuning on validation**: the decision threshold maximizes validation F1.
4. **Frozen threshold on test**: test labels are never used for tuning or model selection.
5. **Honest accounting**: every value is read from executed artifacts; missing experiments are `NOT RUN`.

## Metrics

### Detection performance

- **F1** — primary metric under class imbalance.
- **Precision** — false-alarm control.
- **Recall** — missed-theft control.
- **ROC-AUC** — threshold-independent discrimination.
- **PR-AUC** — informative under rare positives.
- **Confusion matrix** — TN/FP/FN/TP.

Final quick test values:

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| FedAvg | 1.0000 | 0.9877 | 0.9938 | 0.9973 |
| PowerShield-FL | 1.0000 | 0.9938 | 0.9969 | 0.9969 |

These are single-seed synthetic results; no significance claim is made.

### Communication overhead

`CommunicationMeter` records every upload and download using **actual serialized bytes**:

```text
payload = 4-byte header length + UTF-8 JSON header + compressed arrays
```

This includes quantization scales, top-k indices, tensor shapes, and metadata. It does not estimate bytes as `parameter_count × dtype_size`. The report computes:

```text
reduction = 100 × (1 - PowerShield_bytes / FedAvg_bytes)
```

Final measured values: FedAvg **3,453,520 bytes**, PowerShield-FL **902,449 bytes**, reduction **73.87%**.

### Computational resources

- Wall-clock federated training time.
- Inference latency per sample.
- Process RSS delta and inference peak Python memory.
- Trainable parameter count: LightweightMLP **10,689**, centralized MLP 25,473, CNN-LSTM 21,617, RF 21,862 (tree nodes/coefficients counted as parameters).
- Model checkpoint size.

Timing and memory are machine-specific (CPU-only host), not portable benchmarks. They are reported as measured, not normalised.

### Privacy / utility

- Clipping norm `C=1.0`.
- Gaussian noise multiplier `σ=0.001` by default.
- Single-release epsilon estimate `sqrt(2 ln(1.25/δ))/σ`; **no composition across rounds**.
- Simulated secure aggregation is not cryptography and is disabled by default.
- Privacy sweep values are in `results/reports/final_experiment_report.md`; the observed F1 is flat at 0.9814 for the four DP settings and 0.9907 for the privacy-off run. With one seed and 4 rounds, this is **not** evidence of a monotonic trade-off.

### Heterogeneity and scalability

- Per-client sample count, theft ratio, n_consumers, and local performance.
- Theft-ratio standard deviation and size coefficient of variation.
- Partition coverage: the final quick partition assigns all 8,736 training windows exactly once; quantity skew moves consumers rather than dropping windows. Sizes are `[2016, 3640, 336, 2408, 336]` and one client has zero theft windows.
- Client counts `{2, 5, 10, 20}` in the executed full sweep.
- Results must be compared at equal round counts; the 5-client sweep uses 4 rounds while the main proposed run uses 8.

## Statistical testing

Implemented in `evaluation/statistical_tests.py`:

- mean, standard deviation, 95% bootstrap confidence interval;
- paired t-test and Wilcoxon signed-rank test when assumptions permit;
- if fewer than two paired seeds exist, the report emits:

```text
Paired test over 3 seeds: not significant (p = 0.4038).
```

The executed suite has one seed, so the report does not claim significance.

## Claim rules

- `SUPPORTED` only when measured evidence directly supports the claim.
- `NO ADVANTAGE OBSERVED` when the proposed method is not better on the measured metric.
- `NOT RUN` for absent experiments (including full mode).
- `EXPERIMENT FAILED` when a stage raised an exception, with the error retained.
- No hard-coded metric is accepted in the report/dashboard.
