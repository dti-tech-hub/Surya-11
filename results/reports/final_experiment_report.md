# PowerShield-FL — Final Experiment Report

> This report is generated automatically from `results/metrics/summary.json`. Every value below comes from an executed run; absent sections are marked **NOT RUN**.

> **DATASET: SYNTHETIC DEMO DATA** — results from this dataset are a software demonstration, not evidence about real utility theft rates.

- Generated: 2026-10-25T01:46:23.794390+00:00
- Suite ID: `experiment_20261025_014623_793809_suite_s42`
- Mode: `quick`; demo: `False`
- Seeds: [42]
- Dataset source: **SYNTHETIC DEMO DATA**

## 1. Implementation status

| Stage | Status |
|---|---|
| dataset | PASS |
| partitions | PASS |
| centralized_s42 | PASS |
| fedavg_s42 | PASS |
| proposed_s42 | PASS |
| explainability_s42 | PASS |
| privacy_sweep | PASS |
| communication_sweep | PASS |
| scalability | PASS |
| tables | PASS |
| figures | PASS |

## 2. Dataset

- Source: **SYNTHETIC DEMO DATA**
- Consumers: train 156, validation 36, test 48
- Windows: train 8736, validation 2016, test 2688
- Features: 133
- Raw missing values: 4244
- Raw duplicate rows: 0
- Split unit: consumer (no consumer appears in two splits)

### Table 1 — Dataset statistics

| statistic                 | value                                                   |
|:--------------------------|:--------------------------------------------------------|
| dataset_source            | SYNTHETIC DEMO DATA                                     |
| synthetic_demo_data       | True                                                    |
| consumers_total           | 240                                                     |
| consumers_train           | 156                                                     |
| consumers_validation      | 36                                                      |
| consumers_test            | 48                                                      |
| raw_rows                  | 2073600                                                 |
| raw_missing_values        | 4244                                                    |
| raw_duplicate_rows        | 0                                                       |
| window_samples_train      | 8736                                                    |
| window_samples_validation | 2016                                                    |
| window_samples_test       | 2688                                                    |
| feature_count             | 133                                                     |
| theft_ratio_train_windows | 0.08081501831501832                                     |
| theft_ratio_test_windows  | 0.060267857142857144                                    |
| split_unit                | consumer                                                |
| label_rule                | window-level labels respect theft_start_day (synthetic) |

## 3. Preprocessing

- Cleaning report: `{"duplicates_dropped": 0, "negative_values": 0, "infinite_values": 0, "noise": {"enabled": true, "window": 3, "spike_threshold": 6.0, "min_deviation_frac": 0.5, "values_replaced": 2034}, "missing": {"missing_before": 4244, "strategy": "interpolate", "missing_after": 0, "interpolated_or_filled": 4244}, "global_median_source": "train consumers only", "rows_after_cleaning": 2073600}`
- Missing-value strategy: `interpolate`
- Noise handling: Hampel-style rolling median (enabled=True)
- Outlier handling: iqr bounds fitted on training consumers, action=clip (records are not deleted)
- Scaling: standard, fitted on training consumers only
- Class imbalance: {"normal_samples": 8030, "theft_samples": 706, "total": 8736, "theft_ratio": 0.08081501831501832, "imbalance_ratio": 11.373937677053824}
- Theft anomalies are preserved; robust clipping only limits extreme numerical outliers.

## 4. Client simulation

- Partition strategy: `non_iid`
- Clients: 5
- Heterogeneity: theft-ratio std=0.0581, size CV=0.7273

### Per-client summary

| client_id   |   n_samples |   n_consumers |   n_theft_windows |   theft_ratio |
|:------------|------------:|--------------:|------------------:|--------------:|
| client_00   |        2016 |            36 |               346 |        0.1716 |
| client_01   |        3640 |            65 |               329 |        0.0904 |
| client_02   |         336 |             6 |                18 |        0.0536 |
| client_03   |        2408 |            43 |                 0 |        0      |
| client_04   |         336 |             6 |                13 |        0.0387 |

## 5. Models

- Lightweight MLP: hidden=[64, 32], parameters=10689
- CNN-LSTM baseline: executed
- Centralized baselines: centralized_logistic_regression, centralized_random_forest, centralized_mlp, centralized_lightweight_mlp, cnn_lstm_baseline
- Federated local model is the same lightweight MLP for FedAvg and PowerShield-FL so aggregation/privacy effects are not confounded by architecture.

## 6. Federated configuration

- Rounds: 8 (sweeps use their own configured round counts)
- Local epochs: 2
- Batch size: 128
- Learning rate: 0.001
- Participation rate: 1.0
- Aggregation: FedAvg = sample-count weighted mean; PowerShield-FL = `adaptive`

## 7. Privacy configuration

- Enabled: True
- Mechanism: `dp` (L2 clip + Gaussian noise)
- Clipping norm: 1.0
- Noise multiplier: 0.001
- Secure aggregation: False (simulation of additive masking, **not production cryptography**)
- The reported epsilon is a single-release classical estimate. Composition across rounds is not implemented; no end-to-end formal differential-privacy guarantee is claimed.

## 8. Communication method

- Compression enabled: True
- Method: `quantize`; top-k fraction=0.2; int8 quantization
- Payload sizes are measured from actual serialized bytes (including indices and headers), not estimated.

## 9. Evaluation protocol

- Consumer-level stratified train/validation/test split; scalers fitted on train only.
- Decision threshold tuned on the validation split and frozen for test evaluation.
- Emphasis on precision, recall, F1, ROC-AUC and PR-AUC under class imbalance.

## 10. Actual results

### Table 2 — Centralized model comparison

| method                          |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |   train_time_s |   n_params |
|:--------------------------------|-----------:|------------:|---------:|-------:|----------:|---------:|---------------:|-----------:|
| centralized_logistic_regression |     0.9993 |      0.9938 |   0.9938 | 0.9938 |    0.9986 |   0.9952 |         0.5045 |        134 |
| centralized_random_forest       |     0.9996 |      1      |   0.9938 | 0.9969 |    0.995  |   0.9942 |         5.1712 |      21862 |
| centralized_mlp                 |     0.9829 |      0.7816 |   0.9938 | 0.875  |    0.9996 |   0.9967 |        15.4467 |      25473 |
| centralized_lightweight_mlp     |     0.9933 |      0.9045 |   0.9938 | 0.9471 |    0.9991 |   0.9958 |         7.5176 |      10689 |
| cnn_lstm_baseline               |     0.9769 |      1      |   0.6173 | 0.7634 |    0.9444 |   0.7797 |        36.1622 |      21617 |

### Table 3 — Federated comparison

| method          |   clients |   rounds |     f1 |   precision |   recall |   roc_auc |   train_time_s |   total_bytes |   bytes_per_round |
|:----------------|----------:|---------:|-------:|------------:|---------:|----------:|---------------:|--------------:|------------------:|
| fedavg_baseline |         5 |        8 | 0.9938 |           1 |   0.9877 |    0.9973 |        15.5091 |       3453520 |            431690 |
| powershield_fl  |         5 |        8 | 0.9969 |           1 |   0.9938 |    0.9969 |        15.9652 |        902449 |            112806 |

- Measured communication: FedAvg 3453520 bytes vs PowerShield-FL 902449 bytes (reduction 73.87%).

### Table 4 — Privacy / utility

| privacy_level   |   noise_multiplier |   epsilon_estimate |     f1 |   precision |   recall |   train_time_s |   total_bytes |
|:----------------|-------------------:|-------------------:|-------:|------------:|---------:|---------------:|--------------:|
| off             |              0     |            nan     | 0.9907 |      0.9938 |   0.9877 |         7.3283 |        451241 |
| noise_0         |              0     |            nan     | 0.9814 |      0.9875 |   0.9753 |         7.5765 |        451235 |
| noise_0.001     |              0.001 |           4844.81  | 0.9814 |      0.9875 |   0.9753 |         7.4271 |        451222 |
| noise_0.003     |              0.003 |           1614.94  | 0.9814 |      0.9875 |   0.9753 |         7.4017 |        451243 |
| noise_0.006     |              0.006 |            807.467 | 0.9814 |      0.9875 |   0.9753 |         7.836  |        451242 |

### Communication sweep

| compression_label   | compression   |     f1 |   roc_auc |   total_bytes |   communication_reduction_pct |   train_time_s |
|:--------------------|:--------------|-------:|----------:|--------------:|------------------------------:|---------------:|
| none                | none          | 0.9814 |    0.9971 |       1726760 |                        0      |         7.2649 |
| quantize_8bit       | quantize      | 0.9814 |    0.997  |        451222 |                       73.8689 |         6.5778 |
| topk_20pct          | topk          | 0.9272 |    0.995  |        700440 |                       59.4362 |         7.4897 |
| topk_quantize       | topk_quantize | 0.9236 |    0.9948 |        451227 |                       73.8686 |         7.6511 |

### Table 5 — Scalability

|   clients |   rounds |   train_time_s |   total_bytes |   rss_delta_mb |     f1 |   accuracy |
|----------:|---------:|---------------:|--------------:|---------------:|-------:|-----------:|
|         2 |        4 |         6.558  |        180497 |         5.3164 | 0.9846 |     0.9981 |
|         5 |        4 |         7.6042 |        451222 |         4.1055 | 0.9814 |     0.9978 |
|        10 |        4 |         8.9537 |        902445 |         5.3125 | 0.838  |     0.9829 |

## 11. Comparison and claim support

| Claim | Evidence | Status | Note |
|---|---|---|---|
| PowerShield-FL detection F1 vs FedAvg | FedAvg mean F1=0.9938, PowerShield-FL mean F1=0.9969, difference=+0.0031 | **SUPPORTED** | measured on this synthetic demo dataset; single-seed differences are not statistical evidence |
| Communication reduction vs standard FedAvg | FedAvg total=3453520 bytes, PowerShield-FL total=902449 bytes, measured reduction=73.87% | **SUPPORTED** | bytes are actual serialized payloads measured by the communication layer |
| Privacy/utility trade-off | F1 range across measured privacy settings: 0.9814–0.9907 | **MEASURED** | noise multipliers are small because per-coordinate Gaussian noise is applied to a high-dimensional update; see privacy caveat |
| Scalability across client counts | measured client counts: 2, 5, 10 | **MEASURED** | quick mode uses up to the configured client count; high counts may hit CPU/RAM limits |

The objective is not to force PowerShield-FL to win. The table above states whether the measured F1/communication results support each claim on this dataset. Single-seed results are descriptive, not inferential.

## 12. Statistical analysis

- fedavg_baseline: F1 mean=0.9938, std=N/A, 95% CI=[N/A, N/A]
- powershield_fl: F1 mean=0.9969, std=N/A, 95% CI=[N/A, N/A]
- Paired comparison: `{"n": 1, "note": "STATISTICAL TEST NOT RUN \u2014 fewer than 2 paired seeds"}`

## 13. Figures and tables

- Tables: {
  "table1_dataset_statistics": {
    "csv": "B:\\surya-11\\results\\tables\\table1_dataset_statistics.csv",
    "markdown": "B:\\surya-11\\results\\tables\\table1_dataset_statistics.md"
  },
  "table2_model_comparison": {
    "csv": "B:\\surya-11\\results\\tables\\table2_model_comparison.csv",
    "markdown": "B:\\surya-11\\results\\tables\\table2_model_comparison.md"
  },
  "table3_federated_comparison": {
    "csv": "B:\\surya-11\\results\\tables\\table3_federated_comparison.csv",
    "markdown": "B:\\surya-11\\results\\tables\\table3_federated_comparison.md"
  },
  "table4_privacy_comparison": {
    "csv": "B:\\surya-11\\results\\tables\\table4_privacy_comparison.csv",
    "markdown": "B:\\surya-11\\results\\tables\\table4_privacy_comparison.md"
  },
  "table5_scalability": {
    "csv": "B:\\surya-11\\results\\tables\\table5_scalability.csv",
    "markdown": "B:\\surya-11\\results\\tables\\table5_scalability.md"
  }
}
- Figures: {
  "class_distribution": "B:\\surya-11\\results\\figures\\class_distribution.png",
  "client_distribution": "B:\\surya-11\\results\\figures\\client_distribution.png",
  "f1_comparison": "B:\\surya-11\\results\\figures\\f1_comparison.png",
  "precision_recall_comparison": "B:\\surya-11\\results\\figures\\precision_recall_comparison.png",
  "training_time_comparison": "B:\\surya-11\\results\\figures\\training_time_comparison.png",
  "memory_comparison": "B:\\surya-11\\results\\figures\\memory_comparison.png",
  "training_convergence": "B:\\surya-11\\results\\figures\\training_convergence.png",
  "communication_comparison": "B:\\surya-11\\results\\figures\\communication_comparison.png",
  "confusion_matrix": "B:\\surya-11\\results\\figures\\confusion_matrix.png",
  "roc_curve": "B:\\surya-11\\results\\figures\\roc_curve.png",
  "privacy_utility_tradeoff": "B:\\surya-11\\results\\figures\\privacy_utility_tradeoff.png",
  "communication_accuracy_tradeoff": "B:\\surya-11\\results\\figures\\communication_accuracy_tradeoff.png",
  "scalability": "B:\\surya-11\\results\\figures\\scalability.png"
}

## 14. Limitations

- The default dataset is SYNTHETIC DEMO DATA; it does not represent a real utility's theft prevalence or tariff structure.
- The demo uses simulated clients, not physically distributed organizations with real network links.
- Secure aggregation is a masking simulation, not a cryptographic protocol; it provides no malicious-party guarantees.
- The epsilon estimate covers one release of a client update and does not compose across rounds.
- The lightweight MLP is evaluated against a CNN-LSTM on a small synthetic dataset; the CNN-LSTM may be under-trained relative to its capacity.
- Wall time and memory are machine-specific (CPU-only host); they are not portable benchmarks.
- QUICK mode uses small client/round counts; FULL mode is configurable but was not necessarily executed for this report.
- Client-level labels are inferred from theft activation in synthetic data; real datasets may have noisy labels.

- Full-mode status: **NOT RUN (this report is quick mode)**

## 15. Reproducibility

- Python: 3.14.5
- OS: Windows-11-10.0.26200-SP0
- CPU cores: 10 physical / 12 logical
- RAM: 15.72 GB
- GPU: CPU only
- Package versions: `{"numpy": "2.4.4", "pandas": "3.0.1", "sklearn": "1.8.0", "torch": "2.14.1+cpu", "matplotlib": "3.10.8", "scipy": "1.17.1", "shap": "0.53.0", "streamlit": "1.65.0", "yaml": "6.0.3"}`
- Run directories: ["B:\\surya-11\\results\\raw\\experiment_20261025_014756_354071_fedavg_s42", "B:\\surya-11\\results\\raw\\experiment_20261025_014812_413345_powershield_s42"]
- Re-run: `python main.py --mode quick` (or `--mode full`), seeds are recorded in the configuration.

## 16. Claim discipline

The framework reports **NOT RUN** for absent experiments and **EXPERIMENT FAILED** for failed stages. No metric in this report was entered manually.
