# PowerShield-FL — Final Experiment Report

> This report is generated automatically from `results/metrics/summary.json`. Every value below comes from an executed run; absent sections are marked **NOT RUN**.

> **DATASET: SYNTHETIC DEMO DATA** — results from this dataset are a software demonstration, not evidence about real utility theft rates.

- Generated: 2026-10-28T13:26:46.352018+00:00
- Suite ID: `experiment_20261028_132646_351780_suite_s42`
- Mode: `quick`; demo: `True`
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
| tables | PASS |
| figures | PASS |

## 2. Dataset

- Source: **SYNTHETIC DEMO DATA**
- Consumers: train 117, validation 27, test 36
- Windows: train 4212, validation 972, test 1296
- Features: 133
- Raw missing values: 2021
- Raw duplicate rows: 0
- Split unit: consumer (no consumer appears in two splits)

### Table 1 — Dataset statistics

| statistic                 | value                                                   |
|:--------------------------|:--------------------------------------------------------|
| dataset_source            | SYNTHETIC DEMO DATA                                     |
| synthetic_demo_data       | True                                                    |
| consumers_total           | 180                                                     |
| consumers_train           | 117                                                     |
| consumers_validation      | 27                                                      |
| consumers_test            | 36                                                      |
| raw_rows                  | 1036800                                                 |
| raw_missing_values        | 2021                                                    |
| raw_duplicate_rows        | 0                                                       |
| window_samples_train      | 4212                                                    |
| window_samples_validation | 972                                                     |
| window_samples_test       | 1296                                                    |
| feature_count             | 133                                                     |
| theft_ratio_train_windows | 0.08808167141500475                                     |
| theft_ratio_test_windows  | 0.0941358024691358                                      |
| split_unit                | consumer                                                |
| label_rule                | window-level labels respect theft_start_day (synthetic) |

## 3. Preprocessing

- Cleaning report: `{"duplicates_dropped": 0, "negative_values": 0, "infinite_values": 0, "noise": {"enabled": true, "window": 3, "spike_threshold": 6.0, "min_deviation_frac": 0.5, "values_replaced": 1056}, "missing": {"missing_before": 2021, "strategy": "interpolate", "missing_after": 0, "interpolated_or_filled": 2021}, "global_median_source": "train consumers only", "rows_after_cleaning": 1036800}`
- Missing-value strategy: `interpolate`
- Noise handling: Hampel-style rolling median (enabled=True)
- Outlier handling: iqr bounds fitted on training consumers, action=clip (records are not deleted)
- Scaling: standard, fitted on training consumers only
- Class imbalance: {"normal_samples": 3841, "theft_samples": 371, "total": 4212, "theft_ratio": 0.08808167141500475, "imbalance_ratio": 10.35309973045822}
- Theft anomalies are preserved; robust clipping only limits extreme numerical outliers.

## 4. Client simulation

- Partition strategy: `non_iid`
- Clients: 4
- Heterogeneity: theft-ratio std=0.0926, size CV=0.9079

### Per-client summary

| client_id   |   n_samples |   n_consumers |   n_theft_windows |   theft_ratio |
|:------------|------------:|--------------:|------------------:|--------------:|
| client_00   |        1008 |            28 |               145 |        0.1438 |
| client_01   |         252 |             7 |                11 |        0.0437 |
| client_02   |         324 |             9 |                88 |        0.2716 |
| client_03   |        2628 |            73 |               127 |        0.0483 |

## 5. Models

- Lightweight MLP: hidden=[64, 32], parameters=10689
- CNN-LSTM baseline: NOT RUN
- Centralized baselines: centralized_lightweight_mlp, centralized_logistic_regression, centralized_mlp
- Federated local model is the same lightweight MLP for FedAvg and PowerShield-FL so aggregation/privacy effects are not confounded by architecture.

## 6. Federated configuration

- Rounds: 3 (sweeps use their own configured round counts)
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

Results are averaged over 1 seed(s): `[42]`. Per-seed rows are retained where they show run-to-run variability.

### Table 2 — Centralized model comparison (seed-averaged)

| method                          |   accuracy_mean |   precision_mean |   recall_mean |   f1_mean |   f1_std |   roc_auc_mean |   pr_auc_mean |   train_time_s_mean |   n_params_mean |
|:--------------------------------|----------------:|-----------------:|--------------:|----------:|---------:|---------------:|--------------:|--------------------:|----------------:|
| centralized_lightweight_mlp     |          0.9931 |           0.9913 |        0.9344 |    0.962  |      nan |         0.9922 |        0.9723 |              1.0859 |           10689 |
| centralized_logistic_regression |          0.9869 |           1      |        0.8607 |    0.9251 |      nan |         0.9912 |        0.9665 |              0.1229 |             134 |
| centralized_mlp                 |          0.9969 |           1      |        0.9672 |    0.9833 |      nan |         0.9942 |        0.9819 |              4.4467 |           25473 |

### Table 3 — Federated comparison (seed-averaged)

| method          |   accuracy_mean |   precision_mean |   recall_mean |   f1_mean |   f1_std |   roc_auc_mean |   pr_auc_mean |   train_time_s_mean |   total_bytes_mean |   bytes_per_round_mean |
|:----------------|----------------:|-----------------:|--------------:|----------:|---------:|---------------:|--------------:|--------------------:|-------------------:|-----------------------:|
| fedavg_baseline |          0.9236 |           0.5816 |        0.6721 |    0.6236 |      nan |         0.8859 |        0.7482 |              1.7173 |        1.03606e+06 |               345352   |
| powershield_fl  |          0.9074 |           0.5063 |        0.6557 |    0.5714 |      nan |         0.8378 |        0.7212 |              1.5652 |   270742           |                90247.3 |

- Measured communication: FedAvg 1036056 bytes vs PowerShield-FL 270742 bytes (reduction 73.87%).

### Table 4 — Privacy / utility

NOT RUN

### Communication sweep

NOT RUN

### Table 5 — Scalability

NOT RUN

### Table 6 — Consumer-level test results

| method          |   seed |   precision |   recall |     f1 |   roc_auc |   tp |   fp |   tn |   fn |   n_consumers |
|:----------------|-------:|------------:|---------:|-------:|----------:|-----:|-----:|-----:|-----:|--------------:|
| fedavg_baseline |     42 |         0.6 |      0.5 | 0.5455 |    0.9    |    3 |    2 |   28 |    3 |            36 |
| powershield_fl  |     42 |         0.6 |      0.5 | 0.5455 |    0.8444 |    3 |    2 |   28 |    3 |            36 |

## 11. Comparison and claim support

| Claim | Evidence | Status | Note |
|---|---|---|---|
| PowerShield-FL detection F1 vs FedAvg | FedAvg mean F1=0.6236, PowerShield-FL mean F1=0.5714, difference=-0.0521 | **NO ADVANTAGE OBSERVED** | measured on this synthetic demo dataset; single-seed differences are not statistical evidence |
| Communication reduction vs standard FedAvg | FedAvg total=1036056 bytes, PowerShield-FL total=270742 bytes, measured reduction=73.87% | **SUPPORTED** | bytes are actual serialized payloads measured by the communication layer |
| Privacy/utility trade-off | privacy sweep not executed | **NOT RUN** |  |
| Scalability | scalability sweep not executed | **NOT RUN** |  |

The objective is not to force PowerShield-FL to win. The table above states whether the measured F1/communication results support each claim on this dataset. Single-seed results are descriptive, not inferential.

## 12. Statistical analysis

- fedavg_baseline: F1 mean=0.6236, std=N/A, 95% CI=[N/A, N/A], per-seed values=[]
- powershield_fl: F1 mean=0.5714, std=N/A, 95% CI=[N/A, N/A], per-seed values=[]
- Paired FedAvg − PowerShield-FL F1 difference: mean=NOT RUN, std=NOT RUN, n=1; paired t p=NOT RUN, Wilcoxon p=NOT RUN; significant at alpha=0.05: **None**
- Caveat: 

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
  "roc_curve": "B:\\surya-11\\results\\figures\\roc_curve.png"
}

## 14. Limitations

- The default dataset is SYNTHETIC DEMO DATA; it does not represent a real utility's theft prevalence or tariff structure.
- The demo uses simulated clients, not physically distributed organizations with real network links.
- Secure aggregation is a masking simulation, not a cryptographic protocol; it provides no malicious-party guarantees.
- The epsilon estimate covers one release of a client update and does not compose across rounds.
- The lightweight MLP is evaluated against a CNN-LSTM on a synthetic dataset; the CNN-LSTM is unstable across seeds and may be under-trained relative to its capacity.
- Wall time and memory are machine-specific (CPU-only host); they are not portable benchmarks.
- This report covers `quick` mode with 1 seed(s); other configured modes/seeds remain NOT RUN until executed.
- Client-level labels are inferred from theft activation in synthetic data; real datasets may have noisy labels.
- The consumer-level decision rule is a fixed mean-probability threshold of 0.5; it is not tuned on validation consumers.

- Full-mode status: **NOT RUN (this report is quick mode)**

## 15. Reproducibility

- Python: 3.14.5
- OS: Windows-11-10.0.26200-SP0
- CPU cores: 10 physical / 12 logical
- RAM: 15.72 GB
- GPU: CPU only
- Package versions: `{"numpy": "2.4.4", "pandas": "3.0.1", "sklearn": "1.8.0", "torch": "2.14.1+cpu", "matplotlib": "3.10.8", "scipy": "1.17.1", "shap": "0.53.0", "streamlit": "1.65.0", "yaml": "6.0.3"}`
- Run directories: ["B:\\surya-11\\results\\raw\\experiment_20261028_132658_536888_fedavg_s42", "B:\\surya-11\\results\\raw\\experiment_20261028_132701_053069_powershield_s42"]
- Re-run: `python main.py --mode quick` (or `--mode full`), seeds are recorded in the configuration.

## 16. Claim discipline

The framework reports **NOT RUN** for absent experiments and **EXPERIMENT FAILED** for failed stages. No metric in this report was entered manually.
