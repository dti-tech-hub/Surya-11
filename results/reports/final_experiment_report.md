# PowerShield-FL — Final Experiment Report

> This report is generated automatically from `results/metrics/summary.json`. Every value below comes from an executed run; absent sections are marked **NOT RUN**.

> **DATASET: SYNTHETIC DEMO DATA** — results from this dataset are a software demonstration, not evidence about real utility theft rates.

- Generated: 2026-10-28T07:27:46.856639+00:00
- Suite ID: `experiment_20261028_072746_856067_suite_s42`
- Mode: `full`; demo: `False`
- Seeds: [42, 123, 2026]
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
| centralized_s123 | PASS |
| fedavg_s123 | PASS |
| proposed_s123 | PASS |
| centralized_s2026 | PASS |
| fedavg_s2026 | PASS |
| proposed_s2026 | PASS |
| privacy_sweep | PASS |
| communication_sweep | PASS |
| scalability | PASS |
| tables | PASS |
| figures | PASS |

## 2. Dataset

- Source: **SYNTHETIC DEMO DATA**
- Consumers: train 390, validation 90, test 120
- Windows: train 46020, validation 10620, test 14160
- Features: 133
- Raw missing values: 20802
- Raw duplicate rows: 0
- Split unit: consumer (no consumer appears in two splits)

### Table 1 — Dataset statistics

| statistic                 | value                                                   |
|:--------------------------|:--------------------------------------------------------|
| dataset_source            | SYNTHETIC DEMO DATA                                     |
| synthetic_demo_data       | True                                                    |
| consumers_total           | 600                                                     |
| consumers_train           | 390                                                     |
| consumers_validation      | 90                                                      |
| consumers_test            | 120                                                     |
| raw_rows                  | 10512000                                                |
| raw_missing_values        | 20802                                                   |
| raw_duplicate_rows        | 0                                                       |
| window_samples_train      | 46020                                                   |
| window_samples_validation | 10620                                                   |
| window_samples_test       | 14160                                                   |
| feature_count             | 133                                                     |
| theft_ratio_train_windows | 0.05999565406345067                                     |
| theft_ratio_test_windows  | 0.06024011299435028                                     |
| split_unit                | consumer                                                |
| label_rule                | window-level labels respect theft_start_day (synthetic) |

## 3. Preprocessing

- Cleaning report: `{"duplicates_dropped": 0, "negative_values": 0, "infinite_values": 0, "noise": {"enabled": true, "window": 3, "spike_threshold": 6.0, "min_deviation_frac": 0.5, "values_replaced": 10595}, "missing": {"missing_before": 20802, "strategy": "interpolate", "missing_after": 0, "interpolated_or_filled": 20802}, "global_median_source": "train consumers only", "rows_after_cleaning": 10512000}`
- Missing-value strategy: `interpolate`
- Noise handling: Hampel-style rolling median (enabled=True)
- Outlier handling: iqr bounds fitted on training consumers, action=clip (records are not deleted)
- Scaling: standard, fitted on training consumers only
- Class imbalance: {"normal_samples": 43259, "theft_samples": 2761, "total": 46020, "theft_ratio": 0.05999565406345067, "imbalance_ratio": 15.667873958710612}
- Theft anomalies are preserved; robust clipping only limits extreme numerical outliers.

## 4. Client simulation

- Partition strategy: `non_iid`
- Clients: 10
- Heterogeneity: theft-ratio std=0.0655, size CV=1.6110

### Per-client summary

| client_id   |   n_samples |   n_consumers |   n_theft_windows |   theft_ratio |
|:------------|------------:|--------------:|------------------:|--------------:|
| client_00   |         826 |             7 |                 0 |        0      |
| client_01   |        2714 |            23 |               151 |        0.0556 |
| client_02   |        1652 |            14 |               117 |        0.0708 |
| client_03   |         826 |             7 |               206 |        0.2494 |
| client_04   |       12272 |           104 |               312 |        0.0254 |
| client_05   |         826 |             7 |                69 |        0.0835 |
| client_06   |         826 |             7 |                88 |        0.1065 |
| client_07   |         826 |             7 |                37 |        0.0448 |
| client_08   |       24426 |           207 |              1756 |        0.0719 |
| client_09   |         826 |             7 |                25 |        0.0303 |

## 5. Models

- Lightweight MLP: hidden=[64, 32], parameters=10689
- CNN-LSTM baseline: executed
- Centralized baselines: centralized_lightweight_mlp, centralized_logistic_regression, centralized_mlp, centralized_random_forest, cnn_lstm_baseline
- Federated local model is the same lightweight MLP for FedAvg and PowerShield-FL so aggregation/privacy effects are not confounded by architecture.

## 6. Federated configuration

- Rounds: 25 (sweeps use their own configured round counts)
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

Results are averaged over 3 seed(s): `[42, 123, 2026]`. Per-seed rows are retained where they show run-to-run variability.

### Table 2 — Centralized model comparison (seed-averaged)

| method                          |   accuracy_mean |   precision_mean |   recall_mean |   f1_mean |   f1_std |   roc_auc_mean |   pr_auc_mean |   train_time_s_mean |   n_params_mean |
|:--------------------------------|----------------:|-----------------:|--------------:|----------:|---------:|---------------:|--------------:|--------------------:|----------------:|
| centralized_lightweight_mlp     |          0.9984 |           0.9995 |        0.9688 |    0.9839 |   0.0102 |         0.9949 |        0.9846 |             49.749  |         10689   |
| centralized_logistic_regression |          0.9986 |           0.9991 |        0.9739 |    0.9863 |   0.0057 |         0.995  |        0.9849 |              2.8013 |           134   |
| centralized_mlp                 |          0.9982 |           1      |        0.9646 |    0.9819 |   0.0103 |         0.994  |        0.9833 |             43.1313 |         25473   |
| centralized_random_forest       |          0.9965 |           0.9863 |        0.9468 |    0.966  |   0.0134 |         0.9962 |        0.9857 |             43.815  |         52499.3 |
| cnn_lstm_baseline               |          0.9866 |           0.9529 |        0.77   |    0.8476 |   0.0858 |         0.9696 |        0.8689 |            348.449  |         21617   |

### Table 2b — Centralized results by seed

| method                          |   seed |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |   train_time_s |   n_params |
|:--------------------------------|-------:|-----------:|------------:|---------:|-------:|----------:|---------:|---------------:|-----------:|
| centralized_logistic_regression |     42 |     0.9989 |      0.9988 |   0.9836 | 0.9911 |    0.9951 |   0.9884 |         3.0681 |        134 |
| centralized_random_forest       |     42 |     0.996  |      1      |   0.9332 | 0.9654 |    0.9953 |   0.9855 |        48.3634 |      48762 |
| centralized_mlp                 |     42 |     0.9982 |      1      |   0.9695 | 0.9845 |    0.9928 |   0.9835 |        48.7243 |      25473 |
| centralized_lightweight_mlp     |     42 |     0.9987 |      1      |   0.9777 | 0.9887 |    0.9948 |   0.9865 |        36.2309 |      10689 |
| cnn_lstm_baseline               |     42 |     0.9931 |      0.9666 |   0.9168 | 0.941  |    0.9862 |   0.9548 |       455.423  |      21617 |
| centralized_logistic_regression |    123 |     0.9982 |      1      |   0.9608 | 0.98   |    0.9921 |   0.9764 |         3.459  |        134 |
| centralized_random_forest       |    123 |     0.9956 |      0.9602 |   0.9458 | 0.953  |    0.9946 |   0.9795 |        43.8905 |      61664 |
| centralized_mlp                 |    123 |     0.9973 |      1      |   0.9428 | 0.9705 |    0.9904 |   0.9733 |        40.4821 |      25473 |
| centralized_lightweight_mlp     |    123 |     0.9975 |      0.9984 |   0.9473 | 0.9722 |    0.9917 |   0.975  |        36.1126 |      10689 |
| cnn_lstm_baseline               |    123 |     0.9856 |      0.9323 |   0.747  | 0.8294 |    0.9606 |   0.8346 |       315.273  |      21617 |
| centralized_logistic_regression |   2026 |     0.9988 |      0.9985 |   0.9772 | 0.9877 |    0.9979 |   0.99   |         1.8767 |        134 |
| centralized_random_forest       |   2026 |     0.998  |      0.9985 |   0.9615 | 0.9797 |    0.9987 |   0.9923 |        39.1912 |      47072 |
| centralized_mlp                 |   2026 |     0.9991 |      1      |   0.9815 | 0.9906 |    0.9987 |   0.993  |        40.1876 |      25473 |
| centralized_lightweight_mlp     |   2026 |     0.9991 |      1      |   0.9815 | 0.9906 |    0.9983 |   0.9923 |        76.9034 |      10689 |
| cnn_lstm_baseline               |   2026 |     0.9811 |      0.9597 |   0.6462 | 0.7724 |    0.9621 |   0.8173 |       274.651  |      21617 |

### Table 3 — Federated comparison (seed-averaged)

| method          |   accuracy_mean |   precision_mean |   recall_mean |   f1_mean |   f1_std |   roc_auc_mean |   pr_auc_mean |   train_time_s_mean |   total_bytes_mean |   bytes_per_round_mean |
|:----------------|----------------:|-----------------:|--------------:|----------:|---------:|---------------:|--------------:|--------------------:|-------------------:|-----------------------:|
| fedavg_baseline |          0.9982 |           1      |        0.9653 |    0.9823 |   0.0071 |          0.993 |        0.9816 |             284.237 |        2.15845e+07 |                 863380 |
| powershield_fl  |          0.998  |           0.9989 |        0.9622 |    0.9802 |   0.009  |          0.993 |        0.9816 |             316.62  |        5.64022e+06 |                 225609 |

### Table 3b — Federated results by seed

| method          |   seed |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |   train_time_s |   total_bytes |   bytes_per_round |
|:----------------|-------:|-----------:|------------:|---------:|-------:|----------:|---------:|---------------:|--------------:|------------------:|
| fedavg_baseline |     42 |     0.9981 |      1      |   0.9683 | 0.9839 |    0.9888 |   0.9775 |        329.326 |      21584500 |            863380 |
| fedavg_baseline |    123 |     0.9977 |      1      |   0.9503 | 0.9745 |    0.9924 |   0.9769 |        280.49  |      21584500 |            863380 |
| fedavg_baseline |   2026 |     0.9989 |      1      |   0.9772 | 0.9885 |    0.9979 |   0.9904 |        242.896 |      21584500 |            863380 |
| powershield_fl  |     42 |     0.9975 |      1      |   0.9578 | 0.9784 |    0.9866 |   0.9747 |        306.664 |       5640150 |            225606 |
| powershield_fl  |    123 |     0.9975 |      0.9968 |   0.9488 | 0.9722 |    0.9939 |   0.9771 |        326.217 |       5640337 |            225613 |
| powershield_fl  |   2026 |     0.999  |      1      |   0.98   | 0.9899 |    0.9985 |   0.993  |        316.979 |       5640188 |            225608 |

- Measured communication: FedAvg 21584500 bytes vs PowerShield-FL 5640225 bytes (reduction 73.87%).

### Table 4 — Privacy / utility

| privacy_level   |   noise_multiplier |   epsilon_estimate |     f1 |   precision |   recall |   train_time_s |   total_bytes |
|:----------------|-------------------:|-------------------:|-------:|------------:|---------:|---------------:|--------------:|
| off             |              0     |            nan     | 0.9735 |           1 |   0.9484 |        47.079  |        902359 |
| noise_0         |              0     |            nan     | 0.9692 |           1 |   0.9402 |        33.3511 |        902405 |
| noise_0.001     |              0.001 |           4844.81  | 0.9692 |           1 |   0.9402 |        58.0736 |        902464 |
| noise_0.003     |              0.003 |           1614.94  | 0.9692 |           1 |   0.9402 |        51.2933 |        902483 |
| noise_0.006     |              0.006 |            807.467 | 0.9692 |           1 |   0.9402 |        57.4226 |        902493 |

### Communication sweep

| compression_label   | compression   |     f1 |   roc_auc |   total_bytes |   communication_reduction_pct |   train_time_s |
|:--------------------|:--------------|-------:|----------:|--------------:|------------------------------:|---------------:|
| none                | none          | 0.9679 |    0.9737 |       3453520 |                        0      |        46.4168 |
| quantize_8bit       | quantize      | 0.9692 |    0.9734 |        902464 |                       73.8683 |        32.5914 |
| topk_20pct          | topk          | 0.9522 |    0.9729 |       1400880 |                       59.4362 |        48.5697 |
| topk_quantize       | topk_quantize | 0.9542 |    0.9735 |        902486 |                       73.8676 |        49.2478 |

### Table 5 — Scalability

|   clients |   rounds |   train_time_s |      total_bytes |   rss_delta_mb |     f1 |   accuracy |
|----------:|---------:|---------------:|-----------------:|---------------:|-------:|-----------:|
|         2 |        4 |        36.957  | 180495           |        28.1211 | 0.9754 |     0.9971 |
|         5 |        4 |        32.6173 | 451256           |        22.2656 | 0.9735 |     0.9969 |
|        10 |        4 |        33.6418 | 902464           |        18.918  | 0.9692 |     0.9964 |
|        20 |        4 |        48.0006 |      1.80481e+06 |        26.2461 | 0.9494 |     0.994  |

### Table 6 — Consumer-level test results

| method          |   seed |   precision |   recall |     f1 |   roc_auc |   tp |   fp |   tn |   fn |   n_consumers |
|:----------------|-------:|------------:|---------:|-------:|----------:|-----:|-----:|-----:|-----:|--------------:|
| fedavg_baseline |     42 |           1 |   0.5714 | 0.7273 |         1 |    8 |    0 |  106 |    6 |           120 |
| fedavg_baseline |    123 |           1 |   0.2143 | 0.3529 |         1 |    3 |    0 |  106 |   11 |           120 |
| fedavg_baseline |   2026 |           1 |   0.4286 | 0.6    |         1 |    6 |    0 |  106 |    8 |           120 |
| powershield_fl  |     42 |           1 |   0.5    | 0.6667 |         1 |    7 |    0 |  106 |    7 |           120 |
| powershield_fl  |    123 |           1 |   0.2143 | 0.3529 |         1 |    3 |    0 |  106 |   11 |           120 |
| powershield_fl  |   2026 |           1 |   0.4286 | 0.6    |         1 |    6 |    0 |  106 |    8 |           120 |

## 11. Comparison and claim support

| Claim | Evidence | Status | Note |
|---|---|---|---|
| PowerShield-FL detection F1 vs FedAvg | FedAvg mean F1=0.9823, PowerShield-FL mean F1=0.9802, difference=-0.0021 | **NO ADVANTAGE OBSERVED** | mean over 3 seeds; paired t-test p=0.4038 (significant at alpha=0.05: False); low statistical power with few seeds; treat non-significance as 'inconclusive', not 'no difference' |
| Communication reduction vs standard FedAvg | FedAvg total=21584500 bytes, PowerShield-FL total=5640225 bytes, measured reduction=73.87% | **SUPPORTED** | bytes are actual serialized payloads measured by the communication layer; mean of 3 seeds |
| Privacy/utility trade-off | F1 range across measured privacy settings: 0.9692–0.9735 | **MEASURED** | noise multipliers are small because per-coordinate Gaussian noise is applied to a high-dimensional update; see privacy caveat |
| Scalability across client counts | measured client counts: 2, 5, 10, 20 | **MEASURED** | the sweep uses its own configured round count; high client counts may hit CPU/RAM limits |

The objective is not to force PowerShield-FL to win. The table above states whether the measured F1/communication results support each claim on this dataset. Results are averaged over 3 seeds; the paired test and its caveat are reported below.

## 12. Statistical analysis

- fedavg_baseline: F1 mean=0.9823, std=0.0071, 95% CI=[0.9646, 1.0000], per-seed values=[0.9839189994044074, 0.9745173745173745, 0.9884559884559885]
- powershield_fl: F1 mean=0.9802, std=0.0090, 95% CI=[0.9579, 1.0025], per-seed values=[0.9784431137724551, 0.9722222222222222, 0.9899135446685879]
- Paired FedAvg − PowerShield-FL F1 difference: mean=0.0021, std=0.0035, n=3; paired t p=0.4038, Wilcoxon p=0.5000; significant at alpha=0.05: **False**
- Caveat: low statistical power with few seeds; treat non-significance as 'inconclusive', not 'no difference'

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
  "confusion_matrix": "NOT RUN: ValueError('Found empty input array (e.g., `y_true` or `y_pred`) while a minimum of 1 sample is required.')",
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
- The lightweight MLP is evaluated against a CNN-LSTM on a synthetic dataset; the CNN-LSTM is unstable across seeds and may be under-trained relative to its capacity.
- Wall time and memory are machine-specific (CPU-only host); they are not portable benchmarks.
- This report covers `full` mode with 3 seed(s); other configured modes/seeds remain NOT RUN until executed.
- Client-level labels are inferred from theft activation in synthetic data; real datasets may have noisy labels.
- The consumer-level decision rule is a fixed mean-probability threshold of 0.5; it is not tuned on validation consumers.

- Full-mode status: **EXECUTED**

## 15. Reproducibility

- Python: 3.14.5
- OS: Windows-11-10.0.26200-SP0
- CPU cores: 10 physical / 12 logical
- RAM: 15.72 GB
- GPU: CPU only
- Package versions: `{"numpy": "2.4.4", "pandas": "3.0.1", "sklearn": "1.8.0", "torch": "2.14.1+cpu", "matplotlib": "3.10.8", "scipy": "1.17.1", "shap": "0.53.0", "streamlit": "1.65.0", "yaml": "6.0.3"}`
- Run directories: ["B:\\surya-11\\results\\raw\\experiment_20261028_073942_304208_fedavg_s42", "B:\\surya-11\\results\\raw\\experiment_20261028_074512_651688_powershield_s42", "B:\\surya-11\\results\\raw\\experiment_20261028_075947_320605_fedavg_s123", "B:\\surya-11\\results\\raw\\experiment_20261028_080428_902766_powershield_s123", "B:\\surya-11\\results\\raw\\experiment_20261028_081921_137344_fedavg_s2026", "B:\\surya-11\\results\\raw\\experiment_20261028_082324_936553_powershield_s2026"]
- Re-run: `python main.py --mode quick` (or `--mode full`), seeds are recorded in the configuration.

## 16. Claim discipline

The framework reports **NOT RUN** for absent experiments and **EXPERIMENT FAILED** for failed stages. No metric in this report was entered manually.
