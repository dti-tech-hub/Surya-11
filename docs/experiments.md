# PowerShield-FL — Experiments Guide

## Running Experiments

### Fast demo (always available, synthetic data)

```bash
python main.py --demo
```

4 clients, 3 rounds, 2 local epochs, 120-day synthetic data. It runs preprocessing, centralized baselines, FedAvg, PowerShield-FL, explainability, figures, and the report.

### Quick mode (full suite, fast settings)

```bash
python main.py --mode quick
```

Quick mode: 240 synthetic consumers × 180 days, 5 non-IID clients, 8 rounds, all sweeps. Full mode: 600 consumers × 365 days, 10 clients, 25 rounds, seeds [42, 123, 2026], 20-client scalability.

### Full mode (executed: `experiment_20261028_072746_856067_suite_s42`)

```bash
python main.py --mode full
```

600 consumers × 365 days, 10 clients, 25 rounds, seeds `[42, 123, 2026]`, scalability up to 20 clients. It is deliberately not claimed as executed.

## Individual Experiment Stages

```bash
python main.py --experiment centralized
python main.py --experiment fedavg
python main.py --experiment proposed
python main.py --experiment privacy
python main.py --experiment communication
python main.py --experiment scalability
```

## Script-Level Entry Points

```bash
python scripts/prepare_dataset.py     # load/synthetic sample + statistics
python scripts/create_clients.py      # partition training consumers
python scripts/train_centralized.py   # centralized baselines
python scripts/train_fedavg.py        # FedAvg
python scripts/train_proposed.py      # PowerShield-FL + explanations
python scripts/run_experiments.py     # suite
python scripts/generate_report.py     # regenerate report from summary.json
python scripts/run_all_checks.py      # end-to-end validation
```

## Configuration Overrides

Any dotted config path can be overridden:

```bash
# 10 clients, 5 rounds
python main.py --mode quick --override clients.num_clients=10 --override federated.rounds=5

# Disable privacy
python main.py --demo --override privacy.enabled=false

# Try top-k + int8 instead of the default dense int8
python main.py --demo --override communication.method=topk_quantize

# Reproduce the final quick run
python main.py --mode quick --seed 42
```

## Experiment Outputs

```text
results/
├── raw/
│   ├── experiment_<TIMESTAMP>_<name>_s<seed>/
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── metrics.csv
│   │   ├── logs.txt
│   │   ├── model/checkpoint.pt
│   │   ├── model/checkpoint.meta.json
│   │   ├── test_predictions.csv
│   │   └── explainability/
│   └── experiment_metadata.json
├── metrics/summary.json
├── metrics/dataset_statistics.json
├── figures/*.png                 (15 figures)
├── tables/table1..table5.{csv,md}
└── reports/final_experiment_report.md
```

Every `metrics.json` includes the full round history, measured communication, threshold information, model resources, and client heterogeneity. `summary.json` is the only input to the report generator.

## Experiment Details

### A. Centralized baselines

All training consumers are pooled (the privacy-centralization scenario the project compares against).

- Logistic Regression
- Random Forest (200 trees, max depth 12, class weights)
- sklearn MLP
- LightweightMLP (same architecture as federated)
- CNN-LSTM (1D convolution + LSTM on 14×48 sequences)

### B. Standard FedAvg

- `aggregation.method = fedavg`
- privacy disabled
- compression disabled
- sample-count weights
- full participation, 8 rounds, 2 local epochs

### C. PowerShield-FL

- same LightweightMLP, partitions, seed, rounds, and local epochs
- `aggregation.method = adaptive`
- L2 clipping norm 1.0, Gaussian noise multiplier 0.001
- `communication.method = quantize` (int8)
- threshold tuned on validation, frozen for test
- SHAP/permutation/gradient×input explanations

The final default is dense int8 quantization, not top-k. The communication sweep measured `quantize` at the same validation/test F1 as no compression (4 rounds) while top-k lost utility. Both methods remain implemented and configurable.

The non-IID quantity-skew stage moves **whole consumers** between clients; it does not thin client lists or discard windows. In the final quick partition, all 8,736 training windows are assigned exactly once. Client sizes are `[2016, 3640, 336, 2408, 336]`, and one 43-consumer client has zero theft windows. The coverage property is asserted in `tests/test_partition.py` and in notebook 3.

## Sweeps

### Privacy / utility

Noise multipliers `[0, 0.001, 0.003, 0.006]`; 4 rounds; all other factors fixed. The report includes the single-release epsilon estimate and explicitly warns that composition is not implemented. On the executed synthetic run the F1 was flat at 0.9814 for all nonzero/zero-noise DP settings and 0.9907 with privacy off; no monotonic degradation claim is made.

### Communication / accuracy

Methods `{none, quantize, topk, topk_quantize}`; 4 rounds. Byte counts are actual serialized payload sizes. Measured values:

| Setting | F1 | Validation F1 | Total bytes |
|---|---:|---:|---:|
| none | 0.9814 | 0.9885 | 1,726,760 |
| quantize | 0.9814 | 0.9885 | 451,222 |
| topk 20% | 0.9272 | 0.9767 | 700,440 |
| topk + quantize | 0.9236 | 0.9738 | 451,227 |

### Scalability

Client counts `{2, 5, 10}` in quick mode, 4 rounds. The 5-client row is the same 4-round setting as the sweeps, so its F1 differs from the 8-round main proposed run. Report actual values; do not compare rounds implicitly.

| Clients | F1 | Validation F1 | Time (s) | Bytes |
|---:|---:|---:|---:|---:|
| 2 | 0.9846 | 0.9943 | 6.56 | 180,497 |
| 5 | 0.9814 | 0.9885 | 7.60 | 451,222 |
| 10 | 0.8380 | 0.9290 | 8.95 | 902,445 |

## Interpreting Results

| Question | Evidence |
|---|---|
| Does FL match centralized? | Table 2 vs Table 3 |
| Does compression reduce bytes? | measured `total_bytes` |
| Does privacy hurt F1? | privacy sweep; with only one seed and flat results, say no conclusion |
| Is adaptive better than FedAvg? | Table 3; final quick test F1 0.9969 vs 0.9938, single seed |
| Does scale hurt performance? | Table 5; quick client counts are small and rounds differ from the main run |

### Honest interpretation

- The final quick run is **one seed**; differences are descriptive, not statistically significant.
- The results use synthetic data, not real utility records.
- Full mode and real-data experiments are `NOT RUN` / `DATASET NOT CONFIGURED`.
- Failed stages are recorded as `EXPERIMENT FAILED` with the exception; absent stages are `NOT RUN`.
