# Results

This directory contains the curated final evidence for the PowerShield-FL quick suite.

**Authoritative suite:** `experiment_20261025_014623_793809_suite_s42`  
**Mode:** quick; **seed:** 42; **dataset:** `SYNTHETIC DEMO DATA`

## Committed artifacts

| Path | Contents |
|---|---|
| `metrics/summary.json` | Machine-readable final summary used by the dashboard/report |
| `metrics/dataset_statistics.json` | Dataset, split, cleaning, and preprocessing statistics |
| `reports/final_experiment_report.md` | Generated final report |
| `tables/table1..table5.{csv,md}` | Dataset, centralized, federated, privacy, scalability tables |
| `figures/*.png` | 15 generated figures |

## Not committed

`results/raw/` contains per-run directories with checkpoints, prediction CSVs, logs, and full metrics. These are generated artifacts and are intentionally ignored because they are large and reproducible with:

```bash
python main.py --mode quick --seed 42
```

No real smart-meter data is included. All executed values are from `SYNTHETIC DEMO DATA`.

## Reproduce

```bash
python main.py --mode quick --seed 42
```

The run writes a new suite under `results/raw/` and overwrites the curated summary/report/tables/figures with freshly measured values.
