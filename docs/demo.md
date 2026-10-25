# PowerShield-FL — Demonstration Guide

## One-Command Demo

```bash
python main.py --demo
```

The demo is a self-contained smoke test for a student laptop. It needs no external dataset, GPU, or network access.

### Demo configuration

| Setting | Value |
|---|---|
| Dataset | `SYNTHETIC DEMO DATA` |
| Consumers | 180 |
| Days | 120 |
| Readings | 48 per day (half-hourly) |
| Federated clients | 4 |
| Federated rounds | 3 |
| Local epochs | 2 |
| Centralized models | Logistic Regression, MLP, centralized LightweightMLP |
| CNN-LSTM | Disabled in demo |
| Sweeps | Disabled in demo |
| Proposed aggregation | Adaptive |
| Privacy | L2 clipping (C=1.0) + Gaussian noise (σ=0.001) |
| Communication | int8 quantization (`quantize`) |

The demo is intentionally short. It proves that the code paths execute; it is **not** the final experiment and its 3-round metrics must not be compared with the 8-round quick suite.

## What Runs

1. **Synthetic generation** — 180 consumers × 120 days, clearly labelled `SYNTHETIC DEMO DATA`.
2. **Leakage-safe preprocessing** — consumer-level train/validation/test split first; cleaning; train-fitted imputation/outlier/scaler objects; 14-day windows and 133 features.
3. **Client simulation** — 4 non-IID clients; raw windows stay local.
4. **Centralized baselines** — Logistic Regression, MLP, LightweightMLP on pooled data.
5. **FedAvg** — same lightweight model, sample-weighted, uncompressed, no privacy.
6. **PowerShield-FL** — adaptive aggregation + clipping/Gaussian noise + int8 quantization.
7. **Detection** — validation-tuned threshold, test predictions, consumer postprocessing.
8. **Explainability** — permutation importance, SHAP KernelExplainer, local gradient×input explanations.
9. **Artifacts** — real metrics, figures, tables, report, and a dashboard-ready checkpoint.

## Sample Console Output

The final local run completed with exit code 0. A representative output shape:

```text
=======================================================
             POWERSHIELD-FL DEMO
=======================================================
Environment              : OK
Dataset                  : SYNTHETIC DEMO DATA
Clients                  : 4
Federated Rounds         : 3

Preprocessing            : OK
Client Simulation        : OK
Centralized Baselines    : OK
Local Training / FedAvg  : OK
Privacy Layer            : OK
Communication Layer      : OK
Adaptive Aggregation     : OK
Global Model             : OK
Detection                : OK
Explainability           : OK

Results generated at:
  B:\surya-11\results

Dashboard:
  streamlit run dashboard/app.py
  http://localhost:8501
=======================================================
             DEMO COMPLETED
=======================================================
```

The demo was smoke-tested locally with exit code 0 after the final partition correction. Its three-round metrics are deliberately **not** reported as final evidence: the demo writes to the same `results/` directory and is overwritten by the quick suite. The repository currently contains the final 8-round quick run, not a demo run. If you execute the demo, re-run `python main.py --mode quick` afterwards before quoting the comparison tables.

The full quick suite (8 rounds, 240 consumers) is the reported comparison. The demo is only a smoke test of the code paths.

## Dashboard

```bash
streamlit run dashboard/app.py
# open http://localhost:8501
```

Tabs:

1. **Overview** — proposed/FedAvg/centralized cards and measured communication reduction.
2. **Dataset** — source badge, dataset statistics, class distribution, cleaning report.
3. **Federated Clients** — per-client sample counts, theft ratios, heterogeneity.
4. **Training** — actual round history (loss, validation F1/accuracy, bytes per round).
5. **Model Comparison** — detection, timing, memory, parameter, and communication columns.
6. **Privacy** — noise/clipping settings, single-release epsilon estimate, sweep rows.
7. **Communication** — compression sweep and actual serialized-byte comparison.
8. **Detection** — choose a held-out test window; view verdict, P(theft), confidence, true label, and model-derived factors; optional CSV feature-row upload.
9. **Explainability** — permutation importance, SHAP summary, local explanation records.
10. **Report** — live Markdown report and generated CSV tables.

### Honest empty states

- No `results/metrics/summary.json` → **“No experimental result available. Run the experiment first.”**
- Missing/corrupt detection checkpoint → an error, never a fabricated prediction.
- Missing sweep → `NOT RUN` with the command needed to generate it.
- Synthetic data → a permanent `SYNTHETIC DEMO DATA` warning.

The dashboard was validated with Streamlit `AppTest` (10 tabs, no exceptions). A live headless server on port `8511` returned HTTP `200` / `ok` from `/_stcore/health`.

## Demo vs Quick vs Full

| | Demo | Quick | Full |
|---|---|---|---|
| Consumers | 180 | 240 | 600 |
| Days | 120 | 180 | 365 |
| Clients | 4 | 5 | 10 |
| Rounds | 3 | 8 | 25 |
| Seeds | 1 | 1 | 3 |
| Centralized models | LR, MLP, lightweight | LR, RF, MLP, lightweight, CNN-LSTM | same |
| Privacy/communication/scalability sweeps | No | Yes | Yes |
| Purpose | smoke test | reported quick result | final thesis run |
| Executed here | Yes | Yes | **NOT RUN** |

## Troubleshooting

```bash
# Re-run with a smaller dataset
python main.py --demo --override data.synthetic.consumers=80 --override data.synthetic.days=60

# Disable the privacy layer
python main.py --demo --override privacy.enabled=false

# Try top-k + int8 instead of the default dense int8
python main.py --demo --override communication.method=topk_quantize

# Validate the whole environment
python scripts/run_all_checks.py
```

If SHAP fails, the run records `PARTIAL`/`EXPERIMENT FAILED` in the explainability status; the dashboard shows that status rather than pretending explanations exist.
