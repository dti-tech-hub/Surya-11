# PowerShield-FL — Viva Voce Preparation (50 Code-Aligned Questions)

This guide answers questions an examiner can verify against this repository. Values quoted here come from the final QUICK run (`experiment_20261025_014623_793809_suite_s42`). The executed data is **SYNTHETIC DEMO DATA**.

---

## 1. Fundamentals

### Q1. What is electricity theft, and why is it hard to detect?
**Answer:** Electricity theft (non-technical loss) is unauthorized consumption caused by meter bypass, tampering, line tapping, or billing fraud. It causes revenue loss, distorts demand forecasting, and creates safety risks. Detection is hard because theft patterns are heterogeneous, labelled incidents are rare, and raw consumption data is private. A classifier therefore has to handle class imbalance and weak supervision, not just fit a clean binary problem.

### Q2. Why use machine learning instead of rule-based thresholds?
**Answer:** A simple rule such as “flag any daily drop greater than 50%” creates many false alarms for normal seasonal or occupancy changes. ML models combine many engineered signals — daily drops, volatility, night/day ratios, low-consumption days, and intraday blocks — and learn a joint decision boundary. The project still reports precision and recall separately, because a high F1 alone does not tell a utility how many inspections are wasted.

### Q3. What is Federated Learning, and why is it relevant to smart meters?
**Answer:** Federated Learning trains a shared model across clients while raw data remains local. In a utility setting, regional retailers or substations can train locally and send model updates to a coordinator. This avoids pooling household-level consumption records, which can reveal occupancy, sleep schedules, and appliance usage. Federated Learning reduces the centralization risk, but it does **not** automatically provide differential privacy; that is why PowerShield-FL adds an explicit update-protection layer.

### Q4. What is the difference between centralized ML, standard FedAvg, and PowerShield-FL in this project?
**Answer:**
- **Centralized:** all training consumers are pooled; the server sees raw data. It is the performance reference, not the privacy-preserving deployment.
- **FedAvg:** raw data stays with clients; only model deltas are sent; aggregation weights are sample counts; no clipping/noise/compression.
- **PowerShield-FL:** same lightweight model, partitions, rounds, and seed as FedAvg, but adds adaptive aggregation, clipping + Gaussian noise, int8 update quantization, and explainability.

### Q5. What is the research question?
**Answer:** Can a lightweight, privacy-preserving, communication-efficient Federated Learning framework detect electricity theft effectively while reducing computational and communication overhead and handling heterogeneous client data? The project answers with measured synthetic experiments and explicitly labels what was not run.

---

## 2. Architecture and Implementation

### Q6. What are the main modules?
**Answer:** `preprocessing/` (loader, cleaning, outliers, scaling, features), `models/` (lightweight MLP, CNN-LSTM, baselines), `federated/` (client, server, simulator, partitioning, aggregation, compression), `privacy/` (clipping/noise, simulated secure aggregation), `detection/` (threshold, verdicts), `explainability/` (SHAP, permutation, local attribution), `evaluation/` (metrics, tables, figures, report), `experiments/` (drivers and suite), `dashboard/`, `scripts/`, and `tests/`.

### Q7. Why was a custom simulator used instead of Flower?
**Answer:** The project needs deterministic, CPU-only, inspectable control over the exact bytes sent, the privacy mechanism, and adaptive weights. Flower adds a distributed runtime and dependency surface that is unnecessary for a single-machine research comparison. The custom simulator exposes every round and every `ClientUpdate`, and the tests can verify the privacy contract directly. A production deployment could replace the transport layer with Flower or gRPC without changing the model or metrics.

### Q8. Describe the LightweightMLP.
**Answer:** It is a 3-layer MLP:

```text
133 inputs → Linear(64) → ReLU → Dropout(0.2)
           → Linear(32) → ReLU → Dropout(0.2)
           → Linear(1)  → logit
```

It has **10,689 trainable parameters** (measured by `model.n_params`), and its checkpoint is under 45 KB. The 133 inputs come from the feature pipeline, not from raw half-hourly values.

### Q9. Why this model instead of the CNN-LSTM?
**Answer:** The research question asks for a lightweight, communication-efficient method. The CNN-LSTM has 21,617 parameters and takes far longer on CPU (36.16 s centralized vs 7.52 s for the lightweight MLP in the final quick run). It is retained as a centralized research baseline. The federated comparison uses the same lightweight model for FedAvg and PowerShield-FL so architecture does not confound the aggregation/privacy result.

### Q10. How is the global model updated?
**Answer:** The server holds the global state. A client computes `Δ = local_weights - global_weights`. After privacy and compression, the server receives deltas and computes a weighted mean. The global state is updated by adding the scaled weighted delta. For full participation the scale is 1; partial participation scales by `selected_clients / total_clients` to avoid shrinking the update.

### Q11. What exactly does a client transmit?
**Answer:** Only a `ClientUpdate`: model delta tensors, `num_samples`, `train_loss`, `val_loss`, aggregate `val_metrics`, `num_positive`, `num_negative`, `update_norm`, and small metadata. Raw windows, timestamps, customer IDs, and consumption sequences are not fields in the contract. `ClientUpdate.validate()` rejects unexpected fields, and `tests/test_privacy.py` enforces the boundary.

### Q12. How is non-IID heterogeneity simulated?
**Answer:** Training consumers are partitioned with a Dirichlet distribution (`dirichlet_alpha=0.5`) and a quantity-skew factor. Quantity skew moves whole consumers between clients; it does not thin or drop windows. The final partition assigns all 8,736 training windows exactly once, with client sizes `[2016, 3640, 336, 2408, 336]`. Per-client counts, theft ratios, size CV, and theft-ratio standard deviation are recorded in `data/partitions/partitions_seed42_n5.npz` and the summary. One client has zero theft windows; this is handled explicitly by the adaptive-quality fallback.

### Q13. What is the role of the server validation split?
**Answer:** The server holds a validation set from consumers that were split off before training. It is used to monitor global F1/AUC each round, select the best checkpoint (`best_checkpoint_metric=f1`), and tune the decision threshold. Test consumers are untouched until the final evaluation.

### Q14. Where are checkpoints and metadata stored?
**Answer:** `models/checkpoints/` for the final named checkpoints, and every run also stores `results/raw/<run_id>/model/checkpoint.pt` plus `checkpoint.meta.json`. The meta file contains feature names, threshold, model configuration, and run context needed by the dashboard. A checkpoint is loaded only if it exists; the dashboard never invents a prediction.

### Q15. What configuration controls the experiment?
**Answer:** `config/config.yaml` holds the mode presets and model/communication/evaluation settings; `config/datasets.yaml`, `clients.yaml`, `privacy.yaml`, `aggregation.yaml`, and `experiments.yaml` hold component settings. Any dotted path can be overridden, e.g. `--override federated.rounds=4` or `--override communication.method=topk_quantize`.

---

## 3. Data and Leakage Prevention

### Q16. Why split by consumer rather than by row or window?
**Answer:** A single consumer's windows are highly correlated. If Monday is in train and Tuesday in test, the model can memorize the consumer and report inflated performance. The pipeline splits **consumer IDs first** into train/validation/test, then builds windows. No consumer appears in more than one split. This is enforced by the `split_unit=consumer` record and tests.

### Q17. What preprocessing is performed?
**Answer:** The pipeline loads raw data, splits consumers, cleans each consumer's series (duplicates, invalid values, Hampel denoising, missing values), fits IQR outlier bounds and a scaler on training consumers only, transforms validation/test with those fitted objects, and builds 14-day windows with a 3-day stride. It produces 133 features: statistical, ratio, trend, temporal, and intraday block features. The exact report is stored in `dataset_statistics.json` and the run's `config.json`.

### Q18. Why is the imputation median fitted only on training consumers?
**Answer:** If the median were computed over validation/test rows, their distribution would leak into preprocessing. The training median is passed explicitly into cleaning for all splits. The same rule applies to IQR bounds and scaler parameters. This is the central leakage-prevention rule in the implementation.

### Q19. Why are outliers clipped rather than removed?
**Answer:** Theft itself can appear as an extreme drop or anomalous zero-consumption period. Deleting outliers could delete the positive class. The pipeline computes training IQR bounds and clips values outside them, preserving the direction and occurrence of an anomaly while limiting numerical instability. The cleaning report records how many values were affected.

### Q20. How is class imbalance handled?
**Answer:** The project measures the positive ratio (8.08% in the quick training windows, 6.03% in test windows) and uses class-weighted binary cross-entropy. It tunes the decision threshold on validation instead of assuming 0.5, and evaluates precision, recall, F1, ROC-AUC, and PR-AUC. Accuracy alone is not treated as sufficient evidence.

### Q21. How is missing data handled?
**Answer:** The cleaning stage interpolates within a consumer's time series where possible and uses the training-fitted median for remaining gaps. The synthetic generator inserts missing values and spikes so the path is exercised; the run record shows the raw missing count (4,244 in the final quick dataset). No future rows are used to fill past values.

### Q22. Why is the dataset labelled synthetic?
**Answer:** The executed results use the built-in generator, not a real utility feed. The source is recorded as `SYNTHETIC DEMO DATA` in `summary.json`, every report, the dashboard, and the data tables. Real SGCC/long-CSV loaders exist, but no real dataset is bundled or downloaded. The project therefore does not claim real-world theft rates.

---

## 4. Privacy

### Q23. How does the privacy layer work?
**Answer:** For each client delta `Δ`:

```text
Δ_clipped = Δ · min(1, C / ||Δ||₂)          # L2 clipping
Δ_protected = Δ_clipped + N(0, (σC)² I)     # per-coordinate Gaussian noise
```

With default `C=1.0` and `σ=0.001`. The optional secure-aggregation simulation then adds pairwise masks. The client-side code applies clipping and noise before the update leaves the client; the server never sees the unperturbed delta.

### Q24. Does this provide formal differential privacy?
**Answer:** The code computes a **single-release** Gaussian-mechanism estimate:

```text
ε ≈ sqrt(2 ln(1.25/δ)) / σ
```

and labels it an estimate. It does **not** implement RDP composition across rounds, privacy amplification by client sampling, or a formal end-to-end guarantee. With `σ=0.001`, the estimate is large (about 4,845 at δ=1e-5); the project explicitly says this is a privacy demonstration, not a deployment-grade DP claim.

### Q25. Why is clipping necessary before adding noise?
**Answer:** Gaussian-mechanism analysis needs bounded sensitivity. Without clipping, one client's update could have an arbitrarily large norm and would require unbounded noise to hide it. Clipping bounds the contribution to `C`; the noise scale is then tied to `C`. In this implementation the clipping norm is 1.0 because measured update norms are around 0.5–1.2.

### Q26. Is secure aggregation real?
**Answer:** No. `privacy/secure_aggregation.py` implements **simulated additive pairwise masking**: paired clients receive opposite masks and the masks cancel in the sum. It demonstrates the interface and overhead but uses a seeded PRNG rather than Diffie-Hellman key agreement, has no dropout-recovery protocol, and provides no malicious-server guarantee. It is disabled by default. The report/dashboard carry a warning.

### Q27. What did the privacy sweep measure?
**Answer:** Four rounds with noise multipliers `[0, 0.001, 0.003, 0.006]`. The measured test F1 was 0.9814 for all four DP settings and 0.9907 for the privacy-off run; validation F1 was 0.9885 for the DP settings and 0.9914 with privacy off. With one seed and a short 4-round run, this is **not** a monotonic privacy–utility trade-off, and the project does not claim one. The correct statement is: no clear degradation was observed in this synthetic setup.

---

## 5. Communication and Aggregation

### Q28. What compression methods are implemented?
**Answer:** `none` (float32), `topk` (top 20% of each tensor by magnitude + int32 indices), `quantize` (symmetric int8 with a per-tensor scale), and `topk_quantize` (top-k followed by int8). The final default is **dense int8 `quantize`**, selected from measured validation utility; top-k remains fully implemented and is measured in the sweep.

### Q29. How does int8 quantization work?
**Answer:** For each tensor, the maximum absolute value gives `scale = max(|v|) / 127`. Values are mapped to `q = clip(round(v / scale), -127, 127)` and stored as `int8`; the scale is stored in the JSON header. Decompression reconstructs `v ≈ q · scale`. The header and scale overhead are included in the byte measurement.

### Q30. Why did top-k lose accuracy here?
**Answer:** Top-k keeps only the largest 20% of coordinates per tensor. In this non-IID, short-round setup it discarded many small but jointly useful updates; test F1 fell from 0.9814 (no compression / quantize) to 0.9272 (top-k) at four rounds. The project reports the result rather than hiding it. The default therefore uses quantization, which preserved utility while still cutting bytes.

### Q31. How is communication measured?
**Answer:** `CommunicationMeter` serializes the actual header and arrays for every upload and download, then measures `len(serialized_buffer)`. It does not estimate `parameter_count × 4`. The header includes method, tensor shapes, scales, sparsity flags, and metadata; top-k includes int32 index arrays. The reported total includes both directions. This is why top-k + int8 is not dramatically smaller than plain int8: int32 indices offset the sparsity gain.

### Q32. What was the measured communication reduction?
**Answer:** FedAvg: **3,453,520 bytes** over 8 rounds × 5 clients × 2 directions. PowerShield-FL: **902,449 bytes**, a **73.87%** reduction. The number comes from `results/metrics/summary.json` and the serialized payloads, not a theoretical estimate.

### Q33. How does adaptive aggregation differ from FedAvg?
**Answer:** FedAvg uses `n_i / Σn`. PowerShield-FL uses:

```text
w_i ∝ (n_i / Σn)^α
      × quality_i^β
      × reliability_i^reliability_power
      × balance_i
```

where:

```text
quality_i = 0.5·val_F1_i + 0.5·val_AUC_i   if both validation classes exist
          = 0.5                            for a single-class local validation split
reliability_i = exp(-| ||Δ_i|| / median(||Δ||) - 1 |)
balance_i = (1-λ) + λ·2·sqrt(p_i(1-p_i)),  p_i = local theft ratio
```

Defaults: `α=1`, `β=0.5`, reliability power `0.5`, `λ=0.2`, quality floor `0.1`, minimum weight `0.01`. Weights are normalized and optionally clipped at 3× the mean.

### Q34. Why is the zero-positive client fallback important?
**Answer:** One quick-run client has no theft windows. Its local F1 is 0 by the `zero_division=0` convention, but that does not mean its normal consumption data is worthless. Before the fix, its weight collapsed to the 0.01 floor. The fixed logic uses neutral quality 0.5 when the local validation split has only one class, and a dedicated test (`test_zero_positive_client_is_not_collapsed_to_min_weight`) prevents regression.

### Q35. Was adaptive aggregation actually better than FedAvg?
**Answer:** In the final quick run, PowerShield-FL reached test F1 **0.9969** versus FedAvg **0.9938**, with precision 1.0000 for both and recall 0.9938 vs 0.9877. This is one seed and synthetic data, so it is a descriptive result, not a statistical claim. The report says exactly that. The comparison is also not purely “adaptive vs FedAvg”: the proposed row also includes privacy and quantization, while FedAvg has neither.

---

## 6. Evaluation and Results

### Q36. What evaluation protocol is used?
**Answer:** Consumer-level 65/15/20 train/validation/test split; train-fitted preprocessing; validation-only threshold tuning (candidate grid 0.05–0.95 in 0.05 steps); frozen threshold on test; metrics computed once on the held-out test windows. The test set is never used to tune the model, threshold, or compression method.

### Q37. Which metrics are reported?
**Answer:** Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, per-consumer verdicts, actual serialized bytes, bytes per round, training time, inference latency, memory, parameter count, and model size. The final quick results are:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Centralized LR | 0.9938 | 0.9938 | 0.9938 | 0.9986 |
| Centralized RF | 1.0000 | 0.9938 | 0.9969 | 0.9950 |
| Centralized MLP | 0.7816 | 0.9938 | 0.8750 | 0.9996 |
| Centralized lightweight | 0.9045 | 0.9938 | 0.9471 | 0.9991 |
| CNN-LSTM | 1.0000 | 0.6173 | 0.7634 | 0.9444 |
| FedAvg | 1.0000 | 0.9877 | 0.9938 | 0.9973 |
| PowerShield-FL | 1.0000 | 0.9938 | 0.9969 | 0.9969 |

### Q38. Why is centralized Random Forest competitive with the federated model?
**Answer:** Random Forest reaches F1 0.9969 on pooled data, matching PowerShield-FL's 0.9969 in this single synthetic run; FedAvg reaches 0.9938. The settings are intentionally not apples-to-apples: the centralized model sees pooled data, uses a different architecture, has no client partition, no privacy noise, and sends no updates. It is a performance reference, not the privacy-preserving deployment. The fair FL comparison is FedAvg vs PowerShield-FL, which use the same lightweight model, partitions, rounds, and seed.

### Q39. What does the privacy/utility sweep show?
**Answer:** See Q27. The honest conclusion is “no clear monotonic degradation in this single-seed four-round synthetic run.” The project does not claim that DP is free or that a particular epsilon is achieved. The epsilon is a single-release estimate only.

### Q40. What does the scalability sweep show?
**Answer:** Four rounds, quick mode: 2 clients F1 0.9846, 5 clients 0.9814, 10 clients 0.8380. The 5-client row is not the same as the 8-round main run; comparing them directly would be misleading. The full mode is configured for 2/5/10/20 clients but **NOT RUN**. Time and bytes are machine-specific.

### Q41. Is the result statistically significant?
**Answer:** No claim of significance is made. The executed quick suite has one seed (`[42]`). The code supports multiple seeds, mean/std, bootstrap confidence intervals, paired t-test, and Wilcoxon, and the report emits `STATISTICAL TEST NOT RUN — fewer than 2 paired seeds`. The correct language is “in this single synthetic run, the proposed F1 was higher,” not “the method is significantly better.”

### Q42. How is explainability implemented?
**Answer:** SHAP `KernelExplainer` for the torch model, with TreeExplainer/LinearExplainer for RF/LR. DeepExplainer is deliberately not used because it was brittle with the installed SHAP/PyTorch versions. Permutation importance is a manual feature-shuffle ROC-AUC drop on validation rows, repeated with a seeded RNG. Local explanations use gradient×input for the lightweight model and are exposed as top positive/negative factors with human-readable names.

### Q43. How does the dashboard avoid fabricating results?
**Answer:** It reads `results/metrics/summary.json` and per-run artifacts only. If the summary is absent, it stops with “No experimental result available. Run the experiment first.” If a checkpoint or test sample file is missing, detection shows an error. Missing sweeps show `NOT RUN`. The app also displays a permanent synthetic-data warning when `dataset.is_synthetic` is true.

### Q44. What tests exist?
**Answer:** 33 tests cover config loading, seeding, preprocessing leakage, outliers, partitioning, model save/load, aggregation, adaptive weighting (including zero-positive clients), privacy contract, clipping/noise, secure aggregation simulation, communication round-trip and byte measurement, detection thresholds, explainability, reporting, and an end-to-end synthetic pipeline. Final command: `python -m pytest tests -q` → **33 passed**.

### Q45. What are the main limitations?
**Answer:**
1. Executed results use synthetic data; no real SGCC/IRISE run is claimed.
2. Simulated clients, not a real distributed deployment.
3. Secure aggregation is additive-mask simulation, not cryptography.
4. Epsilon is single-release, without composition.
5. One seed in the executed quick result; no significance claim.
6. Full mode (600 consumers, 10 clients, 25 rounds, 3 seeds) is configured but **NOT RUN**.
7. Top-k lost utility in this setup; int8 quantization was selected from validation utility.
8. Wall time/memory are machine-specific.
9. The CNN-LSTM is a research baseline and may be under-trained.

### Q46. What are the next steps?
**Answer:** Run the full 3-seed suite on real AMI data; implement formal RDP composition and production secure aggregation; use error-feedback top-k or index-efficient sparsification to recover utility; evaluate personalized FL for cross-utility heterogeneity; and integrate the model behind an authenticated utility alert workflow with human review.

---

## 7. Reproducibility and Claim Discipline

### Q47. How is reproducibility enforced?
**Answer:** Every stochastic step receives a seed through `utilities.seed.set_seed` / `make_rng`; partitions, client sampling, noise, dropout, and permutation importance are seeded. Every run records config, seed, timestamp, OS, CPU, RAM, library versions, and environment in `results/raw/experiment_metadata.json`; each run has `config.json`, `metrics.json`, `metrics.csv`, and `logs.txt`. Re-run with `python main.py --mode quick --seed 42`.

### Q48. What is the difference between QUICK and FULL mode?
**Answer:**
- **QUICK:** 240 synthetic consumers × 180 days, 5 clients, 8 rounds, 2 local epochs, one seed; includes privacy/communication/scalability sweeps at 4 rounds. Runs on CPU in a few minutes.
- **FULL:** 600 synthetic consumers × 365 days, 10 clients, 25 rounds, seeds `[42, 123, 2026]`, scalability to 20 clients. It is **configured but NOT RUN** on this host.

### Q49. What does “never fabricate results” mean in practice?
**Answer:** Every reported number is read from a real artifact under `results/`. If an experiment was not run, the report says `NOT RUN`; if a dataset is absent, it says `DATASET NOT CONFIGURED`; if a stage raises, it records `EXPERIMENT FAILED` with the exception. The report generator accepts only `summary.json`; it does not insert placeholder metrics. The dashboard refuses missing checkpoints instead of showing invented predictions.

### Q50. Why are demo metrics not used in the final comparison?
**Answer:** The demo is only 3 rounds with 180 consumers and 4 clients; the final quick suite uses 8 rounds and 240 consumers. The demo is a smoke test of the code paths, not a tuned experiment, and its output overwrites `results/metrics/summary.json`. The repository therefore contains the final quick run, not demo metrics. Quoting a three-round smoke test against the eight-round suite would be misleading; re-run quick mode after the demo before using the comparison tables.

---

## One-Sentence Defence Summary

PowerShield-FL keeps raw smart-meter windows local, transmits only clipped/noisy int8-quantized model deltas, aggregates them with size/quality/reliability/balance weights, and evaluates the result on held-out consumers; on the executed **synthetic, single-seed quick run** it reached F1 **0.9969** versus FedAvg **0.9938** while reducing measured communication by **73.87%**, with every limitation — no real data, no formal end-to-end DP, no significance test, and full mode not run — stated explicitly.
