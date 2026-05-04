# Experiment Plan

This file is the runbook for the next phase: use `dataset.csv` only to pick
methods and hyperparameters, then use `test.csv` only once for the final chosen
binary classifier.

## Why Colab T4 First

The expensive step is **hidden-state extraction**, not probe training.

From the current local run:

- `results.json` reports `extract_time_s = 9331.43`
- that is about **2.6 hours** just to extract train-set features on CPU

Once the hidden-state cache is written, the later probe sweeps are cheap and
can run on CPU. So the right workflow is:

1. Use Colab T4 to build the cache from `dataset.csv`
2. Run all method sweeps from the cache
3. Choose the best method
4. Only then run the final prediction pass on `test.csv`

## New Scripts

### `method0_diagnostics.py`

Builds a reusable cache and produces:

- `artifacts/cache/dataset_hidden_cache.npz`
- `artifacts/method0/pca_summary.csv`
- `artifacts/method0/silhouette_heatmap.png`
- `artifacts/method0/silhouette_trends.png`
- `artifacts/method0/top_pca_scatter.png`
- `artifacts/method0/top_pairs.csv`
- `artifacts/method0/layer_rankings.csv`

### `experiment_utils.py`

Provides shared extraction utilities for later methods:

- response-preserving token truncation
- reusable cache extraction
- cached representations for:
  - `last_token`
  - `response_last`
  - `response_mean`
  - `icr_norms`
  - `icr_cosines`
  - `spectrum`
  - `spectrum_logdet`

## Colab Commands

```bash
git clone <your-repo-url>
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python method0_diagnostics.py --batch-size 2 --cache-dtype float16 --max-length 512
```

If the runtime is stable and you want to push throughput:

```bash
python method0_diagnostics.py --batch-size 4 --cache-dtype float16 --max-length 512
```

Quick smoke test:

```bash
python method0_diagnostics.py --subset 80 --batch-size 2 --cache-dtype float16 --overwrite-cache
```

## Method Execution Order

### Stage 1 — Method 0 diagnostics

Goal:

- find which layers and token summaries separate truthful vs hallucinated
- decide the shortlist of layers for Methods 1, 2, and 4

Read these files first:

- `artifacts/method0/silhouette_heatmap.png`
- `artifacts/method0/silhouette_trends.png`
- `artifacts/method0/top_pairs.csv`

Decision rule:

- keep the top `3-5` layers by `max_silhouette`
- note which token mode wins:
  - `last_token`
  - `response_last`
  - `response_mean`

### Stage 2 — Method 1: SAPLMA baseline

Features:

- cached `response_last` or `last_token`
- single best layer first
- then top-3-layer concatenation

Probe:

- small MLP only
- start with:
  - `896 -> 128 -> 1`
  - dropout `0.3`
  - Adam
  - BCEWithLogitsLoss

Sweep:

- token mode: `response_last`, `last_token`, `response_mean`
- layers: best 1 layer, best 3 layers
- hidden size: `64`, `128`
- dropout: `0.2`, `0.3`, `0.4`

Success criterion:

- beat the current Method 0 probe on validation AUROC

### Stage 3 — Method 2: SEP

Features:

- same cached layer/token views as Method 1
- optional PCA compression before logistic regression

Probe:

- `LogisticRegression(class_weight="balanced")`

Sweep:

- token mode: `response_last`, `response_mean`
- layer choice: best 1 layer, mean of best 3 layers
- PCA: none, `64`, `128`
- `C`: `0.1`, `1.0`, `10.0`

Priority:

- this is the most likely strong method for OOD generalization

### Stage 4 — Method 3: ICR Probe

Features:

- `icr_norms`
- `icr_norms + icr_cosines`

Probe:

- logistic regression first
- then very small MLP: `24/47 -> 32 -> 1`

Sweep:

- norms only vs norms+cosines
- logistic regression vs tiny MLP

Priority:

- very cheap
- should be run early once the cache exists

### Stage 5 — Method 4: LLM-Check

Features:

- `spectrum[layer]`
- `spectrum[layer] + spectrum_logdet[layer]`
- optionally concatenate top 3 layers from Method 0

Probe:

- logistic regression

Sweep:

- layer: best 1, best 3
- `top_k`: `8`, `16`, `32`
- spectrum only vs spectrum + logdet

Priority:

- useful complementary signal
- probably weaker alone than SEP, but worth testing for ensembling

## Recommended Comparison Table

Track every run in one CSV with these columns:

- `method`
- `token_mode`
- `layers`
- `feature_dim`
- `model`
- `val_accuracy`
- `val_f1`
- `val_auroc`
- `test_accuracy`
- `test_f1`
- `test_auroc`
- `notes`

## Decision Logic After Colab

1. If SEP clearly wins, use SEP as the final core model.
2. If ICR is competitive, try concatenating SEP + ICR features.
3. If LLM-Check adds complementary gain, try SEP + ICR + LLM-Check.
4. Keep SAPLMA only as a baseline unless it clearly beats the linear probes.

## What To Avoid Right Now

- Do not use `test.csv` for method selection.
- Do not spend time on very large MLPs.
- Do not rerun hidden-state extraction separately for each method.
  Use one cache and sweep probes from it.
