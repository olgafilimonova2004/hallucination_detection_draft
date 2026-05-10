# Method 1 — SAPLMA

This folder contains a reproduction of the **SAPLMA** MLP approach adapted to
the SMILES task setup:

- base model: `Qwen/Qwen2.5-0.5B`
- data source: `data/dataset.csv`
- label space: `0 = truthful`, `1 = hallucinated`
- sequence handling: full tokenized `prompt + response`, with truncation disabled

## What is reproduced from `./SAPLMA`

The local `./SAPLMA` repository uses two core steps:

1. extract one hidden-state vector from a selected model layer
2. feed that vector into an MLP with architecture:

```text
input -> 256 -> 128 -> 64 -> 1
```

with:

- `ReLU` activations
- `Adam`
- `5` training epochs
- `batch_size = 32`

That same architecture is implemented in [probe.py](/root/SMILES_2026/SMILES-2026-Hallucination-Detection/method1_saplma/probe.py:1). The Method 1 adaptation also adds optional:

- `dropout`
- combined `L1 + L2` regularization
- smaller MLP depths for the ablation study

## How layers are chosen

SAPLMA itself does not give a universal best layer for every model. In this
project, layers are chosen from `layer_rankings.csv`, which comes from the
Method 0 diagnostic sweep.

Default behavior:

- `run_method1.py --layers auto --auto-top-k 1`
- read `layer_rankings.csv`
- sort layers by `max_silhouette`, then `mean_silhouette`
- take the top layer

For your current run, that resolves to **layer 15**.

This means the first Method 1 baseline is:

- token mode: `response_last`
- layer set: `[15]`

Why `response_last`:

- SAPLMA is a **single-token hidden-state probe**
- `response_last` is the closest match to the method on your QA-style data,
  because it uses the last non-EOS token from the model's answer

You can override the layer set later, for example:

```bash
python method1_saplma/run_method1.py --layers 12,14,15
```

## Commands

Smoke test:

```bash
python method1_saplma/run_method1.py --subset 80 --layers auto --auto-top-k 1
```

Full run:

```bash
python method1_saplma/run_method1.py --layers auto --auto-top-k 1
```

Logistic-regression baseline:

```bash
python method1_saplma/run_method1.py --classifier logistic --layers auto --auto-top-k 1
```

Regularized variants:

```bash
python method1_saplma/run_method1.py --layers 15 --dropout-p 0.3
python method1_saplma/run_method1.py --layers 15 --l1-lambda 1e-5 --l2-weight-decay 1e-4
python method1_saplma/run_method1.py --layers 15 --hidden-dims 256,128
python method1_saplma/run_method1.py --layers 15 --hidden-dims 256
```

Layer ablation:

```bash
python method1_saplma/run_method1.py --layers 12
python method1_saplma/run_method1.py --layers 14
python method1_saplma/run_method1.py --layers 15
python method1_saplma/run_method1.py --layers 16
```

Token-mode ablation:

```bash
python method1_saplma/run_method1.py --layers 15 --token-mode last_token
python method1_saplma/run_method1.py --layers 15 --token-mode response_last
python method1_saplma/run_method1.py --layers 15 --token-mode response_mean
```

Regularization and depth ablation sweep:

```bash
python method1_saplma/run_ablation.py --layers auto --auto-top-k 1
```

The ablation runner evaluates the requested grid:

- depths: `4`, `3`, `2` linear layers
- dropout: `0.0`, `0.3`
- regularization: `off`, `L1 + L2`

That gives `12` configurations in total. The best one is selected by **mean validation AUROC**, not by test AUROC.

Binary-classifier ablation sweep:

```bash
python method1_saplma/run_binary_ablation.py --layers auto --auto-top-k 1
```

This runner compares exactly three classifier configurations on the same SAPLMA
features:

- logistic regression
- 2-hidden-layer MLP with `dropout=0.3` and `L2`
- 2-hidden-layer MLP with `dropout=0.3` and `L1+L2`

## Outputs

Artifacts are written to:

- `method1_saplma/artifacts/cache/`
- `method1_saplma/artifacts/method1_results.json`
- `method1_saplma/artifacts/method1_results_metadata.json`
- `method1_saplma/artifacts/ablation/ablation_results.csv`
- `method1_saplma/artifacts/ablation/ablation_results.json`
- `method1_saplma/artifacts/ablation/best_config.json`
- `method1_saplma/artifacts/ablation/best_results.json`

## Colab Notebook

Use:

- `method1_saplma/SAPLMA_Binary_Ablation_Colab.ipynb`

That notebook syncs the repo into a Drive-backed Colab workspace, installs the
project dependencies, runs `run_binary_ablation.py`, and displays the
leaderboard plus the selected best configuration.
