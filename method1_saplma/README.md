# Method 1 — SAPLMA

This folder contains a reproduction of the **SAPLMA** MLP approach adapted to
the SMILES task setup:

- base model: `Qwen/Qwen2.5-0.5B`
- data source: `data/dataset.csv`
- label space: `0 = truthful`, `1 = hallucinated`

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

That same architecture is implemented in [probe.py](/root/SMILES_2026/SMILES-2026-Hallucination-Detection/method1_saplma/probe.py:1).

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

## Outputs

Artifacts are written to:

- `method1_saplma/artifacts/cache/`
- `method1_saplma/artifacts/method1_results.json`
- `method1_saplma/artifacts/method1_results_metadata.json`
