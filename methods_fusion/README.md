# Methods Fusion — Late Fusion

This folder implements the requested final fusion experiments across the three
standalone methods:

- SAPLMA + ICR
- SAPLMA + LLM-Check
- ICR + LLM-Check
- SAPLMA + ICR + LLM-Check

The primary metric tracked for ranking is **accuracy**. Model selection uses
internal validation accuracy from `splitting.py`; the external
`data/test.csv` is not used.

## Fusion Design

The implementation is late-fusion, not raw-feature fusion:

```text
raw method feature -> trained standalone MLP branch -> frozen penultimate embedding
frozen embeddings -> concatenate -> final fusion MLP -> hallucination label
```

Each branch MLP is trained only on the current fold training split, then frozen
before extracting embeddings for train/validation/internal-test rows.

## Branch Contributions

The branch feature bookkeeping is:

| Branch | Raw feature vector | Raw dim | Frozen fusion vector | Fusion dim |
|---|---:|---:|---|---:|
| SAPLMA | `response_last` hidden state at layer 15 | 896 | penultimate MLP activation | 64 |
| ICR Probe | layerwise mean ICR vector | 24 | penultimate MLP activation | 32 |
| LLM-Check | attention-score vector over layers 2-24 | 23 | penultimate MLP activation | 32 |

Therefore the four final MLP input dimensions are:

| Experiment | Concatenated frozen vectors | Final input dim |
|---|---|---:|
| SAPLMA + ICR | `64 + 32` | 96 |
| SAPLMA + LLM-Check | `64 + 32` | 96 |
| ICR + LLM-Check | `32 + 32` | 64 |
| SAPLMA + ICR + LLM-Check | `64 + 32 + 32` | 128 |

## Selected Standalone Configurations

SAPLMA:

```text
raw dim 896 -> 256 -> 128 -> 64 -> 1
ReLU, dropout p=0.3, L1=1e-5, L2=1e-4
```

ICR Probe:

```text
raw dim 24 -> 128 -> 64 -> 32 -> 1
BatchNorm1d, LeakyReLU(0.01), dropout p=0.3, L2=1e-4
```

LLM-Check Attention:

```text
raw dim 23 -> 64 -> 32 -> 1
ReLU, dropout p=0.3, L2=1e-4
```

The LLM-Check branch preserves the current Method 3 implementation exactly.

## Run

Smoke-test imports and syntax:

```bash
python3 -m py_compile methods_fusion/common.py methods_fusion/run_fusion.py
```

Full fusion run on `data/dataset.csv`:

```bash
python3 methods_fusion/run_fusion.py
```

Colab-friendly full run:

```bash
python3 methods_fusion/run_fusion.py \
  --batch-size 1 \
  --saplma-batch-size 2 \
  --saplma-cache-dtype float16 \
  --reduced-cache-dtype float32
```

If the method caches already exist, the fusion runner reuses them. If a cache
is missing, it builds the reduced feature cache for that method.

## Output

The runner writes a single JSON file:

```text
methods_fusion/artifacts/fusion_results.json
```

The JSON contains:

- all four experiments
- fold-level metrics
- leaderboard ranked by mean validation accuracy
- internal held-out fold accuracy/F1/AUROC
- exact raw feature dimensions
- exact frozen embedding dimensions
- concat order and fusion feature dimension

