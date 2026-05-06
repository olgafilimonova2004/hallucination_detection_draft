# Method 2 — ICR Probe

This folder contains the **adapted ICR Probe** implementation requested in
`instructions.md`.

## What is implemented

The extractor follows the paper repo score path from
`/root/SMILES_2026/ICR_Probe_paper_repo/src/icr_score.py`, with the exact
adaptations required for `Qwen/Qwen2.5-0.5B`:

- keep **all attention heads**
- mean-pool attention across all heads
- use fixed `top_k = 10`
- disable `top_p`
- keep the repo-style **z-score normalization before softmax**
- compute the JS divergence between:
  - top-k pooled attention weights
  - top-k hidden-state projection scores derived from the residual update

The extractor does **not** save raw attentions. It computes the reduced feature
vector on the fly and stores only the final per-sample ICR vector.

The default probe now matches the paper MLP path from
`/root/SMILES_2026/ICR_Probe_paper_repo/src/utils.py`:

- `L -> 128 -> 64 -> 32 -> 1`
- `BatchNorm1d` after each hidden linear layer
- `LeakyReLU(0.01)`
- `Dropout(p=0.3)`
- `Sigmoid` output

Two regularization variants are exposed through the runner:

- `l2` via `--l2-weight-decay`
- `l1 + l2` via `--l1-lambda` plus `--l2-weight-decay`

## ChatML adaptation

The original repo zeros attention outside three "core" regions:

- user prompt
- response
- everything else removed

For SMILES, prompts are ChatML strings. This implementation maps the repo's
core positions to:

- `user prompt` = tokens inside the `<|im_start|>user ... <|im_end|>` body
- `response` = the preserved tokenized `response` column

System tokens and the assistant wrapper prompt are zeroed before top-k
selection, matching the repo's "core positions only" behavior as closely as
possible in this dataset format.

## Fixed per-sample vector

The paper repo stores token-wise ICR scores with shape:

- `[layer, output_token]`

The empirical notebook in the paper repo converts that into a sample-level
vector by averaging across output tokens:

- `[layer, output_token] -> [layer]` via `mean(axis=-1)`

This implementation uses that same aggregation. Each sample therefore gets one
ICR score per transformer layer, so the feature vector length is `24`.

The response span is the full preserved response token span returned by the
tokenizer. Because the dataset responses already include `<|endoftext|>`, the
terminal EOS token remains in the span when present, which is closer to the
paper repo's generation-time output-token handling than stripping it out.

## Important implementation detail

Qwen does not expose attentions under the default `sdpa` path on this machine.
Method 2 therefore loads the model with:

```python
attn_implementation="eager"
```

Without that change, `output_attentions=True` returns no attention tensors and
the ICR score cannot be computed.

## Commands

Smoke test:

```bash
PYTHONPATH=/root/methodologist/.venv/lib/python3.12/site-packages \
python3 method2_icr_probe/run_method2.py \
  --subset 40 \
  --batch-size 1 \
  --max-length 256 \
  --cache-dtype float32
```

Full MLP run with `l2` regularization:

```bash
python method2_icr_probe/run_method2.py \
  --classifier mlp \
  --hidden-dims 128,64,32 \
  --dropout-p 0.3 \
  --l2-weight-decay 1e-4 \
  --batch-size 1 \
  --cache-dtype float32
```

MLP run with `l1 + l2` regularization:

```bash
python method2_icr_probe/run_method2.py \
  --classifier mlp \
  --hidden-dims 128,64,32 \
  --dropout-p 0.3 \
  --l1-lambda 1e-5 \
  --l2-weight-decay 1e-4 \
  --batch-size 1 \
  --cache-dtype float32 \
  --output-file method2_icr_probe/artifacts/method2_l1_l2.json
```

Optional logistic-regression baseline:

```bash
python method2_icr_probe/run_method2.py \
  --classifier logistic \
  --batch-size 1 \
  --cache-dtype float32 \
  --output-file method2_icr_probe/artifacts/method2_logistic.json
```

## Outputs

Artifacts are written to:

- `method2_icr_probe/artifacts/cache/`
- `method2_icr_probe/artifacts/method2_results.json`
- `method2_icr_probe/artifacts/method2_results_metadata.json`

The cached `.npz` file contains:

- `icr_vector`: shape `(n_samples, 24)`
- `prompt_token_length`
- `response_token_length`
- `top_k`
- `z_normalize`
- `max_length`
- `labels`
