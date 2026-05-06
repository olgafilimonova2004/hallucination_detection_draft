# Method 3 — LLM-Check Attention Score

This folder contains the **Method 3 LLM-Check adaptation** for the SMILES task,
restricted to the **attention score only**, as requested.

## What is implemented

The implementation follows the attention-score path from
`/root/SMILES_2026/LLM_Check_paper_repo/common_utils.py`, specifically
`get_attn_eig_prod`, but adapted to the SMILES project structure and the Qwen
runtime.

Important clarification:

- despite the helper name `get_attn_eig_prod`, the repo does **not** compute an
  eigendecomposition
- the implemented score is the mean log of the diagonal attention values on the
  selected token span, summed across heads

For a layer `l` and attention head `h`, with response-span attention matrix
`A^{(l,h)}_resp`, the score is:

```text
score_l = sum_h mean(log(diag(A^{(l,h)}_resp)))
```

This implementation builds a **sample-level feature vector across layers** by
computing that score for every selected layer.

## Alignment to the paper repo

The LLM-Check repo uses:

```python
for layer_num in range(1, len(attns[0])):
    ...
```

So it **skips the first attention layer** when collecting per-layer scores.

To match that behavior, Method 3 also skips attention layer index `0`. For
`Qwen/Qwen2.5-0.5B`, which has 24 transformer layers, the resulting feature
vector has length `23`, corresponding to zero-based layer indices `1..23`
(or transformer layers `2..24` in one-based numbering).

## Token span choice

The repo supports a `--use_toklens` option that slices to the response span.
For SMILES, this adaptation uses the **response-only span by default**, because
that is the most direct equivalent of the repo's `tok_lens` setting and is the
best fit for this task.

The response span is produced by the shared response-preserving tokenization
logic already used elsewhere in the repo:

- keep the response tail intact
- crop the start of the prompt first if truncation is needed
- score the attention only on the preserved response tokens

The dataset response column already contains the final generated text, so this
method does not need hidden-state caches or additional generations.

## Numerical detail

The original repo applies `log(diagonal_attention)` directly.

This implementation uses a small clamp before the log:

```python
diag = diag.clamp_min(1e-12)
```

This is only a numerical safeguard against `log(0)` and does not change the
feature definition in any meaningful way.

## Qwen-specific runtime detail

Qwen does not expose attention tensors under the default `sdpa` path on this
machine. Method 3 therefore loads the model with:

```python
attn_implementation="eager"
```

Without that change, `output_attentions=True` does not provide the matrices
needed for the score.

## Classifier ablation

The requested ablation study compares exactly two classifier configurations on
top of the Method 3 feature vector:

1. `logistic_regression`
2. `mlp_dropout0.3_l2`

The MLP configuration uses:

- input dim = `23`
- hidden dims = `64 -> 32`
- `ReLU`
- `Dropout(p=0.3)`
- `Adam` with `L2` regularization via `weight_decay`

The combined ablation output is written to one JSON file:

- `method3_llm_check/artifacts/ablation/ablation_results.json`

That file contains both configurations and their cross-fold summary metrics.

## Commands

Smoke test:

```bash
PYTHONPATH=/root/methodologist/.venv/lib/python3.12/site-packages \
python3 method3_llm_check/run_ablation.py \
  --subset 40 \
  --batch-size 1 \
  --max-length 256 \
  --cache-dtype float32 \
  --output-dir method3_llm_check/artifacts/ablation_smoke
```

Single logistic-regression run:

```bash
python method3_llm_check/run_method3.py \
  --classifier logistic \
  --batch-size 1 \
  --cache-dtype float32
```

Single MLP run:

```bash
python method3_llm_check/run_method3.py \
  --classifier mlp \
  --hidden-dims 64,32 \
  --dropout-p 0.3 \
  --l2-weight-decay 1e-4 \
  --batch-size 1 \
  --cache-dtype float32
```

Requested two-config ablation:

```bash
python method3_llm_check/run_ablation.py \
  --batch-size 1 \
  --cache-dtype float32
```

## Outputs

Artifacts are written to:

- `method3_llm_check/artifacts/cache/`
- `method3_llm_check/artifacts/method3_results.json`
- `method3_llm_check/artifacts/method3_results_metadata.json`
- `method3_llm_check/artifacts/ablation/ablation_results.csv`
- `method3_llm_check/artifacts/ablation/ablation_results.json`

The cached `.npz` file contains:

- `attention_score_vector`: shape `(n_samples, 23)`
- `selected_layer_indices_zero_based`
- `selected_transformer_layers_1based`
- `prompt_token_length`
- `response_token_length`
- `response_truncated`
- `max_length`
- `labels`
