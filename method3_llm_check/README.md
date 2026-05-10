# Method 3 — LLM-Check Feature-Family Ablation

This folder now supports a broader **LLM-Check ablation study** based on the
feature families implemented in
`/root/SMILES_2026/LLM_Check_paper_repo/common_utils.py`.

The implementation keeps the SMILES project adaptation:

- base model: `Qwen/Qwen2.5-0.5B`
- full tokenized `prompt + response`, with truncation disabled
- evaluation on `data/dataset.csv` only
- 5-fold CV with the repo's `splitting.py`

## Feature Families Implemented

The paper repo exposes three feature families:

1. `logit`
2. `hidden`
3. `attns`

This adaptation extracts all three in a **single cache build** and then forms
different feature sets by concatenation.

### `logit` features

The logit family is a 3-dimensional vector:

- `perplexity`
- `window_entropy_w1`
- `logit_entropy_top50`

These follow the formulas in the paper repo:

- perplexity over response tokens using the previous-token logits
- maximum single-token entropy window with `w=1`
- entropy over the top-50 logits on response tokens

So:

```text
logit dim = 3
```

### `hidden` features

The hidden family computes one centered SVD score per transformer layer.

For Qwen with 24 transformer layers, this produces:

```text
hidden dim = 24
```

This matches the paper-repo loop:

```python
for layer_num in range(1, len(hidden_acts[0])):
```

which skips the embedding entry in `hidden_states` and keeps transformer layers
`1..24` in one-based numbering.

### `attns` features

The attention family computes one score per selected attention layer:

```text
score_l = sum_h mean(log(diag(A_resp^(l,h))))
```

For Qwen with 24 attention layers, the paper repo uses:

```python
for layer_num in range(1, len(attns[0])):
```

which skips the first actual attention layer. Therefore:

```text
attns dim = 23
```

corresponding to transformer layers `2..24` in one-based numbering.

## Feature Sets Compared

The ablation runner compares these feature sets:

- `logit`
- `hidden`
- `attns`
- `logit_hidden`
- `logit_attns`
- `hidden_attns`
- `logit_hidden_attns`

Their dimensions for Qwen are:

- `logit`: `3`
- `hidden`: `24`
- `attns`: `23`
- `logit_hidden`: `27`
- `logit_attns`: `26`
- `hidden_attns`: `47`
- `logit_hidden_attns`: `50`

## Classifiers Compared

Each feature set is evaluated with:

1. `logistic` regression
2. lightweight `mlp`

The lightweight MLP is:

```text
input -> 64 -> 32 -> 1
```

with:

- `ReLU`
- `Dropout(p=0.3)`
- `Adam`
- `L2` regularization via `weight_decay=1e-4`

## Commands

Single run on one feature set:

```bash
python method3_llm_check/run_method3.py \
  --feature-set attns \
  --classifier logistic \
  --batch-size 1 \
  --cache-dtype float32
```

Single MLP run on a combined feature set:

```bash
python method3_llm_check/run_method3.py \
  --feature-set logit_hidden_attns \
  --classifier mlp \
  --hidden-dims 64,32 \
  --dropout-p 0.3 \
  --l2-weight-decay 1e-4 \
  --batch-size 1 \
  --cache-dtype float32
```

Full ablation study:

```bash
python method3_llm_check/run_ablation.py \
  --feature-sets logit,hidden,attns,logit_hidden,logit_attns,hidden_attns,logit_hidden_attns \
  --batch-size 1 \
  --cache-dtype float32
```

Smoke test:

```bash
PYTHONPATH=/root/methodologist/.venv/lib/python3.12/site-packages \
python3 method3_llm_check/run_ablation.py \
  --subset 40 \
  --batch-size 1 \
  --cache-dtype float32 \
  --output-dir method3_llm_check/artifacts/ablation_smoke
```

## Colab Notebook

Use:

- `method3_llm_check/LLMCheck_Ablation_Colab.ipynb`

That notebook runs the full ablation study and saves one combined JSON file
that can be inspected to identify the best configuration.

## Outputs

Artifacts are written to:

- `method3_llm_check/artifacts/cache/`
- `method3_llm_check/artifacts/method3_results.json`
- `method3_llm_check/artifacts/method3_results_metadata.json`
- `method3_llm_check/artifacts/ablation/ablation_results.csv`
- `method3_llm_check/artifacts/ablation/ablation_results.json`

The shared cache now stores:

- `logit_metrics_vector`: shape `(n_samples, 3)`
- `hidden_score_vector`: shape `(n_samples, 24)`
- `attention_score_vector`: shape `(n_samples, 23)`
- `logit_metric_names`
- hidden-layer index metadata
- attention-layer index metadata
- `prompt_token_length`
- `response_token_length`
- `response_truncated`
- `max_length`
- `labels`
