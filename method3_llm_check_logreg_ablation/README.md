# Method 3 — LLM-Check Logistic-Regularization Ablation

This folder is a dedicated continuation of `method3_llm_check/` focused only on
**logistic-regression variants** for the existing LLM-Check feature families.

It keeps the Method 3 feature formulas unchanged while enforcing the study
constraints from `llm_check_ablation.md`:

- base model: `Qwen/Qwen2.5-0.5B`
- full tokenized `prompt + response`
- truncation disabled
- response-only scoring region
- float32 cache values and float32 model forward pass
- evaluation on `data/dataset.csv` only
- 5-fold CV via the repository-wide `splitting.py`

## Files

- `aggregation.py`: loads or builds the shared float32 Method 3 cache and
  assembles the requested feature set.
- `probe.py`: defines the logistic probe families and the regularization grid.
- `splitting.py`: delegates to the root `split_data()` implementation.
- `run_ablation.py`: runs the logistic-only ablation and writes the leaderboard
  plus the selected best config.
- `solution.py`: runs the standard evaluation loop with the selected final
  logistic config and saves `results.json`.
- `LLMCheck_LogReg_Ablation_Colab.ipynb`: Drive-backed Colab runner for the
  ablation.

## Feature Sets

Available feature sets:

- `logit`
- `hidden`
- `attns`
- `logit_hidden`
- `logit_attns`
- `hidden_attns`
- `logit_hidden_attns`

Default primary study:

- `logit_hidden`

Optional secondary feature sets:

- `logit`
- `hidden`
- `attns`
- `logit_hidden_attns`

## Logistic Families Compared

The new ablation excludes MLPs and compares only:

- `L2` logistic regression with `C in {0.01, 0.1, 1.0, 10.0}` and solvers
  `liblinear`, `lbfgs`
- `L1` logistic regression with `C in {0.01, 0.1, 1.0, 10.0}` and solvers
  `liblinear`, `saga`
- `Elastic-Net` logistic regression with `C in {0.01, 0.1, 1.0}`,
  `l1_ratio in {0.25, 0.5, 0.75}`, solver `saga`
- class weighting modes: `balanced`, `none`

All probes:

- standardize inputs with `StandardScaler`
- keep validation-based threshold tuning
- log penalty, solver, `C`, `l1_ratio`, `class_weight`, feature set, and
  feature dimension

## Why MLP Is Excluded

The prior Method 3 ablation already compared logistic regression against MLP
variants. This continuation isolates the classifier-family question:

- whether stronger or different logistic regularization reduces overfitting
- whether the chosen logistic config is more stable than the historical Method 3
  MLP rows

Removing the MLP branch keeps the study aligned with the current task instead
of widening the search space again.

## How To Run

Smoke test:

```bash
python method3_llm_check_logreg_ablation/run_ablation.py --subset 40 --batch-size 1
```

Primary logistic-only ablation on `logit_hidden`:

```bash
python method3_llm_check_logreg_ablation/run_ablation.py --batch-size 1
```

Primary study plus optional secondary replay on additional feature sets:

```bash
python method3_llm_check_logreg_ablation/run_ablation.py \
  --batch-size 1 \
  --secondary-feature-sets logit,hidden,attns,logit_hidden_attns \
  --secondary-top-k 3
```

Final evaluation using the selected best config:

```bash
python method3_llm_check_logreg_ablation/solution.py --batch-size 1
```

If `best_config.json` is missing, `solution.py` rebuilds it before running the
final evaluation.

## Ranking And Overfitting Diagnostics

The ablation ranks rows by:

1. `mean_val_accuracy`
2. `mean_val_auroc`
3. stronger regularization
4. simpler penalty / solver combination

When multiple configs are near-tied on validation accuracy, the final pick also
prefers:

1. smaller train/validation gap
2. smaller train/validation AUROC gap
3. stronger regularization
4. simpler penalty / solver

Each row logs:

- `mean_train_accuracy`
- `mean_val_accuracy`
- `mean_test_accuracy`
- `mean_train_auroc`
- `mean_val_auroc`
- `mean_test_auroc`
- `train_val_accuracy_gap`
- `train_test_accuracy_gap`
- `train_val_auroc_gap`
- `train_test_auroc_gap`

## Outputs

Artifacts are written to:

- `method3_llm_check_logreg_ablation/artifacts/cache/method3_llm_check_logreg_cache_float32.npz`
- `method3_llm_check_logreg_ablation/artifacts/ablation/ablation_results.csv`
- `method3_llm_check_logreg_ablation/artifacts/ablation/ablation_results.json`
- `method3_llm_check_logreg_ablation/artifacts/ablation/best_config.json`
- `method3_llm_check_logreg_ablation/artifacts/ablation/best_results.json`
- `method3_llm_check_logreg_ablation/artifacts/results.json`
- `method3_llm_check_logreg_ablation/artifacts/results_metadata.json`

## Colab Notebook

Use:

- `method3_llm_check_logreg_ablation/LLMCheck_LogReg_Ablation_Colab.ipynb`

That notebook:

- clones or pulls the GitHub repo in a Drive-backed Colab workspace
- installs dependencies with `pip install -r requirements.txt`
- runs the new logistic-regression ablation
- displays the leaderboard
- prints the chosen best config
- leaves the combined `ablation_results.json` in the repo workspace
