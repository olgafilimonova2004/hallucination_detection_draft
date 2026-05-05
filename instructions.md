# Instructions For The Next Codex Session

## Purpose

Continue the SMILES-2026 hallucination-detection project without restarting
from scratch.

The user wants the next session to:

1. Inspect the current repo state first.
2. Inspect the sibling paper repos next:
   - `/root/SMILES_2026/ICR_Probe_paper_repo`
   - `/root/SMILES_2026/LLM_Check_paper_repo`
3. Implement **two more methods** in the main project repo:
   - ICR Probe
   - LLM-Check
4. Create Colab notebooks for the new methods.
5. Compare standalone results for:
   - SAPLMA
   - ICR Probe
   - LLM-Check
6. Build **4 fusion experiments**:
   - SAPLMA + ICR
   - SAPLMA + LLM-Check
   - ICR + LLM-Check
   - SAPLMA + ICR + LLM-Check
7. Keep track of exactly which feature vectors are fused.
8. Do **not** use `data/test.csv` yet for model selection.

The user explicitly wants the work to continue in the same practical style:
inspect first, implement in separate method folders, create Colab notebooks,
verify with smoke tests, commit, and push to GitHub so Colab can `git pull`.

## Main Paths

Main git repo:

- `/root/SMILES_2026/SMILES-2026-Hallucination-Detection`

Sibling resources outside the git repo:

- `/root/SMILES_2026/METHODS_DETAILED.md`
- `/root/SMILES_2026/layer_rankings.csv`
- `/root/SMILES_2026/SAPLMA`
- `/root/SMILES_2026/ICR_Probe_paper_repo`
- `/root/SMILES_2026/LLM_Check_paper_repo`

Git remote:

- `origin https://github.com/olgafilimonova2004/hallucination_detection_draft.git`

## Read These Files First

Read in this order before changing code:

1. `README.md` in the main repo.
2. `/root/SMILES_2026/METHODS_DETAILED.md`
3. `EXPERIMENTS.md`
4. `experiment_utils.py`
5. `evaluate.py`
6. `splitting.py`
7. `method1_saplma/README.md`
8. `method1_saplma/run_method1.py`
9. `method1_saplma/run_ablation.py`

Then inspect the paper repos:

### ICR repo

Read these first:

1. `/root/SMILES_2026/ICR_Probe_paper_repo/README.md`
2. `/root/SMILES_2026/ICR_Probe_paper_repo/src/icr_score.py`
3. `/root/SMILES_2026/ICR_Probe_paper_repo/src/utils.py`
4. `/root/SMILES_2026/ICR_Probe_paper_repo/src/icr_probe.py`
5. `/root/SMILES_2026/ICR_Probe_paper_repo/scripts/empirical_study.ipynb` if needed

### LLM-Check repo

Read these first:

1. `/root/SMILES_2026/LLM_Check_paper_repo/README.md`
2. `/root/SMILES_2026/LLM_Check_paper_repo/common_utils.py`
3. `/root/SMILES_2026/LLM_Check_paper_repo/run_detection_combined.py`
4. `/root/SMILES_2026/LLM_Check_paper_repo/Hallucination_Detection_NeurIPS_24.pdf` if the code leaves ambiguity

## Current Project State

Recent important commits before this handoff:

- `29cd06a` Reduce Method 0 cache memory usage
- `58b0465` Add Colab notebook for Method 0 runs
- `9368193` Add Method 1 SAPLMA baseline
- `4af0ed8` Add Method 1 ablation sweep

What already exists:

- `method0_diagnostics.py`
- `SMILES_Method0_Colab.ipynb`
- `experiment_utils.py`
- `method1_saplma/`
- `SOLUTION.md`

Important status notes:

- Method 0 was only a diagnostic.
- The user already ran Method 0 in Colab and reported poor separation.
- The reported best silhouette was `0.0456`.
- PCA clusters were mixed.
- The current best layer from `/root/SMILES_2026/layer_rankings.csv` is **layer 15**.

Important path caveat:

- `layer_rankings.csv` is currently a **sibling file outside the git repo**.
- In Colab, that sibling file may not exist after `git pull`.
- For new method runners, either:
  - add a fallback to `layer 15`, or
  - move/copy the rankings file into the repo if needed.

Current split strategy:

- `splitting.py` uses 5-fold stratified CV with an inner validation split.
- Keep using the same split logic for fair comparison across methods.

Very important evaluation caveat:

- `results.json` is produced from internal splits on `data/dataset.csv`.
- The `test_*` fields in `results.json` are **internal held-out fold metrics**, not the external competition `data/test.csv`.
- Do not confuse those with the competition test set.

## Existing Utilities You Should Reuse

`experiment_utils.py` already provides:

- response-preserving truncation
- Qwen hidden-state caching
- cached token summaries:
  - `last_token`
  - `response_last`
  - `response_mean`
- optional reduced features:
  - `icr_norms`
  - `icr_cosines`
  - `spectrum`
  - `spectrum_logdet`

But there are two critical warnings:

1. `icr_norms` and `icr_cosines` are **simplified features**, not automatically
   the exact ICR paper method.
2. `spectrum` and `spectrum_logdet` are useful geometric summaries, but they are
   **not yet guaranteed to match the exact LLM-Check formulas from the paper
   repo**.

So:

- reuse `experiment_utils.py` where it truly matches the needed computation
- otherwise implement a new reduced-feature extractor for the exact paper score
- do **not** claim “ICR” or “LLM-Check” unless the formulas match the repo/paper
  or you clearly document the approximation

## Method 1 Status

SAPLMA is already implemented in `method1_saplma/`.

Files to inspect:

- `method1_saplma/probe.py`
- `method1_saplma/common.py`
- `method1_saplma/run_method1.py`
- `method1_saplma/run_ablation.py`
- `method1_saplma/SAPLMA_Method1_Colab.ipynb`

Method 1 currently supports:

- layer auto-selection from rankings
- fallback to layer 15
- token-mode choices
- ablations over:
  - MLP depth
  - dropout `p=0.3`
  - combined `L1 + L2`

Keep Method 1 as the baseline for comparisons and fusion.

## Environment Notes

The user previously asked to use a root `.venv`.

However, in this workspace, plain `python3` did **not** have the required
packages during local verification. The local runs that worked used:

```bash
PYTHONPATH=/root/methodologist/.venv/lib/python3.12/site-packages python3 ...
```

So in the next session:

1. First check whether the requested `.venv` is actually visible.
2. If it is not, use the working package path above for local smoke tests.
3. In Colab notebooks, continue using `pip install -r requirements.txt`.

## Implementation Rules For The Next Session

1. Do not mix new work into `method0_diagnostics.py` or `method1_saplma/`.
2. Create separate folders for the next methods.
3. Keep experiments on `data/dataset.csv` only.
4. Do not run `data/test.csv` until the user explicitly wants the final binary
   classifier.
5. After meaningful changes, commit and push to GitHub `main`.
6. Create/update Colab notebooks so the user can pull and run them.
7. Smoke-test every new runner locally before pushing if possible.
8. Keep feature extraction memory-efficient.
9. Do not cache full attentions for all samples if a reduced per-sample vector
   can be computed on the fly.

## Recommended New Folder Layout

Create these folders inside the main repo:

- `method2_icr_probe/`
- `method3_llm_check/`
- `method_fusion/`

Recommended contents for each method folder:

- `README.md`
- `common.py` if needed
- `run_method*.py`
- `run_ablation.py` if needed
- `*_Colab.ipynb`

Recommended contents for the fusion folder:

- `README.md`
- `run_fusion.py`
- `Fusion_Colab.ipynb`

## Method 2: ICR Probe

### What to inspect

In the ICR paper repo, the key file is:

- `src/icr_score.py`

The exact repo formula is more involved than simple L2 norms.

From the repo code, the score is built roughly as:

1. preprocess hidden states and attentions
2. optionally detect induction heads
3. pool attention maps across selected heads
4. for each layer and output token:
   - select top-k attended positions
   - compute residual update `hs_diff = h_l(token) - h_{l-1}(token)`
   - compute a projection-based vector `w_i`
   - standardize `w_i` and attention weights
   - softmax both
   - compute JS divergence between them

The repo implementation specifically uses:

- top-k / top-p filtering
- pooled attentions
- JS divergence
- optional induction-head filtering

This is **not** the same as the simplified `icr_norms` in `experiment_utils.py`.

### What to implement

Implement Method 2 in a separate folder.

Strong recommendation:

- build a reduced-feature extractor that computes the **exact ICR-derived
  sample-level vector** directly from hidden states + attentions during
  extraction
- save only the final reduced feature vectors, not the raw full attention cache

Important open point to resolve from the repo/paper:

- how to aggregate token-wise ICR scores into a fixed per-sample vector for our
  dataset

Do not guess silently. Inspect the repo and paper and document the final choice
in `method2_icr_probe/README.md`.

Suggested ablations:

- exact repo-style ICR vector
- exact ICR + a very small MLP
- exact ICR + logistic regression
- optional comparison against simplified `icr_norms` / `icr_cosines`, but label
  that as an approximation, not the paper method

Suggested classifiers:

- logistic regression first
- then tiny MLP if justified

Suggested notebook name:

- `method2_icr_probe/ICR_Method2_Colab.ipynb`

## Method 3: LLM-Check

### What to inspect

In the LLM-Check repo, the key file is:

- `common_utils.py`

The main geometric hidden-state formula in that repo is:

1. take token-wise hidden activations for a selected layer and token span
2. center them with `J = I - (1/n)11^T`
3. compute `Sigma = Z^T J Z + alpha I`
4. compute singular values of `Sigma`
5. score = mean of `log(svdvals(Sigma))`

The helper functions to inspect are:

- `centered_svd_val`
- `get_svd_eval`
- `get_attn_eig_prod`
- `perplexity`
- `logit_entropy`
- `window_logit_entropy`

### User priority for this method

The user explicitly said:

- “LLM-Check (the last one is geometric features)”

So the first target should be the **geometric / eigen-analysis features**.

Recommended first implementation:

- hidden-state geometric score vector across layers on the **response token
  span**

Optional later ablations:

- attention geometric score from `get_attn_eig_prod`
- logit uncertainty scores if useful

Important implementation detail:

- the repo frequently uses token-length slicing to exclude the prompt prefix
- for SMILES, the response-only slice is likely the correct default
- be explicit about the chosen span in the README and metadata

Suggested notebook name:

- `method3_llm_check/LLMCheck_Method3_Colab.ipynb`

## Comparison Phase

After implementing Methods 2 and 3, compare standalone methods on
`data/dataset.csv` only:

1. SAPLMA
2. ICR Probe
3. LLM-Check

For fair comparison:

- use the same split logic
- use validation AUROC for model/config selection
- keep a common comparison table

Recommended comparison file:

- `method_fusion/artifacts/standalone_comparison.csv`

Recommended columns:

- `method`
- `config_name`
- `feature_type`
- `token_mode`
- `layers`
- `feature_dim`
- `mean_val_accuracy`
- `mean_val_f1`
- `mean_val_auroc`
- `mean_test_accuracy`
- `mean_test_f1`
- `mean_test_auroc`
- `notes`

## Fusion Phase

This part is easy to get confused about. Be explicit.

The user wants **feature-vector fusion**, not an ambiguous mixture of internal
states and classifier outputs.

Use this rule:

- For each standalone method, define one **selected feature vector**
  corresponding to the best standalone configuration.

For example:

- SAPLMA selected vector = the exact hidden-state feature block that feeds the
  best Method 1 classifier
- ICR selected vector = the exact per-sample ICR feature vector from Method 2
- LLM-Check selected vector = the exact geometric score vector from Method 3

Do **not** silently switch between:

- raw feature vectors
- probe hidden activations
- final scalar probabilities

If you also try late fusion of scalar scores, label it clearly as a separate
ablation.

### Required fusion experiments

Run these 4:

1. SAPLMA + ICR
2. SAPLMA + LLM-Check
3. ICR + LLM-Check
4. SAPLMA + ICR + LLM-Check

### Strong recommendation for fusion hygiene

Before concatenating feature blocks:

1. save a metadata file for each selected standalone vector
2. include:
   - method name
   - source runner
   - token mode
   - layers
   - feature dimension
   - normalization assumptions
   - exact config name
3. standardize each block or use a global `StandardScaler` in a clean sklearn
   pipeline

Do not concatenate raw blocks without considering scale differences.

Suggested fusion probe:

- logistic regression first

Reason:

- it keeps the fusion comparison simple
- it reduces overfitting risk
- it makes method comparison easier to interpret

## Colab Notebook Pattern

Follow the existing notebook style already used in:

- `SMILES_Method0_Colab.ipynb`
- `method1_saplma/SAPLMA_Method1_Colab.ipynb`

Each new notebook should include:

1. Google Drive mount
2. clone-or-update logic for an existing repo on Drive
3. auto-stash before pull if dirty
4. dependency install
5. runtime/GPU check
6. smoke test cell
7. full run cell
8. artifact inspection cell

For the fusion stage, create one notebook for combined experiments rather than
mixing fusion into the method-specific notebooks.

## Final Integration Reminder

The competition repo still ultimately expects the official pipeline:

- `aggregation.py`
- `probe.py`
- `splitting.py`
- `solution.py`

Right now, Methods 0 and 1 are mostly in separate experimental scripts and
folders. That is fine for the experimentation phase.

But after the best standalone or fusion method is selected, the winning method
still needs to be ported back into the official starter contract so that:

- `solution.py` runs end to end
- `results.json` is generated
- final `predictions.csv` can be generated from `data/test.csv`
- `SOLUTION.md` can be updated with the final method

Do not do this port too early. First finish:

1. Method 2
2. Method 3
3. standalone comparison
4. the 4 fusion experiments

Then choose the winner and integrate.

## Git And Push Workflow

At the beginning of the next session:

```bash
git -C /root/SMILES_2026/SMILES-2026-Hallucination-Detection pull --ff-only origin main
```

After meaningful progress:

1. stage only the intended files
2. commit with a clear message
3. push to `origin main`

The user explicitly wants pushes after changes so Colab can fetch them.

## Suggested Immediate Next Steps

1. Pull the repo and inspect the files listed above.
2. Read the exact ICR formulas from `src/icr_score.py`.
3. Decide how to turn repo-style ICR into a fixed sample-level vector for
   SMILES.
4. Implement `method2_icr_probe/`.
5. Add its Colab notebook.
6. Read the exact LLM-Check hidden geometry formulas from `common_utils.py`.
7. Implement `method3_llm_check/`.
8. Add its Colab notebook.
9. Build a common comparison table.
10. Implement `method_fusion/` and run the 4 required fusion experiments.
11. Push changes after each meaningful milestone.

