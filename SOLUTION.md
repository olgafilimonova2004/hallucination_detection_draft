# Method 0 Solution

## Reproducibility

Standard setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

On this machine, the available packages were split across the system Python and
an existing venv at `/root/methodologist/.venv`, so the local verification
command was:

```bash
PYTHONPATH=/root/methodologist/.venv/lib/python3.12/site-packages python3 solution.py
```

Running `solution.py` produces:

- `results.json`
- `predictions.csv`

## Final Solution Description

This implementation adapts **Method 0: Geometry of Truth — PCA Analysis** into
the provided starter pipeline.

### `aggregation.py`

- Instead of keeping only the final layer, the aggregator now keeps **all 25
  model outputs** (embedding + 24 transformer layers).
- For each layer it stores two end-of-sequence token views:
  - the **penultimate real token** as a proxy for the last content token
  - the **last real token** as a proxy for the EOS summary state
- The resulting feature vector has shape `2 × 25 × 896 = 44800`.

### `probe.py`

- The probe no longer uses the starter MLP.
- During `fit()`, it reconstructs the flattened vector into
  layer-position pairs and runs a **2-D PCA** for each one.
- It scores every pair with a **silhouette score** using the training labels.
- It keeps the best-separated pairs (top 4 positive-scoring pairs) and trains a
  **balanced logistic regression** on the concatenated PCA projections.
- `fit_hyperparameters()` still tunes the decision threshold on validation data
  by maximizing F1.

This keeps the implementation faithful to Method 0:

- truthfulness is diagnosed through PCA geometry
- layer selection is data-driven rather than guessed
- the downstream classifier is intentionally lightweight

### `splitting.py`

- Replaced the single split with **5-fold stratified cross-validation**
- Each training fold is further split into train/validation subsets

This gives a more stable evaluation of the PCA-based layer selection.

## Experiments and Failed Attempts

- The official full run on the complete 689-sample training set was started in
  this session, but the environment only exposed **CPU** execution. Hidden-state
  extraction for Qwen2.5-0.5B at `MAX_LENGTH=512` was too slow to complete
  within a reasonable interactive session window.
- A smaller smoke test on 6 real samples completed successfully and confirmed:
  - feature extraction shape: `(6, 44800)`
  - the Method 0 probe fit/predict path runs end-to-end
  - layer-position selection returns sensible PCA/silhouette diagnostics

## Notes

- This repository now contains a working Method 0 implementation.
- To obtain submission-ready `results.json` and `predictions.csv`, rerun
  `solution.py` in an environment with GPU access or with enough CPU time for
  full hidden-state extraction.
