"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a lightweight classifier that follows
Method 0 from ``METHODS_DETAILED.md``:

  1. Split the flattened feature vector back into per-layer hidden states for
     each tracked token position.
  2. Run a 2-D PCA for every layer-position pair.
  3. Score the pair with a silhouette score using the training labels.
  4. Keep the best-separated pairs and train a small linear classifier on the
     concatenated PCA projections.

The class still exposes the same public API expected by ``evaluate.py``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from aggregation import TOKEN_SUMMARIES

EXPECTED_HIDDEN_DIM = 896
EXPECTED_LAYER_OUTPUTS = 25
PCA_COMPONENTS = 2
MAX_SELECTED_PAIRS = 4
RANDOM_STATE = 42


class HallucinationProbe(nn.Module):
    """Method-0-inspired binary classifier over hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        self._classifier: LogisticRegression | None = None
        self._projection_scaler = StandardScaler()
        self._selected_pairs: list[dict[str, object]] = []
        self._threshold: float = 0.5
        self._core_dim = EXPECTED_HIDDEN_DIM * EXPECTED_LAYER_OUTPUTS * len(TOKEN_SUMMARIES)

        # Exposed for diagnostics / reporting.
        self.layer_diagnostics_: list[dict[str, float | int | str]] = []
        self.selected_layer_pairs_: list[dict[str, float | int | str]] = []

    def _validate_and_prepare(self, X: np.ndarray) -> np.ndarray:
        """Convert *X* to a float 2-D array and validate the expected layout."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected a 2-D feature matrix, got shape {X_arr.shape}.")
        if X_arr.shape[1] < self._core_dim:
            raise ValueError(
                "Method 0 expects concatenated per-layer features for "
                f"{EXPECTED_LAYER_OUTPUTS} outputs and {len(TOKEN_SUMMARIES)} token "
                f"positions ({self._core_dim} dims), got {X_arr.shape[1]}."
            )
        return X_arr

    def _slice_pair(
        self,
        X: np.ndarray,
        token_idx: int,
        layer_idx: int,
    ) -> np.ndarray:
        """Return the hidden-state block for one token-position/layer pair."""
        block_width = EXPECTED_LAYER_OUTPUTS * EXPECTED_HIDDEN_DIM
        start = token_idx * block_width + layer_idx * EXPECTED_HIDDEN_DIM
        end = start + EXPECTED_HIDDEN_DIM
        return X[:, start:end]

    def _score_layer_pairs(self, X: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
        """Fit a PCA for every pair and rank pairs by silhouette score."""
        diagnostics: list[dict[str, object]] = []

        for token_idx, token_name in enumerate(TOKEN_SUMMARIES):
            for layer_idx in range(EXPECTED_LAYER_OUTPUTS):
                layer_features = self._slice_pair(X, token_idx, layer_idx)
                pca = PCA(
                    n_components=PCA_COMPONENTS,
                    svd_solver="randomized",
                    random_state=RANDOM_STATE,
                )
                projected = pca.fit_transform(layer_features)

                try:
                    score = float(silhouette_score(projected, y))
                except ValueError:
                    score = float("-inf")

                diagnostics.append(
                    {
                        "token_idx": token_idx,
                        "token_name": token_name,
                        "layer_idx": layer_idx,
                        "silhouette": score,
                        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                        "pca": pca,
                    }
                )

        diagnostics.sort(
            key=lambda item: (float(item["silhouette"]), float(item["explained_variance"])),
            reverse=True,
        )
        return diagnostics

    def _select_pairs(self, diagnostics: list[dict[str, object]]) -> list[dict[str, object]]:
        """Keep the best-separated layer-position pairs for the final classifier."""
        positive_pairs = [
            pair for pair in diagnostics if math.isfinite(float(pair["silhouette"])) and float(pair["silhouette"]) > 0.0
        ]
        selected = positive_pairs[:MAX_SELECTED_PAIRS] or diagnostics[:1]
        return selected

    def _project_selected_pairs(self, X: np.ndarray) -> np.ndarray:
        """Project all selected layer-position pairs into PCA space."""
        if not self._selected_pairs:
            raise RuntimeError("No PCA pairs have been selected. Call fit() first.")

        parts: list[np.ndarray] = []
        for pair in self._selected_pairs:
            token_idx = int(pair["token_idx"])
            layer_idx = int(pair["layer_idx"])
            pca = pair["pca"]
            layer_features = self._slice_pair(X, token_idx, layer_idx)
            parts.append(pca.transform(layer_features))

        if X.shape[1] > self._core_dim:
            parts.append(X[:, self._core_dim :])

        return np.hstack(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for *x* by routing through ``predict_proba``."""
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs)).astype(np.float32)
        return torch.from_numpy(logits)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the Method 0 probe on labelled feature vectors."""
        X_arr = self._validate_and_prepare(X)
        y_arr = np.asarray(y, dtype=int)

        diagnostics = self._score_layer_pairs(X_arr[:, : self._core_dim], y_arr)
        self._selected_pairs = self._select_pairs(diagnostics)
        self.layer_diagnostics_ = [
            {
                "token_name": str(item["token_name"]),
                "layer_idx": int(item["layer_idx"]),
                "silhouette": float(item["silhouette"]),
                "explained_variance": float(item["explained_variance"]),
            }
            for item in diagnostics
        ]
        self.selected_layer_pairs_ = [
            {
                "token_name": str(item["token_name"]),
                "layer_idx": int(item["layer_idx"]),
                "silhouette": float(item["silhouette"]),
                "explained_variance": float(item["explained_variance"]),
            }
            for item in self._selected_pairs
        ]

        projected = self._project_selected_pairs(X_arr)
        projected_scaled = self._projection_scaler.fit_transform(projected)

        self._classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        self._classifier.fit(projected_scaled, y_arr)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1."""
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            score = f1_score(y_val, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        X_arr = self._validate_and_prepare(X)
        projected = self._project_selected_pairs(X_arr)
        projected_scaled = self._projection_scaler.transform(projected)
        prob_pos = self._classifier.predict_proba(projected_scaled)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
