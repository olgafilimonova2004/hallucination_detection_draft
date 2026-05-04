"""
probe.py — SAPLMA-style MLP classifier for Method 1.

This mirrors the core architecture used in the SAPLMA repository:

  input -> 256 -> 128 -> 64 -> 1

with ReLU activations and 5 epochs of Adam training.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class SAPLMAProbe(nn.Module):
    """MLP probe that follows the SAPLMA architecture."""

    def __init__(
        self,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 32,
        random_state: int = 42,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._net: nn.Sequential | None = None
        self._threshold: float = 0.5

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_network(self, input_dim: int) -> None:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self._net = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network not initialized. Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAPLMAProbe":
        self._set_seed()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self._build_network(X_arr.shape[1])
        assert self._net is not None

        X_t = torch.from_numpy(X_arr).to(self.device)
        y_t = torch.from_numpy(y_arr).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.eval()
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "SAPLMAProbe":
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
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X_arr).to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).detach().cpu().numpy()

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
