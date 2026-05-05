"""Probe classifiers for Method 2 ICR vectors."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class _ThresholdedProbe(nn.Module):
    """Shared threshold-tuning helpers for sklearn-style binary probes."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "_ThresholdedProbe":
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


class ICRLogisticProbe(_ThresholdedProbe):
    """Scaled logistic-regression baseline over the adapted ICR vectors."""

    def __init__(
        self,
        c: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._classifier: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ICRLogisticProbe":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        X_scaled = self._scaler.fit_transform(X_arr)
        self._classifier = LogisticRegression(
            C=self.c,
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="liblinear",
        )
        self._classifier.fit(X_scaled, y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X_arr)
        prob_pos = self._classifier.predict_proba(X_scaled)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)


class ICRMLPProbe(_ThresholdedProbe):
    """Tiny MLP ablation over the adapted ICR vectors."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32,),
        lr: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 32,
        dropout_p: float = 0.3,
        l2_weight_decay: float = 1e-4,
        random_state: int = 42,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.l2_weight_decay = l2_weight_decay
        self.random_state = random_state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._scaler = StandardScaler()
        self._net: nn.Sequential | None = None

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
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(p=self.dropout_p))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self._net = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network not initialized. Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ICRMLPProbe":
        self._set_seed()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        X_scaled = self._scaler.fit_transform(X_arr)

        self._build_network(X_scaled.shape[1])
        assert self._net is not None

        X_t = torch.from_numpy(X_scaled).to(self.device)
        y_t = torch.from_numpy(y_arr).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self.lr,
            weight_decay=self.l2_weight_decay,
        )

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X_arr)
        X_t = torch.from_numpy(X_scaled).to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).detach().cpu().numpy()

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
