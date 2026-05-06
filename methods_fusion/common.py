"""Shared utilities for late fusion over Methods 1, 2, and 3.

The fusion design is deliberately late-fusion:

1. train each selected standalone MLP branch on the current training fold,
2. freeze the branch,
3. extract its penultimate hidden representation,
4. concatenate those frozen representations,
5. train one final MLP classifier on the concatenated vector.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from method1_saplma.probe import SAPLMAProbe
from method2_icr_probe.probe import ICRMLPProbe
from method3_llm_check.common import LLMCheckAttentionMLPProbe


BRANCH_ORDER: tuple[str, ...] = ("saplma", "icr", "llm_check")
FUSION_EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "saplma_icr": ("saplma", "icr"),
    "saplma_llm_check": ("saplma", "llm_check"),
    "icr_llm_check": ("icr", "llm_check"),
    "saplma_icr_llm_check": ("saplma", "icr", "llm_check"),
}


@dataclass(frozen=True)
class BranchMetadata:
    """Bookkeeping for one frozen branch contribution."""

    name: str
    raw_feature_name: str
    raw_dim: int
    branch_architecture: list[int]
    fusion_embedding_dim: int
    fusion_vector: str
    activation: str
    dropout_p: float
    l1_lambda: float
    l2_weight_decay: float


def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-dim string."""
    hidden_dims = tuple(int(item) for item in hidden_dims_arg.split(",") if item.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one width, e.g. 64,32")
    return hidden_dims


def format_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    """Format hidden dims for metadata and command lines."""
    return ",".join(str(width) for width in hidden_dims)


def summarize_fold_results(fold_results: list[dict]) -> dict[str, float]:
    """Compute mean metrics for one fusion experiment."""
    metric_keys = [
        "baseline_accuracy",
        "baseline_f1",
        "train_accuracy",
        "train_f1",
        "train_auroc",
        "val_accuracy",
        "val_f1",
        "val_auroc",
        "test_accuracy",
        "test_f1",
        "test_auroc",
    ]
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [fold.get(key, float("nan")) for fold in fold_results]
        summary[f"mean_{key}"] = float(np.nanmean(values))
    return summary


class FusionMLPProbe(nn.Module):
    """Final MLP trained on concatenated frozen branch embeddings."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
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
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(p=self.dropout_p))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self._net = nn.Sequential(*layers).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network not initialized. Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FusionMLPProbe":
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

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "FusionMLPProbe":
        """Tune threshold on validation accuracy, with F1 as a tie-breaker."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if acc > best_accuracy or (acc == best_accuracy and f1 > best_f1):
                best_accuracy = float(acc)
                best_f1 = float(f1)
                best_threshold = float(threshold)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

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


def build_branch_metadata(raw_dims: dict[str, int]) -> dict[str, BranchMetadata]:
    """Return fixed branch metadata for the selected best standalone configs."""
    return {
        "saplma": BranchMetadata(
            name="SAPLMA",
            raw_feature_name="response_last_hidden_state_layer_15",
            raw_dim=raw_dims["saplma"],
            branch_architecture=[raw_dims["saplma"], 256, 128, 64, 1],
            fusion_embedding_dim=64,
            fusion_vector="penultimate_hidden_activation",
            activation="ReLU",
            dropout_p=0.3,
            l1_lambda=1e-5,
            l2_weight_decay=1e-4,
        ),
        "icr": BranchMetadata(
            name="ICR Probe",
            raw_feature_name="layerwise_mean_icr",
            raw_dim=raw_dims["icr"],
            branch_architecture=[raw_dims["icr"], 128, 64, 32, 1],
            fusion_embedding_dim=32,
            fusion_vector="penultimate_hidden_activation_after_last_dropout",
            activation="BatchNorm1d + LeakyReLU(0.01)",
            dropout_p=0.3,
            l1_lambda=0.0,
            l2_weight_decay=1e-4,
        ),
        "llm_check": BranchMetadata(
            name="LLM-Check Attention",
            raw_feature_name="attention_diagonal_log_score_layers_2_to_24",
            raw_dim=raw_dims["llm_check"],
            branch_architecture=[raw_dims["llm_check"], 64, 32, 1],
            fusion_embedding_dim=32,
            fusion_vector="penultimate_hidden_activation",
            activation="ReLU",
            dropout_p=0.3,
            l1_lambda=0.0,
            l2_weight_decay=1e-4,
        ),
    }


def train_branch_probe(
    branch: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    args,
):
    """Train one standalone branch probe with the selected best config."""
    if branch == "saplma":
        return SAPLMAProbe(
            hidden_dims=(256, 128, 64),
            lr=args.branch_learning_rate,
            epochs=args.saplma_epochs,
            batch_size=args.branch_batch_size,
            dropout_p=0.3,
            l1_lambda=1e-5,
            l2_weight_decay=1e-4,
            random_state=args.random_state,
        ).fit(X_train, y_train)

    if branch == "icr":
        return ICRMLPProbe(
            hidden_dims=(128, 64, 32),
            lr=args.branch_learning_rate,
            epochs=args.branch_epochs,
            batch_size=args.branch_batch_size,
            dropout_p=0.3,
            l1_lambda=0.0,
            l2_weight_decay=1e-4,
            random_state=args.random_state,
        ).fit(X_train, y_train)

    if branch == "llm_check":
        return LLMCheckAttentionMLPProbe(
            hidden_dims=(64, 32),
            lr=args.branch_learning_rate,
            epochs=args.branch_epochs,
            batch_size=args.branch_batch_size,
            dropout_p=0.3,
            l2_weight_decay=1e-4,
            random_state=args.random_state,
        ).fit(X_train, y_train)

    raise ValueError(f"Unknown branch: {branch}")


def extract_branch_embedding(branch: str, probe, X: np.ndarray) -> np.ndarray:
    """Extract the frozen penultimate embedding for one trained branch probe."""
    X_arr = np.asarray(X, dtype=np.float32)

    if branch == "saplma":
        if probe._net is None:
            raise RuntimeError("SAPLMA branch has not been fitted.")
        X_t = torch.from_numpy(X_arr).to(probe.device)
        probe.eval()
        with torch.no_grad():
            out = X_t
            for layer in list(probe._net.children())[:-1]:
                out = layer(out)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)

    if branch == "icr":
        if probe._net is None:
            raise RuntimeError("ICR branch has not been fitted.")
        X_t = torch.from_numpy(X_arr).to(probe.device)
        probe.eval()
        with torch.no_grad():
            out = X_t
            for layer in list(probe._net.network.children())[:-2]:
                out = layer(out)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)

    if branch == "llm_check":
        if probe._net is None:
            raise RuntimeError("LLM-Check branch has not been fitted.")
        X_scaled = probe._scaler.transform(X_arr)
        X_t = torch.from_numpy(X_scaled.astype(np.float32, copy=False)).to(probe.device)
        probe.eval()
        with torch.no_grad():
            out = X_t
            for layer in list(probe._net.children())[:-1]:
                out = layer(out)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)

    raise ValueError(f"Unknown branch: {branch}")


def build_fusion_matrix(
    branches: tuple[str, ...],
    branch_probes: dict[str, object],
    raw_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, dict]]:
    """Concatenate frozen branch embeddings for one fusion experiment."""
    embeddings = []
    contribution_details: dict[str, dict] = {}

    for branch in branches:
        embedding = extract_branch_embedding(branch, branch_probes[branch], raw_features[branch])
        embeddings.append(embedding)
        contribution_details[branch] = {
            "embedding_dim": int(embedding.shape[1]),
            "fusion_vector": "penultimate_hidden_activation",
        }

    return np.concatenate(embeddings, axis=1), contribution_details

