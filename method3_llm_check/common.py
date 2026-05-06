"""Shared utilities for Method 3 LLM-Check attention-score runs."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_utils import (
    build_response_preserving_batch,
    get_best_available_device,
    load_feature_cache,
    save_feature_cache,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATA_FILE = ROOT / "data" / "dataset.csv"
DEFAULT_CACHE_FILE = ROOT / "method3_llm_check" / "artifacts" / "cache" / "method3_llm_check_cache.npz"
DEFAULT_OUTPUT_FILE = ROOT / "method3_llm_check" / "artifacts" / "method3_results.json"
DEFAULT_ABLATION_DIR = ROOT / "method3_llm_check" / "artifacts" / "ablation"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MLP_HIDDEN_DIMS = (64, 32)
DEFAULT_DROPOUT_P = 0.3
DEFAULT_L2_WEIGHT_DECAY = 1e-4


def maybe_take_subset(df: pd.DataFrame, subset_size: int | None) -> pd.DataFrame:
    """Take a roughly stratified subset for smoke tests."""
    if subset_size is None or subset_size >= len(df):
        return df.reset_index(drop=True)

    parts = []
    label_counts = df["label"].value_counts(normalize=True).sort_index()
    for label, frac in label_counts.items():
        n_label = max(1, int(round(subset_size * frac)))
        label_df = df[df["label"] == label]
        parts.append(label_df.sample(n=min(n_label, len(label_df)), random_state=42))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)


def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-dim string."""
    hidden_dims = tuple(int(item) for item in hidden_dims_arg.split(",") if item.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one width, e.g. 64,32")
    return hidden_dims


def format_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    """Format hidden dims for logs and metadata."""
    return ",".join(str(width) for width in hidden_dims)


def summarize_fold_results(fold_results: list[dict]) -> dict[str, float]:
    """Compute cross-fold mean metrics for leaderboard reporting."""
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


def build_ablation_configs(
    hidden_dims: tuple[int, ...],
    dropout_p: float,
    l2_weight_decay: float,
) -> list[dict]:
    """Return the two requested Method 3 classifier configurations."""
    return [
        {
            "name": "logistic_regression",
            "classifier": "logistic",
            "hidden_dims": (),
            "dropout_p": 0.0,
            "l2_weight_decay": 0.0,
        },
        {
            "name": "mlp_dropout0.3_l2",
            "classifier": "mlp",
            "hidden_dims": hidden_dims,
            "dropout_p": dropout_p,
            "l2_weight_decay": l2_weight_decay,
        },
    ]


def get_llm_check_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen with eager attention so per-layer attentions are available."""
    print(f"[Method 3] Loading '{model_name}' with eager attention ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def _compute_attention_layer_score(
    layer_attention: torch.Tensor,
    response_span: tuple[int, int],
) -> torch.Tensor:
    """Compute the repo-style attention score for one layer on the response span."""
    response_start, response_end = response_span
    span_attention = layer_attention[:, response_start:response_end, response_start:response_end]
    if span_attention.size(-1) == 0:
        return layer_attention.new_tensor(0.0)

    # The LLM-Check repo applies log to the diagonal attention values directly.
    # Clamp is used here only as a numerical guard against log(0).
    diagonal = torch.diagonal(span_attention, offset=0, dim1=-2, dim2=-1).clamp_min(1e-12)
    return torch.log(diagonal).mean(dim=-1).sum()


def extract_attention_score_cache(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    device: torch.device | None = None,
    cache_dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """Extract reduced LLM-Check attention-score vectors for all rows in *df*."""
    if device is None:
        device = get_best_available_device()

    model, tokenizer = get_llm_check_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    n_samples = len(df)
    n_layers = model.config.num_hidden_layers
    selected_layer_indices = np.arange(1, n_layers, dtype=np.int32)
    prompts = df["prompt"].tolist()
    responses = df["response"].tolist()

    cache: dict[str, np.ndarray] = {
        "attention_score_vector": np.empty((n_samples, len(selected_layer_indices)), dtype=cache_dtype),
        "selected_layer_indices_zero_based": selected_layer_indices,
        "selected_transformer_layers_1based": selected_layer_indices + 1,
        "prompt_token_length": np.empty(n_samples, dtype=np.int32),
        "response_token_length": np.empty(n_samples, dtype=np.int32),
        "response_truncated": np.empty(n_samples, dtype=np.int8),
        "max_length": np.asarray([max_length], dtype=np.int32),
    }

    for start in tqdm(range(0, n_samples, batch_size), desc="Caching Method 3 attention scores", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        batch_responses = responses[start : start + batch_size]
        input_ids, attention_mask, response_spans = build_response_preserving_batch(
            tokenizer=tokenizer,
            prompts=batch_prompts,
            responses=batch_responses,
            max_length=max_length,
            device=device,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        batch_attention_mask = attention_mask.cpu()
        for sample_idx in range(input_ids.size(0)):
            row_idx = start + sample_idx
            seq_len = int(batch_attention_mask[sample_idx].sum().item())
            response_start, response_end = response_spans[sample_idx]
            response_end = min(response_end, seq_len)
            response_span = (response_start, response_end)

            layer_scores: list[torch.Tensor] = []
            for layer_idx in selected_layer_indices.tolist():
                layer_attention = outputs.attentions[layer_idx][sample_idx, :, :seq_len, :seq_len]
                layer_attention = layer_attention.detach().to(device="cpu", dtype=torch.float32)
                layer_scores.append(_compute_attention_layer_score(layer_attention, response_span))

            cache["attention_score_vector"][row_idx] = (
                torch.stack(layer_scores).numpy().astype(cache_dtype, copy=False)
            )
            cache["prompt_token_length"][row_idx] = response_start
            cache["response_token_length"][row_idx] = response_end - response_start
            cache["response_truncated"][row_idx] = int(response_start == 0)

        del outputs
        del input_ids
        del attention_mask
        del batch_attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "label" in df.columns and df["label"].notna().all():
        cache["labels"] = df["label"].astype(int).to_numpy(dtype=np.int32)

    return cache


def build_feature_matrix(cache: dict[str, np.ndarray]) -> np.ndarray:
    """Return the Method 3 feature matrix."""
    return cache["attention_score_vector"].astype(np.float32, copy=False)


def load_or_build_cache(
    df: pd.DataFrame,
    cache_file: Path,
    data_file: Path,
    batch_size: int,
    max_length: int,
    cache_dtype: str,
    overwrite_cache: bool,
    subset_size: int | None,
) -> dict[str, np.ndarray]:
    """Load the cached attention-score vectors or build them once."""
    should_rebuild_cache = overwrite_cache or subset_size is not None or not cache_file.exists()

    if not should_rebuild_cache:
        print(f"[Method 3] Loading cache from {cache_file}")
        cache = load_feature_cache(cache_file)
        labels_match = "labels" in cache and len(cache["labels"]) == len(df)
        max_length_match = int(cache.get("max_length", np.asarray([-1], dtype=np.int32))[0]) == max_length
        if not (labels_match and max_length_match):
            print("[Method 3] Cache metadata mismatch detected. Rebuilding cache.")
            should_rebuild_cache = True

    if should_rebuild_cache:
        print(f"[Method 3] Building cache from {data_file}")
        cache = extract_attention_score_cache(
            df=df,
            batch_size=batch_size,
            max_length=max_length,
            cache_dtype=np.float16 if cache_dtype == "float16" else np.float32,
        )
        if subset_size is None:
            save_feature_cache(cache_file, cache)
            print(f"[Method 3] Saved cache to {cache_file}")
        else:
            print("[Method 3] Subset run detected. Cache was not persisted.")
        return cache

    return cache


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


class LLMCheckAttentionLogisticProbe(_ThresholdedProbe):
    """Scaled logistic-regression baseline over the Method 3 attention-score vectors."""

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LLMCheckAttentionLogisticProbe":
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


class LLMCheckAttentionMLPProbe(_ThresholdedProbe):
    """Small MLP ablation over the Method 3 attention-score vectors."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = DEFAULT_MLP_HIDDEN_DIMS,
        lr: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 32,
        dropout_p: float = DEFAULT_DROPOUT_P,
        l2_weight_decay: float = DEFAULT_L2_WEIGHT_DECAY,
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LLMCheckAttentionMLPProbe":
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
