"""Shared utilities for Method 1 SAPLMA runs and ablations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import extract_feature_cache, load_feature_cache, save_feature_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = ROOT / "data" / "dataset.csv"
DEFAULT_CACHE_FILE = ROOT / "method1_saplma" / "artifacts" / "cache" / "method1_hidden_cache.npz"
DEFAULT_OUTPUT_FILE = ROOT / "method1_saplma" / "artifacts" / "method1_results.json"
DEFAULT_LAYER_RANKINGS = ROOT.parent / "layer_rankings.csv"
DEFAULT_ABLATION_DIR = ROOT / "method1_saplma" / "artifacts" / "ablation"
DEFAULT_BINARY_ABLATION_DIR = ROOT / "method1_saplma" / "artifacts" / "binary_ablation"
DEFAULT_AUTO_LAYER = 15
DEFAULT_ARCHITECTURES = ((256, 128, 64), (256, 128), (256,))
DEFAULT_ABLATION_DROPOUT_P = 0.3
DEFAULT_ABLATION_L1 = 1e-5
DEFAULT_ABLATION_L2 = 1e-4
DEFAULT_BINARY_MLP_HIDDEN_DIMS = (256, 128)


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


def resolve_layers(
    layers_arg: str,
    layer_rankings_file: Path,
    auto_top_k: int,
) -> list[int]:
    """Resolve explicit layers or derive them from Method 0 rankings."""
    if layers_arg != "auto":
        return [int(item) for item in layers_arg.split(",") if item.strip()]

    if layer_rankings_file.exists():
        rankings = pd.read_csv(layer_rankings_file)
        top_layers = rankings.sort_values(
            ["max_silhouette", "mean_silhouette"],
            ascending=[False, False],
        )["layer"].head(auto_top_k)
        return [int(layer) for layer in top_layers.tolist()]

    if auto_top_k == 1:
        print(
            "[Method 1] "
            f"{layer_rankings_file} not found. Falling back to layer {DEFAULT_AUTO_LAYER}."
        )
        return [DEFAULT_AUTO_LAYER]

    raise FileNotFoundError(
        "layer_rankings.csv is required when --layers=auto and --auto-top-k > 1."
    )


def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-dim string."""
    hidden_dims = tuple(int(item) for item in hidden_dims_arg.split(",") if item.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one width, e.g. 256,128,64")
    return hidden_dims


def parse_architectures(architectures_arg: str) -> list[tuple[int, ...]]:
    """Parse semicolon-separated hidden-layer architectures."""
    architectures = [
        parse_hidden_dims(chunk.strip())
        for chunk in architectures_arg.split(";")
        if chunk.strip()
    ]
    if not architectures:
        raise ValueError("architectures must contain at least one entry")
    return architectures


def format_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    """Format hidden dims for logs and metadata."""
    return ",".join(str(width) for width in hidden_dims)


def build_feature_matrix(
    cache: dict[str, np.ndarray],
    token_mode: str,
    layers: list[int],
) -> np.ndarray:
    """Select token-mode features from the chosen layers and concatenate them."""
    mode_features = cache[token_mode].astype(np.float32, copy=False)
    selected = [mode_features[:, layer_idx, :] for layer_idx in layers]
    return np.concatenate(selected, axis=1)


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
    """Load the cached hidden states or build them once."""
    should_rebuild_cache = overwrite_cache or subset_size is not None or not cache_file.exists()

    if not should_rebuild_cache:
        print(f"[Method 1] Loading cache from {cache_file}")
        cache = load_feature_cache(cache_file)
        if (
            "labels" not in cache
            or len(cache["labels"]) != len(df)
            or int(cache.get("truncation_disabled", np.asarray([0], dtype=np.int8))[0]) != 1
        ):
            print("[Method 1] Cache shape mismatch detected. Rebuilding cache.")
            should_rebuild_cache = True

    if should_rebuild_cache:
        print(f"[Method 1] Building cache from {data_file}")
        cache = extract_feature_cache(
            df=df,
            batch_size=batch_size,
            max_length=max_length,
            cache_dtype=np.float16 if cache_dtype == "float16" else np.float32,
            include_icr=False,
            include_spectrum=False,
        )
        if subset_size is None:
            save_feature_cache(cache_file, cache)
            print(f"[Method 1] Saved cache to {cache_file}")
        else:
            print("[Method 1] Subset run detected. Cache was not persisted.")
        return cache

    return cache


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
    architectures: list[tuple[int, ...]],
    dropout_p: float,
    l1_lambda: float,
    l2_weight_decay: float,
) -> list[dict]:
    """Build the Method 1 ablation grid requested for SAPLMA."""
    dropout_values = [0.0]
    if dropout_p > 0.0:
        dropout_values.append(dropout_p)

    regularization_values = [(0.0, 0.0)]
    if l1_lambda > 0.0 or l2_weight_decay > 0.0:
        regularization_values.append((l1_lambda, l2_weight_decay))

    configs: list[dict] = []
    for hidden_dims in architectures:
        n_linear_layers = len(hidden_dims) + 1
        dims_label = "x".join(str(width) for width in hidden_dims)
        for current_dropout in dropout_values:
            for current_l1, current_l2 in regularization_values:
                uses_dropout = current_dropout > 0.0
                uses_regularization = current_l1 > 0.0 or current_l2 > 0.0
                drop_label = "dropout" if uses_dropout else "no_dropout"
                reg_label = "l1_l2" if uses_regularization else "no_reg"
                configs.append(
                    {
                        "name": f"{n_linear_layers}layer_{dims_label}_{drop_label}_{reg_label}",
                        "hidden_dims": hidden_dims,
                        "n_linear_layers": n_linear_layers,
                        "dropout_p": current_dropout,
                        "l1_lambda": current_l1,
                        "l2_weight_decay": current_l2,
                        "uses_dropout": uses_dropout,
                        "uses_regularization": uses_regularization,
                    }
                )
    return configs
