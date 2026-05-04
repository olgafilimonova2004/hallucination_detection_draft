"""
run_method1.py — reproduce the SAPLMA MLP approach on SMILES dataset.csv.

The script:
1. Loads or builds a lightweight hidden-state cache for Qwen/Qwen2.5-0.5B.
2. Selects one or more layers.
3. Extracts the chosen token representation from those layers.
4. Trains a SAPLMA-style MLP and evaluates it with the repo's split/eval code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import print_summary, run_evaluation, save_results
from experiment_utils import extract_feature_cache, load_feature_cache, save_feature_cache
from method1_saplma.probe import SAPLMAProbe
from splitting import split_data


DEFAULT_DATA_FILE = ROOT / "data" / "dataset.csv"
DEFAULT_CACHE_FILE = ROOT / "method1_saplma" / "artifacts" / "cache" / "method1_hidden_cache.npz"
DEFAULT_OUTPUT_FILE = ROOT / "method1_saplma" / "artifacts" / "method1_results.json"
DEFAULT_LAYER_RANKINGS = ROOT.parent / "layer_rankings.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 1 (SAPLMA) on dataset.csv.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--layer-rankings-file", default=str(DEFAULT_LAYER_RANKINGS))
    parser.add_argument(
        "--layers",
        default="auto",
        help='Comma-separated layer indices, or "auto" to read the best layers from layer_rankings.csv.',
    )
    parser.add_argument(
        "--auto-top-k",
        type=int,
        default=1,
        help="Number of top-ranked layers to keep when --layers=auto.",
    )
    parser.add_argument(
        "--token-mode",
        choices=("response_last", "last_token", "response_mean"),
        default="response_last",
        help="Hidden-state view fed to the MLP.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def maybe_take_subset(df: pd.DataFrame, subset_size: int | None) -> pd.DataFrame:
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
    if layers_arg != "auto":
        return [int(item) for item in layers_arg.split(",") if item.strip()]

    rankings = pd.read_csv(layer_rankings_file)
    top_layers = rankings.sort_values(
        ["max_silhouette", "mean_silhouette"],
        ascending=[False, False],
    )["layer"].head(auto_top_k)
    return [int(layer) for layer in top_layers.tolist()]


def build_feature_matrix(
    cache: dict[str, np.ndarray],
    token_mode: str,
    layers: list[int],
) -> np.ndarray:
    mode_features = cache[token_mode].astype(np.float32, copy=False)
    selected = [mode_features[:, layer_idx, :] for layer_idx in layers]
    return np.concatenate(selected, axis=1)


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    cache_file = Path(args.cache_file)
    output_file = Path(args.output_file)
    layer_rankings_file = Path(args.layer_rankings_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df = maybe_take_subset(df, args.subset)

    should_rebuild_cache = args.overwrite_cache or args.subset is not None or not cache_file.exists()

    if not should_rebuild_cache:
        print(f"[Method 1] Loading cache from {cache_file}")
        cache = load_feature_cache(cache_file)
        if "labels" not in cache or len(cache["labels"]) != len(df):
            print("[Method 1] Cache shape mismatch detected. Rebuilding cache.")
            should_rebuild_cache = True

    if should_rebuild_cache:
        print(f"[Method 1] Building cache from {data_file}")
        cache = extract_feature_cache(
            df=df,
            batch_size=args.batch_size,
            max_length=args.max_length,
            cache_dtype=np.float16 if args.cache_dtype == "float16" else np.float32,
            include_icr=False,
            include_spectrum=False,
        )
        save_feature_cache(cache_file, cache)
        print(f"[Method 1] Saved cache to {cache_file}")

    layers = resolve_layers(
        layers_arg=args.layers,
        layer_rankings_file=layer_rankings_file,
        auto_top_k=args.auto_top_k,
    )
    print(f"[Method 1] Using token mode: {args.token_mode}")
    print(f"[Method 1] Using layers: {layers}")

    X = build_feature_matrix(cache=cache, token_mode=args.token_mode, layers=layers)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)

    probe_factory = lambda: SAPLMAProbe(
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.mlp_batch_size,
    )

    fold_results = run_evaluation(splits, X, y, probe_factory)
    print_summary(fold_results, X.shape[1], len(X), extract_time=0.0)
    save_results(fold_results, X.shape[1], len(X), extract_time=0.0, output_file=str(output_file))

    metadata = {
        "method": "SAPLMA",
        "token_mode": args.token_mode,
        "layers": layers,
        "cache_file": str(cache_file),
        "data_file": str(data_file),
        "subset": args.subset,
        "epochs": args.epochs,
        "mlp_batch_size": args.mlp_batch_size,
        "learning_rate": args.learning_rate,
    }
    metadata_file = output_file.with_name(output_file.stem + "_metadata.json")
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"[Method 1] Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
