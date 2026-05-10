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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import print_summary, run_evaluation, save_results
from method1_saplma.common import (
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_LAYER_RANKINGS,
    DEFAULT_OUTPUT_FILE,
    build_feature_matrix,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_hidden_dims,
    resolve_layers,
)
from method1_saplma.probe import SAPLMALogisticProbe, SAPLMAProbe
from splitting import split_data


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
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Deprecated compatibility flag. Truncation is disabled and the full prompt+response is used.",
    )
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--classifier",
        choices=("mlp", "logistic"),
        default="mlp",
        help="Probe family used on top of the extracted SAPLMA features.",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        default="256,128,64",
        help="Comma-separated hidden widths. Example: 256,128,64",
    )
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--l1-lambda", type=float, default=0.0)
    parser.add_argument("--l2-weight-decay", type=float, default=0.0)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


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
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    cache = load_or_build_cache(
        df=df,
        cache_file=cache_file,
        data_file=data_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dtype=args.cache_dtype,
        overwrite_cache=args.overwrite_cache,
        subset_size=args.subset,
    )

    layers = resolve_layers(
        layers_arg=args.layers,
        layer_rankings_file=layer_rankings_file,
        auto_top_k=args.auto_top_k,
    )
    print(f"[Method 1] Using token mode: {args.token_mode}")
    print(f"[Method 1] Using layers: {layers}")
    print(f"[Method 1] Classifier: {args.classifier}")
    if args.classifier == "mlp":
        print(f"[Method 1] Hidden dims: {hidden_dims}")
        print(f"[Method 1] Dropout p: {args.dropout_p}")
        print(f"[Method 1] L1 lambda: {args.l1_lambda}")
        print(f"[Method 1] L2 weight decay: {args.l2_weight_decay}")

    X = build_feature_matrix(cache=cache, token_mode=args.token_mode, layers=layers)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)

    if args.classifier == "logistic":
        probe_factory = lambda: SAPLMALogisticProbe(c=args.logistic_c)
    else:
        probe_factory = lambda: SAPLMAProbe(
            hidden_dims=hidden_dims,
            lr=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.mlp_batch_size,
            dropout_p=args.dropout_p,
            l1_lambda=args.l1_lambda,
            l2_weight_decay=args.l2_weight_decay,
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
        "classifier": args.classifier,
        "logistic_c": args.logistic_c,
        "epochs": args.epochs,
        "mlp_batch_size": args.mlp_batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dims": list(hidden_dims),
        "dropout_p": args.dropout_p,
        "l1_lambda": args.l1_lambda,
        "l2_weight_decay": args.l2_weight_decay,
        "hidden_dims_text": format_hidden_dims(hidden_dims),
        "truncation": "disabled",
    }
    metadata_file = output_file.with_name(output_file.stem + "_metadata.json")
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"[Method 1] Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
