"""Run Method 2 (adapted ICR Probe) on dataset.csv."""

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
from method2_icr_probe.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_MLP_HIDDEN_DIMS,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_TOP_K,
    build_feature_matrix,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_hidden_dims,
)
from method2_icr_probe.probe import ICRLogisticProbe, ICRMLPProbe
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 2 (adapted ICR Probe) on dataset.csv.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--disable-z-normalize",
        action="store_true",
        help="Disable the repo-style z-score normalization before softmax.",
    )
    parser.add_argument(
        "--classifier",
        choices=("logistic", "mlp"),
        default="logistic",
        help="Probe family used on top of the extracted ICR vector.",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--hidden-dims", default=format_hidden_dims(DEFAULT_MLP_HIDDEN_DIMS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-p", type=float, default=0.3)
    parser.add_argument("--l2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    cache_file = Path(args.cache_file)
    output_file = Path(args.output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df = maybe_take_subset(df, args.subset)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    z_normalize = not args.disable_z_normalize

    cache = load_or_build_cache(
        df=df,
        cache_file=cache_file,
        data_file=data_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.top_k,
        z_normalize=z_normalize,
        cache_dtype=args.cache_dtype,
        overwrite_cache=args.overwrite_cache,
        subset_size=args.subset,
    )

    print(f"[Method 2] Classifier: {args.classifier}")
    print(f"[Method 2] top_k: {args.top_k}")
    print(f"[Method 2] z_normalize: {z_normalize}")
    if args.classifier == "mlp":
        print(f"[Method 2] Hidden dims: {hidden_dims}")
        print(f"[Method 2] Dropout p: {args.dropout_p}")
        print(f"[Method 2] L2 weight decay: {args.l2_weight_decay}")

    X = build_feature_matrix(cache)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)

    if args.classifier == "logistic":
        probe_factory = lambda: ICRLogisticProbe(c=args.logistic_c)
    else:
        probe_factory = lambda: ICRMLPProbe(
            hidden_dims=hidden_dims,
            lr=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.mlp_batch_size,
            dropout_p=args.dropout_p,
            l2_weight_decay=args.l2_weight_decay,
        )

    fold_results = run_evaluation(splits, X, y, probe_factory)
    print_summary(fold_results, X.shape[1], len(X), extract_time=0.0)
    save_results(fold_results, X.shape[1], len(X), extract_time=0.0, output_file=str(output_file))

    metadata = {
        "method": "ICR Probe",
        "feature_type": "layerwise_mean_icr",
        "cache_file": str(cache_file),
        "data_file": str(data_file),
        "subset": args.subset,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "cache_dtype": args.cache_dtype,
        "top_k": args.top_k,
        "z_normalize": z_normalize,
        "classifier": args.classifier,
        "logistic_c": args.logistic_c,
        "hidden_dims": list(hidden_dims),
        "hidden_dims_text": format_hidden_dims(hidden_dims),
        "epochs": args.epochs,
        "mlp_batch_size": args.mlp_batch_size,
        "learning_rate": args.learning_rate,
        "dropout_p": args.dropout_p,
        "l2_weight_decay": args.l2_weight_decay,
        "feature_dim": int(X.shape[1]),
        "attention_pooling": "mean_all_heads",
        "use_induction_head": False,
        "token_aggregation": "mean_over_response_tokens_per_layer",
        "response_span": "full_response_tokens_preserved_by_truncation",
    }
    metadata_file = output_file.with_name(output_file.stem + "_metadata.json")
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"[Method 2] Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
