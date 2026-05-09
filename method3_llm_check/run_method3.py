"""Run Method 3 (LLM-Check features) on dataset.csv."""

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
from method3_llm_check.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_DROPOUT_P,
    DEFAULT_FEATURE_SET,
    DEFAULT_FEATURE_SETS,
    DEFAULT_L2_WEIGHT_DECAY,
    DEFAULT_LOGIT_METRIC_NAMES,
    DEFAULT_MLP_HIDDEN_DIMS,
    DEFAULT_OUTPUT_FILE,
    LLMCheckLogisticProbe,
    LLMCheckMLPProbe,
    build_feature_matrix,
    build_feature_set_metadata,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_hidden_dims,
)
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 3 (LLM-Check features) on dataset.csv.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--feature-set",
        choices=DEFAULT_FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
        help="Which LLM-Check feature family or concatenated family to evaluate.",
    )
    parser.add_argument(
        "--classifier",
        choices=("logistic", "mlp"),
        default="logistic",
        help="Probe family used on top of the extracted LLM-Check features.",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--hidden-dims", default=format_hidden_dims(DEFAULT_MLP_HIDDEN_DIMS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_DROPOUT_P)
    parser.add_argument("--l2-weight-decay", type=float, default=DEFAULT_L2_WEIGHT_DECAY)
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

    feature_metadata = build_feature_set_metadata(cache, args.feature_set)
    print(f"[Method 3] Feature set: {args.feature_set}")
    print(f"[Method 3] Components: {feature_metadata['components']}")
    print(f"[Method 3] Feature dim: {feature_metadata['feature_dim']}")
    print(f"[Method 3] Classifier: {args.classifier}")
    if args.classifier == "mlp":
        print(f"[Method 3] Hidden dims: {hidden_dims}")
        print(f"[Method 3] Dropout p: {args.dropout_p}")
        print(f"[Method 3] L2 weight decay: {args.l2_weight_decay}")

    X = build_feature_matrix(cache, feature_set=args.feature_set)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)

    if args.classifier == "logistic":
        probe_factory = lambda: LLMCheckLogisticProbe(c=args.logistic_c)
    else:
        probe_factory = lambda: LLMCheckMLPProbe(
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
        "method": "LLM-Check",
        "feature_set": args.feature_set,
        "description": feature_metadata["description"],
        "components": feature_metadata["components"],
        "component_dims": feature_metadata["component_dims"],
        "cache_file": str(cache_file),
        "data_file": str(data_file),
        "subset": args.subset,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "cache_dtype": args.cache_dtype,
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
        "response_span": feature_metadata["response_span"],
        "logit_metric_names": cache["logit_metric_names"].tolist(),
        "selected_hidden_state_indices_hf": cache["selected_hidden_state_indices_hf"].astype(int).tolist(),
        "selected_hidden_transformer_layers_zero_based": cache[
            "selected_hidden_transformer_layers_zero_based"
        ].astype(int).tolist(),
        "selected_hidden_transformer_layers_1based": cache[
            "selected_hidden_transformer_layers_1based"
        ].astype(int).tolist(),
        "selected_attention_layer_indices_zero_based": cache[
            "selected_attention_layer_indices_zero_based"
        ].astype(int).tolist(),
        "selected_attention_transformer_layers_1based": cache[
            "selected_attention_transformer_layers_1based"
        ].astype(int).tolist(),
        "feature_formulas": {
            "logit": {
                "metrics": list(DEFAULT_LOGIT_METRIC_NAMES),
                "perplexity": "exp(-mean(log p(correct_response_token | previous_context)))",
                "window_entropy_w1": "max_token mean(-p log p) over full-vocab softmax",
                "logit_entropy_top50": "mean(-p log p) over top-50 logits on response tokens",
            },
            "hidden": "per-layer mean(log(svdvals(Z^T J Z + alpha I))) over response tokens",
            "attns": "per-layer sum_heads(mean(log(diag(attention_layer[:, response, response]))))",
        },
    }
    metadata_file = output_file.with_name(output_file.stem + "_metadata.json")
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"[Method 3] Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
