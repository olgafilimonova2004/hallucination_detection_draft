"""Run the requested Method 3 classifier ablation study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import run_evaluation
from method3_llm_check.common import (
    DEFAULT_ABLATION_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_DROPOUT_P,
    DEFAULT_L2_WEIGHT_DECAY,
    DEFAULT_MLP_HIDDEN_DIMS,
    LLMCheckAttentionLogisticProbe,
    LLMCheckAttentionMLPProbe,
    build_ablation_configs,
    build_feature_matrix,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_hidden_dims,
    summarize_fold_results,
)
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 3 LLM-Check attention-score ablations.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--hidden-dims", default=format_hidden_dims(DEFAULT_MLP_HIDDEN_DIMS))
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_DROPOUT_P)
    parser.add_argument("--l2-weight-decay", type=float, default=DEFAULT_L2_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def selection_key(row: dict) -> tuple[float, float, int]:
    """Rank ablations by validation accuracy, then AUROC, then simplicity."""
    val_accuracy = row.get("mean_val_accuracy", float("-inf"))
    val_auroc = row.get("mean_val_auroc", float("-inf"))
    if pd.isna(val_accuracy):
        val_accuracy = float("-inf")
    if pd.isna(val_auroc):
        val_auroc = float("-inf")
    is_logistic = int(row["classifier"] == "logistic")
    return (
        float(val_accuracy),
        float(val_auroc),
        is_logistic,
    )


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    cache_file = Path(args.cache_file)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
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

    X = build_feature_matrix(cache)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)
    configs = build_ablation_configs(
        hidden_dims=hidden_dims,
        dropout_p=args.dropout_p,
        l2_weight_decay=args.l2_weight_decay,
    )

    experiments: list[dict] = []
    for index, config in enumerate(configs, start=1):
        print(
            f"\n[Method 3 Ablation] {index}/{len(configs)} "
            f"{config['name']}  "
            f"classifier={config['classifier']}"
        )

        if config["classifier"] == "logistic":
            probe_factory = lambda: LLMCheckAttentionLogisticProbe(c=args.logistic_c)
        else:
            probe_factory = lambda config=config: LLMCheckAttentionMLPProbe(
                hidden_dims=config["hidden_dims"],
                lr=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.mlp_batch_size,
                dropout_p=config["dropout_p"],
                l2_weight_decay=config["l2_weight_decay"],
            )

        fold_results = run_evaluation(splits, X, y, probe_factory)
        summary = summarize_fold_results(fold_results)
        row = {
            "name": config["name"],
            "classifier": config["classifier"],
            "feature_dim": int(X.shape[1]),
            "hidden_dims": format_hidden_dims(config["hidden_dims"]) if config["hidden_dims"] else "",
            "dropout_p": config["dropout_p"],
            "l2_weight_decay": config["l2_weight_decay"],
            "response_span": "response_only",
            "selected_layer_indices_zero_based": cache["selected_layer_indices_zero_based"].astype(int).tolist(),
            "selected_transformer_layers_1based": cache["selected_transformer_layers_1based"].astype(int).tolist(),
            **summary,
        }
        experiments.append(
            {
                "config": config,
                "row": row,
                "fold_results": fold_results,
            }
        )

    leaderboard = pd.DataFrame([experiment["row"] for experiment in experiments]).sort_values(
        ["mean_val_accuracy", "mean_val_auroc", "mean_test_accuracy"],
        ascending=[False, False, False],
    )
    leaderboard.to_csv(output_dir / "ablation_results.csv", index=False)

    best_experiment = max(experiments, key=lambda item: selection_key(item["row"]))
    best_row = best_experiment["row"]
    payload = {
        "method": "LLM-Check Attention",
        "feature_type": "attention_diagonal_log_score",
        "selection_metric": "mean_val_accuracy",
        "response_span": "response_only",
        "feature_dim": int(X.shape[1]),
        "selected_layer_indices_zero_based": cache["selected_layer_indices_zero_based"].astype(int).tolist(),
        "selected_transformer_layers_1based": cache["selected_transformer_layers_1based"].astype(int).tolist(),
        "configs": leaderboard.to_dict(orient="records"),
        "best_config": best_row,
        "recommended_command": (
            "python method3_llm_check/run_method3.py "
            f"--classifier {best_row['classifier']} "
            + (
                f"--hidden-dims {best_row['hidden_dims']} "
                f"--dropout-p {best_row['dropout_p']} "
                f"--l2-weight-decay {best_row['l2_weight_decay']} "
                if best_row["classifier"] == "mlp"
                else f"--logistic-c {args.logistic_c} "
            )
            + f"--batch-size {args.batch_size} --cache-dtype {args.cache_dtype}"
        ),
    }
    (output_dir / "ablation_results.json").write_text(json.dumps(payload, indent=2))

    print("\n[Method 3 Ablation] Results")
    display_columns = [
        "name",
        "classifier",
        "hidden_dims",
        "dropout_p",
        "l2_weight_decay",
        "mean_val_accuracy",
        "mean_val_auroc",
        "mean_test_accuracy",
        "mean_test_auroc",
    ]
    print(leaderboard[display_columns].to_string(index=False))
    print(f"\n[Method 3 Ablation] Combined JSON saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
