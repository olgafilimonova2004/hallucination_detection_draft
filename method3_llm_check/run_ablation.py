"""Run the Method 3 LLM-Check feature-family ablation study."""

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
    DEFAULT_FEATURE_SETS,
    DEFAULT_L2_WEIGHT_DECAY,
    DEFAULT_MLP_HIDDEN_DIMS,
    LLMCheckLogisticProbe,
    LLMCheckMLPProbe,
    build_ablation_configs,
    build_feature_matrix,
    build_feature_set_metadata,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_feature_sets,
    parse_hidden_dims,
    summarize_fold_results,
)
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 3 LLM-Check feature-family ablations.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--feature-sets",
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature families to compare. Example: logit,hidden,attns,logit_hidden_attns",
    )
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


def selection_key(row: dict) -> tuple[float, float, int, int]:
    """Rank ablations by validation accuracy, then AUROC, then simplicity."""
    val_accuracy = row.get("mean_val_accuracy", float("-inf"))
    val_auroc = row.get("mean_val_auroc", float("-inf"))
    if pd.isna(val_accuracy):
        val_accuracy = float("-inf")
    if pd.isna(val_auroc):
        val_auroc = float("-inf")
    is_logistic = int(row["classifier"] == "logistic")
    smaller_feature_set = -int(row["feature_dim"])
    return (
        float(val_accuracy),
        float(val_auroc),
        is_logistic,
        smaller_feature_set,
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
    feature_sets = parse_feature_sets(args.feature_sets)

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

    y = cache["labels"].astype(int)
    splits = split_data(y, df)
    configs = build_ablation_configs(
        feature_sets=feature_sets,
        hidden_dims=hidden_dims,
        dropout_p=args.dropout_p,
        l2_weight_decay=args.l2_weight_decay,
    )

    experiments: list[dict] = []
    for index, config in enumerate(configs, start=1):
        feature_metadata = build_feature_set_metadata(cache, config["feature_set"])
        X = build_feature_matrix(cache, feature_set=config["feature_set"])
        print(
            f"\n[Method 3 Ablation] {index}/{len(configs)} "
            f"{config['name']}  "
            f"classifier={config['classifier']}  "
            f"feature_set={config['feature_set']}  "
            f"feature_dim={X.shape[1]}"
        )

        if config["classifier"] == "logistic":
            probe_factory = lambda: LLMCheckLogisticProbe(c=args.logistic_c)
        else:
            probe_factory = lambda config=config: LLMCheckMLPProbe(
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
            "feature_set": config["feature_set"],
            "description": feature_metadata["description"],
            "components": ",".join(feature_metadata["components"]),
            "component_dims": json.dumps(feature_metadata["component_dims"], sort_keys=True),
            "feature_dim": int(X.shape[1]),
            "classifier": config["classifier"],
            "hidden_dims": format_hidden_dims(config["hidden_dims"]) if config["hidden_dims"] else "",
            "dropout_p": config["dropout_p"],
            "l2_weight_decay": config["l2_weight_decay"],
            "response_span": feature_metadata["response_span"],
            **summary,
        }
        experiments.append(
            {
                "config": config,
                "row": row,
                "feature_metadata": feature_metadata,
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
    best_config = best_experiment["config"]
    best_feature_metadata = best_experiment["feature_metadata"]
    payload = {
        "method": "LLM-Check Feature-Family Ablation",
        "selection_metric": "mean_val_accuracy",
        "data_file": str(data_file),
        "cache_file": str(cache_file),
        "subset": args.subset,
        "response_span": "response_only",
        "feature_sets_compared": feature_sets,
        "feature_set_definitions": {
            feature_set: build_feature_set_metadata(cache, feature_set)
            for feature_set in feature_sets
        },
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
        "configs": leaderboard.to_dict(orient="records"),
        "best_config": {
            **best_row,
            "components_list": best_feature_metadata["components"],
            "feature_metadata": best_feature_metadata,
        },
        "recommended_command": (
            "python method3_llm_check/run_method3.py "
            f"--feature-set {best_config['feature_set']} "
            f"--classifier {best_config['classifier']} "
            + (
                f"--hidden-dims {format_hidden_dims(best_config['hidden_dims'])} "
                f"--dropout-p {best_config['dropout_p']} "
                f"--l2-weight-decay {best_config['l2_weight_decay']} "
                if best_config["classifier"] == "mlp"
                else f"--logistic-c {args.logistic_c} "
            )
            + f"--batch-size {args.batch_size} --cache-dtype {args.cache_dtype}"
        ),
    }
    (output_dir / "ablation_results.json").write_text(json.dumps(payload, indent=2))

    print("\n[Method 3 Ablation] Results")
    display_columns = [
        "name",
        "feature_set",
        "classifier",
        "feature_dim",
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
