"""Run a binary-classifier ablation for Method 1 SAPLMA."""

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
    DEFAULT_ABLATION_DROPOUT_P,
    DEFAULT_ABLATION_L1,
    DEFAULT_ABLATION_L2,
    DEFAULT_BINARY_ABLATION_DIR,
    DEFAULT_BINARY_MLP_HIDDEN_DIMS,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_LAYER_RANKINGS,
    build_feature_matrix,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_hidden_dims,
    resolve_layers,
    summarize_fold_results,
)
from method1_saplma.probe import SAPLMALogisticProbe, SAPLMAProbe
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 1 SAPLMA binary-classifier ablations.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_BINARY_ABLATION_DIR))
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
        help="Hidden-state view fed to the classifier.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Deprecated compatibility flag. Truncation is disabled and the full prompt+response is used.",
    )
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--hidden-dims", default=format_hidden_dims(DEFAULT_BINARY_MLP_HIDDEN_DIMS))
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_ABLATION_DROPOUT_P)
    parser.add_argument("--l1-lambda", type=float, default=DEFAULT_ABLATION_L1)
    parser.add_argument("--l2-weight-decay", type=float, default=DEFAULT_ABLATION_L2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def selection_key(row: dict) -> tuple[float, float, int]:
    """Rank binary-classifier configs by validation accuracy, then AUROC, then simplicity."""
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
    layer_rankings_file = Path(args.layer_rankings_file)

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

    layers = resolve_layers(
        layers_arg=args.layers,
        layer_rankings_file=layer_rankings_file,
        auto_top_k=args.auto_top_k,
    )
    X = build_feature_matrix(cache=cache, token_mode=args.token_mode, layers=layers)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)

    configs = [
        {
            "name": "logistic_regression",
            "classifier": "logistic",
            "hidden_dims": (),
            "dropout_p": 0.0,
            "l1_lambda": 0.0,
            "l2_weight_decay": 0.0,
            "regularization": "none",
        },
        {
            "name": "mlp_2layer_dropout0.3_l2",
            "classifier": "mlp",
            "hidden_dims": hidden_dims,
            "dropout_p": args.dropout_p,
            "l1_lambda": 0.0,
            "l2_weight_decay": args.l2_weight_decay,
            "regularization": "l2",
        },
        {
            "name": "mlp_2layer_dropout0.3_l1_l2",
            "classifier": "mlp",
            "hidden_dims": hidden_dims,
            "dropout_p": args.dropout_p,
            "l1_lambda": args.l1_lambda,
            "l2_weight_decay": args.l2_weight_decay,
            "regularization": "l1+l2",
        },
    ]

    experiments: list[dict] = []
    for index, config in enumerate(configs, start=1):
        print(
            f"\n[Method 1 Binary Ablation] {index}/{len(configs)} "
            f"{config['name']}  "
            f"classifier={config['classifier']}"
        )

        if config["classifier"] == "logistic":
            probe_factory = lambda: SAPLMALogisticProbe(c=args.logistic_c)
        else:
            probe_factory = lambda config=config: SAPLMAProbe(
                hidden_dims=config["hidden_dims"],
                lr=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.mlp_batch_size,
                dropout_p=config["dropout_p"],
                l1_lambda=config["l1_lambda"],
                l2_weight_decay=config["l2_weight_decay"],
            )

        fold_results = run_evaluation(splits, X, y, probe_factory)
        summary = summarize_fold_results(fold_results)
        row = {
            "name": config["name"],
            "classifier": config["classifier"],
            "token_mode": args.token_mode,
            "layers": ",".join(str(layer) for layer in layers),
            "feature_dim": int(X.shape[1]),
            "hidden_dims": format_hidden_dims(config["hidden_dims"]) if config["hidden_dims"] else "",
            "dropout_p": config["dropout_p"],
            "l1_lambda": config["l1_lambda"],
            "l2_weight_decay": config["l2_weight_decay"],
            "regularization": config["regularization"],
            "truncation": "disabled",
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
    (output_dir / "ablation_results.json").write_text(
        json.dumps(leaderboard.to_dict(orient="records"), indent=2)
    )

    best_experiment = max(experiments, key=lambda item: selection_key(item["row"]))
    best_row = best_experiment["row"]
    best_config = best_experiment["config"]
    if best_config["classifier"] == "logistic":
        recommended_command = (
            "python method1_saplma/run_method1.py "
            f"--classifier logistic --logistic-c {args.logistic_c} "
            f"--layers {','.join(str(layer) for layer in layers)} "
            f"--token-mode {args.token_mode} "
            f"--batch-size {args.batch_size} "
            f"--cache-dtype {args.cache_dtype}"
        )
    else:
        recommended_command = (
            "python method1_saplma/run_method1.py "
            f"--classifier mlp "
            f"--layers {','.join(str(layer) for layer in layers)} "
            f"--token-mode {args.token_mode} "
            f"--hidden-dims {format_hidden_dims(best_config['hidden_dims'])} "
            f"--dropout-p {best_config['dropout_p']} "
            f"--l1-lambda {best_config['l1_lambda']} "
            f"--l2-weight-decay {best_config['l2_weight_decay']} "
            f"--batch-size {args.batch_size} "
            f"--cache-dtype {args.cache_dtype}"
        )

    best_payload = {
        "selection_metric": "mean_val_accuracy",
        "layers": layers,
        "token_mode": args.token_mode,
        "feature_dim": int(X.shape[1]),
        "compared_classifiers": [config["name"] for config in configs],
        "best_config": best_row,
        "recommended_command": recommended_command,
    }
    (output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2))

    print("\n[Method 1 Binary Ablation] Results")
    display_columns = [
        "name",
        "classifier",
        "hidden_dims",
        "regularization",
        "mean_val_accuracy",
        "mean_val_auroc",
        "mean_test_accuracy",
        "mean_test_auroc",
    ]
    print(leaderboard[display_columns].to_string(index=False))
    print("\n[Method 1 Binary Ablation] Best configuration summary")
    print(json.dumps(best_payload, indent=2))

    best_results_file = output_dir / "best_results.json"
    print_summary(best_experiment["fold_results"], X.shape[1], len(X), extract_time=0.0)
    save_results(
        best_experiment["fold_results"],
        X.shape[1],
        len(X),
        extract_time=0.0,
        output_file=str(best_results_file),
    )
    print(f"[Method 1 Binary Ablation] Leaderboard saved to {output_dir / 'ablation_results.csv'}")
    print(f"[Method 1 Binary Ablation] Best config saved to {output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
