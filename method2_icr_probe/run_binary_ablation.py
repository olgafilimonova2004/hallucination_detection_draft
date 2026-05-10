"""Run a binary-classifier ablation for Method 2 ICR Probe."""

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
    DEFAULT_ABLATION_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BINARY_MLP_HIDDEN_DIMS,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_TOP_K,
    build_feature_matrix,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_hidden_dims,
    summarize_fold_results,
)
from method2_icr_probe.probe import ICRLogisticProbe, ICRMLPProbe
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 2 ICR binary-classifier ablations.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Deprecated compatibility flag. Truncation is disabled and the full prompt+response is used.",
    )
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--disable-z-normalize",
        action="store_true",
        help="Disable the repo-style z-score normalization before softmax.",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--hidden-dims", default=format_hidden_dims(DEFAULT_BINARY_MLP_HIDDEN_DIMS))
    parser.add_argument("--dropout-p", type=float, default=0.3)
    parser.add_argument("--l1-lambda", type=float, default=1e-5)
    parser.add_argument("--l2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=25)
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

    output_dir.mkdir(parents=True, exist_ok=True)
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

    X = build_feature_matrix(cache)
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
            f"\n[Method 2 Binary Ablation] {index}/{len(configs)} "
            f"{config['name']}  "
            f"classifier={config['classifier']}"
        )

        if config["classifier"] == "logistic":
            probe_factory = lambda: ICRLogisticProbe(c=args.logistic_c)
        else:
            probe_factory = lambda config=config: ICRMLPProbe(
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
            "feature_dim": int(X.shape[1]),
            "hidden_dims": format_hidden_dims(config["hidden_dims"]) if config["hidden_dims"] else "",
            "dropout_p": config["dropout_p"],
            "l1_lambda": config["l1_lambda"],
            "l2_weight_decay": config["l2_weight_decay"],
            "regularization": config["regularization"],
            "top_k": args.top_k,
            "z_normalize": z_normalize,
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
            "python method2_icr_probe/run_method2.py "
            f"--classifier logistic --logistic-c {args.logistic_c} "
            f"--top-k {args.top_k} "
            + ("--disable-z-normalize " if not z_normalize else "")
            + f"--batch-size {args.batch_size} --cache-dtype {args.cache_dtype}"
        )
    else:
        recommended_command = (
            "python method2_icr_probe/run_method2.py "
            f"--classifier mlp "
            f"--top-k {args.top_k} "
            + ("--disable-z-normalize " if not z_normalize else "")
            + f"--hidden-dims {format_hidden_dims(best_config['hidden_dims'])} "
            f"--dropout-p {best_config['dropout_p']} "
            f"--l1-lambda {best_config['l1_lambda']} "
            f"--l2-weight-decay {best_config['l2_weight_decay']} "
            f"--batch-size {args.batch_size} "
            f"--cache-dtype {args.cache_dtype}"
        )

    best_payload = {
        "selection_metric": "mean_val_accuracy",
        "feature_dim": int(X.shape[1]),
        "top_k": args.top_k,
        "z_normalize": z_normalize,
        "compared_classifiers": [config["name"] for config in configs],
        "best_config": best_row,
        "recommended_command": recommended_command,
    }
    (output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2))

    print("\n[Method 2 Binary Ablation] Results")
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
    print("\n[Method 2 Binary Ablation] Best configuration summary")
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
    print(f"[Method 2 Binary Ablation] Leaderboard saved to {output_dir / 'ablation_results.csv'}")
    print(f"[Method 2 Binary Ablation] Best config saved to {output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
