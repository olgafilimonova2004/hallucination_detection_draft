"""run_ablation.py — sweep SAPLMA regularization and depth variants."""

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
    DEFAULT_ABLATION_DIR,
    DEFAULT_ABLATION_DROPOUT_P,
    DEFAULT_ABLATION_L1,
    DEFAULT_ABLATION_L2,
    DEFAULT_ARCHITECTURES,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_LAYER_RANKINGS,
    build_ablation_configs,
    build_feature_matrix,
    format_hidden_dims,
    load_or_build_cache,
    maybe_take_subset,
    parse_architectures,
    resolve_layers,
    summarize_fold_results,
)
from method1_saplma.probe import SAPLMAProbe
from splitting import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 1 SAPLMA ablations.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_ABLATION_DIR))
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
    parser.add_argument(
        "--architectures",
        default=";".join(format_hidden_dims(hidden_dims) for hidden_dims in DEFAULT_ARCHITECTURES),
        help="Semicolon-separated hidden-layer widths. Example: 256,128,64;256,128;256",
    )
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_ABLATION_DROPOUT_P)
    parser.add_argument("--l1-lambda", type=float, default=DEFAULT_ABLATION_L1)
    parser.add_argument("--l2-weight-decay", type=float, default=DEFAULT_ABLATION_L2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def selection_key(row: dict) -> tuple[float, float, int, int, int]:
    """Select the best configuration by validation performance, then simplicity."""
    val_auroc = row.get("mean_val_auroc", float("-inf"))
    val_f1 = row.get("mean_val_f1", float("-inf"))
    if pd.isna(val_auroc):
        val_auroc = float("-inf")
    if pd.isna(val_f1):
        val_f1 = float("-inf")
    return (
        float(val_auroc),
        float(val_f1),
        -int(row["n_linear_layers"]),
        -int(row["uses_regularization"]),
        -int(row["uses_dropout"]),
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
    architectures = parse_architectures(args.architectures)

    print(f"[Method 1 Ablation] Using token mode: {args.token_mode}")
    print(f"[Method 1 Ablation] Using layers: {layers}")
    print(f"[Method 1 Ablation] Architectures: {architectures}")

    X = build_feature_matrix(cache=cache, token_mode=args.token_mode, layers=layers)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)
    configs = build_ablation_configs(
        architectures=architectures,
        dropout_p=args.dropout_p,
        l1_lambda=args.l1_lambda,
        l2_weight_decay=args.l2_weight_decay,
    )

    experiments: list[dict] = []
    for index, config in enumerate(configs, start=1):
        print(
            f"\n[Method 1 Ablation] {index}/{len(configs)} "
            f"{config['name']}  "
            f"hidden_dims={config['hidden_dims']}  "
            f"dropout={config['dropout_p']}  "
            f"l1={config['l1_lambda']}  "
            f"l2={config['l2_weight_decay']}"
        )

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
            "token_mode": args.token_mode,
            "layers": ",".join(str(layer) for layer in layers),
            "hidden_dims": format_hidden_dims(config["hidden_dims"]),
            "n_linear_layers": config["n_linear_layers"],
            "dropout_p": config["dropout_p"],
            "l1_lambda": config["l1_lambda"],
            "l2_weight_decay": config["l2_weight_decay"],
            "uses_dropout": int(config["uses_dropout"]),
            "uses_regularization": int(config["uses_regularization"]),
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
        ["mean_val_auroc", "mean_val_f1"],
        ascending=[False, False],
    )
    leaderboard.to_csv(output_dir / "ablation_results.csv", index=False)
    (output_dir / "ablation_results.json").write_text(
        json.dumps(leaderboard.to_dict(orient="records"), indent=2)
    )

    best_experiment = max(experiments, key=lambda item: selection_key(item["row"]))
    best_row = best_experiment["row"]
    best_config = best_experiment["config"]
    best_command = (
        "python method1_saplma/run_method1.py "
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
        "selection_metric": "mean_val_auroc",
        "layers": layers,
        "token_mode": args.token_mode,
        "architectures_searched": [list(hidden_dims) for hidden_dims in architectures],
        "dropout_p_tested": args.dropout_p,
        "l1_lambda_tested": args.l1_lambda,
        "l2_weight_decay_tested": args.l2_weight_decay,
        "best_config": {
            **best_row,
            "hidden_dims_list": list(best_config["hidden_dims"]),
        },
        "recommended_command": best_command,
    }
    (output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2))

    print("\n[Method 1 Ablation] Top configurations")
    display_columns = [
        "name",
        "hidden_dims",
        "dropout_p",
        "l1_lambda",
        "l2_weight_decay",
        "mean_val_auroc",
        "mean_test_auroc",
    ]
    print(leaderboard[display_columns].head(10).to_string(index=False))

    print("\n[Method 1 Ablation] Best configuration summary")
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
    print(f"[Method 1 Ablation] Leaderboard saved to {output_dir / 'ablation_results.csv'}")
    print(f"[Method 1 Ablation] Best config saved to {output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
