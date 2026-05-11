"""Run the Method 3 logistic-regression regularization ablation study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import print_summary, run_evaluation, save_results
from method3_llm_check_logreg_ablation.aggregation import (
    DEFAULT_ABLATION_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_FEATURE_SET,
    DEFAULT_FEATURE_SETS,
    build_feature_matrix,
    build_feature_set_metadata,
    load_or_build_cache,
    maybe_take_subset,
    parse_feature_sets,
)
from method3_llm_check_logreg_ablation.probe import (
    DEFAULT_CLASS_WEIGHT_TOKENS,
    HallucinationLogisticProbe,
    LogisticProbeConfig,
    build_primary_logistic_configs,
    format_class_weight,
    format_probe_name,
    parse_class_weights,
    probe_config_from_record,
    probe_config_to_metadata,
    probe_signature,
    regularization_sort_fields,
    simplicity_sort_fields,
)
from method3_llm_check_logreg_ablation.splitting import split_data


HISTORICAL_ABLATION_FILE = ROOT / "method3_llm_check" / "ablation_results.csv"
DEFAULT_NEAR_TIE_ACCURACY_TOL = 0.005
DEFAULT_NEAR_TIE_AUROC_TOL = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Method 3 logistic-regression regularization ablation."
    )
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Compatibility flag only. Truncation is disabled and the full prompt+response is used.",
    )
    parser.add_argument(
        "--primary-feature-set",
        choices=DEFAULT_FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
    parser.add_argument(
        "--secondary-feature-sets",
        default="",
        help=(
            "Optional comma-separated extra feature sets. "
            "Only the top-k primary logistic configs are replayed on these sets."
        ),
    )
    parser.add_argument(
        "--secondary-top-k",
        type=int,
        default=0,
        help="How many top primary logistic configs to replay on the secondary feature sets.",
    )
    parser.add_argument(
        "--class-weights",
        default=",".join(DEFAULT_CLASS_WEIGHT_TOKENS),
        help="Comma-separated class-weight modes to compare. Supported: balanced, none.",
    )
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--near-tie-accuracy-tol",
        type=float,
        default=DEFAULT_NEAR_TIE_ACCURACY_TOL,
        help="Validation-accuracy tolerance used to define near-tied finalists.",
    )
    parser.add_argument(
        "--near-tie-auroc-tol",
        type=float,
        default=DEFAULT_NEAR_TIE_AUROC_TOL,
        help="Validation-AUROC tolerance used to define near-tied finalists.",
    )
    return parser.parse_args()


def summarize_fold_results(fold_results: list[dict]) -> dict[str, float]:
    """Compute cross-fold mean metrics for the leaderboard."""
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


def derive_gap_metrics(summary: dict[str, float]) -> dict[str, float]:
    """Compute explicit overfitting-gap diagnostics."""
    return {
        "train_val_accuracy_gap": float(
            summary["mean_train_accuracy"] - summary["mean_val_accuracy"]
        ),
        "train_test_accuracy_gap": float(
            summary["mean_train_accuracy"] - summary["mean_test_accuracy"]
        ),
        "train_val_auroc_gap": float(
            summary["mean_train_auroc"] - summary["mean_val_auroc"]
        ),
        "train_test_auroc_gap": float(
            summary["mean_train_auroc"] - summary["mean_test_auroc"]
        ),
    }


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records."""
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def build_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort the leaderboard by validation metrics, regularization, and simplicity."""
    return frame.sort_values(
        [
            "mean_val_accuracy",
            "mean_val_auroc",
            "regularization_c_rank",
            "regularization_l1_ratio_rank",
            "simplicity_penalty_rank",
            "simplicity_solver_rank",
            "weighting_rank",
            "feature_dim",
            "abs_train_val_accuracy_gap",
            "abs_train_val_auroc_gap",
        ],
        ascending=[False, False, True, True, True, True, True, True, True, True],
    ).reset_index(drop=True)


def select_top_primary_probe_configs(
    primary_leaderboard: pd.DataFrame,
    top_k: int,
) -> list[LogisticProbeConfig]:
    """Select the top-k unique primary probe configs for the secondary study."""
    selected: list[LogisticProbeConfig] = []
    seen: set[tuple[object, ...]] = set()
    for record in dataframe_records(primary_leaderboard):
        config = probe_config_from_record(record)
        signature = probe_signature(config)
        if signature in seen:
            continue
        selected.append(config)
        seen.add(signature)
        if len(selected) >= top_k:
            break
    return selected


def compare_rows_for_stability(
    selected_row: dict,
    reference_row: dict,
) -> dict[str, object]:
    """Compare the selected config against a reference MLP row."""
    return {
        "reference_name": reference_row["name"],
        "reference_feature_set": reference_row["feature_set"],
        "delta_mean_val_accuracy": float(
            selected_row["mean_val_accuracy"] - reference_row["mean_val_accuracy"]
        ),
        "delta_mean_test_accuracy": float(
            selected_row["mean_test_accuracy"] - reference_row["mean_test_accuracy"]
        ),
        "delta_train_val_accuracy_gap": float(
            abs(selected_row["train_val_accuracy_gap"])
            - abs(reference_row["train_val_accuracy_gap"])
        ),
        "delta_train_test_accuracy_gap": float(
            abs(selected_row["train_test_accuracy_gap"])
            - abs(reference_row["train_test_accuracy_gap"])
        ),
        "delta_mean_val_auroc": float(
            selected_row["mean_val_auroc"] - reference_row["mean_val_auroc"]
        ),
        "delta_mean_test_auroc": float(
            selected_row["mean_test_auroc"] - reference_row["mean_test_auroc"]
        ),
        "delta_train_val_auroc_gap": float(
            abs(selected_row["train_val_auroc_gap"])
            - abs(reference_row["train_val_auroc_gap"])
        ),
        "delta_train_test_auroc_gap": float(
            abs(selected_row["train_test_auroc_gap"])
            - abs(reference_row["train_test_auroc_gap"])
        ),
        "selected_more_stable_on_accuracy_gap": bool(
            abs(selected_row["train_val_accuracy_gap"])
            <= abs(reference_row["train_val_accuracy_gap"])
        ),
        "selected_more_stable_on_auroc_gap": bool(
            abs(selected_row["train_val_auroc_gap"])
            <= abs(reference_row["train_val_auroc_gap"])
        ),
    }


def load_historical_method3_comparison(best_row: dict) -> dict[str, object] | None:
    """Compare the chosen logistic config against the existing Method 3 MLP rows."""
    if not HISTORICAL_ABLATION_FILE.exists():
        return None

    historical = pd.read_csv(HISTORICAL_ABLATION_FILE)
    if historical.empty or "classifier" not in historical.columns:
        return None

    mlp_rows = historical[historical["classifier"] == "mlp"].copy()
    if mlp_rows.empty:
        return None

    mlp_rows["train_val_accuracy_gap"] = (
        mlp_rows["mean_train_accuracy"] - mlp_rows["mean_val_accuracy"]
    )
    mlp_rows["train_test_accuracy_gap"] = (
        mlp_rows["mean_train_accuracy"] - mlp_rows["mean_test_accuracy"]
    )
    mlp_rows["train_val_auroc_gap"] = (
        mlp_rows["mean_train_auroc"] - mlp_rows["mean_val_auroc"]
    )
    mlp_rows["train_test_auroc_gap"] = (
        mlp_rows["mean_train_auroc"] - mlp_rows["mean_test_auroc"]
    )
    mlp_rows = mlp_rows.sort_values(
        ["mean_val_accuracy", "mean_val_auroc"],
        ascending=[False, False],
    )

    best_mlp = dataframe_records(mlp_rows.head(1))[0]
    same_feature_rows = mlp_rows[mlp_rows["feature_set"] == best_row["feature_set"]]
    same_feature_mlp = (
        dataframe_records(same_feature_rows.head(1))[0] if not same_feature_rows.empty else None
    )

    comparison = {
        "historical_file": str(HISTORICAL_ABLATION_FILE),
        "historical_best_mlp": best_mlp,
        "comparison_to_historical_best_mlp": compare_rows_for_stability(best_row, best_mlp),
    }
    if same_feature_mlp is not None:
        comparison["historical_same_feature_mlp"] = same_feature_mlp
        comparison["comparison_to_historical_same_feature_mlp"] = compare_rows_for_stability(
            best_row,
            same_feature_mlp,
        )
    return comparison


def run_feature_set_configs(
    *,
    cache: dict[str, np.ndarray],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]],
    feature_set: str,
    probe_configs: list[LogisticProbeConfig],
    study_phase: str,
    feature_set_index: int,
    feature_set_total: int,
) -> list[dict]:
    """Run all requested probe configs on one feature set."""
    X = build_feature_matrix(cache, feature_set=feature_set)
    feature_metadata = build_feature_set_metadata(cache, feature_set)
    experiments: list[dict] = []

    for config_idx, config in enumerate(probe_configs, start=1):
        config_name = format_probe_name(feature_set, config)
        print(
            f"\n[Method 3 LogReg Ablation] feature_set={feature_set} "
            f"({feature_set_index}/{feature_set_total}) "
            f"config={config_idx}/{len(probe_configs)} "
            f"{config_name}"
        )
        print(
            f"  penalty={config.penalty} solver={config.solver} "
            f"C={config.c} l1_ratio={config.l1_ratio} "
            f"class_weight={format_class_weight(config.class_weight)}"
        )

        probe_factory = lambda config=config: HallucinationLogisticProbe(config=config)
        fold_results = run_evaluation(splits, X, y, probe_factory)
        summary = summarize_fold_results(fold_results)
        gap_metrics = derive_gap_metrics(summary)
        row = {
            "name": config_name,
            "study_phase": study_phase,
            "feature_set": feature_set,
            "description": feature_metadata["description"],
            "components": ",".join(feature_metadata["components"]),
            "component_dims": json.dumps(
                feature_metadata["component_dims"],
                sort_keys=True,
            ),
            "feature_dim": int(X.shape[1]),
            "response_span": feature_metadata["response_span"],
            "truncation": feature_metadata["truncation"],
            **probe_config_to_metadata(config),
            **regularization_sort_fields(config),
            **simplicity_sort_fields(config),
            **summary,
            **gap_metrics,
            "abs_train_val_accuracy_gap": abs(gap_metrics["train_val_accuracy_gap"]),
            "abs_train_test_accuracy_gap": abs(gap_metrics["train_test_accuracy_gap"]),
            "abs_train_val_auroc_gap": abs(gap_metrics["train_val_auroc_gap"]),
            "abs_train_test_auroc_gap": abs(gap_metrics["train_test_auroc_gap"]),
        }
        experiments.append(
            {
                "probe_config": config,
                "row": row,
                "feature_metadata": feature_metadata,
                "fold_results": fold_results,
            }
        )

    return experiments


def select_best_experiment(
    experiments: list[dict],
    near_tie_accuracy_tol: float,
    near_tie_auroc_tol: float,
) -> tuple[pd.DataFrame, dict, list[dict], dict]:
    """Build the final leaderboard and select the best near-tied config."""
    leaderboard = build_leaderboard(pd.DataFrame([item["row"] for item in experiments]))
    top_row = leaderboard.iloc[0]
    top_val_accuracy = float(top_row["mean_val_accuracy"])
    top_val_auroc = float(top_row["mean_val_auroc"])

    finalists = leaderboard[
        leaderboard["mean_val_accuracy"] >= top_val_accuracy - near_tie_accuracy_tol
    ].copy()
    if not math.isnan(top_val_auroc):
        finalists = finalists[
            finalists["mean_val_auroc"] >= top_val_auroc - near_tie_auroc_tol
        ].copy()
    if finalists.empty:
        finalists = leaderboard.head(1).copy()

    finalists = finalists.sort_values(
        [
            "abs_train_val_accuracy_gap",
            "abs_train_val_auroc_gap",
            "regularization_c_rank",
            "regularization_l1_ratio_rank",
            "simplicity_penalty_rank",
            "simplicity_solver_rank",
            "weighting_rank",
            "mean_val_accuracy",
            "mean_val_auroc",
        ],
        ascending=[True, True, True, True, True, True, True, False, False],
    ).reset_index(drop=True)

    best_record = dataframe_records(finalists.head(1))[0]
    experiment_lookup = {item["row"]["name"]: item for item in experiments}
    best_experiment = experiment_lookup[best_record["name"]]
    near_tie_records = dataframe_records(finalists)

    selection_rule = {
        "primary_metric": "mean_val_accuracy",
        "leaderboard_sort": [
            "mean_val_accuracy desc",
            "mean_val_auroc desc",
            "stronger regularization first",
            "simpler penalty/solver first",
        ],
        "near_tie_accuracy_tolerance": near_tie_accuracy_tol,
        "near_tie_auroc_tolerance": near_tie_auroc_tol,
        "near_tie_preference": [
            "smaller train/validation generalization gap",
            "stronger regularization",
            "simpler penalty/solver combination",
        ],
    }
    return leaderboard, best_experiment, near_tie_records, selection_rule


def run_ablation_study(
    *,
    data_file: Path,
    cache_file: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    primary_feature_set: str = DEFAULT_FEATURE_SET,
    secondary_feature_sets: list[str] | None = None,
    secondary_top_k: int = 0,
    class_weights: list[str | None] | None = None,
    subset: int | None = None,
    overwrite_cache: bool = False,
    near_tie_accuracy_tol: float = DEFAULT_NEAR_TIE_ACCURACY_TOL,
    near_tie_auroc_tol: float = DEFAULT_NEAR_TIE_AUROC_TOL,
) -> dict[str, object]:
    """Run the full logistic-regression ablation and persist its artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if secondary_feature_sets is None:
        secondary_feature_sets = []
    if class_weights is None:
        class_weights = parse_class_weights(",".join(DEFAULT_CLASS_WEIGHT_TOKENS))

    df = pd.read_csv(data_file)
    df = maybe_take_subset(df, subset)
    cache = load_or_build_cache(
        df=df,
        cache_file=cache_file,
        data_file=data_file,
        batch_size=batch_size,
        max_length=max_length,
        overwrite_cache=overwrite_cache,
        subset_size=subset,
    )

    y = cache["labels"].astype(int)
    splits = split_data(y, df)
    primary_probe_configs = build_primary_logistic_configs(class_weights=class_weights)

    primary_experiments = run_feature_set_configs(
        cache=cache,
        y=y,
        splits=splits,
        feature_set=primary_feature_set,
        probe_configs=primary_probe_configs,
        study_phase="primary",
        feature_set_index=1,
        feature_set_total=1 + len(secondary_feature_sets),
    )
    experiments = list(primary_experiments)

    if secondary_feature_sets and secondary_top_k > 0:
        primary_leaderboard = build_leaderboard(
            pd.DataFrame([item["row"] for item in primary_experiments])
        )
        secondary_probe_configs = select_top_primary_probe_configs(
            primary_leaderboard=primary_leaderboard,
            top_k=secondary_top_k,
        )
        for offset, feature_set in enumerate(secondary_feature_sets, start=2):
            experiments.extend(
                run_feature_set_configs(
                    cache=cache,
                    y=y,
                    splits=splits,
                    feature_set=feature_set,
                    probe_configs=secondary_probe_configs,
                    study_phase="secondary",
                    feature_set_index=offset,
                    feature_set_total=1 + len(secondary_feature_sets),
                )
            )

    leaderboard, best_experiment, near_tie_records, selection_rule = select_best_experiment(
        experiments=experiments,
        near_tie_accuracy_tol=near_tie_accuracy_tol,
        near_tie_auroc_tol=near_tie_auroc_tol,
    )
    leaderboard_records = dataframe_records(leaderboard)
    best_record = leaderboard_records[
        next(
            idx
            for idx, record in enumerate(leaderboard_records)
            if record["name"] == best_experiment["row"]["name"]
        )
    ]

    historical_comparison = load_historical_method3_comparison(best_record)
    best_payload = {
        "method": "LLM-Check Logistic-Regularization Ablation",
        "selection_metric": "mean_val_accuracy",
        "selection_rule": selection_rule,
        "data_file": str(data_file),
        "cache_file": str(cache_file),
        "subset": subset,
        "batch_size": batch_size,
        "max_length_requested": max_length,
        "cache_dtype": "float32",
        "model_forward_dtype": "float32",
        "primary_feature_set": primary_feature_set,
        "secondary_feature_sets": secondary_feature_sets,
        "secondary_top_k": secondary_top_k,
        "class_weights_tested": [format_class_weight(value) for value in class_weights],
        "best_config": best_record,
        "best_feature_metadata": best_experiment["feature_metadata"],
        "near_tie_candidates": near_tie_records,
        "historical_method3_mlp_comparison": historical_comparison,
        "recommended_command": (
            "python method3_llm_check_logreg_ablation/solution.py "
            f"--best-config-file {output_dir / 'best_config.json'} "
            f"--cache-file {cache_file} --data-file {data_file}"
        ),
    }

    ablation_payload = {
        "method": "LLM-Check Logistic-Regularization Ablation",
        "description": (
            "Logistic-only continuation of Method 3 with float32 feature caching, "
            "full prompt+response tokenization, truncation disabled, and "
            "response-only scoring."
        ),
        "data_file": str(data_file),
        "cache_file": str(cache_file),
        "subset": subset,
        "batch_size": batch_size,
        "max_length_requested": max_length,
        "cache_dtype": "float32",
        "model_forward_dtype": "float32",
        "primary_feature_set": primary_feature_set,
        "secondary_feature_sets": secondary_feature_sets,
        "secondary_top_k": secondary_top_k,
        "class_weights_tested": [format_class_weight(value) for value in class_weights],
        "feature_set_definitions": {
            feature_set: build_feature_set_metadata(cache, feature_set)
            for feature_set in [primary_feature_set, *secondary_feature_sets]
        },
        "selection_rule": selection_rule,
        "historical_method3_mlp_comparison": historical_comparison,
        "experiments": leaderboard_records,
        "best_config": best_record,
    }

    leaderboard.to_csv(output_dir / "ablation_results.csv", index=False)
    (output_dir / "ablation_results.json").write_text(
        json.dumps(ablation_payload, indent=2)
    )
    (output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2))

    print("\n[Method 3 LogReg Ablation] Top configurations")
    display_columns = [
        "name",
        "study_phase",
        "feature_set",
        "penalty",
        "solver",
        "C",
        "l1_ratio",
        "class_weight",
        "mean_val_accuracy",
        "mean_val_auroc",
        "mean_test_accuracy",
        "mean_test_auroc",
        "train_val_accuracy_gap",
        "train_val_auroc_gap",
    ]
    print(leaderboard[display_columns].head(15).to_string(index=False))

    if historical_comparison is not None:
        best_mlp_name = historical_comparison["historical_best_mlp"]["name"]
        mlp_gap_delta = historical_comparison["comparison_to_historical_best_mlp"][
            "delta_train_val_accuracy_gap"
        ]
        print(
            "\n[Method 3 LogReg Ablation] Historical Method 3 comparison"
            f"\n  reference MLP: {best_mlp_name}"
            f"\n  selected config train-val accuracy gap delta vs MLP: {mlp_gap_delta:+.6f}"
        )

    best_results_file = output_dir / "best_results.json"
    print("\n[Method 3 LogReg Ablation] Best configuration summary")
    print(json.dumps(best_payload, indent=2))
    print_summary(
        best_experiment["fold_results"],
        int(best_experiment["row"]["feature_dim"]),
        len(y),
        extract_time=0.0,
    )
    save_results(
        best_experiment["fold_results"],
        int(best_experiment["row"]["feature_dim"]),
        len(y),
        extract_time=0.0,
        output_file=str(best_results_file),
    )
    print(
        f"[Method 3 LogReg Ablation] Leaderboard saved to "
        f"{output_dir / 'ablation_results.csv'}"
    )
    print(
        f"[Method 3 LogReg Ablation] Best config saved to "
        f"{output_dir / 'best_config.json'}"
    )
    return best_payload


def main() -> None:
    args = parse_args()
    secondary_feature_sets = (
        parse_feature_sets(args.secondary_feature_sets)
        if args.secondary_feature_sets.strip()
        else []
    )
    if args.primary_feature_set in secondary_feature_sets:
        secondary_feature_sets = [
            feature_set
            for feature_set in secondary_feature_sets
            if feature_set != args.primary_feature_set
        ]
    class_weights = parse_class_weights(args.class_weights)
    run_ablation_study(
        data_file=Path(args.data_file),
        cache_file=Path(args.cache_file),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        primary_feature_set=args.primary_feature_set,
        secondary_feature_sets=secondary_feature_sets,
        secondary_top_k=max(args.secondary_top_k, 0),
        class_weights=class_weights,
        subset=args.subset,
        overwrite_cache=args.overwrite_cache,
        near_tie_accuracy_tol=args.near_tie_accuracy_tol,
        near_tie_auroc_tol=args.near_tie_auroc_tol,
    )


if __name__ == "__main__":
    main()
