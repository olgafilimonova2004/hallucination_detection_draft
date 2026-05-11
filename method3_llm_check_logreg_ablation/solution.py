"""Final Method 3 logistic-regression runner built on the ablation winner."""

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
from method3_llm_check_logreg_ablation.aggregation import (
    DEFAULT_ABLATION_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_FEATURE_SET,
    DEFAULT_OUTPUT_FILE,
    build_feature_matrix,
    build_feature_set_metadata,
    load_or_build_cache,
    maybe_take_subset,
)
from method3_llm_check_logreg_ablation.probe import (
    DEFAULT_CLASS_WEIGHT_TOKENS,
    HallucinationLogisticProbe,
    parse_class_weights,
    probe_config_from_record,
)
from method3_llm_check_logreg_ablation.run_ablation import run_ablation_study
from method3_llm_check_logreg_ablation.splitting import split_data


DEFAULT_BEST_CONFIG_FILE = DEFAULT_ABLATION_DIR / "best_config.json"
DEFAULT_METADATA_FILE = (
    ROOT / "method3_llm_check_logreg_ablation" / "artifacts" / "results_metadata.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final Method 3 logistic-regression configuration on dataset.csv."
    )
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--metadata-file", default=str(DEFAULT_METADATA_FILE))
    parser.add_argument("--best-config-file", default=str(DEFAULT_BEST_CONFIG_FILE))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Compatibility flag only. Truncation is disabled and the full prompt+response is used.",
    )
    parser.add_argument(
        "--primary-feature-set",
        default=DEFAULT_FEATURE_SET,
        help="Primary feature set to search if best_config.json must be rebuilt.",
    )
    parser.add_argument(
        "--secondary-feature-sets",
        default="",
        help="Optional secondary feature sets to include when rebuilding best_config.json.",
    )
    parser.add_argument(
        "--secondary-top-k",
        type=int,
        default=0,
        help="How many top primary configs to replay on secondary feature sets when rebuilding.",
    )
    parser.add_argument(
        "--class-weights",
        default=",".join(DEFAULT_CLASS_WEIGHT_TOKENS),
        help="Comma-separated class-weight modes to compare if best_config.json must be rebuilt.",
    )
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--refresh-ablation",
        action="store_true",
        help="Rebuild the best_config.json selection before running the final evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    cache_file = Path(args.cache_file)
    output_file = Path(args.output_file)
    metadata_file = Path(args.metadata_file)
    best_config_file = Path(args.best_config_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df = maybe_take_subset(df, args.subset)

    should_refresh_ablation = (
        args.refresh_ablation or args.subset is not None or not best_config_file.exists()
    )
    if should_refresh_ablation:
        secondary_feature_sets = []
        if args.secondary_feature_sets.strip():
            from method3_llm_check_logreg_ablation.aggregation import parse_feature_sets

            secondary_feature_sets = parse_feature_sets(args.secondary_feature_sets)
        class_weights = parse_class_weights(args.class_weights)
        run_ablation_study(
            data_file=data_file,
            cache_file=cache_file,
            output_dir=best_config_file.parent,
            batch_size=args.batch_size,
            max_length=args.max_length,
            primary_feature_set=args.primary_feature_set,
            secondary_feature_sets=secondary_feature_sets,
            secondary_top_k=max(args.secondary_top_k, 0),
            class_weights=class_weights,
            subset=args.subset,
            overwrite_cache=args.overwrite_cache,
        )

    best_payload = json.loads(best_config_file.read_text())
    best_record = best_payload["best_config"]
    probe_config = probe_config_from_record(best_record)

    cache = load_or_build_cache(
        df=df,
        cache_file=cache_file,
        data_file=data_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        overwrite_cache=args.overwrite_cache,
        subset_size=args.subset,
    )
    feature_set = str(best_record["feature_set"])
    feature_metadata = build_feature_set_metadata(cache, feature_set)
    X = build_feature_matrix(cache, feature_set=feature_set)
    y = cache["labels"].astype(int)
    splits = split_data(y, df)

    probe_factory = lambda: HallucinationLogisticProbe(config=probe_config)
    fold_results = run_evaluation(splits, X, y, probe_factory)
    print_summary(fold_results, X.shape[1], len(X), extract_time=0.0)
    save_results(
        fold_results,
        X.shape[1],
        len(X),
        extract_time=0.0,
        output_file=str(output_file),
    )

    metadata = {
        "method": "LLM-Check Logistic-Regularization Final Runner",
        "data_file": str(data_file),
        "cache_file": str(cache_file),
        "output_file": str(output_file),
        "best_config_file": str(best_config_file),
        "subset": args.subset,
        "batch_size": args.batch_size,
        "max_length_requested": args.max_length,
        "cache_dtype": "float32",
        "model_forward_dtype": "float32",
        "selected_probe_config": best_record,
        "feature_metadata": feature_metadata,
        "selection_payload": best_payload,
        "feature_dim": int(X.shape[1]),
        "n_samples": int(len(X)),
        "n_folds": int(len(splits)),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"[Method 3 LogReg] Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
