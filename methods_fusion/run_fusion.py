"""Run late-fusion experiments over SAPLMA, ICR Probe, and LLM-Check."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method1_saplma.common import (
    DEFAULT_CACHE_FILE as DEFAULT_SAPLMA_CACHE_FILE,
    DEFAULT_DATA_FILE,
    DEFAULT_LAYER_RANKINGS,
    build_feature_matrix as build_saplma_feature_matrix,
    load_or_build_cache as load_or_build_saplma_cache,
    maybe_take_subset,
    resolve_layers,
)
from method2_icr_probe.common import (
    DEFAULT_CACHE_FILE as DEFAULT_ICR_CACHE_FILE,
    DEFAULT_TOP_K,
    build_feature_matrix as build_icr_feature_matrix,
    load_or_build_cache as load_or_build_icr_cache,
)
from method3_llm_check.common import (
    DEFAULT_CACHE_FILE as DEFAULT_LLM_CHECK_CACHE_FILE,
    build_feature_matrix as build_llm_check_feature_matrix,
    load_or_build_cache as load_or_build_llm_check_cache,
)
from methods_fusion.common import (
    BRANCH_ORDER,
    FUSION_EXPERIMENTS,
    FusionMLPProbe,
    build_branch_metadata,
    extract_branch_embedding,
    format_hidden_dims,
    parse_hidden_dims,
    summarize_fold_results,
    train_branch_probe,
)
from splitting import split_data


DEFAULT_OUTPUT_FILE = ROOT / "methods_fusion" / "artifacts" / "fusion_results.json"
DEFAULT_FUSION_HIDDEN_DIMS = (64, 32)


def _nanmean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run late-fusion experiments over Methods 1, 2, and 3.")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))

    parser.add_argument("--saplma-cache-file", default=str(DEFAULT_SAPLMA_CACHE_FILE))
    parser.add_argument("--icr-cache-file", default=str(DEFAULT_ICR_CACHE_FILE))
    parser.add_argument("--llm-check-cache-file", default=str(DEFAULT_LLM_CHECK_CACHE_FILE))
    parser.add_argument("--layer-rankings-file", default=str(DEFAULT_LAYER_RANKINGS))

    parser.add_argument("--saplma-layers", default="auto")
    parser.add_argument("--saplma-auto-top-k", type=int, default=1)
    parser.add_argument(
        "--saplma-token-mode",
        choices=("response_last", "last_token", "response_mean"),
        default="response_last",
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--saplma-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--saplma-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--reduced-cache-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--icr-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--disable-icr-z-normalize", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")

    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--saplma-epochs", type=int, default=5)
    parser.add_argument("--branch-epochs", type=int, default=25)
    parser.add_argument("--branch-batch-size", type=int, default=32)
    parser.add_argument("--branch-learning-rate", type=float, default=1e-3)

    parser.add_argument("--fusion-hidden-dims", default=format_hidden_dims(DEFAULT_FUSION_HIDDEN_DIMS))
    parser.add_argument("--fusion-epochs", type=int, default=25)
    parser.add_argument("--fusion-batch-size", type=int, default=32)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fusion-dropout-p", type=float, default=0.3)
    parser.add_argument("--fusion-l2-weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def evaluate_fusion_fold(
    X_fusion: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray | None,
    idx_test: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Train and evaluate the final fusion MLP for one fold."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_fusion[idx_train], y[idx_train])
    y_dummy = dummy.predict(X_fusion[idx_test])

    probe = FusionMLPProbe(
        hidden_dims=parse_hidden_dims(args.fusion_hidden_dims),
        lr=args.fusion_learning_rate,
        epochs=args.fusion_epochs,
        batch_size=args.fusion_batch_size,
        dropout_p=args.fusion_dropout_p,
        l2_weight_decay=args.fusion_l2_weight_decay,
        random_state=args.random_state,
    )
    probe.fit(X_fusion[idx_train], y[idx_train])
    if idx_val is not None:
        probe.fit_hyperparameters(X_fusion[idx_val], y[idx_val])

    metrics = {
        "baseline_accuracy": accuracy_score(y[idx_test], y_dummy),
        "baseline_f1": f1_score(y[idx_test], y_dummy, zero_division=0),
    }

    for split_name, idx_split in [
        ("train", idx_train),
        ("val", idx_val),
        ("test", idx_test),
    ]:
        if idx_split is None:
            continue
        y_true = y[idx_split]
        y_pred = probe.predict(X_fusion[idx_split])
        y_prob = probe.predict_proba(X_fusion[idx_split])[:, 1]
        metrics[f"{split_name}_accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{split_name}_f1"] = f1_score(y_true, y_pred, zero_division=0)
        try:
            metrics[f"{split_name}_auroc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics[f"{split_name}_auroc"] = float("nan")

    return metrics


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    df = maybe_take_subset(df, args.subset)

    saplma_cache_file = Path(args.saplma_cache_file)
    icr_cache_file = Path(args.icr_cache_file)
    llm_check_cache_file = Path(args.llm_check_cache_file)
    for cache_file in [saplma_cache_file, icr_cache_file, llm_check_cache_file]:
        cache_file.parent.mkdir(parents=True, exist_ok=True)

    saplma_layers = resolve_layers(
        layers_arg=args.saplma_layers,
        layer_rankings_file=Path(args.layer_rankings_file),
        auto_top_k=args.saplma_auto_top_k,
    )
    icr_z_normalize = not args.disable_icr_z_normalize

    print("[Fusion] Loading or building SAPLMA cache")
    saplma_cache = load_or_build_saplma_cache(
        df=df,
        cache_file=saplma_cache_file,
        data_file=data_file,
        batch_size=args.saplma_batch_size,
        max_length=args.max_length,
        cache_dtype=args.saplma_cache_dtype,
        overwrite_cache=args.overwrite_cache,
        subset_size=args.subset,
    )

    print("[Fusion] Loading or building ICR cache")
    icr_cache = load_or_build_icr_cache(
        df=df,
        cache_file=icr_cache_file,
        data_file=data_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.icr_top_k,
        z_normalize=icr_z_normalize,
        cache_dtype=args.reduced_cache_dtype,
        overwrite_cache=args.overwrite_cache,
        subset_size=args.subset,
    )

    print("[Fusion] Loading or building LLM-Check cache")
    llm_check_cache = load_or_build_llm_check_cache(
        df=df,
        cache_file=llm_check_cache_file,
        data_file=data_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dtype=args.reduced_cache_dtype,
        overwrite_cache=args.overwrite_cache,
        subset_size=args.subset,
    )

    raw_features = {
        "saplma": build_saplma_feature_matrix(
            cache=saplma_cache,
            token_mode=args.saplma_token_mode,
            layers=saplma_layers,
        ),
        "icr": build_icr_feature_matrix(icr_cache),
        "llm_check": build_llm_check_feature_matrix(llm_check_cache),
    }
    y = saplma_cache["labels"].astype(int)

    if not np.array_equal(y, icr_cache["labels"].astype(int)):
        raise ValueError("SAPLMA and ICR cache labels are not aligned.")
    if not np.array_equal(y, llm_check_cache["labels"].astype(int)):
        raise ValueError("SAPLMA and LLM-Check cache labels are not aligned.")

    raw_dims = {name: int(features.shape[1]) for name, features in raw_features.items()}
    branch_metadata = build_branch_metadata(raw_dims)
    splits = split_data(y, df)

    print("[Fusion] Raw feature dimensions")
    for name, dim in raw_dims.items():
        print(f"  {name}: {dim}")

    print("\n[Fusion] Training and freezing standalone branches once per fold")
    frozen_fold_embeddings: list[dict[str, np.ndarray]] = []
    for fold_idx, (idx_train, _idx_val, _idx_test) in enumerate(splits, start=1):
        print(f"[Fusion] Branch fold {fold_idx}/{len(splits)}")
        branch_embeddings: dict[str, np.ndarray] = {}
        for branch in BRANCH_ORDER:
            probe = train_branch_probe(
                branch=branch,
                X_train=raw_features[branch][idx_train],
                y_train=y[idx_train],
                args=args,
            )
            branch_embeddings[branch] = extract_branch_embedding(
                branch=branch,
                probe=probe,
                X=raw_features[branch],
            )
            print(f"  {branch}: embedding_dim={branch_embeddings[branch].shape[1]}")
        frozen_fold_embeddings.append(branch_embeddings)

    experiments: list[dict] = []
    for experiment_name, branches in FUSION_EXPERIMENTS.items():
        print(f"\n[Fusion] Experiment: {experiment_name} branches={branches}")
        fold_results: list[dict] = []

        for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
            print(
                f"[Fusion] Fold {fold_idx}/{len(splits)} "
                f"train={len(idx_train)} val={len(idx_val) if idx_val is not None else 0} test={len(idx_test)}"
            )
            fold_embeddings = frozen_fold_embeddings[fold_idx - 1]
            X_fusion = np.concatenate([fold_embeddings[branch] for branch in branches], axis=1)
            contribution_details = {
                branch: {
                    "embedding_dim": int(fold_embeddings[branch].shape[1]),
                    "fusion_vector": branch_metadata[branch].fusion_vector,
                }
                for branch in branches
            }
            metrics = evaluate_fusion_fold(
                X_fusion=X_fusion,
                y=y,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                args=args,
            )
            fold_results.append(
                {
                    "fold": fold_idx,
                    "n_train": int(len(idx_train)),
                    "n_val": int(len(idx_val)) if idx_val is not None else 0,
                    "n_test": int(len(idx_test)),
                    "fusion_feature_dim": int(X_fusion.shape[1]),
                    "branch_contributions": contribution_details,
                    **metrics,
                }
            )

        summary = summarize_fold_results(fold_results)
        expected_dim = sum(branch_metadata[branch].fusion_embedding_dim for branch in branches)
        experiments.append(
            {
                "name": experiment_name,
                "branches": list(branches),
                "fusion_type": "late_fusion_concat_penultimate_embeddings",
                "fusion_feature_dim": int(expected_dim),
                "branch_metadata": {
                    branch: branch_metadata[branch].__dict__
                    for branch in branches
                },
                "summary": summary,
                "folds": fold_results,
            }
        )

    leaderboard = sorted(
        experiments,
        key=lambda experiment: (
            experiment["summary"].get("mean_val_accuracy", float("-inf")),
            experiment["summary"].get("mean_val_f1", float("-inf")),
        ),
        reverse=True,
    )

    payload = {
        "method": "Late Fusion",
        "primary_metric": "accuracy",
        "model_selection_metric": "mean_val_accuracy",
        "external_competition_test_used": False,
        "data_file": str(data_file),
        "subset": args.subset,
        "split_strategy": "splitting.split_data 5-fold stratified CV with inner validation",
        "fusion_note": "Each branch MLP is trained on the fold training split, frozen, converted to penultimate embeddings, then concatenated for the final MLP.",
        "saplma_feature_config": {
            "token_mode": args.saplma_token_mode,
            "layers": saplma_layers,
            "raw_feature_dim": raw_dims["saplma"],
            "best_config": "4layer_256x128x64_dropout_l1_l2",
        },
        "icr_feature_config": {
            "feature_type": "layerwise_mean_icr",
            "raw_feature_dim": raw_dims["icr"],
            "top_k": args.icr_top_k,
            "z_normalize": icr_z_normalize,
            "best_config": "mlp_128_64_32_dropout0.3_l2",
        },
        "llm_check_feature_config": {
            "feature_type": "attention_diagonal_log_score",
            "raw_feature_dim": raw_dims["llm_check"],
            "selected_layer_indices_zero_based": llm_check_cache[
                "selected_layer_indices_zero_based"
            ].astype(int).tolist(),
            "selected_transformer_layers_1based": llm_check_cache[
                "selected_transformer_layers_1based"
            ].astype(int).tolist(),
            "best_config": "mlp_dropout0.3_l2",
        },
        "final_fusion_mlp": {
            "hidden_dims": list(parse_hidden_dims(args.fusion_hidden_dims)),
            "dropout_p": args.fusion_dropout_p,
            "l2_weight_decay": args.fusion_l2_weight_decay,
            "epochs": args.fusion_epochs,
            "batch_size": args.fusion_batch_size,
            "learning_rate": args.fusion_learning_rate,
            "threshold_tuning": "validation_accuracy_then_f1_tiebreak",
        },
        "experiments": experiments,
        "leaderboard": [
            {
                "name": experiment["name"],
                "branches": experiment["branches"],
                "fusion_feature_dim": experiment["fusion_feature_dim"],
                "mean_val_accuracy": experiment["summary"].get("mean_val_accuracy", float("nan")),
                "mean_val_f1": experiment["summary"].get("mean_val_f1", float("nan")),
                "mean_internal_test_accuracy": experiment["summary"].get("mean_test_accuracy", float("nan")),
                "mean_internal_test_f1": experiment["summary"].get("mean_test_f1", float("nan")),
                "mean_internal_test_auroc": experiment["summary"].get("mean_test_auroc", float("nan")),
            }
            for experiment in leaderboard
        ],
        "best_by_validation_accuracy": leaderboard[0]["name"],
        "mean_baseline_accuracy": _nanmean(
            [
                experiment["summary"].get("mean_baseline_accuracy", float("nan"))
                for experiment in experiments
            ]
        ),
    }

    output_file.write_text(json.dumps(payload, indent=2))
    print(f"\n[Fusion] Results saved to {output_file}")
    print("[Fusion] Leaderboard by validation accuracy")
    for row in payload["leaderboard"]:
        print(
            f"  {row['name']}: "
            f"val_acc={row['mean_val_accuracy']:.4f} "
            f"internal_test_acc={row['mean_internal_test_accuracy']:.4f} "
            f"dim={row['fusion_feature_dim']}"
        )


if __name__ == "__main__":
    main()
