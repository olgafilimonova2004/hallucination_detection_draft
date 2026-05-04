"""
method0_diagnostics.py — PCA diagnostics for Method 0 (Geometry of Truth).

This script:
1. Extracts reusable hidden-state summaries from ``dataset.csv``.
2. Runs 2-D PCA for each layer and token-summary mode.
3. Computes silhouette scores to quantify class separation.
4. Saves a heatmap, layer-trend plot, and scatter plots for the best pairs.

The outputs are meant to guide layer/token selection for the later probe
methods.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from experiment_utils import (
    SEQUENCE_MODES,
    extract_feature_cache,
    load_feature_cache,
    save_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Method 0 PCA diagnostics.")
    parser.add_argument(
        "--data-file",
        default="./data/dataset.csv",
        help="Labelled CSV used for diagnostics.",
    )
    parser.add_argument(
        "--cache-file",
        default="./artifacts/cache/method0_hidden_cache.npz",
        help="Compressed feature cache reused across experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/method0",
        help="Directory where figures and CSV summaries are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used during hidden-state extraction.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Token budget after response-preserving truncation.",
    )
    parser.add_argument(
        "--spectrum-top-k",
        type=int,
        default=16,
        help="Also cache top-k singular values for later LLM-Check experiments.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Optional stratified subset size for quick smoke tests.",
    )
    parser.add_argument(
        "--top-scatter",
        type=int,
        default=6,
        help="Number of best PCA scatter plots to save.",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for cached hidden-state summaries.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Force cache regeneration even if the cache file already exists.",
    )
    return parser.parse_args()


def maybe_take_subset(df: pd.DataFrame, subset_size: int | None) -> pd.DataFrame:
    """Optionally take a stratified subset for a quick run."""
    if subset_size is None or subset_size >= len(df):
        return df.reset_index(drop=True)
    if "label" not in df.columns:
        return df.sample(n=subset_size, random_state=42).reset_index(drop=True)

    parts = []
    label_counts = df["label"].value_counts(normalize=True).sort_index()
    for label, frac in label_counts.items():
        n_label = max(1, int(round(subset_size * frac)))
        label_df = df[df["label"] == label]
        parts.append(label_df.sample(n=min(n_label, len(label_df)), random_state=42))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_summary(cache: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """Run PCA + silhouette scoring for all sequence modes and layers."""
    if "labels" not in cache:
        raise ValueError("Method 0 diagnostics require labels in the feature cache.")

    labels = cache["labels"]
    summary_rows: list[dict[str, float | int | str]] = []
    projections: dict[tuple[str, int], np.ndarray] = {}

    for mode in SEQUENCE_MODES:
        mode_features = cache[mode]
        for layer_idx in range(mode_features.shape[1]):
            X = mode_features[:, layer_idx, :].astype(np.float32, copy=False)
            pca = PCA(n_components=2, random_state=42)
            projected = pca.fit_transform(X)
            projections[(mode, layer_idx)] = projected

            try:
                silhouette = float(silhouette_score(projected, labels))
            except ValueError:
                silhouette = float("nan")

            summary_rows.append(
                {
                    "mode": mode,
                    "layer": layer_idx,
                    "silhouette": silhouette,
                    "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                    "pc1_var": float(pca.explained_variance_ratio_[0]),
                    "pc2_var": float(pca.explained_variance_ratio_[1]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["silhouette", "explained_variance"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return summary_df, projections


def plot_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Save a silhouette-score heatmap over modes and layers."""
    heatmap = np.full((len(SEQUENCE_MODES), 25), np.nan, dtype=np.float32)
    for mode_idx, mode in enumerate(SEQUENCE_MODES):
        mode_df = summary_df[summary_df["mode"] == mode]
        for _, row in mode_df.iterrows():
            heatmap[mode_idx, int(row["layer"])] = float(row["silhouette"])

    fig, ax = plt.subplots(figsize=(13, 3.8))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(25))
    ax.set_yticks(np.arange(len(SEQUENCE_MODES)))
    ax.set_yticklabels(SEQUENCE_MODES)
    ax.set_xlabel("Layer")
    ax.set_title("Method 0 — Silhouette Score by Layer and Token Summary")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Silhouette score")
    fig.tight_layout()
    fig.savefig(output_dir / "silhouette_heatmap.png", dpi=180)
    plt.close(fig)


def plot_layer_trends(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Save line plots of silhouette trends across layers."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for mode in SEQUENCE_MODES:
        mode_df = summary_df[summary_df["mode"] == mode].sort_values("layer")
        ax.plot(
            mode_df["layer"].to_numpy(),
            mode_df["silhouette"].to_numpy(),
            marker="o",
            linewidth=1.8,
            label=mode,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Method 0 — Layerwise PCA Separation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "silhouette_trends.png", dpi=180)
    plt.close(fig)


def plot_top_scatter_grid(
    summary_df: pd.DataFrame,
    projections: dict[tuple[str, int], np.ndarray],
    labels: np.ndarray,
    output_dir: Path,
    top_scatter: int,
) -> None:
    """Save PCA scatter plots for the strongest layer-mode pairs."""
    top_df = summary_df.head(top_scatter).copy()
    n_plots = len(top_df)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4.2 * n_rows),
        squeeze=False,
    )

    colors = {0: "#1f77b4", 1: "#d62728"}
    labels_map = {0: "truthful", 1: "hallucinated"}

    for ax, (_, row) in zip(axes.flat, top_df.iterrows()):
        mode = str(row["mode"])
        layer = int(row["layer"])
        projection = projections[(mode, layer)]

        for label_value in (0, 1):
            mask = labels == label_value
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=16,
                alpha=0.75,
                c=colors[label_value],
                label=labels_map[label_value],
            )

        ax.set_title(
            f"{mode} | layer {layer} | silhouette={row['silhouette']:.3f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    for ax in axes.flat[n_plots:]:
        ax.axis("off")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "top_pca_scatter.png", dpi=180)
    plt.close(fig)


def save_recommendations(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Save ranked layer recommendations for downstream experiments."""
    top_pairs = summary_df.head(12).copy()
    top_pairs.to_csv(output_dir / "top_pairs.csv", index=False)

    layer_rank = (
        summary_df.groupby("layer", as_index=False)
        .agg(
            mean_silhouette=("silhouette", "mean"),
            max_silhouette=("silhouette", "max"),
        )
        .sort_values(["max_silhouette", "mean_silhouette"], ascending=[False, False])
    )
    layer_rank.to_csv(output_dir / "layer_rankings.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = Path(args.cache_file)

    if cache_file.exists() and not args.overwrite_cache:
        print(f"[Method 0] Loading cache from {cache_file}")
        cache = load_feature_cache(cache_file)
    else:
        print(f"[Method 0] Building cache from {args.data_file}")
        df = pd.read_csv(args.data_file)
        df = maybe_take_subset(df, args.subset)
        cache = extract_feature_cache(
            df=df,
            batch_size=args.batch_size,
            max_length=args.max_length,
            spectrum_top_k=args.spectrum_top_k,
            cache_dtype=np.float16 if args.cache_dtype == "float16" else np.float32,
            include_icr=False,
            include_spectrum=False,
        )
        save_feature_cache(cache_file, cache)
        print(f"[Method 0] Saved cache to {cache_file}")

    summary_df, projections = build_summary(cache)
    summary_df.to_csv(output_dir / "pca_summary.csv", index=False)

    plot_heatmap(summary_df, output_dir)
    plot_layer_trends(summary_df, output_dir)
    plot_top_scatter_grid(
        summary_df=summary_df,
        projections=projections,
        labels=cache["labels"],
        output_dir=output_dir,
        top_scatter=args.top_scatter,
    )
    save_recommendations(summary_df, output_dir)

    print()
    print("[Method 0] Top layer-mode pairs:")
    print(summary_df.head(10).to_string(index=False))
    print()
    print(f"[Method 0] Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
