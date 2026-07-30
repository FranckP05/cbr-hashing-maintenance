"""Create static Matplotlib figures from the Heart Disease retrieval benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .plotting_theme import (
        EXPORT_DPI,
        FIGURE_SIZE,
        METHOD_COLORS,
        METHOD_STYLES,
        PLOT_MARGINS,
        configure_matplotlib,
        save_figure,
    )
except ImportError:
    from src.visualization.plotting_theme import (
        EXPORT_DPI,
        FIGURE_SIZE,
        METHOD_COLORS,
        METHOD_STYLES,
        PLOT_MARGINS,
        configure_matplotlib,
        save_figure,
    )


configure_matplotlib()
import matplotlib.pyplot as plt


METHODS = ("Baseline", "KD-Tree", "LSH")


def summarize(per_seed: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        per_seed.groupby(["method", "k"], sort=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )


def metric_plot(
    per_seed: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    y_limits: tuple[float, float],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    figure.subplots_adjust(**PLOT_MARGINS)
    k_values = sorted(per_seed["k"].unique())

    for method in METHODS:
        data = summarize(per_seed[per_seed["method"] == method], metric)
        axis.errorbar(
            data["k"],
            data["mean"],
            yerr=data["std"],
            label=method,
            color=METHOD_COLORS[method],
            linewidth=2.2,
            markersize=6,
            capsize=4,
            capthick=1.2,
            **METHOD_STYLES[method],
        )

    axis.set_title(title, fontfamily="Space Grotesk", fontsize=15, fontweight="semibold", y=1.02, pad=5)
    axis.set_xlabel("Nombre de voisins k", fontsize=11, labelpad=11)
    axis.set_ylabel(y_label, fontsize=11, labelpad=10)
    axis.set_xticks(k_values)
    axis.set_ylim(*y_limits)
    axis.tick_params(labelsize=10)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=len(labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.7,
    )
    save_figure(figure, output_path)
    plt.close(figure)


def create_figures(input_path: Path, output_dir: Path) -> None:
    per_seed = pd.read_csv(input_path)
    metric_plot(
        per_seed,
        "exact_neighbor_recall_at_k",
        "Fidélité de récupération aux voisins exacts",
        "Recall des voisins exacts@k",
        (0.85, 1.02),
        output_dir / "heart_exact_neighbor_recall.png",
    )
    metric_plot(
        per_seed,
        "ndcg_at_k",
        "Qualité de classement clinique des voisins",
        "nDCG@k (label clinique)",
        (0.60, 0.84),
        output_dir / "heart_label_ndcg.png",
    )
    metric_plot(
        per_seed,
        "f1",
        "Qualité de prédiction par vote pondéré",
        "F1-score",
        (0.60, 0.92),
        output_dir / "heart_prediction_f1.png",
    )
    metric_plot(
        per_seed,
        "query_time_ms",
        "Temps moyen de récupération par requête",
        "Temps de requête (ms)",
        (0.0, 0.30),
        output_dir / "heart_query_time.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path", type=Path, default=Path("results/benchmarks/heart_retrieval_per_seed.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/heart_retrieval"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_figures(args.input_path, args.output_dir)
    print(f"PNG figures written to {args.output_dir}.")
