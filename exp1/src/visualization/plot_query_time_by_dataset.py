"""Figure 3 — Temps de requête moyen (ms) par méthode et par dataset.

KD-Tree et LSH : résumés canoniques de results/benchmarks/ (time_query, une requête
réelle par cas) — heart_disease_retrieval_summary.csv, diabetes130_retrieval_sample10pct_summary.csv,
chestxray14_retrieval_sample10pct_summary.csv.

Baseline : temps individuel mesuré de LinearScanIndex.query() (une requête à la fois,
en boucle, sans batch), 5 seeds × 20 requêtes par dataset — le coût en ligne d'un
balayage séquentiel, scénario flux de nouveaux cas un par un (chapitre 3.2 du plan).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .plotting_theme import (
        EXPORT_DPI,
        FIGURE_SIZE,
        METHOD_COLORS,
        PLOT_MARGINS,
        configure_matplotlib,
        save_figure,
    )
except ImportError:
    from src.visualization.plotting_theme import (
        EXPORT_DPI,
        FIGURE_SIZE,
        METHOD_COLORS,
        PLOT_MARGINS,
        configure_matplotlib,
        save_figure,
    )


configure_matplotlib()
import matplotlib.pyplot as plt


METHODS = ("Baseline", "KD-Tree", "LSH")

DATASETS = (
    ("Heart Disease (13-d)", {"Baseline": 0.05, "KD-Tree": 0.112, "LSH": 0.179}),
    ("Diabetes-130 (390-d)", {"Baseline": 47.00, "KD-Tree": 73.53, "LSH": 143.80}),
    ("ChestX-ray14 (1024-d)", {"Baseline": 100.51, "KD-Tree": 109.23, "LSH": 162.75}),
)


def plot_query_time_by_dataset(output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), dpi=EXPORT_DPI)
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.86, wspace=0.30)

    for axis, (title, values) in zip(axes, DATASETS):
        positions = np.arange(len(METHODS))
        bars = axis.bar(
            positions,
            [values[m] for m in METHODS],
            width=0.58,
            color=[METHOD_COLORS[m] for m in METHODS],
        )
        axis.set_title(title, fontfamily="Space Grotesk", fontsize=12.5, fontweight="semibold")
        axis.set_xticks(positions, [m for m in METHODS], fontsize=8.5)
        axis.tick_params(labelsize=9)
        axis.grid(axis="y", linewidth=0.8)

        y_max = max(values.values())
        for bar, method in zip(bars, METHODS):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{values[method]:.2f} ms".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=METHOD_COLORS[method],
                fontweight="semibold",
            )
        axis.set_ylim(0, y_max * 1.22)

    figure.suptitle(
        "Temps moyen par requête (ms) : Baseline vs KD-Tree vs LSH, par dataset",
        fontfamily="Space Grotesk",
        fontsize=16,
        fontweight="semibold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.015,
        "Baseline : LinearScanIndex.query() individuel, 5 seeds × 20 requêtes (pas de batch). "
        "KD-Tree/LSH : résumés canoniques (time_query, une requête réelle par cas). "
        "5 seeds × k={1,5,10}.",
        ha="center",
        fontsize=9,
        color=METHOD_COLORS["Baseline"],
    )

    save_figure(figure, output_path)
    plt.close(figure)


if __name__ == "__main__":
    plot_query_time_by_dataset(Path("results/figures/query_time_by_dataset.png"))
    print("PNG figure written to results/figures/query_time_by_dataset.png")
