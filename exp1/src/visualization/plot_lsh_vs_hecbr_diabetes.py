"""Figure 4 — LSH vs HeCBR sur Diabetes-130 : précision de classification.

Données issues de la validation finale (journal §22) :
  - LSH   (12 bits, 32 tables, calibré) : précision 0.8801
  - HeCBR (64 bits, distance de Hamming) : précision 0.8644

Réserve portée dans la figure : HeCBR n'a reçu qu'un seul passage d'entraînement,
sans recherche d'hyperparamètres.
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


METHODS = ("LSH", "HeCBR")
VALUES = {"LSH": 0.8801, "HeCBR": 0.8644}


def plot_lsh_vs_hecbr_diabetes(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    figure.subplots_adjust(**PLOT_MARGINS)

    positions = np.arange(len(METHODS))
    bars = axis.bar(
        positions,
        [VALUES[m] for m in METHODS],
        width=0.44,
        color=[METHOD_COLORS[m] for m in METHODS],
    )

    axis.set_title(
        "Précision de classification — Diabetes-130 : LSH vs HeCBR",
        fontfamily="Space Grotesk",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
        pad=5,
    )
    axis.set_ylabel("Précision de classification (2 035 requêtes)", fontsize=11, labelpad=10)
    axis.set_xticks(positions, METHODS, fontsize=11)
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(labelsize=10)

    for bar, method in zip(bars, METHODS):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{VALUES[method]:.4f}".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=METHOD_COLORS[method],
        )

    axis.text(
        0.5,
        0.965,
        "Réserve : HeCBR n'a reçu qu'un seul passage d'entraînement\n"
        "(64 bits fixés a priori), sans recherche d'hyperparamètres.",
        ha="center",
        va="top",
        fontsize=9,
        color=METHOD_COLORS["Baseline"],
        transform=axis.transAxes,
    )

    save_figure(figure, output_path)
    plt.close(figure)


if __name__ == "__main__":
    plot_lsh_vs_hecbr_diabetes(Path("results/figures/lsh_vs_hecbr_diabetes130.png"))
    print("PNG figure written to results/figures/lsh_vs_hecbr_diabetes130.png")
