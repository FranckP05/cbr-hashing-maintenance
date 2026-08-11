"""Figure 2 — Nombre de bits LSH requis par dataset (effet dimensionnel).

Les parametres retenus sont les valeurs canoniques du journal :
  - Heart Disease : num_bits=5, tables=16 (journal §3)
  - Diabetes-130  : num_bits=12, tables=32 (journal §17-18)
  - ChestX-ray14  : num_bits=16, tables=32 (journal §15)

Le point scientifique cle : Diabetes et Chest ont des bases de taille comparable
(81 412 vs 89 696 cas) mais exigent un hachage nettement plus fin en 1024-d
(+4 bits) — effet attribuable a la dimensionnalite, pas a la taille de la base.
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
        TEXT_COLOR,
        configure_matplotlib,
        save_figure,
    )
except ImportError:
    from src.visualization.plotting_theme import (
        EXPORT_DPI,
        FIGURE_SIZE,
        METHOD_COLORS,
        PLOT_MARGINS,
        TEXT_COLOR,
        configure_matplotlib,
        save_figure,
    )


configure_matplotlib()
import matplotlib.pyplot as plt


DATASETS = (
    ("Heart Disease", 5, 237, 13),
    ("Diabetes-130", 12, 81412, 390),
    ("ChestX-ray14", 16, 89696, 1024),
)


def plot_lsh_bits_by_dimension(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    figure.subplots_adjust(**PLOT_MARGINS)

    names = [row[0] for row in DATASETS]
    bits = [row[1] for row in DATASETS]
    n_cases = [row[2] for row in DATASETS]
    dims = [row[3] for row in DATASETS]

    positions = np.arange(len(DATASETS))
    bars = axis.bar(
        positions,
        bits,
        width=0.55,
        color=METHOD_COLORS["LSH"],
        alpha=0.9,
    )

    axis.set_title(
        "Nombre de bits LSH requis pour maintenir le rappel, par dataset",
        fontfamily="Space Grotesk",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
        pad=5,
    )
    axis.set_ylabel("num_bits calibré (2^bits compartiments)", fontsize=11, labelpad=10)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{name}\n({dim}-d)" for name, dim in zip(names, dims)],
        fontsize=10,
    )
    axis.set_ylim(0, 22.5)
    axis.tick_params(labelsize=10)

    for bar, n_case in zip(bars, n_cases):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"n = {n_case:,}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_COLOR,
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.55,
            f"{int(bar.get_height())} bits",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#FFFFFF",
        )

    axis.annotate(
        "n comparable (~81k vs ~90k), pourtant +4 bits nécessaires\n"
        "— effet attribuable à la dimensionnalité (390 → 1024),\n"
        "pas à la taille de la base.",
        xy=(2, bits[2] + 0.7),
        xytext=(1.5, 20.6),
        fontsize=9.5,
        color=METHOD_COLORS["HeCBR"],
        arrowprops=dict(
            arrowstyle="->",
            color=METHOD_COLORS["HeCBR"],
            lw=1.4,
        ),
        ha="center",
        va="center",
    )

    figure.text(
        0.5,
        0.03,
        "Donnée complémentaire — % de candidats LSH avant reclassement :\n"
        "42,9% (13-d) / 44,7% (390-d) / 19,0% (1024-d) — confondu par n, non monotone, non représenté.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=METHOD_COLORS["Baseline"],
    )

    save_figure(figure, output_path)
    plt.close(figure)


if __name__ == "__main__":
    plot_lsh_bits_by_dimension(Path("results/figures/lsh_bits_by_dimension.png"))
    print("PNG figure written to results/figures/lsh_bits_by_dimension.png")
