"""Figure 1 — Maintenance : condensation CBM vs CNN sur les trois datasets.

Données par seed :
  - Heart Disease (13-d)  : plot_cnn_vs_cbm.py / journal §3, §11
  - Diabetes-130 (390-d)  : journal §18
  - ChestX-ray14 (1024-d) : maintenance_chestxray14.log (Étape 8, 5 seeds)
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


DATASETS = (
    ("Heart Disease\n(13-d)", [0.7667, 0.6667, 0.8333, 0.6667, 0.7], [0.7667, 0.7, 0.85, 0.7667, 0.7333]),
    ("Diabetes-130\n(390-d)", [0.7383, 0.7376, 0.7330, 0.7361, 0.7267], [0.8205, 0.8219, 0.8212, 0.8209, 0.8188]),
    ("ChestX-ray14\n(1024-d)", [0.5792, 0.5844, 0.5802, 0.5893, 0.5823], [0.6209, 0.6205, 0.6205, 0.6217, 0.6206]),
)


def plot_maintenance_cbm_vs_cnn(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    figure.subplots_adjust(**PLOT_MARGINS)

    positions = np.arange(len(DATASETS))
    width = 0.34

    cnn_means = []
    cbm_means = []
    cnn_stds = []
    cbm_stds = []
    for _, cnn, cbm in DATASETS:
        cnn = np.asarray(cnn)
        cbm = np.asarray(cbm)
        cnn_means.append(cnn.mean())
        cbm_means.append(cbm.mean())
        cnn_stds.append(cnn.std())
        cbm_stds.append(cbm.std())

    axis.bar(
        positions - width / 2,
        cnn_means,
        width,
        yerr=cnn_stds,
        capsize=3,
        label="CNN",
        color=METHOD_COLORS["LSH"],
    )
    axis.bar(
        positions + width / 2,
        cbm_means,
        width,
        yerr=cbm_stds,
        capsize=3,
        label="CBM",
        color=METHOD_COLORS["HeCBR"],
    )

    axis.set_title(
        "Condensation 1-NN : CBM vs CNN, par dataset",
        fontfamily="Space Grotesk",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
        pad=5,
    )
    axis.set_ylabel("Précision (moyenne sur 5 seeds)", fontsize=11, labelpad=10)
    axis.set_xticks(positions, [name for name, _, _ in DATASETS], fontsize=10)
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(labelsize=10)

    for i, (cnn_mean, cbm_mean) in enumerate(zip(cnn_means, cbm_means)):
        delta = (cbm_mean - cnn_mean) * 100
        axis.text(
            i,
            max(cnn_mean, cbm_mean) + 0.055,
            f"+{delta:.1f} pt",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=METHOD_COLORS["HeCBR"],
        )

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncols=len(labels),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.7,
    )

    save_figure(figure, output_path)
    plt.close(figure)


if __name__ == "__main__":
    plot_maintenance_cbm_vs_cnn(Path("results/figures/maintenance_cbm_vs_cnn.png"))
    print("PNG figure written to results/figures/maintenance_cbm_vs_cnn.png")
