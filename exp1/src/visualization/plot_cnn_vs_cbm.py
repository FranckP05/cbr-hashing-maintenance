"""Static comparison figure for CNN and CBM condensation on Heart Disease."""

from __future__ import annotations

from pathlib import Path

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
import numpy as np


def plot_cnn_vs_cbm(seeds, precisions_cnn, precisions_cbm, output_path):
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=EXPORT_DPI)
    figure.subplots_adjust(**PLOT_MARGINS)
    positions = np.arange(len(seeds))
    width = 0.34

    axis.bar(positions - width / 2, precisions_cnn, width, label="CNN", color=METHOD_COLORS["LSH"])
    axis.bar(positions + width / 2, precisions_cbm, width, label="CBM", color=METHOD_COLORS["HeCBR"])
    axis.set_title(
        "Précision de classification: CNN vs CBM, par seed",
        fontfamily="Space Grotesk",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
        pad=5,
    )
    axis.set_xlabel("Seed", fontsize=11, labelpad=11)
    axis.set_ylabel("Précision", fontsize=11, labelpad=10)
    axis.set_xticks(positions, [str(seed) for seed in seeds])
    axis.set_ylim(0.0, 1.0)
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
    save_figure(figure, Path(output_path))
    plt.close(figure)


if __name__ == "__main__":
    seeds = [42, 7, 123, 2024, 99]
    precisions_cnn = [0.7666666666666667, 0.6666666666666666, 0.8333333333333334, 0.6666666666666666, 0.7]
    precisions_cbm = [0.7666666666666667, 0.7, 0.85, 0.7666666666666667, 0.7333333333333333]

    plot_cnn_vs_cbm(seeds, precisions_cnn, precisions_cbm, "results/figures/cnn_vs_cbm_heart_disease.png")
    print("PNG figure written to results/figures/cnn_vs_cbm_heart_disease.png")
