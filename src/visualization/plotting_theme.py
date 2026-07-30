"""Shared static-figure theme for research outputs."""

from __future__ import annotations

import os
from pathlib import Path

# The execution environment mounts the usual home configuration directory read-only.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib


FIGURE_SIZE = (8, 4.5)
EXPORT_DPI = 300
PLOT_MARGINS = {"left": 0.12, "right": 0.98, "bottom": 0.25, "top": 0.82}
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#0F172A"
GRID_COLOR = "#DCE5EF"

# Restrained, colour-blind-friendly contrast for research figures.
METHOD_COLORS = {
    "HeCBR": "#0F172A",
    "LSH": "#D55E00",
    "KD-Tree": "#0284C7",
    "Baseline": "#64748B",
}
METHOD_STYLES = {
    "Baseline": {"marker": "s", "linestyle": "--"},
    "KD-Tree": {"marker": "o", "linestyle": "-"},
    "LSH": {"marker": "^", "linestyle": "-"},
    "HeCBR": {"marker": "D", "linestyle": "-"},
}


def configure_matplotlib() -> None:
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.facecolor": BACKGROUND_COLOR,
            "savefig.facecolor": BACKGROUND_COLOR,
            "font.family": "Manrope",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
        }
    )


def save_figure(figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=EXPORT_DPI)
