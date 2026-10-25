"""Charte graphique Matplotlib centralisée pour HeCBR et SOTA LSH.

Polices embarquées (visualization/fonts/) :
- Titres / display : Space Grotesk (SemiBold)
- Corps / axes / légendes : Manrope
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"
for _ttf in sorted(_FONTS_DIR.glob("*.ttf")):
    font_manager.fontManager.addfont(str(_ttf))

FONT_DISPLAY = "Space Grotesk"
FONT_BODY    = "Manrope"

FIGURE_SIZE    = (8.0, 4.5)
EXPORT_DPI     = 300

# Couleurs méthodes
COLOR_HECBR    = "#009E73"      # Vert émeraude (HeCBR Deep Hashing)
COLOR_SOTA_LSH = "#D55E00"      # Orange vif (Cross-Polytope LSH)
COLOR_BASELINE = "#0F172A"      # Navy foncé (Baseline Linéaire)

COLOR_BG       = "#FFFFFF"
COLOR_TEXT     = "#0F172A"
COLOR_GRID     = "#DCE5EF"

STYLE_HECBR    = {"color": COLOR_HECBR,    "marker": "o", "linestyle": "-",  "linewidth": 1.8, "markersize": 7}
STYLE_SOTA_LSH = {"color": COLOR_SOTA_LSH, "marker": "^", "linestyle": "-",  "linewidth": 1.8, "markersize": 7}
STYLE_BASELINE = {"color": COLOR_BASELINE, "marker": "s", "linestyle": "--", "linewidth": 1.8, "markersize": 7}

METHOD_LABELS  = {
    "baseline": "Balayage Linéaire Exact",
    "sota_lsh": "Cross-Polytope LSH (Multi-Probe)",
    "hecbr":    "HeCBR (Deep Hashing 64-bit)",
    "hecbr_mp": "HeCBR Multi-Probe",
}


def configure_matplotlib() -> None:
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "figure.facecolor":   COLOR_BG,
            "axes.facecolor":     COLOR_BG,
            "savefig.facecolor":  COLOR_BG,
            "font.family":        "sans-serif",
            "font.sans-serif":    [FONT_BODY, FONT_DISPLAY, "DejaVu Sans", "Arial"],
            "text.color":         COLOR_TEXT,
            "axes.labelcolor":    COLOR_TEXT,
            "axes.titlecolor":    COLOR_TEXT,
            "xtick.color":        COLOR_TEXT,
            "ytick.color":        COLOR_TEXT,
            "axes.titlesize":     13,
            "axes.labelsize":     11,
            "xtick.labelsize":    9,
            "ytick.labelsize":    9,
            "legend.fontsize":    9,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.spines.left":   True,
            "axes.spines.bottom": True,
            "axes.edgecolor":     COLOR_TEXT,
            "axes.linewidth":     0.8,
            "axes.grid":          True,
            "grid.color":         COLOR_GRID,
            "grid.linewidth":     0.8,
            "grid.alpha":         1.0,
            "axes.axisbelow":     True,
        }
    )


def new_figure() -> tuple:
    return plt.subplots(figsize=FIGURE_SIZE)


def set_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, fontsize=13, fontweight="semibold",
                 family=FONT_DISPLAY, pad=10)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0.5, 1.01), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9, color=COLOR_TEXT,
            family=FONT_BODY,
        )


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
