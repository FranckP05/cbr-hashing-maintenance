"""Charte graphique Matplotlib centralisée pour toutes les figures du projet."""

from __future__ import annotations

import os
from pathlib import Path

# Répertoire de configuration Matplotlib compatible avec les environnements restreints
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constantes de la charte graphique
# ---------------------------------------------------------------------------
FIGURE_SIZE    = (8.0, 4.5)    # pouces, 16:9
EXPORT_DPI     = 300

# Couleurs méthodes
COLOR_SIMHASH  = "#D55E00"      # orange vif (SimHash / LSH)
COLOR_BASELINE = "#0F172A"      # navy (Baseline linéaire)

# Couleurs UI
COLOR_BG       = "#FFFFFF"      # fond de figure
COLOR_TEXT     = "#0F172A"      # texte, axes
COLOR_GRID     = "#DCE5EF"      # lignes de grille

# Styles méthodes
STYLE_SIMHASH  = {"color": COLOR_SIMHASH,  "marker": "^", "linestyle": "-",  "linewidth": 1.8, "markersize": 7}
STYLE_BASELINE = {"color": COLOR_BASELINE, "marker": "s", "linestyle": "--", "linewidth": 1.8, "markersize": 7}

METHOD_LABELS  = {
    "simhash":  "SimHash (LSH)",
    "baseline": "Balayage Linéaire",
}


def configure_matplotlib() -> None:
    """Applique la charte graphique globale à Matplotlib."""
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            # ---- Fond ----
            "figure.facecolor":   COLOR_BG,
            "axes.facecolor":     COLOR_BG,
            "savefig.facecolor":  COLOR_BG,

            # ---- Typographie ----
            "font.family":        "sans-serif",
            "font.sans-serif":    ["Manrope", "DejaVu Sans", "Arial"],
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

            # ---- Bordures ----
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.spines.left":   True,
            "axes.spines.bottom": True,
            "axes.edgecolor":     COLOR_TEXT,
            "axes.linewidth":     0.8,

            # ---- Grille ----
            "axes.grid":          True,
            "grid.color":         COLOR_GRID,
            "grid.linewidth":     0.8,
            "grid.alpha":         1.0,
            "axes.axisbelow":     True,
        }
    )


def new_figure() -> tuple:
    """Crée une nouvelle figure avec les dimensions standard.

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    return fig, ax


def set_title(ax, title: str, subtitle: str | None = None) -> None:
    """Applique le titre en Space Grotesk (ou sans-serif de repli).

    Args:
        ax       : Axes Matplotlib.
        title    : Titre principal.
        subtitle : Sous-titre optionnel (taille réduite).
    """
    import matplotlib.font_manager as fm
    # Tente de charger Space Grotesk; repli sur la police par défaut
    title_props = {}
    try:
        fe = fm.FontEntry(fname="SpaceGrotesk-SemiBold.ttf", name="Space Grotesk")
        fm.fontManager.ttflist.append(fe)
        title_props["fontfamily"] = "Space Grotesk"
    except Exception:
        pass

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10, **title_props)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0.5, 1.01), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9, color=COLOR_TEXT,
        )


def save_figure(fig, output_path: Path) -> None:
    """Sauvegarde une figure en PNG 300 DPI.

    Args:
        fig         : Figure Matplotlib.
        output_path : Chemin de destination (.png).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
