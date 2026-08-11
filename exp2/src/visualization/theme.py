"""Thème graphique exp2 — charte « DeepTech Research » (Space Grotesk + Manrope).

Combo établi dans exp1 (generate_theme_previews.py, combo 1) :
  - Titres           : Space Grotesk
  - Corps/labels     : Manrope
Couleurs conservées (palette 4 méthodes + classes) :

  Méthodes (palette exp1 / plotting_theme.py) :
    HeCBR   #0F172A  (navy)   losange
    LSH     #D55E00  (orange) triangle
    KD-Tree #0284C7  (bleu)   cercle
    Baseline #64748B (gris)   carré pointillé

  Données (NB01) :
    classes 0/1   #7AA6C2 / #C97B7B
    dimension    #5A7D9A

Stratégie anti-chevauchement (finalize) :
  - la légende « Figure : FIG-X — description » est enveloppée à width=100 ;
  - la marge basse est MESURÉE (hauteur réelle du texte de la légende après
    rendu) puis réservée par subplots_adjust, en itérant jusqu'à convergence ;
  - la sauvegarde utilise bbox_inches='tight' + pad_inches=0.25 (pas de rognage) ;
  - un auto-contrôle tourne sur le layout final : chevauchements de textes et
    textes réellement coupés par le clipbox de leurs axes.
"""

import os
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

FIGURE_SIZE = (8, 4.5)
EXPORT_DPI = 300
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#0F172A"
GRID_COLOR = "#DCE5EF"

TITLE_FONT = "Space Grotesk"
BODY_FONT = "Manrope"
CAPTION_WIDTH = 100

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

CLASS_COLORS = ["#7AA6C2", "#C97B7B"]
DIM_COLOR = "#5A7D9A"


def apply_theme():
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.facecolor": BACKGROUND_COLOR,
            "savefig.facecolor": BACKGROUND_COLOR,
            "font.family": BODY_FONT,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "figure.dpi": 110,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def title(ax, text, fontsize=10, **kw):
    """Titre d'axe en Space Grotesk (combo établi)."""
    ax.set_title(text, fontsize=fontsize, fontfamily=TITLE_FONT, **kw)
    return ax


def finalize(fig, caption, output_path=None):
    """Titre descriptif en bas, sans chevauchement (marge mesurée).

    Le titre (ex. « Répartition des classes par dataset ») est écrit en gras
    Space Grotesk sous les axes :

    1) enveloppe le titre à width=CAPTION_WIDTH et l'écrit à y=0.02 ;
    2) mesure après rendu la hauteur du titre ET la bande occupée par les
       textes d'axes (ticks, labels) sous les axes, puis réserve la marge basse
       par subplots_adjust, en itérant jusqu'à convergence ;
    3) auto-contrôle sur le layout final : chevauchements / textes coupés ;
    4) sauvegarde en bbox_inches='tight' + pad_inches=0.25.
    """
    wrapped = textwrap.fill(caption, width=CAPTION_WIDTH)
    fig.text(
        0.5, 0.02, wrapped,
        ha="center", va="bottom", fontsize=11, color=TEXT_COLOR,
        fontfamily=TITLE_FONT, fontweight="bold",
    )
    fig.subplots_adjust(top=0.93, bottom=0.30, left=0.10, right=0.98)
    bottom = 0.30
    for _ in range(6):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        _, fig_h = fig.canvas.get_width_height()
        caption_top = _caption_top(fig, wrapped, renderer, fig_h)
        ax_min = _lowest_axes_text(fig, wrapped, renderer, fig_h)
        band = bottom - ax_min if ax_min < bottom else 0.0
        new_bottom = min(caption_top + band + 0.012, 0.5)
        if abs(new_bottom - bottom) < 0.001:
            bottom = new_bottom
            break
        bottom = new_bottom
        fig.subplots_adjust(top=0.93, bottom=bottom, left=0.10, right=0.98)
    fig.canvas.draw()
    _check_overlap(fig)
    if output_path:
        save_figure(fig, output_path)


def _caption_top(fig, wrapped, renderer, fig_h):
    boxes = [t.get_window_extent(renderer) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() == wrapped]
    return max(b.y1 for b in boxes) / fig_h if boxes else 0.0


def _lowest_axes_text(fig, wrapped, renderer, fig_h):
    lows = [t.get_window_extent(renderer).y0
            for t in fig.findobj(matplotlib.text.Text)
            if t is not None and t.get_text().strip() and t.get_visible()
            and t.get_text().strip() != wrapped]
    return min(lows) / fig_h if lows else 0.0


def _check_overlap(fig):
    """Contrôle objectif sur le layout final.

    - chevauchement : deux textes visibles se recouvrent (positions en fraction
      de figure pour diagnostiquer) ;
    - texte coupé   : texte avec clip_on=True (hors titres, labels et tick
      labels d'axes) dont l'étendue dépasse le clip path de son propriétaire —
      les autres textes sont inclus au rendu par bbox_inches='tight'.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figw, figh = fig.canvas.get_width_height()
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible()]
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    problems = []
    for i in range(len(boxes)):
        t1, b1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            t2, b2 = boxes[j]
            if b1.overlaps(b2):
                cx, cy = b1.x0 / figw, b1.y0 / figh
                problems.append(
                    f"chevauchement @({cx:.2f},{cy:.2f}) : "
                    f"'{t1.get_text()[:30]}…' vs '{t2.get_text()[:30]}…'"
                )
    std = set()
    for ax in fig.axes:
        std.add(ax.title)
        std.add(ax.xaxis.label)
        std.add(ax.yaxis.label)
        std.update(ax.xaxis.get_ticklabels())
        std.update(ax.yaxis.get_ticklabels())
    for t, b in boxes:
        if t in std or not t.get_clip_on():
            continue
        cp = t.get_clip_path()
        if cp is None:
            continue
        try:
            cp_bb = cp.get_window_extent(renderer)
        except Exception:
            continue
        if not (cp_bb.contains(b.x0, b.y0) and cp_bb.contains(b.x1, b.y1)):
            problems.append(f"texte coupé : '{t.get_text()[:30]}…'")
    if problems:
        print(f"  [!!] {len(problems)} problème(s) de texte :")
        for p in problems[:12]:
            print("      -", p)
    else:
        print("  [OK] aucun chevauchement ni coupure de texte")


def broken_axes(figsize=(8, 4.5), height_ratios=(3, 1), hspace=0.08):
    """Figure à 2 panneaux empilés partageant l'axe x (échelle cassée).

    Les labels de ticks les plus proches de la cassure sont masqués (bas du
    panneau haut, haut du panneau bas) pour garantir qu'aucun tick label ne
    chevauche l'autre panneau.
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": hspace},
    )
    ax_top.spines.bottom.set_visible(False)
    ax_bottom.spines.top.set_visible(False)
    ax_top.tick_params(labelbottom=False)
    if ax_top.yaxis.get_major_ticks():
        ax_top.yaxis.get_major_ticks()[0].label1.set_visible(False)
    if ax_bottom.yaxis.get_major_ticks():
        ax_bottom.yaxis.get_major_ticks()[-1].label1.set_visible(False)
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color=TEXT_COLOR, lw=1.0, clip_on=False)
    ax_top.plot([-d, d], [-d, d], **kwargs)
    ax_top.plot([1 - d, 1 + d], [-d, d], **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot([-d, d], [1 - d, 1 + d], **kwargs)
    ax_bottom.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)
    return fig, ax_top, ax_bottom


def save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=EXPORT_DPI, facecolor=BACKGROUND_COLOR,
        bbox_inches="tight", pad_inches=0.25,
    )
