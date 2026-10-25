"""Figures simplifiées — Complexité temporelle & maintenance du hachage.

Principe : TITRE + GRAPHE + LÉGENDE uniquement. Pas d'annotations, pas de sous-titres
bavards, pas de repères superposés. Réutilise la charte graphique (fonts, couleurs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.paths import (
    FIGURES_DIR,
    MAINTENANCE_SCALING_CSV, MAINTENANCE_BUILD_CSV,
    MULTIPROBE_TRADEOFF_CSV, MAINTENANCE_ADD10K_CSV,
    MAINTENANCE_BREAKEVEN_CSV,
)
from visualization.theme import (
    configure_matplotlib, new_figure, set_title, save_figure,
    STYLE_HECBR, STYLE_SOTA_LSH, STYLE_BASELINE,
)

configure_matplotlib()


def plot_scaling_latency(df: pd.DataFrame, output: Path = FIGURES_DIR / "fig4_scaling_latency.png") -> None:
    fig, ax = new_figure()
    ax.loglog(df["N"], df["baseline_ms"], **STYLE_BASELINE, label="Baseline (scan linéaire)")
    ax.loglog(df["N"], df["lsh_ms"],      **STYLE_SOTA_LSH, label="LSH aléatoire")
    ax.loglog(df["N"], df["hecbr_mp_ms"], **STYLE_HECBR,    label="HeCBR (data-dependent)")
    ax.set_xlabel("Taille de la base N")
    ax.set_ylabel("Latence par requête (ms)")
    ax.legend()
    set_title(ax, "Latence de requête vs taille de base")
    save_figure(fig, output)


def plot_build_time(df: pd.DataFrame, output: Path = FIGURES_DIR / "fig5_build_time.png") -> None:
    fig, ax = new_figure()
    ax.loglog(df["N"], df["baseline_s"], **STYLE_BASELINE, label="Baseline (scan linéaire)")
    ax.loglog(df["N"], df["lsh_s"],      **STYLE_SOTA_LSH, label="LSH aléatoire")
    ax.loglog(df["N"], df["hecbr_s"],    **STYLE_HECBR,    label="HeCBR (data-dependent)")
    ax.set_xlabel("Taille de la base N")
    ax.set_ylabel("Temps de construction (s)")
    ax.legend()
    set_title(ax, "Temps de construction de l'index")
    save_figure(fig, output)


def plot_multiprobe_tradeoff(df: pd.DataFrame, output: Path = FIGURES_DIR / "fig6_multiprobe_tradeoff.png") -> None:
    fig, ax = new_figure()
    ax.plot(df["lat_ms"], df["recall_hamming"], "o-", color=STYLE_HECBR["color"], linewidth=1.8, markersize=7, label="HeCBR multi-probe")
    ax.axhline(1.0, color=STYLE_BASELINE["color"], linestyle=":", linewidth=1.5, label="Scan exact (Recall=1.0)")
    ax.set_xlabel("Latence par requête (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.legend()
    set_title(ax, "Compromis vitesse / rappel du multi-probe")
    save_figure(fig, output)


def plot_maintenance(df: pd.DataFrame, output: Path = FIGURES_DIR / "fig7_maintenance.png") -> None:
    fig, ax1 = new_figure()
    methods = df["methode"].tolist()
    labels = ["Baseline", "HeCBR"]
    n = len(methods)
    x = np.arange(n)
    w = 0.35

    c1 = [STYLE_BASELINE["color"] if "Baseline" in m else STYLE_HECBR["color"] for m in methods]
    b1 = ax1.bar(x - w/2, df["maj_cost_s"], width=w, color=c1, alpha=0.85, zorder=3, label="Coût de mise à jour")
    ax1.set_ylabel("Coût de mise à jour (+10k pts) (s)")

    ax2 = ax1.twinx()
    c2 = ["#FFC107" if "Baseline" in m else "#00BFA5" for m in methods]
    b2 = ax2.bar(x + w/2, df["lat_after_ms"], width=w, color=c2, alpha=0.85, zorder=3, label="Latence de requête")
    ax2.set_ylabel("Latence de requête après mise à jour (ms)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    set_title(ax1, "Maintenance : coût de mise à jour vs latence de requête")
    save_figure(fig, output)


def plot_breakeven(df: pd.DataFrame, output: Path = FIGURES_DIR / "fig8_breakeven.png") -> None:
    """Coût opérationnel cumulé après mise à jour : update + requêtes × latence.

    Montre le point d'équilibre où le hachage rattrape le surcoût d'indexation.
    Honnête : ne juxtapose pas des grandeurs incompatibles (ponctuel vs par requête),
    mais résume le vrai coût total en fonction du nombre de requêtes lancées.
    """
    cbr = df[df["methode"].str.contains("CBR", case=False)].iloc[0]
    heb = df[df["methode"].str.contains("HeCBR", case=False)].iloc[0]
    u_c, l_c = float(cbr["update_cost_s"]), float(cbr["query_lat_ms"]) / 1000.0
    u_h, l_h = float(heb["update_cost_s"]), float(heb["query_lat_ms"]) / 1000.0

    q = np.linspace(0, 120, 400)
    cost_cbr = u_c + q * l_c
    cost_heb = u_h + q * l_h
    q_star = (u_h - u_c) / (l_c - l_h)

    fig, ax = new_figure()
    ax.plot(q, cost_cbr, **STYLE_BASELINE, label="CBR exacte (insertion triviale)")
    ax.plot(q, cost_heb, **STYLE_HECBR, label="HeCBR (encoder + index)")
    ax.axvline(q_star, color=STYLE_SOTA_LSH["color"], linestyle=":", linewidth=1.5,
               label="Point d'équilibre ≈ %.0f requêtes" % q_star)
    ax.set_xlabel("Requêtes lancées après la mise à jour")
    ax.set_ylabel("Coût cumulé (s)")
    ax.legend(loc="upper left")
    set_title(ax, "Maintenance : coût opérationnel cumulé")
    save_figure(fig, output)


def generate_all() -> None:
    plot_scaling_latency(pd.read_csv(MAINTENANCE_SCALING_CSV))
    plot_build_time(pd.read_csv(MAINTENANCE_BUILD_CSV))
    plot_multiprobe_tradeoff(pd.read_csv(MULTIPROBE_TRADEOFF_CSV))
    plot_maintenance(pd.read_csv(MAINTENANCE_ADD10K_CSV))
    plot_breakeven(pd.read_csv(MAINTENANCE_BREAKEVEN_CSV))
    print("Figures simplifiées générées dans", FIGURES_DIR)


if __name__ == "__main__":
    generate_all()
