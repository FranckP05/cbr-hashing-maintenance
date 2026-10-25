"""Génération des figures comparatives du benchmark (Matplotlib, 300 DPI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import FIGURES_DIR, BENCHMARK_TIMING_CSV, BENCHMARK_QUALITY_CSV, CALIBRATION_CSV
from visualization.theme import (
    configure_matplotlib, new_figure, set_title, save_figure,
    COLOR_SIMHASH, COLOR_BASELINE,
    STYLE_SIMHASH, STYLE_BASELINE,
    METHOD_LABELS,
)

# Force l'application du thème à l'import
configure_matplotlib()


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les espaces superflus dans les noms de colonnes et valeurs string."""
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


# =========================================================================== #
# Figure 1 — Comparaison des latences médianes (+ intervalles percentiles)
# =========================================================================== #

def plot_latency_comparison(
    df_timing: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig1_latency_comparison.png",
) -> None:
    """Barres de latence médiane avec moustaches p25-p75 et p95."""
    df_timing = _clean_df(df_timing)
    methods  = ["baseline", "simhash"]
    labels   = [METHOD_LABELS[m] for m in methods]

    # Agrégation inter-seeds
    agg = df_timing.groupby("method").agg(
        median_ms=("median_ms", "median"),
        p25_ms   =("p25_ms",    "median"),
        p75_ms   =("p75_ms",    "median"),
        p95_ms   =("p95_ms",    "median"),
    ).reindex(methods)

    fig, ax = new_figure()
    x = np.arange(len(methods))
    colors = [STYLE_BASELINE["color"], STYLE_SIMHASH["color"]]

    bars = ax.bar(
        x, agg["median_ms"].values,
        color=colors, width=0.5, alpha=0.85, zorder=3,
    )
    # Moustaches p25–p75
    ax.errorbar(
        x, agg["median_ms"].values,
        yerr=[
            agg["median_ms"].values - agg["p25_ms"].values,
            agg["p75_ms"].values - agg["median_ms"].values,
        ],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4,
    )
    # Marqueur p95
    ax.scatter(x, agg["p95_ms"].values, marker="D", color="black",
               zorder=5, s=30, label="p95")

    # Valeurs au-dessus des barres
    for i, v in enumerate(agg["median_ms"].values):
        ax.text(i, v + 15, f"{v:.1f} ms", ha="center", fontweight="bold", fontsize=10, color=STYLE_BASELINE["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latence de requête (ms)")
    ax.legend(loc="upper left")
    set_title(ax, "Latence médiane de requête — Camelyon16 (3.77M patches, 512-D)",
              subtitle="Médiane inter-seeds · Moustaches : [p25, p75] · Losange = p95")

    save_figure(fig, output_path)


# =========================================================================== #
# Figure 2 — Compromis Vitesse / Qualité (latence vs Recall@10 et F1)
# =========================================================================== #

def plot_speed_quality_tradeoff(
    df_quality: pd.DataFrame,
    df_timing:  pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig2_speed_quality.png",
) -> None:
    """Scatter plot latence médiane vs Recall@10 et Macro-F1."""
    df_timing  = _clean_df(df_timing)
    df_quality = _clean_df(df_quality)

    fig, ax = new_figure()

    for method, color, marker, label in [
        ("baseline", STYLE_BASELINE["color"], STYLE_BASELINE["marker"], METHOD_LABELS["baseline"]),
        ("simhash",  STYLE_SIMHASH["color"],  STYLE_SIMHASH["marker"],  METHOD_LABELS["simhash"]),
    ]:
        dm = df_timing[df_timing["method"] == method]
        dq = df_quality[df_quality["method"] == method]
        ax.scatter(
            dm["median_ms"].values,
            dq["recall_at_k"].values,
            color=color,
            marker=marker,
            label=f"{label} (Recall@10)",
            s=80, zorder=3,
        )

    ax.set_xlabel("Latence médiane de requête (ms)")
    ax.set_ylabel("Rappel@10")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1, label="Seuil Recall=0.95")
    ax.legend(loc="center right")
    set_title(ax, "Compromis Vitesse / Qualité — SimHash vs Balayage Linéaire",
              subtitle="Chaque point SimHash = 1 seed de hachage indépendant (3.77M points, 512-D)")

    save_figure(fig, output_path)


# =========================================================================== #
# Figure 3 — Distribution des percentiles de latence
# =========================================================================== #

def plot_latency_distribution(
    df_timing: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig3_latency_distribution.png",
) -> None:
    """Comparaison de la dispersion des latences."""
    df_timing = _clean_df(df_timing)
    fig, ax = new_figure()

    for method, color, marker, label in [
        ("baseline", STYLE_BASELINE["color"], STYLE_BASELINE["marker"], METHOD_LABELS["baseline"]),
        ("simhash",  STYLE_SIMHASH["color"],  STYLE_SIMHASH["marker"],  METHOD_LABELS["simhash"]),
    ]:
        dm = df_timing[df_timing["method"] == method]
        seeds = dm["seed"].unique()
        for seed in sorted(seeds):
            row = dm[dm["seed"] == seed].iloc[0]
            ax.plot(
                [label], [row["median_ms"]],
                color=color, marker=marker, markersize=8, zorder=3, alpha=0.8,
            )

    ax.set_ylabel("Latence médiane par requête (ms)")
    set_title(ax, "Dispersion inter-seeds des latences de requête",
              subtitle="Chaque marqueur = 1 seed de hachage indépendant")

    save_figure(fig, output_path)


# =========================================================================== #
# Figure 4 — Surface de calibration (k, L) sur la validation
# =========================================================================== #

def plot_calibration_surface(
    df_calib: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig4_calibration_surface.png",
) -> None:
    """Heatmap du temps de requête / Recall en fonction de (k, L) sur la validation."""
    df_calib = _clean_df(df_calib)
    import matplotlib.pyplot as plt

    pivot_time = df_calib.pivot(index="num_tables", columns="num_bits", values="median_ms")

    fig, ax = new_figure()
    cmap = plt.get_cmap("YlOrRd")
    im   = ax.imshow(pivot_time.values, aspect="auto", cmap=cmap, origin="lower")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Latence médiane (ms)")

    ax.set_xticks(range(len(pivot_time.columns)))
    ax.set_xticklabels(pivot_time.columns)
    ax.set_yticks(range(len(pivot_time.index)))
    ax.set_yticklabels(pivot_time.index)
    ax.set_xlabel("k (bits par table)")
    ax.set_ylabel("L (nombre de tables)")
    set_title(ax, "Surface de calibration SimHash — Latence vs (k, L)",
              subtitle="Validation sur Camelyon16 (3.77M patches, 512-D)")

    save_figure(fig, output_path)


# =========================================================================== #
# Génération de toutes les figures depuis les CSV de résultats
# =========================================================================== #

def generate_all_figures(
    timing_csv:  Path = BENCHMARK_TIMING_CSV,
    quality_csv: Path = BENCHMARK_QUALITY_CSV,
    calib_csv:   Path = CALIBRATION_CSV,
) -> None:
    """Charge les CSV de résultats et génère toutes les figures."""
    df_timing  = pd.read_csv(timing_csv)
    df_quality = pd.read_csv(quality_csv)
    df_calib   = pd.read_csv(calib_csv)

    plot_latency_comparison(df_timing)
    plot_speed_quality_tradeoff(df_quality, df_timing)
    plot_latency_distribution(df_timing)
    plot_calibration_surface(df_calib)
