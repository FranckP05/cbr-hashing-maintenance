"""Génération des figures 1-3 du benchmark (Baseline vs SOTA LSH vs HeCBR).

Style : TITRE + GRAPHE + LÉGENDE uniquement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.paths import FIGURES_DIR, BENCHMARK_TIMING_CSV, BENCHMARK_QUALITY_CSV
from visualization.theme import (
    configure_matplotlib, new_figure, set_title, save_figure,
    COLOR_HECBR, COLOR_SOTA_LSH, COLOR_BASELINE,
    STYLE_HECBR, STYLE_SOTA_LSH, STYLE_BASELINE,
    METHOD_LABELS,
)

configure_matplotlib()

METHODS = ["baseline", "sota_lsh", "hecbr_mp", "hecbr"]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _method_style(m: str):
    if m == "baseline":
        return STYLE_BASELINE["color"]
    if m == "sota_lsh":
        return STYLE_SOTA_LSH["color"]
    if m == "hecbr" or m == "hecbr_mp":
        return STYLE_HECBR["color"]
    return STYLE_BASELINE["color"]


def plot_latency_comparison(
    df_timing: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig1_latency_comparison.png",
) -> None:
    """Figure 1 : Latence médiane de requête par méthode."""
    df_timing = _clean_df(df_timing)
    methods  = [m for m in METHODS if m in df_timing["method"].values]
    labels   = [METHOD_LABELS.get(m, m) for m in methods]

    agg = df_timing.groupby("method")["median_ms"].median().reindex(methods)
    agg = agg.dropna()

    fig, ax = new_figure()
    x = np.arange(len(agg))
    ax.bar(x, agg.values, color=[_method_style(m) for m in agg.index], width=0.5, alpha=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in agg.index], rotation=12, ha="right")
    ax.set_ylabel("Latence médiane de requête (ms)")
    set_title(ax, "Latence de requête — Baseline vs LSH vs HeCBR")
    save_figure(fig, output_path)


def plot_speed_quality_tradeoff(
    df_quality: pd.DataFrame,
    df_timing:  pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig2_speed_quality.png",
) -> None:
    """Figure 2 : Compromis Vitesse (latence) vs Qualité (Recall@k). Recall mesuré par méthode
    sur sa propre vérité terrain (L2 pour baseline/LSH, Hamming pour HeCBR)."""
    df_timing  = _clean_df(df_timing)
    df_quality = _clean_df(df_quality)
    fig, ax = new_figure()

    for method in METHODS:
        if method not in df_timing["method"].values:
            continue
        dm = df_timing[df_timing["method"] == method]
        dq = df_quality[df_quality["method"] == method]
        marker = {"baseline": "s", "sota_lsh": "^", "hecbr_mp": "D", "hecbr": "o"}[method]
        ax.scatter(
            dm["median_ms"].values, dq["recall_at_k"].values,
            color=_method_style(method), marker=marker,
            label=METHOD_LABELS.get(method, method), s=90, zorder=3,
        )

    ax.set_xlabel("Latence médiane de requête (ms)")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="Seuil Recall=0.90")
    ax.legend(loc="lower right", fontsize=8)
    set_title(ax, "Compromis Vitesse / Qualité")
    save_figure(fig, output_path)


def plot_memory_footprint(
    n_points: int,
    output_path: Path = FIGURES_DIR / "fig3_memory_footprint.png",
) -> None:
    """Figure 3 : Empreinte mémoire de l'index selon N points en base."""
    mem = {
        "baseline":  n_points * 512 * 4,
        "sota_lsh":  n_points * (512 * 4 + 10 * 4),
        "hecbr_mp":  n_points * (8 + 8 * 8),
        "hecbr":     n_points * 8,
    }
    names = [m for m in METHODS]
    values = [mem[m] / (1024 * 1024) for m in names]
    labels = [METHOD_LABELS.get(m, m) for m in names]

    fig, ax = new_figure()
    x = np.arange(len(names))
    ax.bar(x, values, color=[_method_style(m) for m in names], width=0.5, alpha=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Empreinte mémoire de l'index (Mo)")
    set_title(ax, "Empreinte mémoire de l'index")
    save_figure(fig, output_path)


def generate_all_figures(
    timing_csv:  Path = BENCHMARK_TIMING_CSV,
    quality_csv: Path = BENCHMARK_QUALITY_CSV,
    n_points: int | None = None,
) -> None:
    df_timing  = pd.read_csv(timing_csv)
    df_quality = pd.read_csv(quality_csv)
    plot_latency_comparison(df_timing)
    plot_speed_quality_tradeoff(df_quality, df_timing)
    plot_memory_footprint(n_points or 3772657)
