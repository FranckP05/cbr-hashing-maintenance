"""Figures NB06 : scaling — FIG-R3 (temps de requête) et FIG-R4 (temps de construction).

Lecture unique de scaling_timing_quality.csv (aucune valeur en dur).
Panneaux Diabetes-130 / ChestX-ray14 (Heart exclu : pas de scaling, n=237).
Agrégation : moyenne sur 6 seeds et k ; barres d'erreur = écart type inter-seeds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp2.src.visualization.theme import (
    METHOD_COLORS,
    METHOD_STYLES,
    apply_theme,
    finalize,
    title,
)

PANELS = ["diabetes130", "chestxray14"]
METHODS = ["kdtree", "linear", "lsh"]
DATASET_LABELS = {"diabetes130": "Diabetes-130", "chestxray14": "ChestX-ray14"}
METHOD_LABELS = {"kdtree": "KD-Tree", "linear": "Baseline", "lsh": "LSH"}


def color(key):
    return {
        "kdtree": METHOD_COLORS["KD-Tree"],
        "linear": METHOD_COLORS["Baseline"],
        "lsh": METHOD_COLORS["LSH"],
    }[key]


def aggregate(df):
    df = df.copy()
    df["seed"] = df["seed"].astype(str)
    return df.groupby(["dataset", "base_size", "method", "seed"]).agg(
        query_ms=("query_ms", "mean"),
        build_ms=("build_ms", "mean"),
    ).reset_index()


def panel_data(per_seed, metric):
    data = per_seed.groupby(["dataset", "base_size", "method"])[metric].agg(["mean", "std"]).reset_index()
    return data[data["mean"].notna()]


def add_curves(ax, data, metric, err):
    for method in METHODS:
        rows = data[data["method"] == method]
        style = METHOD_STYLES[METHOD_LABELS[method]]
        values = rows["mean"] / 1e3 if metric == "build_ms" else rows["mean"]
        ax.plot(
            rows["base_size"], values, color=color(method),
            marker=style["marker"], linestyle=style["linestyle"],
            label=METHOD_LABELS[method],
        )
        if err:
            ax.errorbar(
                rows["base_size"], values, yerr=rows["std"],
                fmt="none", ecolor=color(method), capsize=2, linewidth=0.8,
            )


def build_figure(per_seed, metric, ylabel, caption, output_path):
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dataset in zip(axes, PANELS):
        data = panel_data(per_seed, metric)
        data = data[data["dataset"] == dataset]
        add_curves(ax, data, metric, err=metric != "build_ms")
        sizes = data["base_size"].unique()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels(["10k", "50k", "full"], fontsize=8)
        ax.set_ylim((1, 1000) if metric == "query_ms" else (0.1, 100))
        ax.set_xlabel("Taille de base")
        ax.set_ylabel(ylabel)
        title(ax, DATASET_LABELS[dataset])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower right", ncol=1, fontsize=8)
    finalize(fig, caption, output_path)


def main():
    parser = argparse.ArgumentParser(description="Figures FIG-R3 / FIG-R4 (NB06).")
    parser.add_argument("--csv", required=True, help="scaling_timing_quality.csv")
    parser.add_argument("--outdir", required=True, help="dossier de sortie")
    args = parser.parse_args()

    per_seed = aggregate(pd.read_csv(args.csv))
    outdir = Path(args.outdir)
    build_figure(
        per_seed, "query_ms", "Temps de requête (ms)",
        "FIGURE FIG-R3 — Temps de requête vs taille de base, échelle log-log (moyenne sur 6 seeds).",
        outdir / "FIG-R3_scaling_temps_requete.png",
    )
    build_figure(
        per_seed, "build_ms", "Temps de construction (s)",
        "FIGURE FIG-R4 — Temps de construction d'index vs taille de base, échelle log-log.",
        outdir / "FIG-R4_scaling_temps_construction.png",
    )


if __name__ == "__main__":
    main()
