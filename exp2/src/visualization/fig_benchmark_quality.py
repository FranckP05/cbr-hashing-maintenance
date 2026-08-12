"""Figures NB04 : qualité de récupération (FIG-R1) et temps de requête (FIG-R2).

Lecture unique de benchmark_timing_quality.csv (aucune valeur en dur).
Agrégation : moyenne sur les seeds et k ∈ {1,5,10} ; barres d'erreur = écart
type inter-seeds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp2.src.visualization.theme import (
    FIGURE_SIZE,
    METHOD_COLORS,
    apply_theme,
    finalize,
    title,
)

DATASETS = ["heart", "diabetes130", "chestxray14"]
METHODS = ["hecbr", "kdtree", "linear", "lsh"]
DATASET_LABELS = {
    "heart": "Heart Disease",
    "diabetes130": "Diabetes-130",
    "chestxray14": "ChestX-ray14",
}
METHOD_LABELS = {
    "hecbr": "HeCBR",
    "kdtree": "KD-Tree",
    "linear": "Linear",
    "lsh": "LSH",
}


def color(key):
    return {
        "hecbr": METHOD_COLORS["HeCBR"],
        "kdtree": METHOD_COLORS["KD-Tree"],
        "linear": METHOD_COLORS["Baseline"],
        "lsh": METHOD_COLORS["LSH"],
    }[key]


def aggregate(df):
    df = df.copy()
    df["seed"] = df["seed"].astype(str)
    return df.groupby(["dataset", "method", "seed"]).agg(
        exact_recall=("exact_recall_at_k", "mean"),
        ndcg=("ndcg_at_k", "mean"),
        macro_f1=("macro_f1", "mean"),
        query_ms=("query_ms", "mean"),
    ).reset_index()


def bars_for(per_seed, metric):
    data = per_seed.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).reset_index()
    return data[data["mean"].notna()]


def add_method_bars(ax, data, group_pos, width=0.2):
    for j, method in enumerate(METHODS):
        rows = data[data["method"] == method]
        if rows.empty:
            continue
        vals = [rows.loc[rows.dataset == d, "mean"].iloc[0] if len(rows[rows.dataset == d]) else np.nan
                for d in DATASETS]
        errs = [rows.loc[rows.dataset == d, "std"].iloc[0] if len(rows[rows.dataset == d]) else np.nan
                for d in DATASETS]
        ax.bar(
            group_pos + (j - 1.5) * width, vals, width, yerr=errs, capsize=2,
            color=color(method), label=METHOD_LABELS[method],
            edgecolor="white", linewidth=0.3,
        )


def format_group_axis(ax):
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=7)


def fig_r1(per_seed, output_path):
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specs = [
        ("exact_recall", "Rappel voisins exacts", (0.90, 1.02)),
        ("ndcg", "nDCG@k", (0.55, 0.90)),
        ("macro_f1", "macro-F1", (0.40, 0.95)),
    ]
    for ax, (metric, label, ylim) in zip(axes, specs):
        data = bars_for(per_seed, metric)
        add_method_bars(ax, data, np.arange(len(DATASETS)))
        format_group_axis(ax)
        ax.set_ylim(*ylim)
        title(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", ncol=2, fontsize=7)
    caption = (
        "FIGURE FIG-R1 — Rappel des voisins exacts, nDCG@k et macro-F1 par méthode et "
        "dataset (moyenne sur 6 seeds, k = 1, 5, 10 ; HeCBR figé, 1 seed)."
    )
    finalize(fig, caption, output_path)


def fig_r2(per_seed, output_path):
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    data = per_seed.groupby(["dataset", "method"])["query_ms"].agg(["mean", "std"]).reset_index()
    data = data[data["mean"].notna()]
    add_method_bars(ax, data, np.arange(len(DATASETS)))
    ax.set_yscale("log")
    ax.set_ylim(0.01, 500)
    format_group_axis(ax)
    ax.set_ylabel("Temps de requête (ms, échelle log)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", ncol=2, fontsize=8)
    title(ax, "Temps de requête par méthode")
    caption = (
        "FIGURE FIG-R2 — Temps de requête par méthode et dataset, échelle log "
        "(protocole unique, moyenne sur 6 seeds, k = 1, 5, 10)."
    )
    finalize(fig, caption, output_path)


def main():
    parser = argparse.ArgumentParser(description="Figures FIG-R1 / FIG-R2 (NB04).")
    parser.add_argument("--csv", required=True, help="benchmark_timing_quality.csv")
    parser.add_argument("--outdir", required=True, help="dossier de sortie")
    args = parser.parse_args()

    per_seed = aggregate(pd.read_csv(args.csv))
    outdir = Path(args.outdir)
    fig_r1(per_seed, outdir / "FIG-R1_qualite_recuperation.png")
    fig_r2(per_seed, outdir / "FIG-R2_temps_requete.png")


if __name__ == "__main__":
    main()
