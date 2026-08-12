"""Figures NB08 : maintenance — FIG-R5 (accord rétention), FIG-R6 (qualité
post-maintenance), FIG-R7 (temps post-maintenance).

Lecture unique de retention_agreement.csv et condensation_quality.csv.
Agrégation : moyenne sur 6 seeds ; barres d'erreur = écart type inter-seeds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp2.src.visualization.theme import (
    METHOD_COLORS,
    apply_theme,
    finalize,
    title,
)

DATASETS = ["heart", "diabetes130", "chestxray14"]
DATASET_LABELS = {
    "heart": "Heart Disease",
    "diabetes130": "Diabetes-130",
    "chestxray14": "ChestX-ray14",
}
STRATEGIES = ["cbm", "bucket", "cnn", "random"]
STRATEGY_LABELS = {
    "cbm": "CBM",
    "bucket": "Bucket",
    "cnn": "CNN",
    "random": "Aléatoire",
}
STRATEGY_COLORS = {
    "cbm": METHOD_COLORS["HeCBR"],
    "bucket": METHOD_COLORS["LSH"],
    "cnn": METHOD_COLORS["KD-Tree"],
    "random": METHOD_COLORS["Baseline"],
}
RATES = [0.2, 0.5]

COLOR_EXACT = "#0284C7"


def aggregate_retention(df):
    df = df.copy()
    df["seed"] = df["seed"].astype(str)
    return df.groupby(["dataset", "seed"]).agg(
        red_lsh=("reduction_rate_lsh", "mean"),
        red_exact=("reduction_rate_exact", "mean"),
        agreement=("agreement", "mean"),
    ).reset_index()


def aggregate_condensation(df):
    df = df.copy()
    df["seed"] = df["seed"].astype(str)
    return df.groupby(["dataset", "rate", "strategy", "seed"]).agg(
        macro_f1=("nn_macro_f1", "mean"),
        query_ms=("query_ms_linear", "mean"),
    ).reset_index()


def format_group_axis(ax, ticks, labels, fontsize=7):
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fontsize)


def fig_r5(ret, output_path):
    apply_theme()
    fig, (ax_red, ax_agree) = plt.subplots(1, 2, figsize=(11, 4.2))
    stats = ret.groupby("dataset").agg(
        red_lsh=("red_lsh", "mean"), std_lsh=("red_lsh", "std"),
        red_exact=("red_exact", "mean"), std_exact=("red_exact", "std"),
        agreement=("agreement", "mean"),
    ).reset_index()
    x = np.arange(len(DATASETS))
    width = 0.35
    ax_red.bar(
        x - width / 2, stats["red_lsh"], width, yerr=stats["std_lsh"], capsize=2,
        color=METHOD_COLORS["LSH"], label="LSH",
    )
    ax_red.bar(
        x + width / 2, stats["red_exact"], width, yerr=stats["std_exact"], capsize=2,
        color=COLOR_EXACT, label="Exacte",
    )
    ax_red.set_ylim(0, 0.45)
    ax_red.set_ylabel("Taux de réduction")
    ax_red.legend(loc="upper right", fontsize=8)
    title(ax_red, "Taux de réduction")
    ax_agree.bar(x, stats["agreement"], 0.5, color=METHOD_COLORS["HeCBR"])
    ax_agree.set_ylim(0, 1.15)
    ax_agree.set_ylabel("Accord cas par cas")
    title(ax_agree, "Accord cas par cas")
    for ax in (ax_red, ax_agree):
        format_group_axis(ax, x, [DATASET_LABELS[d] for d in DATASETS])
    caption = (
        "FIGURE FIG-R5 — Rétention LSH vs exacte : taux de réduction identique et "
        "accord cas par cas de 100 % (moyenne sur 6 seeds)."
    )
    finalize(fig, caption, output_path)


def _cond_stats(cond, dataset, metric):
    stats = cond[cond["dataset"] == dataset].groupby(["rate", "strategy"])[metric].agg(["mean", "std"]).reset_index()
    return stats


def _draw_cond(ax, cond, dataset, metric, ylim, ylabel, raw_label):
    stats = _cond_stats(cond, dataset, metric)
    raw = stats[stats["strategy"] == "raw"]
    width = 0.18
    x = np.arange(len(RATES))
    for j, strategy in enumerate(STRATEGIES):
        rows = stats[stats["strategy"] == strategy]
        vals = [rows.loc[rows.rate == r, "mean"].iloc[0] for r in RATES]
        errs = [rows.loc[rows.rate == r, "std"].iloc[0] for r in RATES]
        ax.bar(
            x + (j - 1.5) * width, vals, width, yerr=errs, capsize=2,
            color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy],
            edgecolor="white", linewidth=0.3,
        )
    raw_mean = raw["mean"].iloc[0]
    ax.axhline(raw_mean, color="#0F172A", linestyle="--", linewidth=1.2,
               label="raw (base brute)")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    format_group_axis(ax, x, ["0.2", "0.5"])
    title(ax, DATASET_LABELS[dataset])


def fig_r6(cond, output_path):
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    ylims = {"heart": (0.55, 0.90), "diabetes130": (0.35, 0.65), "chestxray14": (0.45, 0.75)}
    for ax, ds in zip(axes, DATASETS):
        _draw_cond(ax, cond, ds, "macro_f1", ylims[ds], "macro-F1", None)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper center", ncol=3, fontsize=7)
    caption = (
        "FIGURE FIG-R6 — Qualité (macro-F1) du k-NN après condensation à taux de "
        "réduction égal (0.2 / 0.5) vs base brute, moyenne sur 6 seeds."
    )
    finalize(fig, caption, output_path)


def fig_r7(cond, output_path):
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    ylims = {"heart": (0.01, 0.1), "diabetes130": (0.05, 10), "chestxray14": (0.1, 20)}
    for ax, ds in zip(axes, DATASETS):
        _draw_cond(ax, cond, ds, "query_ms", ylims[ds], "Temps de requête (ms)", None)
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", ncol=3, fontsize=7)
    caption = (
        "FIGURE FIG-R7 — Temps de requête (scan linéaire) après condensation vs base "
        "brute, échelle log, taux 0.2 / 0.5, moyenne sur 6 seeds."
    )
    finalize(fig, caption, output_path)


def main():
    parser = argparse.ArgumentParser(description="Figures FIG-R5 / FIG-R6 / FIG-R7 (NB08).")
    parser.add_argument("--retention", required=True, help="retention_agreement.csv")
    parser.add_argument("--condensation", required=True, help="condensation_quality.csv")
    parser.add_argument("--outdir", required=True, help="dossier de sortie")
    args = parser.parse_args()

    ret = aggregate_retention(pd.read_csv(args.retention))
    cond = aggregate_condensation(pd.read_csv(args.condensation))
    outdir = Path(args.outdir)
    fig_r5(ret, outdir / "FIG-R5_accord_retention.png")
    fig_r6(cond, outdir / "FIG-R6_qualite_post_maintenance.png")
    fig_r7(cond, outdir / "FIG-R7_temps_post_maintenance.png")


if __name__ == "__main__":
    main()
