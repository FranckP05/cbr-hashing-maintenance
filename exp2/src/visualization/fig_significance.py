"""Figure NB11 : matrice de significativité (FIG-R10).

Lecture unique de significance_tests.csv (Wilcoxon signed-rank, 6 seeds
appariés, k agrégé). Une matrice méthode × méthode par (dataset, métrique) :
orange = p < 0.05, gris = non significatif, « ns » sinon.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp2.src.visualization.theme import (
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
METHODS = ["linear", "kdtree", "lsh"]
METHOD_LABELS = {"linear": "Linear", "kdtree": "KD-Tree", "lsh": "LSH"}
METRICS = ["query_ms", "exact_recall_at_k", "ndcg_at_k", "macro_f1"]
METRIC_LABELS = {
    "query_ms": "Temps requête",
    "exact_recall_at_k": "Rappel voisins exacts",
    "ndcg_at_k": "nDCG@k",
    "macro_f1": "macro-F1",
}

COLOR_SIG = "#D55E00"
COLOR_NS = "#E2E8F0"
COLOR_TEXT = "#0F172A"


def load(path):
    df = pd.read_csv(path)
    return df[df["k"] == "agg"]


def build_matrix(rows, dataset, metric):
    rows = rows[(rows["dataset"] == dataset) & (rows["metric"] == metric)]
    p = np.full((3, 3), np.nan)
    for _, r in rows.iterrows():
        i = METHODS.index(r["method_a"])
        j = METHODS.index(r["method_b"])
        p[i, j] = r["wilcoxon_p"]
        p[j, i] = r["wilcoxon_p"]
    return p


def render(ax, p):
    img = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            if i == j or np.isnan(p[i, j]):
                img[i, j] = (1, 1, 1)
            elif p[i, j] < 0.05:
                img[i, j] = tuple(int(COLOR_SIG[k:k + 2], 16) / 255 for k in (1, 3, 5))
            else:
                img[i, j] = tuple(int(COLOR_NS[k:k + 2], 16) / 255 for k in (1, 3, 5))
    ax.imshow(img, interpolation="nearest")
    for i in range(3):
        for j in range(3):
            if i == j or np.isnan(p[i, j]):
                text = ""
            elif p[i, j] < 0.05:
                text = f"{p[i, j]:.2f}"
            else:
                text = "ns"
            ax.text(i, j, text, ha="center", va="center", fontsize=7, color=COLOR_TEXT)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=6)
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=6)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)


def fig_r10(rows, output_path):
    apply_theme()
    fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(12, 7))
    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            p = build_matrix(rows, ds, metric)
            render(ax, p)
            if i == 0:
                title(ax, METRIC_LABELS[metric], fontsize=8)
            if j == 0:
                ax.set_ylabel(DATASET_LABELS[ds], fontsize=8)
    caption = (
        "FIGURE FIG-R10 — Matrice de significativité (Wilcoxon signed-rank, 6 seeds "
        "appariés, k agrégé) : orange = p < 0.05, gris = non significatif. "
        "HeCBR exclu (1 seed)."
    )
    finalize(fig, caption, output_path)


def main():
    parser = argparse.ArgumentParser(description="Figure FIG-R10 (NB11).")
    parser.add_argument("--csv", required=True, help="significance_tests.csv")
    parser.add_argument("--output", required=True, help="chemin du PNG")
    args = parser.parse_args()
    fig_r10(load(args.csv), args.output)
    print(f"[OK] figure écrite : {args.output}")


if __name__ == "__main__":
    main()
