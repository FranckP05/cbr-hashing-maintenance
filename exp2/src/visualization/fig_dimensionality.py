"""Figures NB10 : dimensionnalité — FIG-R8 (compartiments vs dimension) et
FIG-R9 (fraction de candidats vs dimension à n fixé).

Lecture unique de dim_calibration.csv + nb09_summary.json (configuration retenue
par dataset, T4.1 : rappel ≥ 0.99 à compartiments minimaux). Aucune valeur en dur.
3 points par figure (13 / 390 / 1024) : le message est le sens de croissance, pas
la forme de la courbe.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from exp2.src.visualization.theme import (
    DIM_COLOR,
    apply_theme,
    finalize,
    title,
)

DATASET_LABELS = {
    "heart": "Heart\n(13-d)",
    "diabetes130": "Diabetes-130\n(390-d)",
    "chestxray14": "ChestX-ray14\n(1024-d)",
}


def load(csv_path, summary_path):
    df = pd.read_csv(csv_path)
    chosen = json.loads(Path(summary_path).read_text())["chosen"]
    rows = []
    for dataset, cfg in chosen.items():
        row = df[
            (df["dataset"] == dataset)
            & (df["num_bits"] == cfg["num_bits"])
            & (df["num_tables"] == cfg["num_tables"])
        ]
        assert len(row) == 1, f"cellule {cfg} introuvable pour {dataset}"
        rows.append(row.iloc[0])
    return pd.DataFrame(rows)


def _annotate(ax, data):
    for _, r in data.iterrows():
        ax.annotate(
            f"{r['num_bits']} bits × {r['num_tables']} tab.",
            (r["dim"], r["compartments"]),
            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7,
        )


def fig_r8(data, output_path):
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(data["dim"], data["compartments"], s=70, color=DIM_COLOR, zorder=3)
    _annotate(ax, data)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(data["dim"])
    ax.set_xticklabels([DATASET_LABELS[d] for d in data["dataset"]], fontsize=8)
    ax.set_xlabel("Dimensionnalité")
    ax.set_ylabel("Compartiments LSH (2^bits × tables)")
    title(ax, "Compartiments LSH pour rappel ≥ 0.99 à base fixée")
    caption = (
        "FIGURE FIG-R8 — Compartiments (2^bits × tables) de la configuration retenue "
        "pour rappel ≥ 0.99 à taille de base fixée, vs dimensionnalité (échelle log-log). "
        "Bits/tables annotés."
    )
    finalize(fig, caption, output_path)


def fig_r9(data, output_path):
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(data["dim"], data["mean_cand_fraction"], s=70, color=DIM_COLOR, zorder=3)
    for _, r in data.iterrows():
        ax.annotate(
            f"{r['mean_cand_fraction']:.0%}",
            (r["dim"], r["mean_cand_fraction"]),
            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xticks(data["dim"])
    ax.set_xticklabels([DATASET_LABELS[d] for d in data["dataset"]], fontsize=8)
    ax.set_xlabel("Dimensionnalité")
    ax.set_ylabel("Fraction de candidats filtrés (moyenne)")
    ax.set_ylim(0, 1.05)
    title(ax, "Fraction de candidats à taille de base fixée")
    caption = (
        "FIGURE FIG-R9 — Fraction de la base examinée par le filtrage LSH (compartiments"
        " / base, moyenne sur les requêtes) à taille de base fixée, vs dimensionnalité."
    )
    finalize(fig, caption, output_path)


def main():
    parser = argparse.ArgumentParser(description="Figures FIG-R8 / FIG-R9 (NB10).")
    parser.add_argument("--csv", required=True, help="dim_calibration.csv")
    parser.add_argument("--summary", required=True, help="nb09_summary.json")
    parser.add_argument("--outdir", required=True, help="dossier de sortie")
    args = parser.parse_args()

    data = load(args.csv, args.summary)
    outdir = Path(args.outdir)
    fig_r8(data, outdir / "FIG-R8_compartiments_dimension.png")
    fig_r9(data, outdir / "FIG-R9_candidats_dimension.png")


if __name__ == "__main__":
    main()
