"""Figure rapport : compromis rappel / coût de calibration LSH (ChestX-ray14).

Tous les couples calibrés, abscisses ordonnées par permissivité croissante
(num_tables, puis num_bits décroissant). Double axe : rappel (moyen/min) à
gauche, temps de construction à droite. Lecture unique des fichiers de
résultats (aucune valeur en dur).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from exp2.src.visualization.theme import (
    FIGURE_SIZE,
    apply_theme,
    finalize,
    title,
)

COLOR_LSH = "#D55E00"
COLOR_KD = "#0284C7"
COLOR_BUILD = "#0F172A"
COLOR_TARGET = "#94A3B8"

DATASET = "chestxray14"


def retained_params(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    return summary["protocol"]["lsh_params_chosen"][DATASET]


def by_permissiveness(calib):
    return calib.sort_values(["num_tables", "num_bits"], ascending=[True, False])


def plot(top, chosen_pos, output_path):
    apply_theme()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax_build = ax.twinx()
    x = list(range(len(top)))
    labels = [f"{int(r['num_bits'])}/{int(r['num_tables'])}" for _, r in top.iterrows()]

    ax.plot(
        x, top["mean_recall"], color=COLOR_LSH, marker="o", linestyle="-",
        label="rappel moyen",
    )
    ax.plot(
        x, top["min_recall"], color=COLOR_KD, marker="o", markerfacecolor="none",
        linestyle="--", label="rappel min",
    )
    ax.axhline(0.95, color=COLOR_TARGET, linestyle=":", linewidth=1.2,
               label="seuil 0.95")
    ax_build.bar(x, top["build_ms"] / 1e3, color=COLOR_BUILD, alpha=0.35,
                 width=0.6, label="temps de construction")

    chosen_recall = top["mean_recall"].iloc[chosen_pos]
    ax.annotate(
        "couple retenu", xy=(chosen_pos, chosen_recall),
        xytext=(chosen_pos, 1.02), color=COLOR_LSH, fontsize=8,
        arrowprops=dict(arrowstyle="-", color=COLOR_LSH, lw=0.8),
    )

    ax.set_xlabel("Couples (bits, buckets), permissivité croissante →")
    ax.set_ylabel("Rappel moyen / min")
    ax_build.set_ylabel("Temps de construction (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax_build.set_ylim(0, 18)
    ax_build.spines["right"].set_visible(True)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_build.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2, labels1 + labels2,
        loc="upper left", ncol=2, fontsize=7,
    )

    title(ax, "Rappel moyen en fonction des couples (bits, buckets)")
    caption = "FIGURE FIG-R11 — Rappel moyen et min par couple (bits, buckets), ChestX-ray14."
    finalize(fig, caption, output_path)


def main():
    parser = argparse.ArgumentParser(description="Figure FIG-R11 (calibration LSH).")
    parser.add_argument("--csv", required=True, help="lsh_calibration.csv (NB03)")
    parser.add_argument("--summary", required=True, help="nb03_summary.json")
    parser.add_argument("--output", required=True, help="chemin du PNG")
    args = parser.parse_args()

    calib = pd.read_csv(args.csv)
    calib = calib[calib["dataset"] == DATASET]
    top = by_permissiveness(calib).reset_index(drop=True)
    retained = retained_params(args.summary)
    chosen_pos = int(
        top[(top["num_bits"] == retained["num_bits"])
            & (top["num_tables"] == retained["num_tables"])].index[0]
    )
    print(top[["num_bits", "num_tables", "mean_recall", "min_recall", "build_ms"]].to_string(index=False))
    print(f"couple retenu à la position x={chosen_pos}")
    plot(top, chosen_pos, args.output)
    print(f"[OK] figure écrite : {args.output}")


if __name__ == "__main__":
    main()
