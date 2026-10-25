"""Panneau maintenance — résultats CBM vs CNN PRÉCALCULÉS (condensation exp2, NB07).

La condensation complète prend plusieurs minutes : la démo affiche des valeurs figées
(static/data/maintenance_snapshot.json) avec un badge explicite « précalculé ». Le fichier
est régénéré par generate_snapshot() depuis condensation_quality.csv — aucun calcul n'est
relancé à l'exécution.
"""

import json
from pathlib import Path

import pandas as pd

STATIC_SNAPSHOT = Path(__file__).resolve().parent.parent / "static" / "data" / "maintenance_snapshot.json"


def load_snapshot():
    """Lit le snapshot JSON figé (valeurs exp2, moyennes inter-seeds NB07)."""
    with open(STATIC_SNAPSHOT, encoding="utf-8") as f:
        return json.load(f)


def generate_snapshot(condensation_csv, out_path=STATIC_SNAPSHOT):
    """Régénère le snapshot depuis condensation_quality.csv (uniquement pour la production)."""
    df = pd.read_csv(condensation_csv)
    records = []
    for key in ["heart", "diabetes130", "chestxray14"]:
        d = df[(df["dataset"] == key) & (df["strategy"].isin(["cbm", "cnn"]))]
        means = d.groupby("strategy")["nn_accuracy"].mean()
        kept = d[d["strategy"] == "cbm"]["achieved_rate"].mean()
        records.append({
            "dataset": key,
            "cnn_precision": round(float(means["cnn"]), 4),
            "cbm_precision": round(float(means["cbm"]), 4),
            "proportion_reduced": round(float(1 - kept), 4),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records
