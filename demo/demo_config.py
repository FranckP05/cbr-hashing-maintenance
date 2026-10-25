"""Constantes de la démo — valeurs calibrées exp2 (voir docs/demo.md pour la provenance).

Ne pas recalculer ces seuils/paramètres au démarrage : ils sont issus des sorties exp2
(NB03 : lsh_calibration.csv ; NB07 : retention_agreement.csv seed 42 ; condensation_quality.csv)
et figés ici pour garantir la cohérence avec les résultats du mémoire.
"""

import sys
from pathlib import Path

from src.datasets import heart_disease

K = 5

HECBR_CODES_PATH = "data/processed/hecbr_diabetes130_codes_final.parquet"

DATASETS = {
    "heart": {
        "label": "Faible dimensionnalité",
        "n_features": 13,
        "n_total": 297,
        "allow_manual_entry": True,
        "retention_threshold": 1.641387,  # NB07 seed 42 (retention_agreement.csv)
        "lsh_params": {"num_bits": 3, "num_tables": 8},  # NB03 (lsh_params_chosen)
        "feature_columns": heart_disease.FEATURE_COLUMNS,
    },
    "diabetes130": {
        "label": "Moyenne dimensionnalité",
        "n_features": 390,
        "n_total": 101766,
        "allow_manual_entry": False,
        "retention_threshold": 2.630000,  # NB07 seed 42 (retention_agreement.csv)
        "lsh_params": {"num_bits": 8, "num_tables": 16},  # NB03 (lsh_params_chosen)
        "feature_columns": None,
    },
    "chestxray14": {
        "label": "Haute dimensionnalité",
        "n_features": 1024,
        "n_total": 112120,
        "allow_manual_entry": False,
        "retention_threshold": 0.972312,  # NB07 seed 42 (retention_agreement.csv)
        "lsh_params": {"num_bits": 10, "num_tables": 8},  # NB03 (lsh_params_chosen)
        "feature_columns": None,
    },
}

# Ordre d'affichage des cartes de dimensionnalité (étape 1).
ORDER = ["heart", "diabetes130", "chestxray14"]


def exp2_path():
    """Chemin absolu du dossier exp2/ (racine d'import des modules src/ et config/)."""
    return str(Path(__file__).resolve().parent.parent / "exp2")
