"""Protocole du benchmark des index seuls (T1) — constantes a priori.

Décisions de protocole, documentées dans NB03 (reproductibles, pas de tuning post-hoc) :

- requêtes par dataset : 60 (heart = tout le test) / 500 (diabetes130) / 300 (chestxray14),
  échantillonnées par seed (déterministe) ;
- timing (T0.5) : warm-up = 1, répétitions = 3 (construction et requête) ;
- calibration LSH (T1.3) : grille bits × tables sur 30 requêtes, seuil de rappel ≥ 0.95 ;
- paramètres LSH par défaut (validés en exp1) utilisés si la calibration ne converge pas ;
- HeCBR : codes pré-entraînés (Diabetes-130 uniquement), recherche restreinte au split
  "train" du parquet, distance de Hamming.
"""

from config.paths import ENV, kaggle_input

LSH_PARAMS_BY_DATASET = {
    "heart": {"num_bits": 5, "num_tables": 16},
    "diabetes130": {"num_bits": 12, "num_tables": 32},
    "chestxray14": {"num_bits": 16, "num_tables": 32},
}

CALIB_BITS_BY_DATASET = {
    # heart : dimension 13 → peu de bits suffisent (grille étendue vers le bas,
    # sinon le rappel plafonne < 0.95 dès 8 bits)
    "heart": [3, 4, 5, 6, 8, 10, 12, 14, 16],
    "diabetes130": [8, 10, 12, 14, 16],
    "chestxray14": [10, 12, 14, 16, 18, 20],
}
CALIB_TABLES = [8, 16, 32]
CALIB_QUERIES = 30
CALIB_K = 10
CALIB_RECALL_TARGET = 0.95
CALIB_SEED = 42

N_QUERIES_BY_DATASET = {"heart": 60, "diabetes130": 500, "chestxray14": 300}
QUERY_WARMUP = 1
QUERY_REPETITIONS = 3
BUILD_WARMUP = 1
BUILD_REPETITIONS = 3

if ENV == "kaggle":
    HECBR_CODES_PATH = str(
        kaggle_input("hecbr-diabetes130-codes") / "hecbr_diabetes130_codes_final.parquet"
    )
else:
    HECBR_CODES_PATH = "data/processed/hecbr_diabetes130_codes_final.parquet"
