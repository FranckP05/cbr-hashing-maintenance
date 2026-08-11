"""Registre des jeux de données exploités (T0.1).

Chaque entrée décrit : label, description, source locale vs Kaggle (slug à créer),
politique de scaling, dimension attendue. Les valeurs n_instances / ratio positif
sont mesurées en NB01 (exploration) et ne sont pas figées ici.

Slugs Kaggle à créer (une fois par dataset, réutilisables pour tous les runs) :
  - "heart-disease"        -> processed.cleveland.data
  - "diabetes130"          -> diabetic_data.csv
  - "chestxray14-embeddings" -> chestxray14_embeddings.parquet + labels.parquet
"""

from config.paths import ENV

DATASETS = {
    "heart": {
        "label": "Heart Disease (Cleveland)",
        "description": "Tabulaire, 13 attributs cliniques, ~46 % positifs",
        "files": {
            "local": "data/raw/heart_disease/processed.cleveland.data",
            "kaggle": "/kaggle/input/heart-disease/processed.cleveland.data",
        },
        "scaling": "standard",
        "n_features": 13,
    },
    "diabetes130": {
        "label": "Diabetes 130-US",
        "description": "Mixte, 8 numériques + one-hot (dim finale 390), 11,2 % positifs",
        "files": {
            "local": "data/raw/diabetes_130/diabetic_data.csv",
            "kaggle": "/kaggle/input/diabetes130/diabetic_data.csv",
        },
        "scaling": "standard+onehot",
        "n_features": 390,
    },
    "chestxray14": {
        "label": "ChestX-ray14 (embeddings DenseNet121)",
        "description": "Multimodal, embeddings 1024-D, 46,2 % positifs",
        "files": {
            "local": "data/processed",
            "kaggle": "/kaggle/input/chestxray14-embeddings",
        },
        "scaling": "none",
        "n_features": 1024,
    },
}

ORDER = ["heart", "diabetes130", "chestxray14"]


def source(key):
    """Renvoie le chemin de la source de données selon l'environnement."""
    return DATASETS[key]["files"]["kaggle" if ENV == "kaggle" else "local"]
