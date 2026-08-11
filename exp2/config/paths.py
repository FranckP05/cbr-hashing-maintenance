"""Chemins de l'expérience, local vs Kaggle (T0.1)."""

import os
from pathlib import Path

# Détection d'environnement
ENV = "kaggle" if os.path.exists("/kaggle/input") else "local"

# code/ est exp2/config/paths.py -> parents[2] == code/
CODE_ROOT = Path(__file__).resolve().parents[2]

if ENV == "kaggle":
    DATA_ROOT = Path("/kaggle/input")
    RUN_ROOT = Path("/kaggle/working")
else:
    DATA_ROOT = CODE_ROOT / "data"
    RUN_ROOT = CODE_ROOT / "exp2" / "results"


def kaggle_input(slug):
    """Racine d'un dataset Kaggle monté par slug.

    Kaggle a deux layouts de montage selon les époques :
      - ancien : /kaggle/input/<slug>/
      - nouveau : /kaggle/input/datasets/<owner>/<slug>/
    On résout en testant les deux (le chemin direct a priorité).
    """
    root = Path("/kaggle/input")
    direct = root / slug
    if direct.is_dir():
        return direct
    datasets_dir = root / "datasets"
    if datasets_dir.is_dir():
        for p in sorted(datasets_dir.glob(f"*/{slug}")):
            if p.is_dir():
                return p
    return direct
