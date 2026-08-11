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
