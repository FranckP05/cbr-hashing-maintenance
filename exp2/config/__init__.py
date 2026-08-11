"""Package de configuration d'exp2 (T0.1-T0.3).

Ce package est purement déclaratif : seeds, chemins (local/Kaggle), registre des
datasets. Il ne charge aucune donnée lourde.
"""

from config import datasets
from config import paths
from config import run
from config import seeds

__all__ = ["datasets", "paths", "run", "seeds"]
