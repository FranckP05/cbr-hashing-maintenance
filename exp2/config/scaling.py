"""Protocole du benchmark de scaling (T2) — constantes a priori.

Documenté dans NB05 :
- bases 10k / 50k / max(train) par sous-échantillonnage déterministe (premiers n cas
  du split train du seed, "max" = tout le train du seed) ;
- heart (237 cas) ne permet pas de scaling : un seul point à taille pleine, référence ;
- HeCBR exclu du scaling (codes fixes, pas de taille de base variable) ;
- k = 10 pour qualité et temps (T2.2), warm-up = 1, répétitions = 3 (T0.5) ;
- méthodes : linear, kdtree, lsh (paramètres LSH validés de config.benchmark, pas de
  re-calibration ici).
"""

BASE_SIZES_BY_DATASET = {
    "heart": ["max"],
    "diabetes130": [10_000, 50_000, "max"],
    "chestxray14": [10_000, 50_000, "max"],
}

SCALING_K = 10
SCALING_METHODS = ["linear", "kdtree", "lsh"]
