"""Paramètres du protocole expérimental et seuils méthodologiques."""

# ---------------------------------------------------------------------------
# Split des lames (Whole Slide Images)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.90   # fraction des lames -> base d'indexation
TEST_RATIO: float = 0.10    # fraction des lames -> requêtes de test
VAL_RATIO: float = 0.15     # fraction de la base d'indexation -> validation calibration

# ---------------------------------------------------------------------------
# Seuil de qualité cible pour la calibration SimHash
# ---------------------------------------------------------------------------
MIN_F1_THRESHOLD: float = 0.90      # Macro-F1 minimum sur l'ensemble de validation
MIN_RECALL_AT_K: float = 0.80       # Recall@k minimum secondaire (informatif)

# ---------------------------------------------------------------------------
# Voisinage k pour la classification CBR
# ---------------------------------------------------------------------------
DEFAULT_K_NEIGHBORS: int = 10

# ---------------------------------------------------------------------------
# Harnais de timing (benchmark/timing.py)
# ---------------------------------------------------------------------------
TIMING_WARMUP: int = 1      # exécutions de chauffe (exclues de la mesure)
TIMING_REPEATS: int = 5     # répétitions chronométrées par requête

# ---------------------------------------------------------------------------
# Nombre de requêtes de validation (calibration) et de test (benchmark final)
# ---------------------------------------------------------------------------
MAX_CALIBRATION_QUERIES: int = 100
MAX_TEST_QUERIES: int = 300

# ---------------------------------------------------------------------------
# Grille d'hyperparamètres SimHash pour l'échelle 10^6+ (N = 3.77M patches)
# Théorie LSH / FALCONN : k ~ log2(N) ~ 14 à 24 bits
# ---------------------------------------------------------------------------
SIMHASH_K_RANGE = [12, 14, 16, 18, 20, 22, 24]
SIMHASH_L_RANGE = [5, 10, 15, 20, 30, 40, 50]
