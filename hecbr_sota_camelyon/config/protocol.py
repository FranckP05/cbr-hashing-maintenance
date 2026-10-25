"""Constantes du protocole expérimental (HeCBR vs SOTA LSH vs Baseline)."""

# ---------------------------------------------------------------------------
# Split des lames (Whole Slide Images)
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.90   # fraction des lames -> base d'indexation
TEST_RATIO: float  = 0.10   # fraction des lames -> requêtes de test
VAL_RATIO: float   = 0.15   # fraction de la base -> validation

# ---------------------------------------------------------------------------
# Voisinage k pour la classification CBR
# ---------------------------------------------------------------------------
DEFAULT_K_NEIGHBORS: int = 10

# ---------------------------------------------------------------------------
# Harnais temporel
# ---------------------------------------------------------------------------
TIMING_WARMUP: int  = 1
TIMING_REPEATS: int = 5
MAX_TEST_QUERIES: int = 300
MAX_CALIBRATION_QUERIES: int = 100

# ---------------------------------------------------------------------------
# Hyperparamètres HeCBR (Deep Hashing)
# ---------------------------------------------------------------------------
HECBR_BIT_LENGTH: int   = 64       # Longueur du code binaire (64 bits = uint64)
HECBR_HIDDEN_DIM: int   = 256      # Dimension de la couche cachée
HECBR_EPOCHS: int       = 15       # Nombre d'époques d'entraînement
HECBR_BATCH_SIZE: int   = 1024     # Taille des mini-lots
HECBR_LR: float         = 0.001    # Taux d'apprentissage Adam
HECBR_QUANT_WEIGHT: float = 0.01   # Poids de la perte de quantification (gamma)

# ---------------------------------------------------------------------------
# Hyperparamètres SOTA LSH (Cross-Polytope avec Multi-Probe)
# ---------------------------------------------------------------------------
CP_NUM_HASHES: int    = 2          # K : nombre de polytopes croisés par table
CP_NUM_TABLES: int    = 10         # L : nombre de tables
CP_NUM_ROTATIONS: int = 2          # Nombre de rotations Hadamard par hash
CP_NUM_PROBES: int    = 16         # M : nombre de sondes multi-probe par table
