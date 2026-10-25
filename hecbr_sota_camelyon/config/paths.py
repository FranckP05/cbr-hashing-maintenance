"""Chemins d'acces aux donnees, modeles et resultats sur Lightning AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Donnees brutes et cache reutilise depuis l'experience precedente
POSSIBLE_DSMIL_ROOTS = [
    Path("/teamspace/studios/this_studio/dsmil-wsi"),
    Path("/home/zeus/content/dsmil-wsi"),
    PROJECT_ROOT.parent / "dsmil-wsi",
]

DSMIL_ROOT = next((p for p in POSSIBLE_DSMIL_ROOTS if p.exists()), POSSIBLE_DSMIL_ROOTS[0])
CAMELYON_ROOT = DSMIL_ROOT / "datasets" / "Camelyon16"

# Cache partage de 4.88M patches
CACHE_FILE = Path("/teamspace/studios/this_studio/lsh_camelyon/results/raw/camelyon16_slides_cache.pkl")
if not CACHE_FILE.exists():
    CACHE_FILE = Path("/home/zeus/content/lsh_camelyon/results/raw/camelyon16_slides_cache.pkl")
if not CACHE_FILE.exists():
    CACHE_FILE = PROJECT_ROOT.parent / "lsh_camelyon" / "results" / "raw" / "camelyon16_slides_cache.pkl"

# Modele HeCBR sauvegarde
MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
HECBR_MODEL_PATH = MODEL_DIR / "hecbr_camelyon16_64bit.pt"

# Resultats bruts et figures
RESULTS_RAW_DIR = PROJECT_ROOT / "results" / "raw"
FIGURES_DIR     = PROJECT_ROOT / "results" / "figures"

BENCHMARK_TIMING_CSV  = RESULTS_RAW_DIR / "benchmark_timing.csv"
BENCHMARK_QUALITY_CSV = RESULTS_RAW_DIR / "benchmark_quality.csv"
GROUND_TRUTH_TEST_NPZ = RESULTS_RAW_DIR / "ground_truth_test.npz"
HECBR_CODES_NPZ = RESULTS_RAW_DIR / "hecbr_codes.npz"
HECBR_CODES_V4_NPZ = RESULTS_RAW_DIR / "hecbr_codes_v4.npz"

# Mesures complexité temporelle & maintenance
MAINTENANCE_SCALING_CSV = RESULTS_RAW_DIR / "maintenance_scaling_latency.csv"
MAINTENANCE_BUILD_CSV   = RESULTS_RAW_DIR / "maintenance_build_time.csv"
MULTIPROBE_TRADEOFF_CSV = RESULTS_RAW_DIR / "multiprobe_tradeoff.csv"
MAINTENANCE_ADD10K_CSV  = RESULTS_RAW_DIR / "maintenance_add10k.csv"
MAINTENANCE_BREAKEVEN_CSV = RESULTS_RAW_DIR / "maintenance_breakeven.csv"
