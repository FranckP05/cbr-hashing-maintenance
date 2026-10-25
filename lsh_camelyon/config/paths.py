"""Chemins d'accès aux données et aux répertoires de sortie sur Lightning AI."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Racine du projet (répertoire lsh_camelyon/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Localisation automatique de la racine DSMIL / Camelyon16
# ---------------------------------------------------------------------------
POSSIBLE_DSMIL_ROOTS = [
    Path("/teamspace/studios/this_studio/dsmil-wsi"),
    Path("/home/zeus/content/dsmil-wsi"),
    PROJECT_ROOT.parent / "dsmil-wsi",
    Path("dsmil-wsi"),
]

DSMIL_ROOT = next((p for p in POSSIBLE_DSMIL_ROOTS if p.exists()), POSSIBLE_DSMIL_ROOTS[0])
CAMELYON_ROOT = DSMIL_ROOT / "datasets" / "Camelyon16"

# Répertoires des lames par classe (un CSV par lame, 512 colonnes)
NORMAL_DIR = CAMELYON_ROOT / "0-normal"
TUMOR_DIR  = CAMELYON_ROOT / "1-tumor"

# Fichiers d'index
NORMAL_INDEX_CSV = CAMELYON_ROOT / "0-normal.csv"
TUMOR_INDEX_CSV  = CAMELYON_ROOT / "1-tumor.csv"
GLOBAL_INDEX_CSV = CAMELYON_ROOT / "Camelyon16.csv"

# ---------------------------------------------------------------------------
# Résultats bruts (CSV, JSON, NPZ)
# ---------------------------------------------------------------------------
RESULTS_RAW_DIR = PROJECT_ROOT / "results" / "raw"

# Calibration
BEST_PARAMS_JSON      = RESULTS_RAW_DIR / "best_params_simhash.json"
CALIBRATION_CSV       = RESULTS_RAW_DIR / "calibration_results.csv"
GROUND_TRUTH_VAL_NPZ  = RESULTS_RAW_DIR / "ground_truth_val.npz"

# Benchmark final
BENCHMARK_TIMING_CSV  = RESULTS_RAW_DIR / "benchmark_timing.csv"
BENCHMARK_QUALITY_CSV = RESULTS_RAW_DIR / "benchmark_quality.csv"
GROUND_TRUTH_TEST_NPZ = RESULTS_RAW_DIR / "ground_truth_test.npz"

# ---------------------------------------------------------------------------
# Figures (PNG 300 DPI)
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
