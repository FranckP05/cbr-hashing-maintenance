"""Point d'entrée CLI : Calibration des hyperparamètres SimHash sur l'ensemble de validation."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import (
    BEST_PARAMS_JSON, CALIBRATION_CSV, RESULTS_RAW_DIR,
)
from config.protocol import DEFAULT_K_NEIGHBORS
from config.seeds import SPLIT_SEED
from data.loader import load_slides, get_slide_ids
from data.splitter import split_slides, build_split_matrices
from calibration.grid_search import run_grid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_calibration")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration SimHash sur Camelyon16 (validation).")
    p.add_argument(
        "--k", type=int, default=DEFAULT_K_NEIGHBORS,
        help=f"Nombre de voisins k (défaut : {DEFAULT_K_NEIGHBORS}).",
    )
    p.add_argument(
        "--normal-dir", type=Path, default=None,
        help="Répertoire ou CSV des lames normales.",
    )
    p.add_argument(
        "--tumor-dir", type=Path, default=None,
        help="Répertoire ou CSV des lames tumorales.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des données
    logger.info("=== Étape 1 : Chargement des données Camelyon16 ===")
    slide_data, slide_labels = load_slides(args.normal_dir, args.tumor_dir)
    slide_ids, _ = get_slide_ids(slide_labels)
    logger.info("Total : %d lames chargées.", len(slide_ids))

    # 2. Partitionnement par lame
    logger.info("=== Étape 2 : Split stratifié 90/10 par lame ===")
    base_ids, val_ids, test_ids = split_slides(slide_ids, slide_labels, seed=SPLIT_SEED)

    # 3. Construction des matrices avec libération immédiate de mémoire
    logger.info("=== Étape 3 : Construction des matrices (base/val) ===")
    X_base, y_base, X_val, y_val, _, _ = build_split_matrices(
        base_ids, val_ids, test_ids=None, slide_data=slide_data, slide_labels=slide_labels, normalize_l2=True
    )
    # slide_data est maintenant vidé pour préserver la RAM
    del slide_data

    logger.info(
        "Données prêtes et normalisées L2 : base=%d patches, val=%d patches.",
        X_base.shape[0], X_val.shape[0],
    )

    # 4. Grid search de calibration (sur validation uniquement)
    logger.info("=== Étape 4 : Calibration SimHash (grid search sur validation) ===")
    best_params = run_grid_search(
        X_base=X_base,
        y_base=y_base,
        X_val=X_val,
        y_val=y_val,
        k=args.k,
        results_csv=CALIBRATION_CSV,
        best_params_json=BEST_PARAMS_JSON,
    )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "=== Calibration terminée en %.1f s. Meilleurs paramètres : %s ===",
        elapsed, best_params,
    )


if __name__ == "__main__":
    main()
