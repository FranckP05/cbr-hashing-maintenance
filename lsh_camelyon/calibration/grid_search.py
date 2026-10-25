"""Recherche sur grille (Grid Search) des hyperparamètres SimHash (k, L)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.protocol import (
    SIMHASH_K_RANGE,
    SIMHASH_L_RANGE,
    MIN_F1_THRESHOLD,
    DEFAULT_K_NEIGHBORS,
    MAX_CALIBRATION_QUERIES,
)
from config.seeds import CALIBRATION_HASH_SEED
from config.paths import CALIBRATION_CSV, BEST_PARAMS_JSON, GROUND_TRUTH_VAL_NPZ
from indexes.linear_scan import LinearScan
from indexes.simhash import SimHash
from benchmark.metrics import (
    cbr_predict, macro_f1, mean_recall_at_k, mean_candidate_fraction
)
from benchmark.timing import benchmark_queries, aggregate_latencies

logger = logging.getLogger(__name__)


def compute_ground_truth(
    X_base: np.ndarray,
    X_val: np.ndarray,
    k: int,
    cache_path: Path = GROUND_TRUTH_VAL_NPZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule (ou charge depuis le cache) la vérité terrain sur la validation."""
    if cache_path.exists():
        logger.info("Chargement de la vérité terrain depuis le cache : %s", cache_path)
        data = np.load(cache_path)
        cached_idx   = data["indices"]
        cached_dists = data["dists"]
        if len(cached_idx) >= len(X_val):
            return cached_idx[:len(X_val)], cached_dists[:len(X_val)]

    logger.info(
        "Calcul de la vérité terrain linéaire sur %d requêtes de validation (BLAS GEMV)...",
        X_val.shape[0],
    )
    scanner = LinearScan()
    scanner.build(X_base)
    all_idx, all_dists = scanner.batch_query(X_val, k)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, indices=all_idx, dists=all_dists)
    logger.info("Vérité terrain sauvegardée dans %s.", cache_path)
    return all_idx, all_dists


def run_grid_search(
    X_base: np.ndarray,
    y_base: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k: int             = DEFAULT_K_NEIGHBORS,
    k_range: List[int] = SIMHASH_K_RANGE,
    l_range: List[int] = SIMHASH_L_RANGE,
    hash_seed: int     = CALIBRATION_HASH_SEED,
    max_queries: int   = MAX_CALIBRATION_QUERIES,
    results_csv: Path  = CALIBRATION_CSV,
    best_params_json: Path = BEST_PARAMS_JSON,
) -> Dict:
    """Lance la recherche sur grille (k, L) de SimHash sur l'ensemble de validation."""
    # Sous-échantillonnage déterministe de la validation
    if max_queries and X_val.shape[0] > max_queries:
        rng = np.random.default_rng(0)
        idx = rng.choice(X_val.shape[0], max_queries, replace=False)
        X_eval = X_val[idx]
        y_eval = y_val[idx]
    else:
        X_eval = X_val
        y_eval = y_val

    # Vérité terrain sur la validation
    gt_indices, _ = compute_ground_truth(X_base, X_eval, k)

    records = []
    n_configs = len(k_range) * len(l_range)
    logger.info(
        "Grid search SimHash : %d configurations (k in %s, L in %s) sur %d requêtes...",
        n_configs, k_range, l_range, X_eval.shape[0],
    )

    t_grid_start = time.perf_counter()
    for i_cfg, num_bits in enumerate(k_range):
        for num_tables in l_range:
            # Construction de l'index
            t_build = time.perf_counter()
            index = SimHash(num_bits=num_bits, num_tables=num_tables, hash_seed=hash_seed)
            index.build(X_base)
            build_time_s = time.perf_counter() - t_build

            # Benchmark sur les requêtes de validation
            all_returned = []
            all_n_cands  = []

            def _query(x, k_):
                return index.query(x, k_)

            latencies, raw_results = benchmark_queries(_query, X_eval, k)

            for res in raw_results:
                idx_ret, _, n_cand = res
                all_returned.append(idx_ret)
                all_n_cands.append(n_cand)

            # Métriques
            recall    = mean_recall_at_k(all_returned, gt_indices, k)
            cand_frac = mean_candidate_fraction(all_n_cands, X_base.shape[0])
            y_pred    = cbr_predict(y_base, all_returned)
            f1        = macro_f1(y_eval, y_pred)
            lat_stats = aggregate_latencies(latencies)

            records.append({
                "num_bits":       num_bits,
                "num_tables":     num_tables,
                "f1_macro":       f1,
                "recall_at_k":    recall,
                "cand_frac":      cand_frac,
                "build_time_s":   build_time_s,
                **lat_stats,
            })
            logger.info(
                "  [%2d/%2d] k=%2d L=%2d | F1=%.4f | Rec@%d=%.4f | cand=%6.4f | lat=%6.2f ms | build=%.2fs",
                len(records), n_configs, num_bits, num_tables, f1, k, recall, cand_frac, lat_stats["median_ms"], build_time_s,
            )

    # Sauvegarde du tableau complet
    df = pd.DataFrame(records)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    logger.info("Résultats de calibration sauvegardés dans %s (durée totale : %.1fs).", results_csv, time.perf_counter() - t_grid_start)

    # Sélection de la meilleure configuration
    valid = df[df["f1_macro"] >= MIN_F1_THRESHOLD].copy()
    if valid.empty:
        logger.warning(
            "Aucune configuration n'atteint F1 >= %.2f. "
            "Sélection par F1 maximale.", MIN_F1_THRESHOLD
        )
        valid = df.copy()

    # Critères de sélection : effort minimal (k*L), puis latence médiane, puis k, puis L
    valid["effort"] = valid["num_bits"] * valid["num_tables"]
    best_row = valid.sort_values(
        ["effort", "median_ms", "num_bits", "num_tables"]
    ).iloc[0]

    best_params = {
        "num_bits":      int(best_row["num_bits"]),
        "num_tables":    int(best_row["num_tables"]),
        "f1_macro_val":  float(best_row["f1_macro"]),
        "recall_val":    float(best_row["recall_at_k"]),
        "median_ms_val": float(best_row["median_ms"]),
        "hash_seed":     hash_seed,
    }
    with open(best_params_json, "w") as fh:
        json.dump(best_params, fh, indent=2)

    logger.info(
        "=== Meilleurs paramètres retenus : k=%d, L=%d (F1=%.4f, Recall=%.4f, lat=%.2f ms) -> %s ===",
        best_params["num_bits"], best_params["num_tables"],
        best_params["f1_macro_val"], best_params["recall_val"],
        best_params["median_ms_val"], best_params_json,
    )
    return best_params
