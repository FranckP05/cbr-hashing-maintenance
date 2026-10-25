"""Point d'entrée CLI : Benchmark comparatif final (Baseline vs SimHash, multi-seeds)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import (
    BEST_PARAMS_JSON, BENCHMARK_TIMING_CSV, BENCHMARK_QUALITY_CSV,
    GROUND_TRUTH_TEST_NPZ, RESULTS_RAW_DIR, FIGURES_DIR,
)
from config.protocol import DEFAULT_K_NEIGHBORS, MAX_TEST_QUERIES
from config.seeds import SPLIT_SEED, BENCHMARK_HASH_SEEDS
from data.loader import load_slides, get_slide_ids
from data.splitter import split_slides, build_split_matrices
from indexes.linear_scan import LinearScan
from indexes.simhash import SimHash
from benchmark.timing import benchmark_queries, aggregate_latencies
from benchmark.metrics import (
    cbr_predict, macro_f1, mean_recall_at_k, mean_candidate_fraction
)
from visualization.plots import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Baseline vs SimHash sur Camelyon16 (test, multi-seeds)."
    )
    p.add_argument(
        "--k", type=int, default=DEFAULT_K_NEIGHBORS,
        help=f"Nombre de voisins k (défaut : {DEFAULT_K_NEIGHBORS}).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Ne pas générer les figures Matplotlib.",
    )
    p.add_argument(
        "--normal-dir", type=Path, default=None,
    )
    p.add_argument(
        "--tumor-dir", type=Path, default=None,
    )
    return p.parse_args()


def _run_baseline(X_base, y_base, X_test, y_test, gt_indices, k):
    """Lance la baseline linéaire et retourne les résultats structurés."""
    logger.info("--- Baseline linéaire ---")
    scanner = LinearScan()
    t_build = time.perf_counter()
    scanner.build(X_base)
    build_time_s = time.perf_counter() - t_build
    logger.info("  Build : %.2f s", build_time_s)

    N_test  = min(MAX_TEST_QUERIES, X_test.shape[0])
    X_eval  = X_test[:N_test]
    y_eval  = y_test[:N_test]
    gt_eval = gt_indices[:N_test]

    latencies, results = benchmark_queries(scanner.query, X_eval, k)
    lat_stats = aggregate_latencies(latencies)

    all_idx = [res[0] for res in results]
    y_pred  = cbr_predict(y_base, all_idx)
    f1      = macro_f1(y_eval, y_pred)
    recall  = mean_recall_at_k(all_idx, gt_eval, k)

    logger.info(
        "  lat_med=%.2f ms | F1=%.4f | Recall@%d=%.4f",
        lat_stats["median_ms"], f1, k, recall,
    )
    return {
        "method":       "baseline",
        "seed":         0,
        "build_time_s": build_time_s,
        "f1_macro":     f1,
        "recall_at_k":  recall,
        "cand_frac":    1.0,
        **lat_stats,
    }


def _run_simhash(X_base, y_base, X_test, y_test, gt_indices, best_params, k):
    """Lance SimHash pour chaque seed de benchmark et retourne les résultats."""
    num_bits   = best_params["num_bits"]
    num_tables = best_params["num_tables"]
    records    = []

    N_test  = min(MAX_TEST_QUERIES, X_test.shape[0])
    X_eval  = X_test[:N_test]
    y_eval  = y_test[:N_test]
    gt_eval = gt_indices[:N_test]

    for seed in BENCHMARK_HASH_SEEDS:
        logger.info("--- SimHash k=%d L=%d seed=%d ---", num_bits, num_tables, seed)
        index = SimHash(num_bits=num_bits, num_tables=num_tables, hash_seed=seed)
        t_build = time.perf_counter()
        index.build(X_base)
        build_time_s = time.perf_counter() - t_build
        logger.info("  Build : %.2f s", build_time_s)

        latencies, results = benchmark_queries(index.query, X_eval, k)
        lat_stats = aggregate_latencies(latencies)

        all_idx    = [res[0] for res in results]
        all_n_cand = [res[2] for res in results]
        y_pred     = cbr_predict(y_base, all_idx)
        f1         = macro_f1(y_eval, y_pred)
        recall     = mean_recall_at_k(all_idx, gt_eval, k)
        cand_frac  = mean_candidate_fraction(all_n_cand, X_base.shape[0])

        logger.info(
            "  lat_med=%.2f ms | F1=%.4f | Recall@%d=%.4f | cand=%.4f",
            lat_stats["median_ms"], f1, k, recall, cand_frac,
        )
        records.append({
            "method":       "simhash",
            "seed":         seed,
            "num_bits":     num_bits,
            "num_tables":   num_tables,
            "build_time_s": build_time_s,
            "f1_macro":     f1,
            "recall_at_k":  recall,
            "cand_frac":    cand_frac,
            **lat_stats,
        })

    return records


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement
    logger.info("=== Étape 1 : Chargement des données Camelyon16 ===")
    slide_data, slide_labels = load_slides(args.normal_dir, args.tumor_dir)
    slide_ids, _ = get_slide_ids(slide_labels)

    # 2. Split
    logger.info("=== Étape 2 : Split 90/10 par lame ===")
    base_ids, val_ids, test_ids = split_slides(slide_ids, slide_labels, seed=SPLIT_SEED)

    # 3. Matrices en mémoire optimisée (on ne charge que base et test)
    logger.info("=== Étape 3 : Construction des matrices (base/test) ===")
    # On concatène val_ids dans base_ids pour le benchmark final (toute la base de cas = 90% des lames)
    full_base_ids = base_ids + val_ids
    X_base, y_base, _, _, X_test, y_test = build_split_matrices(
        base_ids=full_base_ids, val_ids=[], test_ids=test_ids,
        slide_data=slide_data, slide_labels=slide_labels, normalize_l2=True
    )
    del slide_data

    # 4. Vérité terrain sur le test (via baseline exacte, mise en cache)
    logger.info("=== Étape 4 : Calcul de la vérité terrain (test) ===")
    N_test  = min(MAX_TEST_QUERIES, X_test.shape[0])
    X_eval  = X_test[:N_test]

    if GROUND_TRUTH_TEST_NPZ.exists():
        logger.info("  Cache trouvé : %s", GROUND_TRUTH_TEST_NPZ)
        gt_data    = np.load(GROUND_TRUTH_TEST_NPZ)
        gt_indices = gt_data["indices"]
    else:
        scanner = LinearScan()
        scanner.build(X_base)
        gt_indices, _ = scanner.batch_query(X_eval, args.k)
        np.savez_compressed(GROUND_TRUTH_TEST_NPZ, indices=gt_indices)
        logger.info("  Vérité terrain sauvegardée dans %s", GROUND_TRUTH_TEST_NPZ)

    # 5. Chargement des meilleurs paramètres SimHash
    logger.info("=== Étape 5 : Chargement des paramètres calibrés ===")
    if not BEST_PARAMS_JSON.exists():
        raise FileNotFoundError(
            f"{BEST_PARAMS_JSON} introuvable. "
            "Lancer d'abord : python scripts/run_calibration.py"
        )
    with open(BEST_PARAMS_JSON) as fh:
        best_params = json.load(fh)
    logger.info("  Paramètres : %s", best_params)

    # 6. Benchmark
    logger.info("=== Étape 6 : Benchmark comparatif ===")
    all_records = []
    baseline_rec = _run_baseline(X_base, y_base, X_test[:N_test], y_test[:N_test], gt_indices, args.k)
    all_records.append(baseline_rec)

    simhash_recs = _run_simhash(X_base, y_base, X_test[:N_test], y_test[:N_test], gt_indices, best_params, args.k)
    all_records.extend(simhash_recs)

    # 7. Sauvegarde
    df_all = pd.DataFrame(all_records)
    timing_cols  = ["method", "seed", "build_time_s", "median_ms", "mean_ms",
                    "std_ms", "p25_ms", "p75_ms", "p95_ms", "p99_ms"]
    quality_cols = ["method", "seed", "f1_macro", "recall_at_k", "cand_frac"]

    df_all[timing_cols].to_csv(BENCHMARK_TIMING_CSV, index=False)
    df_all[quality_cols].to_csv(BENCHMARK_QUALITY_CSV, index=False)
    logger.info("Résultats bruts sauvegardés dans %s et %s.", BENCHMARK_TIMING_CSV, BENCHMARK_QUALITY_CSV)

    # 8. Figures
    if not args.no_plots:
        logger.info("=== Étape 7 : Génération des figures ===")
        generate_all_figures()

    elapsed = time.perf_counter() - t_start
    logger.info("=== Benchmark complet en %.1f s ===", elapsed)

    # Résumé console
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU BENCHMARK")
    print("=" * 60)
    for _, row in df_all.iterrows():
        print(
            f"  [{row['method']:10s}] seed={row.get('seed', 0):5}  "
            f"lat_med={row['median_ms']:7.2f} ms  "
            f"F1={row['f1_macro']:.4f}  "
            f"Recall={row['recall_at_k']:.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
