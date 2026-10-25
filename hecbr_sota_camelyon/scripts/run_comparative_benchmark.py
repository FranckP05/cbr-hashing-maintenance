"""Point d'entrée principal : Benchmark comparatif (Baseline vs SOTA LSH vs HeCBR, multi-seeds)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import (
    HECBR_MODEL_PATH, BENCHMARK_TIMING_CSV, BENCHMARK_QUALITY_CSV,
    GROUND_TRUTH_TEST_NPZ, RESULTS_RAW_DIR, FIGURES_DIR,
    HECBR_CODES_NPZ, HECBR_CODES_V4_NPZ,
)
from config.protocol import (
    DEFAULT_K_NEIGHBORS, MAX_TEST_QUERIES, HECBR_BIT_LENGTH,
    CP_NUM_HASHES, CP_NUM_TABLES, CP_NUM_PROBES,
)
from config.seeds import SPLIT_SEED, BENCHMARK_HASH_SEEDS
from data.loader import load_slides, get_slide_ids
from data.splitter import split_slides, build_split_matrices
from indexes.linear_scan import LinearScan
from indexes.cross_polytope import CrossPolytopeLSH
from indexes.hecbr_index import HeCBRIndex, HeCBRMultiProbeIndex
from benchmark.timing import benchmark_queries, aggregate_latencies
from benchmark.metrics import (
    cbr_predict, macro_f1, mean_recall_at_k,
)
from visualization.plots import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("comparative_benchmark")

TIMING_COLS = ["method", "seed", "build_time_s", "median_ms", "mean_ms",
               "std_ms", "p25_ms", "p75_ms", "p95_ms", "p99_ms"]
QUALITY_COLS = ["method", "seed", "f1_macro", "recall_at_k"]

# Hyperparamètres de l'index HeCBR multi-probe (calibrés sur serveur)
MP_TABLES = 8
MP_BITS_PER_TABLE = 24
MP_RADIUS = 2
MP_BUDGET = 256


def _load_hashnet_v4(device: str = "cpu"):
    """Charge le modèle de hachage v4 (HashNetV3) depuis le checkpoint courant."""
    import torch
    from scripts.train_hecbr_v4 import HashNetV3
    checkpoint = torch.load(HECBR_MODEL_PATH, map_location=device)
    model = HashNetV3(
        in_dim=checkpoint.get("in_dim", 512),
        hidden=checkpoint.get("hidden_dim", 512),
        bit_length=checkpoint.get("bit_length", 64),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _fit_zscore(X):
    return X.mean(0), X.std(0) + 1e-6


def _encode_v4(model, X, mu, sd, device="cpu", bs=65536):
    """Encode X en codes binaires bool (N, 64) via la normalisation z-score + HashNetV3."""
    import torch
    out = []
    for i in range(0, len(X), bs):
        x = torch.tensor((X[i:i + bs] - mu) / sd, dtype=torch.float32).to(device)
        with torch.no_grad():
            u = model(x)
            out.append((u >= 0).cpu().numpy())
        del x
    return np.concatenate(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Comparatif Baseline vs SOTA LSH vs HeCBR.")
    p.add_argument("--k", type=int, default=DEFAULT_K_NEIGHBORS, help="Nombre de voisins k (defaut: 10).")
    p.add_argument("--n-base", type=int, default=None, help="Plafonne la base a N points (defaut: base complete split).")
    p.add_argument("--n-lsh-seeds", type=int, default=None, help="Limite le nombre de seeds LSH (defaut: tous).")
    p.add_argument("--n-hecbr-seeds", type=int, default=None, help="Limite le nombre de seeds HeCBR (defaut: tous).")
    p.add_argument("--no-plots", action="store_true", help="Ne pas generer les figures.")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Seeds a executer (defaut: tous ceux pas deja enregistres).")
    return p.parse_args()


def _load_done() -> set[tuple[str, int]]:
    """Seeds deja enregistres dans les CSV (mecanisme de reprise)."""
    done: set[tuple[str, int]] = set()
    if BENCHMARK_QUALITY_CSV.exists() and BENCHMARK_QUALITY_CSV.stat().st_size > 0:
        try:
            df = pd.read_csv(BENCHMARK_QUALITY_CSV)
            if {"method", "seed"}.issubset(df.columns):
                done = {
                    (row.method, int(row.seed))
                    for row in df[["method", "seed"]].dropna().itertuples()
                }
        except Exception as exc:
            logger.warning("Lecture CSV de reprise impossible (%s), on repart des seeds manquants.", exc)
    return done


def _append_record(record: dict) -> None:
    """Ecriture immediate (checkpoint) d'une ligne dans les deux CSV."""
    df_t = pd.DataFrame([{c: record.get(c) for c in TIMING_COLS}])
    df_q = pd.DataFrame([{c: record.get(c) for c in QUALITY_COLS}])
    df_t.to_csv(BENCHMARK_TIMING_CSV, mode="a",
                header=not BENCHMARK_TIMING_CSV.exists() or BENCHMARK_TIMING_CSV.stat().st_size == 0,
                index=False)
    df_q.to_csv(BENCHMARK_QUALITY_CSV, mode="a",
                header=not BENCHMARK_QUALITY_CSV.exists() or BENCHMARK_QUALITY_CSV.stat().st_size == 0,
                index=False)


def _run_baseline(X_base, y_base, X_eval, y_eval, k):
    logger.info("=== 1. Baseline Lineaire BLAS (Exacte) ===")
    scanner = LinearScan()
    t_build = time.perf_counter()
    scanner.build(X_base)
    build_time_s = time.perf_counter() - t_build

    latencies, results = benchmark_queries(scanner.query, X_eval, k)
    lat_stats = aggregate_latencies(latencies)

    all_idx = [res[0] for res in results]
    y_pred  = cbr_predict(y_base, all_idx)
    f1      = macro_f1(y_eval, y_pred)
    recall  = 1.0

    logger.info("  Baseline : lat_med=%.2f ms | F1=%.4f | Recall@%d=%.4f", lat_stats["median_ms"], f1, k, recall)
    return {
        "method": "baseline", "seed": 0, "build_time_s": build_time_s,
        "f1_macro": f1, "recall_at_k": recall, **lat_stats,
    }


def _run_sota_lsh(X_base, y_base, X_eval, y_eval, gt_indices, k, seeds):
    logger.info("=== 2. SOTA LSH : Cross-Polytope avec Multi-Probe ===")
    for seed in seeds:
        logger.info("  Cross-Polytope (K=%d, L=%d, M=%d, seed=%d) ...", CP_NUM_HASHES, CP_NUM_TABLES, CP_NUM_PROBES, seed)
        index = CrossPolytopeLSH(
            num_hashes=CP_NUM_HASHES, num_tables=CP_NUM_TABLES,
            num_probes=CP_NUM_PROBES, hash_seed=seed,
        )
        t_build = time.perf_counter()
        index.build(X_base)
        build_time_s = time.perf_counter() - t_build

        latencies, results = benchmark_queries(index.query, X_eval, k)
        lat_stats = aggregate_latencies(latencies)

        all_idx = [res[0] for res in results]
        y_pred  = cbr_predict(y_base, all_idx)
        f1      = macro_f1(y_eval, y_pred)
        recall  = mean_recall_at_k(all_idx, gt_indices, k)

        logger.info("    lat_med=%.2f ms | F1=%.4f | Recall@%d=%.4f | build=%.2fs",
                    lat_stats["median_ms"], f1, k, recall, build_time_s)
        _append_record({
            "method": "sota_lsh", "seed": seed, "build_time_s": build_time_s,
            "f1_macro": f1, "recall_at_k": recall, **lat_stats,
        })


def _encode_v4_codes(X_base, X_eval, mu, sd, device="cpu"):
    """Binarise base + requetes avec le modele v4 (persiste dans hecbr_codes_v4.npz)."""
    global _V4_MODEL
    if HECBR_CODES_V4_NPZ.exists():
        logger.info("  Rechargement des codes v4 depuis %s ...", HECBR_CODES_V4_NPZ)
        data = np.load(HECBR_CODES_V4_NPZ)
        return data["codes_base"], data["codes_eval"]
    logger.info("  Binarisation v4 de la base (%d cas) et des requetes (%d) ...",
                X_base.shape[0], X_eval.shape[0])
    np.savez_compressed(
        HECBR_CODES_V4_NPZ,
        codes_base=_encode_v4(_V4_MODEL, X_base, mu, sd, device=device),
        codes_eval=_encode_v4(_V4_MODEL, X_eval, mu, sd, device=device),
    )
    data = np.load(HECBR_CODES_V4_NPZ)
    return data["codes_base"], data["codes_eval"]


def _hecbr_gt_hamming(codes_base, codes_eval, k):
    """Verite terrain voisinage Hamming exact (scan SWAR) pour les requetes HeCBR."""
    gt_npz = HECBR_CODES_V4_NPZ.with_name("hecbr_gt_hamming.npz")
    if gt_npz.exists():
        return np.load(gt_npz)["indices"]
    scanner = HeCBRIndex(bit_length=HECBR_BIT_LENGTH)
    scanner.build(codes_base)
    gt, _ = scanner.batch_query(codes_eval, k)
    np.savez_compressed(gt_npz, indices=gt)
    return gt


def _run_hecbr(X_base, y_base, X_eval, y_eval, k, seeds, mu, sd, device="cpu"):
    logger.info("=== 3. HeCBR v4 : Deep Hashing 64-bit (scan exact + multi-probe) ===")
    codes_base, codes_eval = _encode_v4_codes(X_base, X_eval, mu, sd, device=device)
    gt_ham = _hecbr_gt_hamming(codes_base, codes_eval, k)

    for seed in seeds:
        # --- a) Scan exact (SWAR) : methode "hecbr" ---
        index = HeCBRIndex(bit_length=HECBR_BIT_LENGTH)
        t_build = time.perf_counter()
        index.build(codes_base)
        build_time_s = time.perf_counter() - t_build

        latencies, results = benchmark_queries(index.query, codes_eval, k)
        lat_stats = aggregate_latencies(latencies)
        all_idx = [res[0] for res in results]
        y_pred = cbr_predict(y_base, all_idx)
        f1 = macro_f1(y_eval, y_pred)
        recall = mean_recall_at_k(all_idx, gt_ham, k)
        logger.info("  HeCBR scan seed=%d : lat_med=%.2f ms | F1=%.4f | Recall(Ham)@%d=%.4f | build=%.2fs",
                    seed, lat_stats["median_ms"], f1, k, recall, build_time_s)
        _append_record({
            "method": "hecbr", "seed": seed, "build_time_s": build_time_s,
            "f1_macro": f1, "recall_at_k": recall, **lat_stats,
        })

        # --- b) Multi-probe : methode "hecbr_mp" ---
        index = HeCBRMultiProbeIndex(bit_length=HECBR_BIT_LENGTH)
        t_build = time.perf_counter()
        index.build(codes_base, num_tables=MP_TABLES, bits_per_table=MP_BITS_PER_TABLE,
                    radius=MP_RADIUS, hash_seed=seed)
        build_time_s = time.perf_counter() - t_build

        latencies, results = benchmark_queries(
            lambda q, kk: index.query(q, kk, max_buckets_per_table=MP_BUDGET),
            codes_eval, k)
        lat_stats = aggregate_latencies(latencies)
        all_idx = [res[0] for res in results]
        y_pred = cbr_predict(y_base, all_idx)
        f1 = macro_f1(y_eval, y_pred)
        recall = mean_recall_at_k(all_idx, gt_ham, k)
        logger.info("  HeCBR MP   seed=%d : lat_med=%.2f ms | F1=%.4f | Recall(Ham)@%d=%.4f | build=%.2fs",
                    seed, lat_stats["median_ms"], f1, k, recall, build_time_s)
        _append_record({
            "method": "hecbr_mp", "seed": seed, "build_time_s": build_time_s,
            "f1_macro": f1, "recall_at_k": recall, **lat_stats,
        })


def _select_seeds(method: str, allowlist: list[int] | None, done: set[tuple[str, int]]) -> list[int]:
    todo, skipped = [], []
    for seed in (allowlist if allowlist is not None else BENCHMARK_HASH_SEEDS):
        if (method, seed) in done:
            skipped.append(seed)
        else:
            todo.append(seed)
    if skipped:
        logger.info("  %s : seeds deja enregistres, skip : %s", method, skipped)
    return todo


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    done = _load_done()

    # 1. Chargement des données
    logger.info("=== Étape 1 : Chargement des données Camelyon16 ===")
    slide_data, slide_labels = load_slides()
    slide_ids, _ = get_slide_ids(slide_labels)

    # 2. Split stratifié
    base_ids, val_ids, test_ids = split_slides(slide_ids, slide_labels, seed=SPLIT_SEED)

    # 3. Construction des matrices
    full_base_ids = base_ids + val_ids
    X_base, y_base, _, _, X_test, y_test = build_split_matrices(
        base_ids=full_base_ids, val_ids=[], test_ids=test_ids,
        slide_data=slide_data, slide_labels=slide_labels, normalize_l2=True
    )
    del slide_data

    N_req = min(MAX_TEST_QUERIES, X_test.shape[0])
    rng = np.random.default_rng(SPLIT_SEED + 7)
    idx_pos = np.flatnonzero(y_test == 1)
    idx_neg = np.flatnonzero(y_test == 0)
    n_pos = max(1, N_req // 2)
    if len(idx_pos) == 0:
        chosen = rng.choice(idx_neg, size=N_req, replace=len(idx_neg) < N_req)
    else:
        pos_sel = idx_pos[rng.choice(len(idx_pos), size=min(n_pos, len(idx_pos)), replace=len(idx_pos) < n_pos)]
        need_neg = N_req - len(pos_sel)
        neg_sel = idx_neg[rng.choice(len(idx_neg), size=min(need_neg, len(idx_neg)), replace=len(idx_neg) < need_neg)]
        chosen = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(chosen)
        chosen = chosen[:N_req]
    X_eval = X_test[chosen]
    y_eval = y_test[chosen]
    logger.info("Requetes test stratifiees: %d (pos=%d, neg=%d)", len(chosen), int((y_eval == 1).sum()), int((y_eval == 0).sum()))

    # Option : plafonne la base (pour maitriser le temps de build LSH en session)
    if args.n_base is not None and X_base.shape[0] > args.n_base:
        rng_cap = np.random.default_rng(SPLIT_SEED + 11)
        keep = rng_cap.choice(X_base.shape[0], size=args.n_base, replace=False)
        X_base, y_base = X_base[keep], y_base[keep]
        logger.info("Base plafonnee a %d points (echantillon deterministe).", X_base.shape[0])

    # 4. Vérité terrain (L2 exacte)
    if GROUND_TRUTH_TEST_NPZ.exists():
        logger.info("Chargement verite terrain test depuis %s", GROUND_TRUTH_TEST_NPZ)
        gt_data = np.load(GROUND_TRUTH_TEST_NPZ)
        gt_indices = gt_data["indices"][:len(chosen)]
    else:
        logger.info("Calcul de la verite terrain test (%d requetes)...", len(chosen))
        scanner = LinearScan()
        scanner.build(X_base)
        gt_indices, _ = scanner.batch_query(X_eval, args.k)
        np.savez_compressed(GROUND_TRUTH_TEST_NPZ, indices=gt_indices)

    # 5. Modèle HeCBR v4 (checkpoint déjà entraîné)
    import torch
    global _V4_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _V4_MODEL = _load_hashnet_v4(device)
    mu, sd_ = _fit_zscore(X_base)

    # 6. Benchmark avec checkpoint par seed (reprise partielle)
    if ("baseline", 0) in done:
        logger.info("=== 1. Baseline deja enregistree, skip ===")
    else:
        _append_record(_run_baseline(X_base, y_base, X_eval, y_eval, args.k))

    lsh_seeds = _select_seeds("sota_lsh", args.seeds, done)
    if args.n_lsh_seeds is not None:
        lsh_seeds = lsh_seeds[:args.n_lsh_seeds]
    if lsh_seeds:
        _run_sota_lsh(X_base, y_base, X_eval, y_eval, gt_indices, args.k, lsh_seeds)
    else:
        logger.info("=== 2. SOTA LSH : tous les seeds deja enregistres ===")

    hecbr_seeds = _select_seeds("hecbr", args.seeds, done)
    if args.n_hecbr_seeds is not None:
        hecbr_seeds = hecbr_seeds[:args.n_hecbr_seeds]
    if hecbr_seeds:
        _run_hecbr(X_base, y_base, X_eval, y_eval, args.k, hecbr_seeds, mu, sd_, device=device)
    else:
        logger.info("=== 3. HeCBR : tous les seeds deja enregistres ===")

    # 7. Sauvegarde consolidee
    df_t = pd.read_csv(BENCHMARK_TIMING_CSV)
    df_q = pd.read_csv(BENCHMARK_QUALITY_CSV)
    logger.info("Resultats cumules : %d lignes (timing), %d lignes (qualite).",
                len(df_t), len(df_q))

    if not args.no_plots:
        generate_all_figures(BENCHMARK_TIMING_CSV, BENCHMARK_QUALITY_CSV,
                             n_points=X_base.shape[0])
        logger.info("Figures generees dans %s.", FIGURES_DIR)

    logger.info("=== Benchmark comparatif termine en %.1f s ===", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
