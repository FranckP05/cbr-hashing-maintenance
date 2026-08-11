"""Harness de scaling (T2.1-T2.2) : temps de requête + construction vs taille de base.

Pour chaque dataset × seed : split stratifié, puis bases croissantes (10k / 50k /
max(train)) = premiers n cas du train du seed. Pour chaque base :
  1. construction de chaque index (protocole T0.5, mesurée 1× par (dataset, taille)) ;
  2. requêtes sur l'échantillon du test (reproductible par seed) ;
  3. métriques (reclassement + rappel voisins exacts + classification) sur k=10 ;
  4. écriture incrémentale (checkpoint CSV).

Heart (237 cas) ne permet pas de scaling : un seul point à taille pleine (référence).
HeCBR exclu (codes fixes, pas de taille de base variable).

Variables d'environnement de réduction locale (sanity) :
  NB05_SEEDS     : liste de seeds, ex. "42" (défaut : les 6 seeds)
  NB05_QUERY_CAP : plafond de requêtes, ex. "60" (défaut : N_QUERIES_BY_DATASET)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import benchmark as bench_cfg
from config import scaling as scaling_cfg
from config import seeds as seed_cfg
from src.benchmark.benchmark_indexes import (
    LinearScanIndex,
    method_factories,
    run_method,
    sample_queries,
)
from src.datasets import adapters
from src.evaluation import timing

_SCALING_COLS = [
    "dataset", "seed", "base_size", "method", "k", "n_queries", "answered",
    "build_ms", "query_ms",
    "exact_recall_at_k", "precision_at_k", "label_recall_at_k", "map_at_k",
    "ndcg_at_k", "hit_at_k", "mrr_at_k",
    "accuracy", "macro_f1", "balanced_accuracy", "auc",
]


def _env_seeds():
    raw = os.environ.get("NB05_SEEDS")
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return list(seed_cfg.ALL_SEEDS)


def _env_query_cap():
    raw = os.environ.get("NB05_QUERY_CAP")
    return int(raw) if raw else None


def _key(r):
    return (r["dataset"], str(r["seed"]), int(r["base_size"]), r["method"],
            int(r["k"]), int(r["n_queries"]))


def _append(csv_path, rows, done):
    """Écrit les lignes (checkpoint) et met à jour l'ensemble des clés traitées."""
    new = [r for r in rows if _key(r) not in done]
    if not new:
        return 0
    df = pd.DataFrame(new)[_SCALING_COLS]
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    done.update(_key(r) for r in new)
    return len(new)


def benchmark_scaling(key, run_dir, progress=print):
    """Boucle seeds × tailles de base × méthodes pour un dataset (checkpoint CSV)."""
    csv_path = run_dir / "scaling_timing_quality.csv"
    done = set()
    if csv_path.exists():
        done = {_key(r) for r in pd.read_csv(csv_path).to_dict("records")}

    cap = _env_query_cap() or bench_cfg.N_QUERIES_BY_DATASET[key]
    k = scaling_cfg.SCALING_K
    build_cache = {}
    written = 0

    for seed in _env_seeds():
        X_train, y_train, X_test, y_test, _, _ = adapters.get_split(key, seed=seed)
        q_idx = sample_queries(X_test, seed, cap)
        queries = X_test[q_idx]
        query_targets = y_test[q_idx].astype(int)

        for base_size in scaling_cfg.BASE_SIZES_BY_DATASET[key]:
            n = len(X_train) if base_size == "max" else min(int(base_size), len(X_train))
            base = X_train[:n]
            yb = y_train[:n]
            progress(f"[{key} seed={seed}] base {n} -> {len(queries)} requêtes")

            exact_index = LinearScanIndex()
            exact_index.build(base)
            exact_by_k = {k: [exact_index.query(q, k) for q in queries]}

            rows = []
            for method, factory in method_factories(key).items():
                build_key = (key, base_size, method)
                if build_key not in build_cache:
                    _, build_cache[build_key], _ = timing.time_index_build(
                        factory, base,
                        warmup=bench_cfg.BUILD_WARMUP,
                        repetitions=bench_cfg.BUILD_REPETITIONS,
                    )
                    progress(f"  build {method} n={n}: {build_cache[build_key]*1e3:.1f} ms")
                index = factory()
                index.build(base)
                build_ms = round(build_cache[build_key] * 1e3, 4)

                med_q, _ = timing.time_queries(
                    index, queries, k,
                    warmup=bench_cfg.QUERY_WARMUP, repetitions=bench_cfg.QUERY_REPETITIONS,
                )
                res = run_method(index, exact_by_k, queries, query_targets, yb, k,
                                 method_is_exact=(method == "linear"))
                rows.append({
                    "dataset": key, "seed": seed, "base_size": int(n),
                    "method": method, "k": k,
                    "build_ms": build_ms, "query_ms": round(med_q * 1e3, 4), **res,
                })
            written += _append(csv_path, rows, done)
    progress(f"[{key}] lignes écrites : {written}")
    return written
