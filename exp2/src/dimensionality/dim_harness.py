"""Harness NB09 — dimensionnalité (T4.1-T4.2).

QD1 : comment les paramètres LSH (bits, tables) doivent-ils croître avec la
dimension pour maintenir un rappel fixe (≥ 0.99), à taille de base fixée ?
QD3 : la fraction de candidats du filtrage LSH est-elle réellement meilleure en
haute dimension, ou confondue par la taille de base ? → mesurée à n fixé.

Pour chaque dataset (dim 13 / 390 / 1024), sur une base fixée (10k, ou tout le
train si plus petit) et des requêtes hors base (test) échantillonnées par seed 42 :
grille bits × tables ; pour chaque cellule, rappel des voisins exacts (k=10),
fraction de candidats (taille des compartiments / base), temps de construction
et de requête. Sortie : `dim_calibration.csv` + `nb09_summary.json`.

Variables d'environnement de réduction locale (sanity) :
  NB09_QUERIES   : nombre de requêtes (défaut : 60)
  NB09_BASE_SIZE : taille de base fixée (défaut : 10000)
  NB09_TABLES    : tables à tester, ex. "8,16" (défaut : 8,16,32,64)
"""

import os
import time

import numpy as np
import pandas as pd

from config import benchmark as bench_cfg
from src.datasets import adapters
from src.indexing.linear_scan import LinearScanIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex

K = 10
RECALL_TARGET = 0.99
QUERY_WARMUP = 1
QUERY_REPETITIONS = 2

BITS_BY_DATASET = {
    "heart": [3, 4, 5, 6, 8, 10, 12, 14, 16],
    "diabetes130": [8, 10, 12, 14, 16, 20],
    "chestxray14": [10, 12, 14, 16, 18, 20, 24],
}


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_tables():
    raw = os.environ.get("NB09_TABLES")
    return [int(x) for x in raw.split(",") if x.strip()] if raw else [8, 16, 32, 64]


def _median_query_ms(lsh, queries, k):
    for _ in range(QUERY_WARMUP):
        lsh.query(queries[0], k)
    times = []
    for _ in range(QUERY_REPETITIONS):
        t0 = time.perf_counter()
        for q in queries:
            lsh.query(q, k)
        times.append((time.perf_counter() - t0) / len(queries))
    return float(np.median(times)) * 1e3


def calibrate_dataset(key, base_size=None, n_queries=None, tables=None, progress=print):
    """Grille bits × tables sur une base fixée ; renvoie le DataFrame des cellules."""
    base_size = base_size or _env_int("NB09_BASE_SIZE", 10000)
    n_queries = n_queries or _env_int("NB09_QUERIES", 60)
    tables = tables if tables is not None else _env_tables()

    X_train, _, X_test, _, _, _ = adapters.get_split(key, seed=42)
    dim = X_train.shape[1]
    rng = np.random.default_rng(42)
    base_idx = rng.choice(len(X_train), size=min(base_size, len(X_train)), replace=False)
    X_base = X_train[base_idx]
    progress(f"[{key}] dim={dim} base={len(X_base)} (voulue {base_size}) test={len(X_test)}")

    q_idx = np.arange(len(X_test)) if len(X_test) <= n_queries else rng.choice(
        len(X_test), size=n_queries, replace=False)
    queries = X_test[q_idx]

    ref = LinearScanIndex()
    ref.build(X_base)
    gt = [set(ref.query(q, K)[0]) for q in queries]
    progress(f"  GT exact (k={K}) calculé sur {len(queries)} requêtes")

    records = []
    for num_bits in BITS_BY_DATASET[key]:
        for num_tables in tables:
            lsh = RandomProjectionLSHIndex(num_tables=num_tables, num_bits=num_bits,
                                           random_seed=42)
            t0 = time.perf_counter()
            lsh.build(X_base)
            build_ms = (time.perf_counter() - t0) * 1e3
            recalls, cand_fracs = [], []
            for q, g in zip(queries, gt):
                idx, _ = lsh.query(q, K)
                recalls.append(len(set(idx) & g) / K)
                cand_fracs.append(lsh.last_candidate_count / len(X_base))
            query_ms = _median_query_ms(lsh, queries, K)
            rec = {
                "dataset": key, "dim": dim, "base_size": len(X_base),
                "num_bits": num_bits, "num_tables": num_tables,
                "compartments": (2 ** num_bits) * num_tables,
                "mean_recall": round(float(np.mean(recalls)), 4),
                "min_recall": round(float(np.min(recalls)), 4),
                "mean_cand_fraction": round(float(np.mean(cand_fracs)), 4),
                "build_ms": round(build_ms, 2),
                "query_ms": round(query_ms, 4),
            }
            records.append(rec)
            progress(
                f"  bits={num_bits:2d} tables={num_tables:2d} "
                f"recall={rec['mean_recall']:.3f} cand={rec['mean_cand_fraction']:.3f} "
                f"build={build_ms:.0f}ms query={query_ms:.2f}ms"
            )
    return pd.DataFrame(records)


def choose_best(df, target=RECALL_TARGET):
    """Configuration retenue : rappel ≥ target à compartiments minimaux.

    À rappel égal (≥ target), on minimise le nombre de compartiments (2^bits × tables)
    puis la fraction de candidats (tie-break, même rappel espéré).
    """
    ok = df[df["mean_recall"] >= target]
    if ok.empty:
        ok = df[df["mean_recall"] == df["mean_recall"].max()]
    ok = ok.sort_values(["compartments", "mean_cand_fraction"])
    best = ok.iloc[0]
    return {
        "num_bits": int(best["num_bits"]),
        "num_tables": int(best["num_tables"]),
        "compartments": int(best["compartments"]),
        "mean_recall": float(best["mean_recall"]),
        "mean_cand_fraction": float(best["mean_cand_fraction"]),
        "reached_target": bool(best["mean_recall"] >= target),
    }
