"""Harness de benchmark des index seuls (T1.1–T1.6).

Pour chaque dataset et chaque seed :
  1. split train/test (adapters.get_split) ;
  2. construction de chaque index (protocole T0.5 : warm-up + médiane) ;
  3. requêtes sur un échantillon fixé par dataset (reproductible par seed) ;
  4. métriques de rangement (T0.4), rappel des voisins exacts (scan linéaire) et
     métriques de classification (vote pondéré par la distance) ;
  5. écriture incrémentale (checkpoint CSV, reprise sans doublon).

HeCBR est évalué à part (Diabetes-130 uniquement, split fixe du parquet) : distance de
Hamming, recherche restreinte au train, qualité de classification uniquement (le rappel
des voisins exacts euclidiens n'a pas de sens dans l'espace des codes).

Variables d'environnement de réduction locale (sanity) :
  NB03_SEEDS     : liste de seeds, ex. "42,1337" (défaut : les 6 seeds)
  NB03_QUERY_CAP : plafond de requêtes, ex. "60" (défaut : N_QUERIES_BY_DATASET)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import benchmark as bench_cfg
from config import seeds as seed_cfg
from src.datasets import adapters
from src.evaluation import classification_metrics as cm
from src.evaluation import retrieval_metrics as rm
from src.evaluation import timing
from src.indexing.hecbr_index import HeCBRIndex
from src.indexing.kdtree_index import KDTreeIndex
from src.indexing.linear_scan import LinearScanIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex

_RESULT_COLS = [
    "dataset", "seed", "method", "k", "n_queries", "answered",
    "build_ms", "query_ms",
    "exact_recall_at_k", "precision_at_k", "label_recall_at_k", "map_at_k",
    "ndcg_at_k", "hit_at_k", "mrr_at_k",
    "accuracy", "macro_f1", "balanced_accuracy", "auc",
]
_MEMORY_COLS = ["dataset", "method", "index_peak_rss_kb",
                "query_tracemalloc_peak_kb", "query_rss_delta_kb"]


def _env_seeds():
    raw = os.environ.get("NB03_SEEDS")
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return list(seed_cfg.ALL_SEEDS)


def _env_query_cap():
    raw = os.environ.get("NB03_QUERY_CAP")
    return int(raw) if raw else None


def method_factories(key):
    """Factories des index euclidiens disponibles pour un dataset (HeCBR à part)."""
    return {
        "linear": lambda: LinearScanIndex(),
        "kdtree": lambda: KDTreeIndex(),
        "lsh": lambda: RandomProjectionLSHIndex(
            **bench_cfg.LSH_PARAMS_BY_DATASET[key], random_seed=42
        ),
    }


def sample_queries(X_test, seed, cap):
    """Indices de requêtes échantillonnés par seed (déterministe)."""
    n = len(X_test)
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def calibrate_lsh(key, X_train, progress=print):
    """Calibration bits/tables (T1.3) : grille × 30 requêtes, rappel ≥ 0.95.

    Retourne (params_retenus, records).
    """
    n_queries = min(bench_cfg.CALIB_QUERIES, len(X_train))
    rng = np.random.default_rng(bench_cfg.CALIB_SEED)
    queries = X_train[rng.choice(len(X_train), size=n_queries, replace=False)]

    ref = LinearScanIndex()
    ref.build(X_train)
    gt = [set(ref.query(q, bench_cfg.CALIB_K)[0]) for q in queries]

    records = []
    for num_bits in bench_cfg.CALIB_BITS_BY_DATASET[key]:
        for num_tables in bench_cfg.CALIB_TABLES:
            lsh = RandomProjectionLSHIndex(
                num_tables=num_tables, num_bits=num_bits, random_seed=bench_cfg.CALIB_SEED
            )
            t0 = time.perf_counter()
            lsh.build(X_train)
            build_ms = (time.perf_counter() - t0) * 1e3
            recalls = []
            for q, g in zip(queries, gt):
                idx, _ = lsh.query(q, bench_cfg.CALIB_K)
                recalls.append(len(set(idx) & g) / bench_cfg.CALIB_K)
            records.append({
                "dataset": key, "num_bits": num_bits, "num_tables": num_tables,
                "mean_recall": float(np.mean(recalls)),
                "min_recall": float(np.min(recalls)),
                "build_ms": round(build_ms, 2),
            })
            progress(
                f"  bits={num_bits:2d} tables={num_tables:2d} "
                f"recall={np.mean(recalls):.4f} build={build_ms:.1f}ms"
            )

    df = pd.DataFrame(records)
    ok = df[df["mean_recall"] >= bench_cfg.CALIB_RECALL_TARGET]
    ok = ok.sort_values(["num_tables", "num_bits"], ascending=[True, False])
    best = ok.iloc[0] if not ok.empty else df.loc[df["mean_recall"].idxmax()]
    params = {"num_bits": int(best["num_bits"]), "num_tables": int(best["num_tables"])}
    return params, df


def _append_csv(path, rows, done, done_keys):
    """Écrit les lignes (checkpoint) et met à jour l'ensemble des clés traitées."""
    new = [r for r in rows if done_keys(r) not in done]
    if not new:
        return 0
    df = pd.DataFrame(new)[_RESULT_COLS]
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    done.update(done_keys(r) for r in new)
    return len(new)


def run_method(index, exact_by_k, queries, query_targets, y_train, k, method_is_exact):
    """Métriques agrégées d'une méthode sur un lot de requêtes (k fixé)."""
    agg, exact_recalls, preds, scores, qys = {}, [], [], [], []
    for i, (q, qy) in enumerate(zip(queries, query_targets)):
        if method_is_exact:
            idx, dist = exact_by_k[k][i]
        else:
            idx, dist = index.query(q, k)
        if len(idx) == 0:
            continue
        qys.append(qy)
        exact_recalls.append(len(np.intersect1d(exact_by_k[k][i][0], idx)) / k)
        for key, val in rm.ranking_metrics(idx, y_train, query_target=int(qy), k=k).items():
            agg[key] = agg.get(key, 0.0) + val
        pred, score = cm.distance_weighted_prediction(dist, y_train[idx])
        preds.append(pred)
        scores.append(score)

    out = {"n_queries": len(queries), "answered": len(qys)}
    if qys:
        n = len(qys)
        out.update({key: val / n for key, val in agg.items()})
        out["exact_recall_at_k"] = float(np.mean(exact_recalls))
        qys = np.asarray(qys)
        preds = np.asarray(preds)
        out["accuracy"] = cm.accuracy(qys, preds)
        out["macro_f1"] = cm.macro_f1(qys, preds)
        out["balanced_accuracy"] = cm.balanced_accuracy(qys, preds)
        out["auc"] = cm.auc(qys, np.asarray(scores))
    else:
        out.update({c: float("nan") for c in [
            "exact_recall_at_k", "precision_at_k", "label_recall_at_k", "map_at_k",
            "ndcg_at_k", "hit_at_k", "mrr_at_k",
            "accuracy", "macro_f1", "balanced_accuracy", "auc"]})
    return out


def benchmark_dataset(key, run_dir, progress=print):
    """Boucle seeds × méthodes × k pour un dataset (checkpoint CSV)."""
    csv_path = run_dir / "benchmark_timing_quality.csv"
    done = set()
    if csv_path.exists():
        done = {_result_key(r) for r in pd.read_csv(csv_path).to_dict("records")}

    build_cache = {}
    written = 0
    for seed in _env_seeds():
        X_train, y_train, X_test, y_test, _, _ = adapters.get_split(key, seed=seed)
        cap = _env_query_cap() or bench_cfg.N_QUERIES_BY_DATASET[key]
        q_idx = sample_queries(X_test, seed, cap)
        queries = X_test[q_idx]
        query_targets = y_test[q_idx].astype(int)
        progress(f"[{key} seed={seed}] split {X_train.shape} -> {len(queries)} requêtes")

        exact_index = LinearScanIndex()
        exact_index.build(X_train)
        exact_by_k = {k: [exact_index.query(q, k) for q in queries]
                      for k in seed_cfg.K_VALUES}

        for method, factory in method_factories(key).items():
            build_key = (key, method)
            if build_key not in build_cache:
                _, build_cache[build_key], _ = timing.time_index_build(
                    factory, X_train,
                    warmup=bench_cfg.BUILD_WARMUP, repetitions=bench_cfg.BUILD_REPETITIONS,
                )
                progress(f"  build {method}: {build_cache[build_key]*1e3:.1f} ms (mesuré 1×)")
            index = factory()
            index.build(X_train)
            build_ms = round(build_cache[build_key] * 1e3, 4)

            rows = []
            for k in seed_cfg.K_VALUES:
                med_q, _ = timing.time_queries(
                    index, queries, k,
                    warmup=bench_cfg.QUERY_WARMUP, repetitions=bench_cfg.QUERY_REPETITIONS,
                )
                res = run_method(index, exact_by_k, queries, query_targets, y_train, k,
                                 method_is_exact=(method == "linear"))
                rows.append({
                    "dataset": key, "seed": seed, "method": method, "k": k,
                    "build_ms": build_ms, "query_ms": round(med_q * 1e3, 4), **res,
                })
            written += _append_csv(csv_path, rows, done, _result_key)
            progress(f"  {method} terminé (k={seed_cfg.K_VALUES})")
    progress(f"[{key}] lignes écrites : {written}")
    return written


def _result_key(r):
    return (r["dataset"], str(r["seed"]), r["method"], int(r["k"]), int(r["n_queries"]))


def benchmark_hecbr(run_dir, progress=print):
    """Benchmark HeCBR (T1.4) — Diabetes-130, split fixe, Hamming, train restreint."""
    path = bench_cfg.HECBR_CODES_PATH
    if not os.path.exists(path):
        progress(f"HeCBR indisponible (fichier manquant : {path})")
        return 0

    csv_path = run_dir / "benchmark_timing_quality.csv"
    done = set()
    if csv_path.exists():
        done = {_result_key(r) for r in pd.read_csv(csv_path).to_dict("records")}

    _, build_med, _ = timing.time_index_build(
        lambda: HeCBRIndex(path), None,
        warmup=bench_cfg.BUILD_WARMUP, repetitions=bench_cfg.BUILD_REPETITIONS,
    )
    index = HeCBRIndex(path)
    index.build(None, restrict_split="train")
    progress(f"  hecbr build: {build_med*1e3:.1f} ms | codes train {index.codes.shape}")

    df = pd.read_parquet(path)
    bit_cols = [c for c in df.columns if c.startswith("bit")]
    all_codes = df[bit_cols].to_numpy(dtype=np.int8)
    all_labels = df["label"].to_numpy(dtype=np.int64)
    test_idx = np.flatnonzero(df["split"].to_numpy(dtype=object) == "test")
    cap = _env_query_cap() or bench_cfg.N_QUERIES_BY_DATASET["diabetes130"]
    rng = np.random.default_rng(42)
    sel = np.sort(rng.choice(len(test_idx), size=min(cap, len(test_idx)), replace=False))
    q_codes = all_codes[test_idx[sel]]
    q_labels = all_labels[test_idx[sel]]

    rows = []
    for k in seed_cfg.K_VALUES:
        med_q, _ = timing.time_queries(
            index, q_codes, k,
            warmup=bench_cfg.QUERY_WARMUP, repetitions=bench_cfg.QUERY_REPETITIONS,
        )
        preds, scores, qys = [], [], []
        for q, qy in zip(q_codes, q_labels):
            idx, dist = index.query(q, k)
            if len(idx) == 0:
                continue
            qys.append(qy)
            pred, score = cm.distance_weighted_prediction(dist, index.labels[idx])
            preds.append(pred)
            scores.append(score)
        qys, preds = np.asarray(qys), np.asarray(preds)
        rows.append({
            "dataset": "diabetes130", "seed": "fixed", "method": "hecbr", "k": k,
            "build_ms": round(build_med * 1e3, 4), "query_ms": round(med_q * 1e3, 4),
            "n_queries": len(q_codes), "answered": len(qys),
            "exact_recall_at_k": float("nan"),
            "precision_at_k": float("nan"), "label_recall_at_k": float("nan"),
            "map_at_k": float("nan"), "ndcg_at_k": float("nan"),
            "hit_at_k": float("nan"), "mrr_at_k": float("nan"),
            "accuracy": cm.accuracy(qys, preds),
            "macro_f1": cm.macro_f1(qys, preds),
            "balanced_accuracy": cm.balanced_accuracy(qys, preds),
            "auc": cm.auc(qys, np.asarray(scores)),
        })
    written = _append_csv(csv_path, rows, done, _result_key)
    progress(f"[hecbr] lignes écrites : {written}")
    return written


def benchmark_memory(key, run_dir, progress=print):
    """Probes mémoire par sous-processus (T1.6) — pic RSS propre par méthode.

    Un sous-processus "baseline" charge uniquement les données (pic RSS absolu) ;
    la contribution d'un index vaut pic(build) - pic(baseline), le pic étant dominé
    par le chargement des données.
    """
    csv_path = run_dir / "benchmark_memory.csv"
    methods = list(method_factories(key))
    if key == "diabetes130" and os.path.exists(bench_cfg.HECBR_CODES_PATH):
        methods.append("hecbr")

    existing = {}
    if csv_path.exists():
        existing = {(r["dataset"], r["method"]) for r in pd.read_csv(csv_path).to_dict("records")}

    probe = Path(__file__).resolve().parent / "benchmark_memory.py"
    exp2_root = Path(__file__).resolve().parents[2]

    def _run_probe(method):
        proc = subprocess.run(
            [sys.executable, str(probe), "--dataset", key, "--method", method,
             "--exp2", str(exp2_root)],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"probe mémoire {key}/{method} échouée:\n{proc.stderr[-2000:]}")
        return json.loads(proc.stdout)

    baseline = None
    if not existing or any(m not in existing for m in methods):
        baseline = _run_probe("baseline")["peak_after_data_kb"]

    rows = []
    for method in methods:
        if (key, method) in existing:
            continue
        result = _run_probe(method)
        rows.append({
            "dataset": key,
            "method": method,
            "index_peak_rss_kb": max(0, result["peak_after_build_kb"] - baseline),
            "query_tracemalloc_peak_kb": result["query_tracemalloc_peak_kb"],
            "query_rss_delta_kb": result["query_rss_delta_kb"],
        })
        progress(f"  {method}: index +{rows[-1]['index_peak_rss_kb']:.0f} Ko, "
                 f"requête tracemalloc {rows[-1]['query_tracemalloc_peak_kb']:.0f} Ko")

    if rows:
        df = pd.DataFrame(rows)[_MEMORY_COLS]
        if csv_path.exists():
            df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
        df.to_csv(csv_path, index=False)
    return len(rows)
