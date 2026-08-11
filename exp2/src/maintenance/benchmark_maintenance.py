"""Harness de maintenance (T3.1-T3.4) : rétention par collisions + condensation.

Rétention (T3.1) : flux de cas (test, par seed) vs base de train ; seuil de redondance
θ = P25 des distances 1-NN du train ; décision LSH vs référence exacte (accord cas par
cas, taux de réduction, coût, taille du bassin de candidats).

Condensation (T3.2-T3.4) : random / CBM / bucket / CNN sur le même sous-échantillon du
train, aux taux cibles 20 % et 50 % ; qualité 1-NN + temps de requête (k=10) sur la base
condensée vs brute ; build mesuré pour chaque condensation.

Variables d'environnement de réduction locale (sanity) :
  NB07_SEEDS        : liste de seeds, ex. "42"
  NB07_QUERY_CAP    : plafond de cas du flux de test, ex. "60"
  NB07_TRAIN_CAP    : plafond de cas du train pour la rétention, ex. "5000"
  NB07_SUBSAMPLE    : taille du sous-échantillon de condensation, ex. "2000"
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config import benchmark as bench_cfg
from config import maintenance as mnt_cfg
from config import seeds as seed_cfg
from src.benchmark.benchmark_indexes import LinearScanIndex, method_factories, sample_queries
from src.datasets import adapters
from src.evaluation import classification_metrics as cm
from src.evaluation import timing
from src.indexing.lsh_index import RandomProjectionLSHIndex

_RETENTION_COLS = [
    "dataset", "seed", "n_stream", "threshold",
    "retained_lsh", "retained_exact", "agreement",
    "reduction_rate_lsh", "reduction_rate_exact",
    "mean_candidates_lsh", "stream_ms_per_case",
]

_COND_COLS = [
    "dataset", "seed", "strategy", "rate", "subsample_size", "condensed_size",
    "achieved_rate", "build_ms",
    "nn_accuracy", "nn_macro_f1", "nn_balanced_accuracy",
    "query_ms_linear", "query_ms_kdtree", "query_ms_lsh",
]


def _env_seeds():
    raw = os.environ.get("NB07_SEEDS")
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return list(seed_cfg.ALL_SEEDS)


def _env_query_cap():
    raw = os.environ.get("NB07_QUERY_CAP")
    return int(raw) if raw else None


def _env_train_cap():
    raw = os.environ.get("NB07_TRAIN_CAP")
    return int(raw) if raw else None


def _env_subsample():
    raw = os.environ.get("NB07_SUBSAMPLE")
    return int(raw) if raw else None


def _append(csv_path, rows, done, key_fn, cols):
    new = [r for r in rows if key_fn(r) not in done]
    if not new:
        return 0
    df = pd.DataFrame(new)[cols]
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    done.update(key_fn(r) for r in new)
    return len(new)


def retention_threshold(train_vectors, percentile=None, n_sample=None, random_seed=42):
    """θ = percentile des distances au plus proche voisin sur un échantillon du train."""
    percentile = percentile or mnt_cfg.RETENTION_NN_PERCENTILE
    n_sample = n_sample or mnt_cfg.RETENTION_THRESHOLD_SAMPLE
    n_sample = min(n_sample, len(train_vectors))
    rng = np.random.default_rng(random_seed)
    sample = train_vectors[rng.choice(len(train_vectors), size=n_sample, replace=False)]
    finder = NearestNeighbors(n_neighbors=2)
    finder.fit(train_vectors)
    distances, _ = finder.kneighbors(sample)
    return float(np.percentile(distances[:, 1], percentile))


def _lsh_factory(key, random_seed=42):
    return RandomProjectionLSHIndex(
        **bench_cfg.LSH_PARAMS_BY_DATASET[key], random_seed=random_seed
    )


def retention(key, run_dir, progress=print):
    """T3.1 — flux de cas : décisions LSH vs exacte (accord, réduction, coût)."""
    csv_path = run_dir / "retention_agreement.csv"
    done = set()
    if csv_path.exists():
        done = {(r["dataset"], str(r["seed"])) for r in pd.read_csv(csv_path).to_dict("records")}

    cap = _env_query_cap() or bench_cfg.N_QUERIES_BY_DATASET[key]
    train_cap = _env_train_cap()
    written = 0
    for seed in _env_seeds():
        if (key, str(seed)) in done:
            continue
        X_train, y_train, X_test, y_test, _, _ = adapters.get_split(key, seed=seed)
        if train_cap:
            X_train, y_train = X_train[:train_cap], y_train[:train_cap]

        theta = retention_threshold(X_train, random_seed=seed)
        q_idx = sample_queries(X_test, seed, cap)
        stream = X_test[q_idx]

        lsh_index = _lsh_factory(key, random_seed=seed)
        exact_index = LinearScanIndex()
        lsh_index.build(X_train)
        exact_index.build(X_train)

        decisions_lsh, decisions_exact, candidates = [], [], []
        t0 = time.perf_counter()
        for q in stream:
            idx_l, dist_l = lsh_index.query(q, mnt_cfg.RETENTION_LSH_K)
            idx_e, dist_e = exact_index.query(q, mnt_cfg.RETENTION_LSH_K)
            redundant_lsh = bool(len(idx_l) and dist_l[0] < theta)
            redundant_exact = bool(len(idx_e) and dist_e[0] < theta)
            decisions_lsh.append(redundant_lsh)
            decisions_exact.append(redundant_exact)
            candidates.append(getattr(lsh_index, "last_candidate_count", 0))
            if not redundant_lsh:
                lsh_index.add_case(q)
            if not redundant_exact:
                exact_index.add_case(q)
        elapsed = time.perf_counter() - t0

        rows = [{
            "dataset": key, "seed": seed, "n_stream": len(stream), "threshold": theta,
            "retained_lsh": len(stream) - int(sum(decisions_lsh)),
            "retained_exact": len(stream) - int(sum(decisions_exact)),
            "agreement": float(np.mean([a == b for a, b in zip(decisions_lsh, decisions_exact)])),
            "reduction_rate_lsh": float(np.mean(decisions_lsh)),
            "reduction_rate_exact": float(np.mean(decisions_exact)),
            "mean_candidates_lsh": float(np.mean(candidates)),
            "stream_ms_per_case": round(elapsed / max(len(stream), 1) * 1e3, 4),
        }]
        written += _append(csv_path, rows, done, lambda r: (r["dataset"], str(r["seed"])),
                           _RETENTION_COLS)
        progress(f"  {key} seed={seed}: θ={theta:.4f} | accord="
                 f"{rows[0]['agreement']:.3f} réduction LSH={rows[0]['reduction_rate_lsh']:.3f}")
    progress(f"[{key}] rétention : lignes écrites {written}")
    return written


def _closest_to_centroid(vectors):
    centroid = vectors.mean(axis=0)
    sq = np.einsum("ij,ij->i", vectors - centroid, vectors - centroid)
    return int(np.argmin(sq))


def _condense_random(vectors, target, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(vectors), size=target, replace=False))


def _condense_cbm(vectors, targets, target, seed):
    from .condensation_cbm import compute_relative_coverage

    theta = retention_threshold(vectors, random_seed=seed)
    rc = compute_relative_coverage(vectors, targets, theta)
    order = np.argsort(rc)
    return np.sort(order[-target:])


def _condense_bucket(vectors, target, seed, key):
    num_bits = mnt_cfg.BUCKET_NUM_BITS_BY_DATASET[key]
    idx = RandomProjectionLSHIndex(num_bits=num_bits, num_tables=1, random_seed=seed)
    idx.build(vectors)
    groups = sorted(idx.buckets[0].values(), key=len, reverse=True)
    kept = []
    for g in groups:
        if len(kept) >= target:
            break
        kept.append(g[_closest_to_centroid(vectors[g])])
    return np.sort(np.array(kept, dtype=int)[:target])


def _condense_cnn(vectors, targets, seed):
    from .condensation_cnn import condense_dataset

    return condense_dataset(vectors, targets, random_seed=seed)


def _nn_quality(reference, reference_y, queries, query_y):
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(reference)
    _, indices = nn.kneighbors(queries)
    pred = reference_y[indices[:, 0]]
    return {
        "nn_accuracy": cm.accuracy(query_y, pred),
        "nn_macro_f1": cm.macro_f1(query_y, pred),
        "nn_balanced_accuracy": cm.balanced_accuracy(query_y, pred),
    }


def _query_time_ms(reference, queries, key, random_seed=42):
    """Temps de requête (k=10) pour linear / kdtree / lsh sur une base donnée."""
    out = {}
    for method, factory in method_factories(key).items():
        index = factory()
        index.build(reference)
        k = min(mnt_cfg.POST_QUERY_K, len(reference))
        med_q, _ = timing.time_queries(
            index, queries, k,
            warmup=bench_cfg.QUERY_WARMUP, repetitions=bench_cfg.QUERY_REPETITIONS,
        )
        out[f"query_ms_{method}"] = round(med_q * 1e3, 4)
    return out


def condensation(key, run_dir, progress=print):
    """T3.2-T3.4 — condensation (20 % / 50 %) + post-maintenance vs base brute."""
    csv_path = run_dir / "condensation_quality.csv"
    done = set()
    if csv_path.exists():
        done = {(r["dataset"], str(r["seed"]), r["strategy"], float(r["rate"]))
                for r in pd.read_csv(csv_path).to_dict("records")}

    cap = _env_query_cap() or bench_cfg.N_QUERIES_BY_DATASET[key]
    subsample = _env_subsample() or mnt_cfg.COND_SUBSAMPLE
    written = 0
    for seed in _env_seeds():
        X_train, y_train, X_test, y_test, _, _ = adapters.get_split(key, seed=seed)
        base, base_y = X_train[:subsample], y_train[:subsample]
        q_idx = sample_queries(X_test, seed, cap)
        queries, query_y = X_test[q_idx], y_test[q_idx].astype(int)

        rows = []
        # Référence brute (même sous-échantillon)
        key0 = (key, str(seed), "raw", 1.0)
        if key0 not in done:
            t0 = time.perf_counter()
            build_ms = (time.perf_counter() - t0) * 1e3
            rows.append({
                "dataset": key, "seed": seed, "strategy": "raw", "rate": 1.0,
                "subsample_size": len(base), "condensed_size": len(base),
                "achieved_rate": 1.0, "build_ms": build_ms,
                **_nn_quality(base, base_y, queries, query_y),
                **_query_time_ms(base, queries, key),
            })

        for rate in mnt_cfg.COND_RATES:
            target = max(int(round(rate * len(base))), 1)
            for strategy in mnt_cfg.COND_STRATEGIES:
                ckey = (key, str(seed), strategy, rate)
                if ckey in done:
                    continue
                t0 = time.perf_counter()
                if strategy == "random":
                    indices = _condense_random(base, target, seed)
                elif strategy == "cbm":
                    indices = _condense_cbm(base, base_y, target, seed)
                elif strategy == "bucket":
                    indices = _condense_bucket(base, target, seed, key)
                else:  # cnn
                    indices = _condense_cnn(base, base_y, seed)
                    if len(indices) > target:
                        indices = indices[:target]
                build_ms = (time.perf_counter() - t0) * 1e3

                condensed = base[indices]
                condensed_y = base_y[indices]
                rows.append({
                    "dataset": key, "seed": seed, "strategy": strategy, "rate": rate,
                    "subsample_size": len(base), "condensed_size": int(len(indices)),
                    "achieved_rate": float(len(indices) / len(base)),
                    "build_ms": round(build_ms, 4),
                    **_nn_quality(condensed, condensed_y, queries, query_y),
                    **_query_time_ms(condensed, queries, key),
                })
                progress(f"  {key} seed={seed} {strategy}@{rate}: "
                         f"{len(indices)}/{len(base)} cas, "
                         f"acc={rows[-1]['nn_accuracy']:.3f}")
        written += _append(csv_path, rows, done,
                           lambda r: (r["dataset"], str(r["seed"]), r["strategy"],
                                      float(r["rate"])),
                           _COND_COLS)
    progress(f"[{key}] condensation : lignes écrites {written}")
    return written
