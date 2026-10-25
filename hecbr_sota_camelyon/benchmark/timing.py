"""Harnais de chronométrage rigoureux (warm-up + 5 répétitions, médiane + percentiles)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from config.protocol import TIMING_WARMUP, TIMING_REPEATS


def timed_query(
    query_fn: Callable,
    *args,
    warmup: int  = TIMING_WARMUP,
    repeats: int = TIMING_REPEATS,
    **kwargs,
) -> Tuple[float, Any]:
    """Chronomètre une fonction de requête avec warm-up."""
    for _ in range(warmup):
        result = query_fn(*args, **kwargs)

    latencies: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = query_fn(*args, **kwargs)
        latencies.append(time.perf_counter() - t0)

    return float(np.median(latencies)), result


def benchmark_queries(
    query_fn: Callable,
    X_queries: np.ndarray,
    k: int,
    warmup: int  = TIMING_WARMUP,
    repeats: int = TIMING_REPEATS,
) -> Tuple[np.ndarray, List[Any]]:
    """Benchmark une méthode sur un ensemble de requêtes."""
    N_q = X_queries.shape[0]
    latencies = np.empty(N_q, dtype=np.float64)
    results: List[Any] = []

    for i in range(N_q):
        x = X_queries[i]
        lat, res = timed_query(query_fn, x, k, warmup=warmup, repeats=repeats)
        latencies[i] = lat
        results.append(res)

    return latencies, results


def aggregate_latencies(latencies_s: np.ndarray) -> Dict[str, float]:
    """Statistiques agrégées en millisecondes."""
    ms = latencies_s * 1000.0
    return {
        "median_ms": float(np.median(ms)),
        "mean_ms":   float(np.mean(ms)),
        "std_ms":    float(np.std(ms)),
        "p25_ms":    float(np.percentile(ms, 25)),
        "p75_ms":    float(np.percentile(ms, 75)),
        "p95_ms":    float(np.percentile(ms, 95)),
        "p99_ms":    float(np.percentile(ms, 99)),
    }
