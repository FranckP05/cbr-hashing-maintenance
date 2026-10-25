"""
Timing harness for query benchmarking (section 9).

For each query:
    - 1 warm-up repetition (excluded from measurement).
    - 5 measured repetitions.
    - Median latency is kept.

Final statistics reported:
    - median, mean, p95 across all queries.
    - inter-seed spread (computed externally by the benchmark runner).

The timing harness does NOT include:
    - dataset loading;
    - index construction;
    - Recall@k or classification computation;
    - disk writes.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Tuple

import numpy as np

from config.protocol import TIMING_WARMUP, TIMING_REPEATS


def timed_query(
    query_fn: Callable,
    *args,
    warmup: int = TIMING_WARMUP,
    repeats: int = TIMING_REPEATS,
    **kwargs,
) -> Tuple[float, Any]:
    """Time a query function with warm-up, return median latency.

    The chrono starts immediately before the function call and stops
    immediately after it returns. It does not include post-processing.

    Args:
        query_fn: Callable to benchmark.
        *args:    Positional arguments forwarded to query_fn.
        warmup:   Number of warm-up calls (not measured).
        repeats:  Number of measured calls.
        **kwargs: Keyword arguments forwarded to query_fn.

    Returns:
        (median_latency_s, last_result)
        where last_result is the return value of the final measured call.
    """
    # Warm-up.
    for _ in range(warmup):
        result = query_fn(*args, **kwargs)

    # Measured repetitions.
    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = query_fn(*args, **kwargs)
        latencies.append(time.perf_counter() - t0)

    return float(np.median(latencies)), result


def aggregate_latencies(per_query_medians: list) -> dict:
    """Aggregate per-query median latencies into summary statistics.

    Args:
        per_query_medians: List of per-query median latencies in seconds.

    Returns:
        Dict with keys: median, mean, p95.
    """
    arr = np.array(per_query_medians)
    return {
        "median_s": float(np.median(arr)),
        "mean_s": float(np.mean(arr)),
        "p95_s": float(np.percentile(arr, 95)),
    }
