"""Protocole de timing (T0.5) — warm-up, médiane sur répétitions, perf_counter."""

import statistics
import time

DEFAULT_WARMUP = 1
DEFAULT_REPETITIONS = 5


def _median(seconds):
    if not seconds:
        return float("nan")
    return statistics.median(seconds)


def time_index_build(
    index_factory, vectors, warmup=DEFAULT_WARMUP, repetitions=DEFAULT_REPETITIONS
):
    """Construit l'index `repetitions` fois après `warmup` itérations.

    Retourne (index_final, médiane_s, times_s).
    """
    for _ in range(warmup):
        index_factory().build(vectors)
    times = []
    index = None
    for _ in range(repetitions):
        index = index_factory()
        start = time.perf_counter()
        index.build(vectors)
        times.append(time.perf_counter() - start)
    return index, _median(times), times


def time_query(
    index, query_vector, k, warmup=DEFAULT_WARMUP, repetitions=DEFAULT_REPETITIONS
):
    """Exécute une requête `repetitions` fois après `warmup` itérations.

    Retourne (indices, distances, médiane_s, times_s) du dernier run.
    """
    for _ in range(warmup):
        index.query(query_vector, k)
    indices, distances, times = None, None, []
    for _ in range(repetitions):
        start = time.perf_counter()
        indices, distances = index.query(query_vector, k)
        times.append(time.perf_counter() - start)
    return indices, distances, _median(times), times


def time_queries(
    index, queries, k, warmup=DEFAULT_WARMUP, repetitions=DEFAULT_REPETITIONS
):
    """Médiane des temps par requête sur un lot, puis médiane agrégée.

    Retourne (médiane_agrégée_s, médianes_par_requête_s).
    """
    medians = [
        time_query(index, q, k, warmup=warmup, repetitions=repetitions)[2]
        for q in queries
    ]
    return _median(medians), medians
