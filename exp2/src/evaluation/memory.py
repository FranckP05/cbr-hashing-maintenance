"""Mesure mémoire (T0.6) — pic RSS du processus + allocations tracemalloc."""

import resource
import tracemalloc

from src.evaluation.timing import time_index_build, time_query


def _rss_kb() -> int:
    """Pic RSS du processus courant (ru_maxrss)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def measure_index_memory(index_factory, vectors):
    """Construit l'index et estime le coût (delta du pic RSS, en Ko).

    NB : ru_maxrss est un maximum cumulatif — le delta n'est fiable que si
    aucune allocation plus lourde n'a eu lieu avant dans le processus.
    Retourne (index, {index_peak_rss_kb}).
    """
    baseline = _rss_kb()
    index = index_factory()
    index.build(vectors)
    return index, {"index_peak_rss_kb": _rss_kb() - baseline}


def measure_query_memory(index, query_vector, k):
    """Pic d'allocations Python d'une requête (tracemalloc) + delta RSS.

    Retourne (indices, distances, {query_tracemalloc_peak_kb, query_rss_delta_kb}).
    """
    rss_before = _rss_kb()
    tracemalloc.start()
    try:
        indices, distances = index.query(query_vector, k)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return indices, distances, {
        "query_tracemalloc_peak_kb": peak_bytes / 1024.0,
        "query_rss_delta_kb": _rss_kb() - rss_before,
    }
