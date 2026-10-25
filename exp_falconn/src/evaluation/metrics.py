"""
Evaluation metrics (section 10 of protocol).

Neighbourhood quality:
    recall_at_k, precision_exact, ndcg_at_k,
    query_coverage, empty_query_rate

CBR quality:
    accuracy, macro_f1, balanced_accuracy, auc_score

LSH-specific:
    candidate_fraction, bucket_stats
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Neighbourhood quality
# --------------------------------------------------------------------------- #

def recall_at_k(
    returned: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Recall@k: fraction of true top-k neighbours that are returned.

    Args:
        returned:     Indices returned by the approximate method (at most k).
        ground_truth: Exact top-k indices from linear search (shape (k,)).
        k:            Neighbourhood size.

    Returns:
        Float in [0, 1].
    """
    if len(returned) == 0:
        return 0.0
    true_set = set(ground_truth[:k].tolist())
    retrieved_set = set(returned[:k].tolist())
    return len(true_set & retrieved_set) / k


def precision_exact(returned: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Precision of exact neighbour match at rank k."""
    true_set = set(ground_truth[:k].tolist())
    return sum(1 for r in returned[:k] if r in true_set) / max(len(returned[:k]), 1)


def ndcg_at_k(
    returned: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Relevance is 1 if the returned neighbour is in the exact top-k, 0 otherwise.
    """
    true_set = set(ground_truth[:k].tolist())
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, idx in enumerate(returned[:k])
        if idx in true_set
    )
    ideal_dcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(true_set))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def query_coverage(n_returned: int, k: int) -> float:
    """Fraction: 1 if the query returned at least k neighbours, 0 otherwise."""
    return float(n_returned >= k)


def empty_query_rate(returned_counts: List[int]) -> float:
    """Fraction of queries that returned zero candidates."""
    return sum(1 for c in returned_counts if c == 0) / len(returned_counts)


# --------------------------------------------------------------------------- #
# CBR quality
# --------------------------------------------------------------------------- #

def _majority_vote(labels: np.ndarray) -> Any:
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def cbr_predictions(y_train: np.ndarray, neighbour_indices: np.ndarray) -> np.ndarray:
    """Majority-vote CBR prediction over retrieved neighbours.

    Args:
        y_train:          Labels of the base set.
        neighbour_indices: (n_queries, k) array of neighbour indices.

    Returns:
        Predicted label for each query (shape (n_queries,)).
    """
    return np.array([
        _majority_vote(y_train[idxs]) if len(idxs) > 0 else -1
        for idxs in neighbour_indices
    ])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: Sequence) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def balanced_accuracy(y_true: np.ndarray, y_pred: Sequence) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def auc_score(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    multi_class: str = "ovr",
) -> float:
    """AUC with correct tie handling for multi-class."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(
            roc_auc_score(y_true, y_scores, multi_class=multi_class, average="macro")
        )
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# LSH-specific measures
# --------------------------------------------------------------------------- #

def candidate_fraction(n_candidates: int, base_size: int) -> float:
    """Fraction of the base that was examined as candidates."""
    return n_candidates / base_size if base_size > 0 else 0.0


def bucket_stats(n_buckets_visited_per_query: List[int]) -> dict:
    arr = np.array(n_buckets_visited_per_query, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
