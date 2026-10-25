"""Métriques de qualité de voisinage (Recall@k) et de classification CBR (Macro-F1)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.metrics import f1_score


def recall_at_k(returned: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Fraction des vrais top-k voisins retrouvés."""
    if len(returned) == 0:
        return 0.0
    true_set = set(ground_truth[:k].tolist())
    ret_set  = set(returned[:k].tolist())
    return len(true_set & ret_set) / k


def mean_recall_at_k(all_returned: List[np.ndarray], all_ground_truth: np.ndarray, k: int) -> float:
    recalls = [
        recall_at_k(all_returned[i], all_ground_truth[i], k)
        for i in range(len(all_returned))
    ]
    return float(np.mean(recalls))


def majority_vote(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def cbr_predict(y_base: np.ndarray, neighbour_indices: List[np.ndarray]) -> np.ndarray:
    preds = []
    for idx in neighbour_indices:
        if len(idx) == 0:
            preds.append(-1)
        else:
            preds.append(majority_vote(y_base[idx]))
    return np.array(preds, dtype=np.int64)


def macro_f1(y_true: np.ndarray, y_pred: Sequence) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
