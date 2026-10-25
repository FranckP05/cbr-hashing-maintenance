"""Métriques de qualité de voisinage (Recall@k) et de classification (Macro-F1)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.metrics import f1_score


# --------------------------------------------------------------------------- #
# Voisinage
# --------------------------------------------------------------------------- #

def recall_at_k(
    returned: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Rappel@k : fraction des vrais top-k voisins exacts retrouvés.

    Args:
        returned     : indices retournés par la méthode approchée (≤ k éléments).
        ground_truth : indices exacts (vérité terrain LinearScan), shape (k,).
        k            : taille du voisinage de référence.

    Returns:
        Float dans [0, 1].
    """
    if len(returned) == 0:
        return 0.0
    true_set = set(ground_truth[:k].tolist())
    ret_set  = set(returned[:k].tolist())
    return len(true_set & ret_set) / k


def mean_recall_at_k(
    all_returned: List[np.ndarray],
    all_ground_truth: np.ndarray,
    k: int,
) -> float:
    """Rappel@k moyen sur l'ensemble des requêtes.

    Args:
        all_returned     : liste de N_q tableaux d'indices retournés.
        all_ground_truth : (N_q, k) indices exacts.
        k                : taille du voisinage.
    """
    recalls = [
        recall_at_k(all_returned[i], all_ground_truth[i], k)
        for i in range(len(all_returned))
    ]
    return float(np.mean(recalls))


def candidate_fraction(n_candidates: int, n_base: int) -> float:
    """Fraction de la base explorée comme candidats."""
    return n_candidates / n_base if n_base > 0 else 0.0


def mean_candidate_fraction(n_cands_per_query: List[int], n_base: int) -> float:
    """Fraction moyenne de candidats explorés sur toutes les requêtes."""
    return float(np.mean([c / n_base for c in n_cands_per_query]))


# --------------------------------------------------------------------------- #
# Classification CBR (k-NN par vote majoritaire)
# --------------------------------------------------------------------------- #

def majority_vote(labels: np.ndarray) -> int:
    """Vote majoritaire sur un tableau de labels."""
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def cbr_predict(
    y_base: np.ndarray,
    neighbour_indices: List[np.ndarray],
) -> np.ndarray:
    """Prédictions CBR par vote majoritaire sur les voisins retournés.

    Args:
        y_base            : labels de la base d'indexation (N_base,).
        neighbour_indices : liste de N_q tableaux d'indices de voisins.

    Returns:
        y_pred : (N_q,) int64 — label prédit par majorité.
    """
    preds = []
    for idx in neighbour_indices:
        if len(idx) == 0:
            preds.append(-1)
        else:
            preds.append(majority_vote(y_base[idx]))
    return np.array(preds, dtype=np.int64)


def macro_f1(y_true: np.ndarray, y_pred: Sequence) -> float:
    """Macro-F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
