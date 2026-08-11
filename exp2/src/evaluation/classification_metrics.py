"""Métriques de classification (T0.4) — calculées sans dépendance externe."""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction d'étiquettes correctes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 moyenne par classe (macro) — convention : F1=0 si P+R=0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if classes.size == 0 or y_true.size == 0:
        return float("nan")
    f1_scores = []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1_scores.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return float(np.mean(f1_scores))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Moyenne des rappels par classe (équivalent balanced accuracy)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if classes.size == 0:
        return float("nan")
    recalls = []
    for c in classes:
        mask = y_true == c
        if np.sum(mask) == 0:
            continue
        recalls.append(float(np.mean(y_pred[mask] == c)))
    return float(np.mean(recalls)) if recalls else float("nan")


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC binaire (aire sous la courbe ROC) par rang — Mann-Whitney U.

    y_score est le score de la classe positive ; les étiquettes attendues
    sont {0, 1}. Retourne NaN si une seule classe est présente.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return float("nan")
    pos = y_true == 1
    neg = ~pos
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, y_true.size + 1)
    sum_pos_ranks = float(np.sum(ranks[pos]))
    auc_val = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc_val)


def distance_weighted_prediction(
    distances: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    if len(targets) == 0:
        return np.nan, np.nan
    weights = 1.0 / np.maximum(distances, np.finfo(float).eps)
    positive_score = float(np.average(targets, weights=weights))
    return float(positive_score >= 0.5), positive_score
