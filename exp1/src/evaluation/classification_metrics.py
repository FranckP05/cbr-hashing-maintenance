import numpy as np


def distance_weighted_prediction(
    distances: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    if len(targets) == 0:
        return np.nan, np.nan
    weights = 1.0 / np.maximum(distances, np.finfo(float).eps)
    positive_score = float(np.average(targets, weights=weights))
    return float(positive_score >= 0.5), positive_score
