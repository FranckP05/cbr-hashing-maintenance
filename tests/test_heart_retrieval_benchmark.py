import numpy as np

from src.heart_retrieval_benchmark import distance_weighted_prediction, ranking_metrics


def test_ranking_metrics_pad_missing_neighbors_as_non_relevant():
    metrics = ranking_metrics(
        np.array([0, 1]), np.array([1, 0, 1, 0]), query_target=1, k=5
    )

    assert metrics["precision_at_k"] == 0.2
    assert metrics["label_recall_at_k"] == 0.5
    assert metrics["hit_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 1.0


def test_distance_weighted_prediction_returns_no_value_without_neighbors():
    prediction, score = distance_weighted_prediction(np.array([]), np.array([]))

    assert np.isnan(prediction)
    assert np.isnan(score)
