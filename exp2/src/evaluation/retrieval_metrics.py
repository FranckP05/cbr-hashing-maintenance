import numpy as np


def ranking_metrics(
    retrieved_indices: np.ndarray,
    train_targets: np.ndarray,
    query_target: int,
    k: int,
) -> dict[str, float]:
    relevance = train_targets[retrieved_indices] == query_target
    padded_relevance = np.zeros(k, dtype=bool)
    padded_relevance[: len(relevance)] = relevance
    relevant_in_base = int(np.sum(train_targets == query_target))
    ranks = np.arange(1, k + 1)

    precision = float(np.sum(padded_relevance) / k)
    recall = float(np.sum(padded_relevance) / relevant_in_base)
    precision_at_rank = np.cumsum(padded_relevance) / ranks
    average_precision = float(
        np.sum(precision_at_rank * padded_relevance) / min(k, relevant_in_base)
    )
    discounts = 1.0 / np.log2(ranks + 1)
    dcg = float(np.sum(padded_relevance * discounts))
    ideal_dcg = float(np.sum(discounts[: min(k, relevant_in_base)]))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    hit = float(np.any(padded_relevance))
    first_relevant = np.flatnonzero(padded_relevance)
    reciprocal_rank = float(1.0 / (first_relevant[0] + 1)) if len(first_relevant) else 0.0

    return {
        "precision_at_k": precision,
        "label_recall_at_k": recall,
        "map_at_k": average_precision,
        "ndcg_at_k": ndcg,
        "hit_at_k": hit,
        "mrr_at_k": reciprocal_rank,
    }
