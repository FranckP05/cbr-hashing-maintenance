"""
Unit tests for the E2LSH index.

Verifies:
  - Returns at most k results after reranking.
  - Distances are non-negative and sorted.
  - Determinism at same hash seed.
  - Variation at different hash seeds.
  - Recall@10 is reasonable on a simple low-dimensional dataset.
"""

import numpy as np
import pytest
from src.indexes.e2lsh import E2LSH
from src.indexes.linear_exact import LinearExact
from src.evaluation.metrics import recall_at_k
from config.seeds import HASH_SEEDS


@pytest.fixture
def dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 16)).astype(np.float32)
    return X


def test_returns_at_most_k(dataset):
    idx = E2LSH(w=4.0, K=4, L=10, n_probes=2, hash_seed=HASH_SEEDS[0])
    idx.build(dataset)
    q = dataset[0]
    indices, dists, n_cand, n_buckets = idx.query(q, k=10)
    assert len(indices) <= 10


def test_distances_non_negative(dataset):
    idx = E2LSH(w=4.0, K=4, L=10, n_probes=2, hash_seed=HASH_SEEDS[0])
    idx.build(dataset)
    q = np.random.default_rng(99).standard_normal(16).astype(np.float32)
    indices, dists, _, _ = idx.query(q, k=5)
    assert np.all(dists >= 0)


def test_distances_sorted(dataset):
    idx = E2LSH(w=4.0, K=4, L=10, n_probes=2, hash_seed=HASH_SEEDS[0])
    idx.build(dataset)
    q = dataset[5]
    _, dists, _, _ = idx.query(q, k=10)
    assert np.all(dists[:-1] <= dists[1:])


def test_determinism(dataset):
    q = dataset[10]
    idx1 = E2LSH(w=4.0, K=4, L=10, n_probes=2, hash_seed=11)
    idx1.build(dataset)
    idx2 = E2LSH(w=4.0, K=4, L=10, n_probes=2, hash_seed=11)
    idx2.build(dataset)
    i1, d1, _, _ = idx1.query(q, k=5)
    i2, d2, _, _ = idx2.query(q, k=5)
    np.testing.assert_array_equal(i1, i2)


def test_variation_across_seeds(dataset):
    q = dataset[10]
    results = []
    for seed in HASH_SEEDS:
        idx = E2LSH(w=4.0, K=4, L=10, n_probes=2, hash_seed=seed)
        idx.build(dataset)
        i, _, _, _ = idx.query(q, k=10)
        results.append(set(i.tolist()))
    # Not all seeds should produce identical candidate sets.
    assert not all(r == results[0] for r in results), "All seeds produced identical results"


def test_reasonable_recall(dataset):
    """Recall@10 should be > 0.5 on a simple Gaussian dataset with generous params."""
    linear = LinearExact()
    linear.build(dataset)
    lsh = E2LSH(w=4.0, K=3, L=20, n_probes=4, hash_seed=11)
    lsh.build(dataset)

    recalls = []
    for q in dataset[:20]:
        gt_idx, _ = linear.query(q, k=10)
        lsh_idx, _, _, _ = lsh.query(q, k=10)
        recalls.append(recall_at_k(lsh_idx, gt_idx, 10))

    assert np.mean(recalls) > 0.5, f"Recall too low: {np.mean(recalls):.2f}"
