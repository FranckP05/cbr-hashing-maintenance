"""
Unit tests for the CrossPolytope (FALCONN-style) index.

Verifies:
  - Rotation preserves the norm of normalised vectors.
  - Hash determinism at the same hash seed.
  - Variation across different hash seeds.
  - Returns at most k results after reranking.
  - Distances are non-negative and sorted ascending.
  - A base point queried against itself is top-1.
  - Recall@10 is high on a well-separated Gaussian dataset on the sphere.
  - Non-power-of-two dimensions are padded without breaking the index.
"""

import numpy as np
import pytest
from src.indexes.cross_polytope import CrossPolytope, _fht_batch, _next_pow2
from src.indexes.linear_exact import LinearExact
from src.evaluation.metrics import recall_at_k
from config.seeds import HASH_SEEDS


@pytest.fixture
def unit_dataset():
    """Random Gaussian points, L2-normalised (cosine arm precondition)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# --------------------------------------------------------------------------- #
# FHT primitive
# --------------------------------------------------------------------------- #

def test_fht_is_orthogonal():
    """H/sqrt(d) should be an orthogonal transform (H^T H = d I)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 1)).astype(np.float32)  # (d, n) layout
    X = X / np.linalg.norm(X)
    Y = _fht_batch(X.copy())
    # Norm is preserved up to the sqrt(d) scaling of the unnormalised H.
    np.testing.assert_allclose(np.linalg.norm(Y), np.sqrt(8) * np.linalg.norm(X), rtol=1e-4)


def test_next_pow2():
    assert _next_pow2(8) == 8
    assert _next_pow2(13) == 16
    assert _next_pow2(1024) == 1024
    assert _next_pow2(1) == 1


# --------------------------------------------------------------------------- #
# Index behaviour
# --------------------------------------------------------------------------- #

def test_rotation_preserves_norm(unit_dataset):
    idx = CrossPolytope(num_hashes=2, num_tables=1, num_rotations=3, hash_seed=11)
    idx.build(unit_dataset)
    Y = idx._rotate_batch_T(idx._base_T, 0, 0)
    norms = np.linalg.norm(Y, axis=0)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


def test_returns_at_most_k(unit_dataset):
    idx = CrossPolytope(num_hashes=2, num_tables=5, num_rotations=3, num_probes=4, hash_seed=HASH_SEEDS[0])
    idx.build(unit_dataset)
    q = unit_dataset[0]
    indices, _, n_cand, n_buckets = idx.query(q, k=10)
    assert len(indices) <= 10
    assert n_cand >= 0
    assert n_buckets >= 1


def test_distances_non_negative(unit_dataset):
    idx = CrossPolytope(num_hashes=2, num_tables=5, num_rotations=3, hash_seed=HASH_SEEDS[0])
    idx.build(unit_dataset)
    q = np.random.default_rng(99).standard_normal(16).astype(np.float32)
    q /= np.linalg.norm(q)
    indices, dists, _, _ = idx.query(q, k=5)
    assert np.all(dists >= 0)


def test_distances_sorted(unit_dataset):
    idx = CrossPolytope(num_hashes=2, num_tables=5, num_rotations=3, hash_seed=HASH_SEEDS[0])
    idx.build(unit_dataset)
    q = unit_dataset[5]
    _, dists, _, _ = idx.query(q, k=10)
    assert np.all(dists[:-1] <= dists[1:])


def test_determinism(unit_dataset):
    q = unit_dataset[10]
    idx1 = CrossPolytope(num_hashes=2, num_tables=5, num_rotations=3, hash_seed=11)
    idx1.build(unit_dataset)
    idx2 = CrossPolytope(num_hashes=2, num_tables=5, num_rotations=3, hash_seed=11)
    idx2.build(unit_dataset)
    i1, d1, _, _ = idx1.query(q, k=5)
    i2, d2, _, _ = idx2.query(q, k=5)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(d1, d2, rtol=1e-6)


def test_variation_across_seeds(unit_dataset):
    q = unit_dataset[10]
    results = []
    for seed in HASH_SEEDS:
        idx = CrossPolytope(num_hashes=2, num_tables=5, num_rotations=3, hash_seed=seed)
        idx.build(unit_dataset)
        i, _, _, _ = idx.query(q, k=10)
        results.append(set(i.tolist()))
    assert not all(r == results[0] for r in results), "All seeds produced identical results"


def test_self_query_is_top1(unit_dataset):
    idx = CrossPolytope(num_hashes=3, num_tables=10, num_rotations=3, num_probes=8, hash_seed=11)
    idx.build(unit_dataset)
    i, dists, _, _ = idx.query(unit_dataset[7], k=1)
    assert i[0] == 7, f"Self query should be top-1, got {i[0]}"
    assert dists[0] < 1e-5


def test_reasonable_recall(unit_dataset):
    """With enough tables and probes, recall@10 on the sphere should be high."""
    linear = LinearExact()
    linear.build(unit_dataset)
    cp = CrossPolytope(num_hashes=2, num_tables=20, num_rotations=3, num_probes=8, hash_seed=11)
    cp.build(unit_dataset)

    recalls = []
    for q in unit_dataset[:20]:
        gt_idx, _ = linear.query(q, k=10)
        cp_idx, _, _, _ = cp.query(q, k=10)
        recalls.append(recall_at_k(cp_idx, gt_idx, 10))

    assert np.mean(recalls) > 0.5, f"Recall too low: {np.mean(recalls):.2f}"


def test_non_power_of_two_dimension():
    """d=13 (heart) is padded to 16 and still returns valid results."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 13)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    linear = LinearExact()
    linear.build(X)
    cp = CrossPolytope(num_hashes=2, num_tables=10, num_rotations=3, num_probes=4, hash_seed=11)
    cp.build(X)
    assert cp._padded_d == 16

    q = X[3]
    i, dists, _, _ = cp.query(q, k=5)
    assert len(i) <= 5
    assert np.all(dists >= 0)
