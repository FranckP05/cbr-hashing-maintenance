"""
Unit tests for the LinearExact index.

Verifies:
  - Results are sorted by ascending distance.
  - Distances are non-negative.
  - Correct top-k returned when query is present in the base.
  - Determinism at identical inputs.
"""

import numpy as np
import pytest
from src.indexes.linear_exact import LinearExact


@pytest.fixture
def small_base():
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 8)).astype(np.float32)


def test_sorted_ascending(small_base):
    idx = LinearExact()
    idx.build(small_base)
    q = small_base[0]
    _, dists = idx.query(q, k=10)
    assert np.all(dists[:-1] <= dists[1:]), "Distances not sorted ascending"


def test_non_negative_distances(small_base):
    idx = LinearExact()
    idx.build(small_base)
    q = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    _, dists = idx.query(q, k=5)
    assert np.all(dists >= 0), "Negative squared distance found"


def test_query_in_base_is_top1(small_base):
    idx = LinearExact()
    idx.build(small_base)
    q = small_base[7]
    indices, dists = idx.query(q, k=1)
    assert indices[0] == 7, f"Query present in base should be top-1, got {indices[0]}"
    assert dists[0] < 1e-5, f"Self-distance should be ~0, got {dists[0]}"


def test_determinism(small_base):
    idx1 = LinearExact()
    idx1.build(small_base)
    idx2 = LinearExact()
    idx2.build(small_base)
    q = small_base[3]
    i1, d1 = idx1.query(q, k=5)
    i2, d2 = idx2.query(q, k=5)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_almost_equal(d1, d2)
