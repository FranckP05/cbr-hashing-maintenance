"""Tests pour CrossPolytopeLSH."""

from __future__ import annotations

import numpy as np
import pytest

from indexes.cross_polytope import CrossPolytopeLSH


def random_l2(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def test_cross_polytope_determinism():
    N, d, k = 500, 32, 5
    X_base = random_l2(N, d, seed=10)
    q = random_l2(1, d, seed=11).squeeze()

    index1 = CrossPolytopeLSH(num_hashes=2, num_tables=4, num_probes=4, hash_seed=42)
    index2 = CrossPolytopeLSH(num_hashes=2, num_tables=4, num_probes=4, hash_seed=42)

    index1.build(X_base)
    index2.build(X_base)

    idx1, _, _ = index1.query(q, k)
    idx2, _, _ = index2.query(q, k)

    assert set(idx1.tolist()) == set(idx2.tolist())


def test_cross_polytope_near_neighbor_recovery():
    d = 32
    q = random_l2(1, d, seed=1).squeeze()
    noise = np.random.default_rng(2).standard_normal(d).astype(np.float32) * 1e-3
    q_near = (q + noise) / np.linalg.norm(q + noise)

    X_base = np.vstack([q_near.reshape(1, -1), random_l2(100, d, seed=3)])
    index = CrossPolytopeLSH(num_hashes=2, num_tables=10, num_probes=8, hash_seed=7)
    index.build(X_base)

    idx, _, n_cand = index.query(q, k=5)
    assert 0 in idx.tolist()
