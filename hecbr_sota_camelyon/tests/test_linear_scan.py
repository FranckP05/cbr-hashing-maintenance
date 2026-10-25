"""Tests pour LinearScan."""

from __future__ import annotations

import numpy as np
import pytest

from indexes.linear_scan import LinearScan


def random_l2(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def test_linear_scan_matches_naive():
    N, d, k = 200, 32, 10
    X_base = random_l2(N, d, seed=1)
    q = random_l2(1, d, seed=2).squeeze()

    index = LinearScan()
    index.build(X_base)
    idx_fast, dists_fast = index.query(q, k)

    # Calcul naïf
    naive_dists = np.sum((X_base - q) ** 2, axis=1)
    naive_order = np.argsort(naive_dists)[:k]

    assert set(idx_fast.tolist()) == set(naive_order.tolist())
    assert np.all(dists_fast >= 0.0)
