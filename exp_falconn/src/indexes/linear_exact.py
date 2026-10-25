"""
Optimised linear exact L2 nearest-neighbour search (section 4.1).

Design constraints from protocol:
  - BLAS-vectorised computation (no Python loop over candidates).
  - Base norms pre-computed at build time.
  - Efficient top-k selection (np.argpartition).
  - No artificial batch amortisation.
  - Results are always sorted by ascending L2 distance.
  - Ground truth computed once per (dataset, split, base_size) and reused.
"""

from __future__ import annotations

import numpy as np
import time
from typing import Tuple


class LinearExact:
    """Exhaustive L2 nearest-neighbour index."""

    def __init__(self) -> None:
        self._base: np.ndarray | None = None
        self._base_norms_sq: np.ndarray | None = None
        self.build_time_s: float = 0.0

    # ---------------------------------------------------------------------- #
    def build(self, X_base: np.ndarray) -> None:
        """Pre-compute squared norms of base vectors.

        Args:
            X_base: float32 array of shape (n_base, d).
        """
        t0 = time.perf_counter()
        self._base = np.asarray(X_base, dtype=np.float32)
        # ||x_i||^2 for each base vector.
        self._base_norms_sq = np.einsum("ij,ij->i", self._base, self._base)
        self.build_time_s = time.perf_counter() - t0

    # ---------------------------------------------------------------------- #
    def query(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k nearest neighbours by squared L2 distance.

        Uses the identity:
          ||x - x_i||^2 = ||x||^2 - 2 x.x_i + ||x_i||^2

        The ||x||^2 term is constant and does not affect ranking.

        Args:
            x: Query vector of shape (d,).
            k: Number of neighbours to return.

        Returns:
            (indices, sq_distances) arrays of shape (k,), sorted ascending.
        """
        assert self._base is not None, "Call build() before query()."
        x = np.asarray(x, dtype=np.float32)
        # Squared L2: ||x||^2 - 2*X_base @ x + ||x_base||^2
        dots = self._base @ x                           # (n_base,)
        sq_dists = self._base_norms_sq - 2.0 * dots + float(np.dot(x, x))
        sq_dists = np.maximum(sq_dists, 0.0)            # numerical safety

        if k >= len(sq_dists):
            idx = np.argsort(sq_dists)
        else:
            idx = np.argpartition(sq_dists, k)[:k]
            idx = idx[np.argsort(sq_dists[idx])]

        return idx, sq_dists[idx]
