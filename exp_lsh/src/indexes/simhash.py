"""
SimHash / random-hyperplane LSH (section 4.2).

This implementation is kept as a *diagnostic ablation* only.
It approximates angular / cosine similarity, not L2 distance.
Do NOT treat SimHash results as a valid L2 approximation.

Hash function:
    h_j(x) = sign(a_j . x)    where a_j ~ N(0, I)

Calibrated parameters (section 8):
    num_bits   : number of hash bits per table
    num_tables : number of independent hash tables
"""

from __future__ import annotations

import numpy as np
import time
from typing import Dict, List, Tuple


class SimHash:
    """Random-hyperplane LSH for angular similarity."""

    def __init__(
        self,
        num_bits: int,
        num_tables: int,
        hash_seed: int,
    ) -> None:
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.hash_seed = hash_seed
        self._projections: List[np.ndarray] = []
        self._tables: List[Dict[int, List[int]]] = []
        self._base: np.ndarray | None = None
        self.build_time_s: float = 0.0

    # ---------------------------------------------------------------------- #
    def _hash(self, x: np.ndarray, table_idx: int) -> int:
        """Compute integer bucket key for a single vector."""
        signs = (self._projections[table_idx] @ x) >= 0  # (num_bits,)
        # Pack bits into an integer.
        key = 0
        for bit in signs:
            key = (key << 1) | int(bit)
        return key

    # ---------------------------------------------------------------------- #
    def build(self, X_base: np.ndarray) -> None:
        """Build hash tables from base vectors."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.hash_seed)
        d = X_base.shape[1]
        self._base = np.asarray(X_base, dtype=np.float32)
        self._projections = [
            rng.standard_normal((self.num_bits, d)).astype(np.float32)
            for _ in range(self.num_tables)
        ]
        self._tables = [{} for _ in range(self.num_tables)]
        for i, vec in enumerate(self._base):
            for t in range(self.num_tables):
                key = self._hash(vec, t)
                self._tables[t].setdefault(key, []).append(i)
        self.build_time_s = time.perf_counter() - t0

    # ---------------------------------------------------------------------- #
    def query(
        self, x: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return top-k neighbours after exact L2 reranking on candidates.

        Returns:
            (indices, sq_distances, n_candidates)
        """
        assert self._base is not None, "Call build() before query()."
        x = np.asarray(x, dtype=np.float32)
        candidates: set = set()
        for t in range(self.num_tables):
            key = self._hash(x, t)
            candidates.update(self._tables[t].get(key, []))

        if not candidates:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

        cand = np.array(sorted(candidates), dtype=np.int64)
        vecs = self._base[cand]
        diff = vecs - x
        sq_dists = np.einsum("ij,ij->i", diff, diff)

        if k >= len(cand):
            order = np.argsort(sq_dists)
        else:
            order = np.argpartition(sq_dists, k)[:k]
            order = order[np.argsort(sq_dists[order])]

        return cand[order], sq_dists[order], len(candidates)
