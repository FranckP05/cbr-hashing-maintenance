"""
E2LSH p-stable locality-sensitive hashing for L2 distance (section 4.3).

Reference:
    Datar, M., Indyk, P., Immorlica, N., Mirrokni, V. S. (2004).
    Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
    Proc. 20th Annual Symposium on Computational Geometry.

Hash function:
    h(x) = floor((a . x + b) / w)

    where:
        a ~ N(0, I_d)   (Gaussian, p-stable for L2)
        b ~ Uniform[0, w]
        w  width of the bucket (calibrated on validation)

Search pipeline:
    1. Hash the query.
    2. Generate multi-probe perturbation list (optional).
    3. Retrieve candidate sets from all tables.
    4. Union of candidates.
    5. Exact L2 reranking on candidates.
    6. Return top-k neighbours.
"""

from __future__ import annotations

import numpy as np
import time
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Multi-probe perturbation helper
# --------------------------------------------------------------------------- #

def _perturbation_keys(base_keys: np.ndarray, n_probes: int) -> List[Tuple]:
    """Generate neighbouring bucket keys for multi-probe LSH.

    Simple ±1 perturbations on each hash component, truncated to n_probes.

    Args:
        base_keys: Integer array of shape (K,) — the query bucket indices.
        n_probes:  Maximum number of additional buckets to visit.

    Returns:
        List of tuples, each representing an adjacent bucket key.
    """
    K = len(base_keys)
    perturbed: List[Tuple] = []
    for i in range(K):
        for delta in (-1, +1):
            candidate = base_keys.copy()
            candidate[i] += delta
            perturbed.append(tuple(candidate.tolist()))
            if len(perturbed) >= n_probes:
                return perturbed
    return perturbed


# --------------------------------------------------------------------------- #
# E2LSH index
# --------------------------------------------------------------------------- #

class E2LSH:
    """E2LSH p-stable index for L2 nearest-neighbour search.

    Parameters
    ----------
    w : float
        Bucket width (key hyper-parameter, calibrated on validation).
    K : int
        Number of hash functions concatenated per table.
    L : int
        Number of independent hash tables.
    n_probes : int
        Number of additional multi-probe buckets visited per table.
        Set to 0 to disable multi-probe.
    hash_seed : int
        Seed for random projections.
    """

    def __init__(
        self,
        w: float,
        K: int,
        L: int,
        n_probes: int = 0,
        hash_seed: int = 42,
    ) -> None:
        self.w = w
        self.K = K
        self.L = L
        self.n_probes = n_probes
        self.hash_seed = hash_seed

        self._a: List[np.ndarray] = []          # projections per table: L x (K, d)
        self._b: List[np.ndarray] = []          # offsets per table: L x (K,)
        self._tables: List[Dict[Tuple, List[int]]] = []
        self._base: Optional[np.ndarray] = None
        self.build_time_s: float = 0.0

    # ---------------------------------------------------------------------- #
    def _hash_vector(self, x: np.ndarray, table_idx: int) -> Tuple[int, ...]:
        """Return the K-dimensional integer bucket key for vector x."""
        raw = (self._a[table_idx] @ x + self._b[table_idx]) / self.w
        return tuple(np.floor(raw).astype(np.int64).tolist())

    # ---------------------------------------------------------------------- #
    def build(self, X_base: np.ndarray) -> None:
        """Build E2LSH tables.

        Args:
            X_base: float32 array of shape (n_base, d).
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.hash_seed)
        d = X_base.shape[1]
        self._base = np.asarray(X_base, dtype=np.float32)
        self._a = []
        self._b = []
        self._tables = []

        for _ in range(self.L):
            a = rng.standard_normal((self.K, d)).astype(np.float32)
            b = rng.uniform(0.0, self.w, size=self.K).astype(np.float32)
            self._a.append(a)
            self._b.append(b)
            self._tables.append({})

        for i, vec in enumerate(self._base):
            for t in range(self.L):
                key = self._hash_vector(vec, t)
                self._tables[t].setdefault(key, []).append(i)

        self.build_time_s = time.perf_counter() - t0

    # ---------------------------------------------------------------------- #
    def query(
        self, x: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Retrieve top-k nearest neighbours.

        Steps:
            1. Hash query.
            2. Enumerate multi-probe buckets.
            3. Union candidates across tables.
            4. Exact L2 reranking.
            5. Return top-k sorted ascending.

        Returns:
            (indices, sq_distances, n_candidates, n_buckets_visited)
        """
        assert self._base is not None, "Call build() before query()."
        x = np.asarray(x, dtype=np.float32)
        candidates: set = set()
        n_buckets_visited = 0

        for t in range(self.L):
            base_key_arr = np.array(
                self._hash_vector(x, t), dtype=np.int64
            )
            # Primary bucket.
            primary = tuple(base_key_arr.tolist())
            hits = self._tables[t].get(primary, [])
            candidates.update(hits)
            n_buckets_visited += 1

            # Multi-probe buckets.
            if self.n_probes > 0:
                for neighbour_key in _perturbation_keys(base_key_arr, self.n_probes):
                    hits = self._tables[t].get(neighbour_key, [])
                    candidates.update(hits)
                    n_buckets_visited += 1

        if not candidates:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                0,
                n_buckets_visited,
            )

        cand = np.array(sorted(candidates), dtype=np.int64)
        vecs = self._base[cand]
        diff = vecs - x
        sq_dists = np.einsum("ij,ij->i", diff, diff)

        if k >= len(cand):
            order = np.argsort(sq_dists)
        else:
            order = np.argpartition(sq_dists, k)[:k]
            order = order[np.argsort(sq_dists[order])]

        return cand[order], sq_dists[order], len(candidates), n_buckets_visited
