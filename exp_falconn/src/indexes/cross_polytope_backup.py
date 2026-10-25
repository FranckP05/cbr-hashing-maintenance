"""
Cross-polytope LSH (FALCONN-style) for angular / cosine similarity (section 5).

Reference:
    Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, R., Schmidt, L. (2015).
    "Practical and Optimal LSH for Angular Distance". NeurIPS 2015.
    Reference implementation: https://github.com/FALCONN-LIB/FALCONN

    Hash function (cross-polytope):
        1. Rotate x by a *pseudo-random rotation*:
               x -> H D_3 H D_2 H D_1 x
           where H is the Hadamard transform (via the Fast Hadamard Transform)
           and D_i are random diagonal +-1 matrices. The default is 2 HD
           stages (FALCONN's ``get_default_parameters``), 3 for stronger
           randomness.
    2. h(x) = argmax_j |(Ax)_j|  (index of the closest vertex of the
       cross-polytope), returned with the sign of that coordinate.

    The hash value is encoded as an integer in [0, 2d):
        value = 2 * index + sign_bit     (sign_bit = 1 if coordinate < 0)

    K hash functions are concatenated per table; L independent tables are
    built. Multi-probe enumerates the most promising alternative buckets
    ordered by decreasing |coordinate| magnitude (FALCONN multi-probe).

Preconditions:
    - Input vectors are L2-normalised (cosine arm, protocol section 5).
    - Ranking is done by exact squared L2 on the unit sphere, which equals
      cosine ranking: ||p - q||^2 = 2 - 2*cos(p, q).

Dimension handling:
    The FHT requires d to be a power of two. Non-power-of-two dimensions
    are padded with zeros to the next power of two; zero-padding preserves
    all inner products, hence all angles.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Fast Walsh-Hadamard transform (vectorised over a batch of rows)
# --------------------------------------------------------------------------- #

def _next_pow2(d: int) -> int:
    """Smallest power of two >= d."""
    return 1 << (d - 1).bit_length()


def _fht_batch(X: np.ndarray) -> np.ndarray:
    """In-place Fast Walsh-Hadamard transform of each column.

    The data is stored transposed, shape (d, n), so the butterflies run
    along the first (contiguous) axis: this is ~3x faster in NumPy than
    the (n, d) layout because the pair access is cache-local.

    Args:
        X: float array of shape (d, n) with d a power of two.

    Returns:
        The transformed array (the input is modified in place).
    """
    d, n = X.shape
    h = 1
    while h < d:
        view = X.reshape(d // (2 * h), 2 * h, n)
        a, b = view[:, :h], view[:, h:]
        t = a.copy()
        np.add(a, b, out=a)
        np.subtract(t, b, out=b)
        h *= 2
    return X


# --------------------------------------------------------------------------- #
# Cross-polytope LSH index
# --------------------------------------------------------------------------- #

class CrossPolytope:
    """Cross-polytope LSH index for cosine / angular similarity.

    Parameters
    ----------
    num_hashes : int
        Number of hash functions concatenated per table (K).
    num_tables : int
        Number of independent hash tables (L).
    num_rotations : int
        Number of HD pseudo-random rotation applications. FALCONN's
        ``get_default_parameters`` uses 2 for dense data (1 when the data
        is "sufficiently dense"); the paper recommends 3 for stronger
        randomness. Default 2 to match FALCONN's shipped heuristic.
    num_probes : int
        Total number of buckets probed per table, including the primary
        bucket. 1 means primary bucket only (no multi-probe).
    last_cp_dimensions : Optional[int]
        If set, the last hash function only considers the first
        `last_cp_dimensions` coordinates ("partial cross-polytope").
    hash_seed : int
        Seed for the random diagonal matrices (projections).
    """

    def __init__(
        self,
        num_hashes: int,
        num_tables: int,
        num_rotations: int = 2,
        num_probes: int = 1,
        last_cp_dimensions: Optional[int] = None,
        hash_seed: int = 42,
    ) -> None:
        self.K = num_hashes
        self.L = num_tables
        self.num_rotations = num_rotations
        self.num_probes = max(1, int(num_probes))
        self.last_cp_dimensions = last_cp_dimensions
        self.hash_seed = hash_seed

        self._D: List[List[List[np.ndarray]]] = []   # L x K x num_rotations
        self._D_all: Optional[np.ndarray] = None     # (L*K, num_rotations, padded_d)
        self._tables: List[Dict[Tuple[int, ...], List[int]]] = []
        self._base: Optional[np.ndarray] = None
        self._base_T: Optional[np.ndarray] = None  # (padded_d, n), unit norm
        self._padded_d: int = 0
        self.build_time_s: float = 0.0

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """Zero-pad the last axis to self._padded_d."""
        if X.shape[1] == self._padded_d:
            return X.copy()
        out = np.zeros((X.shape[0], self._padded_d), dtype=np.float32)
        out[:, : X.shape[1]] = X
        return out

    def _rotate_batch_T(self, X_T: np.ndarray, t: int, j: int) -> np.ndarray:
        """Apply pseudo-random rotation A = H D_3 H D_2 H D_1 to each column.

        Args:
            X_T: padded, transposed array of shape (padded_d, n).
            t: table index.
            j: hash-function index within the table.

        Returns:
            Rotated, unit-norm columns of shape (padded_d, n).
        """
        Y = X_T.copy()
        for D in self._D[t][j]:
            Y = Y * D[:, np.newaxis]
            Y = _fht_batch(Y)
        # Normalise so the rotated vectors lie on the unit sphere
        # (H unscaled grows the norm by d^(num_rotations/2)).
        norms = np.linalg.norm(Y, axis=0, keepdims=True)
        return Y / np.where(norms == 0, 1.0, norms)

    def _rotate_query_all(self, x_pad: np.ndarray) -> np.ndarray:
        """Rotate one padded query under every (table, hash) rotation at once.

        All L*K rotations share the same pseudo-random structure, so the
        FHT butterfly can run over a stacked (padded_d, L*K) buffer: the
        per-column random signs make each column an independent rotation.
        This is much faster than L*K separate single-vector rotations.

        Args:
            x_pad: padded query of shape (padded_d,).

        Returns:
            Rotated, unit-norm columns of shape (padded_d, L*K); column
            (t*K + j) is the rotation used by table t, hash fn j.
        """
        assert self._D_all is not None
        Y = x_pad[:, np.newaxis] * self._D_all[:, 0, :].T  # (padded_d, L*K)
        for r in range(1, self.num_rotations):
            _fht_batch(Y)
            Y = Y * self._D_all[:, r, :].T
        _fht_batch(Y)
        norms = np.linalg.norm(Y, axis=0, keepdims=True)
        return Y / np.where(norms == 0, 1.0, norms)

    def _hash_batch(self, X_T: np.ndarray, t: int) -> np.ndarray:
        """Hash a batch of padded, transposed vectors for table t.

        Args:
            X_T: padded, transposed array of shape (padded_d, n).

        Returns:
            Integer array of shape (n, K), one encoded hash value per
            hash function.
        """
        n = X_T.shape[1]
        keys = np.zeros((n, self.K), dtype=np.int64)
        for j in range(self.K):
            Y_T = self._rotate_batch_T(X_T, t, j)
            dim = self._hash_dim(j)
            Y = Y_T[:dim, :].T  # (n, dim), contiguous
            idx = np.argmax(np.abs(Y), axis=1)
            sign_bit = (Y[np.arange(n), idx] < 0).astype(np.int64)
            keys[:, j] = idx * 2 + sign_bit
        return keys

    def _hash_dim(self, j: int) -> int:
        """Effective cross-polytope dimension for hash function j."""
        if (
            self.last_cp_dimensions is not None
            and j == self.K - 1
            and self.last_cp_dimensions < self._padded_d
        ):
            return int(self.last_cp_dimensions)
        return self._padded_d

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def build(self, X_base: np.ndarray) -> None:
        """Build the hash tables from the base set.

        The base is L2-normalised (cosine arm). A deterministic set of
        random diagonal +-1 matrices is drawn for every (table, hash fn,
        rotation) triple.

        Args:
            X_base: float32 array of shape (n_base, d).
        """
        t0 = time.perf_counter()
        self._base = np.asarray(X_base, dtype=np.float32)
        norms = np.linalg.norm(self._base, axis=1, keepdims=True)
        self._base = self._base / np.where(norms == 0, 1.0, norms)

        d = self._base.shape[1]
        self._padded_d = _next_pow2(d)

        rng = np.random.default_rng(self.hash_seed)
        self._D = []
        self._tables = []
        for _ in range(self.L):
            table_D = []
            for _ in range(self.K):
                rot = [
                    rng.choice([-1.0, 1.0], size=self._padded_d).astype(np.float32)
                    for _ in range(self.num_rotations)
                ]
                table_D.append(rot)
            self._D.append(table_D)
            self._tables.append({})
        # Stacked form for batched query rotations: (L*K, num_rotations, d).
        self._D_all = np.stack(
            [D for table_D in self._D for D in table_D], axis=0
        )

        X_pad = self._pad(self._base)
        self._base_T = X_pad.T.copy()  # (padded_d, n), contiguous
        for t in range(self.L):
            keys = self._hash_batch(self._base_T, t)  # (n, K)
            tbl = self._tables[t]
            for i in range(len(keys)):
                key = tuple(int(v) for v in keys[i])
                lst = tbl.get(key)
                if lst is None:
                    tbl[key] = [i]
                else:
                    lst.append(i)

        self.build_time_s = time.perf_counter() - t0

    def query(
        self, x: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Retrieve top-k neighbours by exact L2 on the unit sphere.

        Steps:
            1. Normalise and hash the query.
            2. Enumerate candidate buckets via multi-probe, ordered by
               decreasing |coordinate| magnitude.
            3. Union candidates across all tables.
            4. Exact L2 reranking on the normalised base.
            5. Return top-k sorted ascending.

        Returns:
            (indices, sq_distances, n_candidates, n_buckets_visited)
        """
        assert self._base is not None, "Call build() before query()."
        x = np.asarray(x, dtype=np.float32)
        norm = np.linalg.norm(x)
        x = x / norm if norm > 0 else x
        x_pad = self._pad(x[np.newaxis, :])[0]

        candidates: set = set()
        n_buckets_visited = 0

        # Rotate the query under every (table, hash) rotation at once.
        Y_all = self._rotate_query_all(x_pad)  # (padded_d, L*K)

        for t in range(self.L):
            primary: List[int] = []
            ys: List[np.ndarray] = []  # per-hash (dim,) rotated vectors
            for j in range(self.K):
                dim = self._hash_dim(j)
                y = Y_all[:dim, t * self.K + j]
                ys.append(y)
                idx = int(np.argmax(np.abs(y)))
                primary.append(idx * 2 + (1 if y[idx] < 0 else 0))

            # Multi-probe: order alternative buckets by |coordinate|.
            probes: Dict[Tuple[int, ...], None] = {tuple(primary): None}
            if self.num_probes > 1:
                # Concatenate |coordinate| scores of every hash function,
                # masking out the primary (max) coordinate of each.
                parts, offsets = [], []
                acc = 0
                for j in range(self.K):
                    mag = np.abs(ys[j]).copy()
                    mag[int(np.argmax(mag))] = -1.0
                    parts.append(mag)
                    offsets.append(acc)
                    acc += len(mag)
                all_scores = np.concatenate(parts)
                for pos in np.argsort(-all_scores):
                    if all_scores[pos] < 0:
                        break
                    j = 0
                    while j + 1 < self.K and pos >= offsets[j + 1]:
                        j += 1
                    i = pos - offsets[j]
                    cand = list(primary)
                    cand[j] = i * 2 + (1 if ys[j][i] < 0 else 0)
                    probes.setdefault(tuple(cand), None)
                    if len(probes) >= self.num_probes:
                        break

            for key in probes:
                hits = self._tables[t].get(key)
                n_buckets_visited += 1
                if hits:
                    candidates.update(hits)

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

    # ---------------------------------------------------------------------- #
    def estimate_memory_bytes(self) -> int:
        """Rough index memory estimate (rotations + bucket entries)."""
        total = 0
        for table_D in self._D:
            for rot in table_D:
                for D in rot:
                    total += D.nbytes
        if self._base is not None:
            total += self._base.nbytes
        for table in self._tables:
            for bucket in table.values():
                total += len(bucket) * 8
        return total
