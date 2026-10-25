"""SOTA LSH — Cross-Polytope LSH avec Fast Hadamard Transform et Multi-Probe (FALCONN)."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def _next_pow2(d: int) -> int:
    """Plus petite puissance de deux >= d."""
    return 1 << (d - 1).bit_length()


def _fht_batch(X: np.ndarray) -> np.ndarray:
    """Transformée rapide de Walsh-Hadamard in-place sur l'axe contigu (d, n)."""
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


class CrossPolytopeLSH:
    """Index LSH Cross-Polytope à exposant optimal rho = 1/(2c^2 - 1) avec Multi-Probe."""

    def __init__(
        self,
        num_hashes: int = 2,
        num_tables: int = 10,
        num_rotations: int = 2,
        num_probes: int = 16,
        hash_seed: int = 42,
    ) -> None:
        self.K = num_hashes
        self.L = num_tables
        self.num_rotations = num_rotations
        self.num_probes = max(1, int(num_probes))
        self.hash_seed = hash_seed

        self._D: List[List[List[np.ndarray]]] = []
        self._D_all_query: Optional[np.ndarray] = None
        # Stockage compact par table : (sorted_order: np.ndarray int32, lookup: Dict[key -> (start, count)])
        self._table_orders: List[np.ndarray] = []
        self._table_lookups: List[Dict[Tuple[int, ...], Tuple[int, int]]] = []

        self._base: Optional[np.ndarray] = None
        self._padded_d: int = 0
        self.n_base: int = 0
        self.dim: int = 0
        self.build_time_s: float = 0.0

    def _pad(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == self._padded_d:
            return X.copy()
        out = np.zeros((X.shape[0], self._padded_d), dtype=np.float32)
        out[:, : X.shape[1]] = X
        return out

    def _rotate_batch_T(self, X_T: np.ndarray, t: int, j: int) -> np.ndarray:
        Y = X_T.copy()
        for D in self._D[t][j]:
            Y = Y * D[:, np.newaxis]
            _fht_batch(Y)
        norms = np.linalg.norm(Y, axis=0, keepdims=True)
        return Y / np.where(norms == 0, 1.0, norms)

    def _rotate_query_all(self, x_pad: np.ndarray) -> np.ndarray:
        assert self._D_all_query is not None
        Y = x_pad[:, np.newaxis] * self._D_all_query[0]
        for r in range(1, self.num_rotations):
            _fht_batch(Y)
            Y = Y * self._D_all_query[r]
        _fht_batch(Y)
        norms = np.linalg.norm(Y, axis=0, keepdims=True)
        return Y / np.where(norms == 0, 1.0, norms)

    def _hash_batch(self, X_T: np.ndarray, t: int) -> np.ndarray:
        n = X_T.shape[1]
        keys = np.zeros((n, self.K), dtype=np.int32)
        for j in range(self.K):
            Y_T = self._rotate_batch_T(X_T, t, j)
            Y = Y_T.T
            idx = np.argmax(np.abs(Y), axis=1)
            sign_bit = (Y[np.arange(n), idx] < 0).astype(np.int32)
            keys[:, j] = idx * 2 + sign_bit
        return keys

    def build(self, X_base: np.ndarray) -> None:
        t0 = time.perf_counter()
        self._base = X_base
        self.n_base = X_base.shape[0]
        self.dim = X_base.shape[1]
        self._padded_d = _next_pow2(self.dim)

        rng = np.random.default_rng(self.hash_seed)
        self._D = []
        for _ in range(self.L):
            table_D = []
            for _ in range(self.K):
                rot = [
                    rng.choice([-1.0, 1.0], size=self._padded_d).astype(np.float32)
                    for _ in range(self.num_rotations)
                ]
                table_D.append(rot)
            self._D.append(table_D)

        D_all = np.stack([D for table_D in self._D for D in table_D], axis=0)
        self._D_all_query = np.ascontiguousarray(D_all.transpose(1, 2, 0))

        # Construction des tables par blocs pour économiser la mémoire vive
        self._table_orders = []
        self._table_lookups = []

        BLOCK = 250_000
        for t in range(self.L):
            all_keys = np.empty((self.n_base, self.K), dtype=np.int32)
            for start in range(0, self.n_base, BLOCK):
                end = min(start + BLOCK, self.n_base)
                chunk_pad = self._pad(self._base[start:end]).T
                all_keys[start:end] = self._hash_batch(chunk_pad, t)

            # Encodage compact des tuples de clés
            # Pack des K composantes en hash
            powers = (1 << (16 * np.arange(self.K, dtype=np.int64)))
            int_keys = all_keys.astype(np.int64) @ powers

            order = np.argsort(int_keys, kind="mergesort").astype(np.int32)
            sorted_keys = int_keys[order]

            unique_k, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
            lookup = {
                int(k): (int(s), int(c))
                for k, s, c in zip(unique_k, starts, counts)
            }

            self._table_orders.append(order)
            self._table_lookups.append(lookup)

        self.build_time_s = time.perf_counter() - t0

    def query(
        self, x: np.ndarray, k: int, max_candidates: int = 50_000
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        assert self._base is not None, "Appeler build() avant query()."
        x = np.asarray(x, dtype=np.float32)
        x_pad = self._pad(x[np.newaxis, :])[0]

        Y_all = self._rotate_query_all(x_pad)
        cand_list: List[np.ndarray] = []

        for t in range(self.L):
            primary: List[int] = []
            ys: List[np.ndarray] = []
            for j in range(self.K):
                y = Y_all[:, t * self.K + j]
                ys.append(y)
                idx = int(np.argmax(np.abs(y)))
                primary.append(idx * 2 + (1 if y[idx] < 0 else 0))

            # Multi-probe generation
            probes: List[Tuple[int, ...]] = [tuple(primary)]
            if self.num_probes > 1:
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
                    cand_k = list(primary)
                    cand_k[j] = i * 2 + (1 if ys[j][i] < 0 else 0)
                    probes.append(tuple(cand_k))
                    if len(probes) >= self.num_probes:
                        break

            powers = (1 << (16 * np.arange(self.K, dtype=np.int64)))
            for pr in probes:
                int_key = int(np.array(pr, dtype=np.int64) @ powers)
                entry = self._table_lookups[t].get(int_key)
                if entry is not None:
                    start, count = entry
                    cand_list.append(self._table_orders[t][start : start + count])

        if not cand_list:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

        cand = np.unique(np.concatenate(cand_list))
        n_candidates = len(cand)
        if n_candidates == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

        if n_candidates > max_candidates:
            cand = cand[:max_candidates]

        # Rerank exact
        vecs = self._base[cand]
        q_norm_sq = float(np.dot(x, x))
        dots = vecs @ x
        sq_d = (1.0 + q_norm_sq) - 2.0 * dots
        sq_d = np.maximum(sq_d, 0.0)

        k_eff = min(k, len(cand))
        if k_eff >= len(cand):
            order = np.argsort(sq_d)
        else:
            order = np.argpartition(sq_d, k_eff)[:k_eff]
            order = order[np.argsort(sq_d[order])]

        return cand[order].astype(np.int64), sq_d[order].astype(np.float32), n_candidates
