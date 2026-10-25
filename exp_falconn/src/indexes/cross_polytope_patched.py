"""
Cross-polytope LSH — version finale, un seul changement cible par rapport a
l'original : self._D_all est precalcule CONTIGU en memoire, sous la forme
(num_rotations, padded_d, L*K), au lieu d'etre reconstruit par une vue
transposee (.T) a chaque etape de rotation de chaque requete.

Ce fichier est deliberement une modification minimale de cross_polytope.py
(pas une reecriture) : tout le reste — rotation Hadamard, definition du
hash, buckets en dict Python, multi-probe — est identique bit pour bit.
C'est un choix assume : le profiling (voir NOTE_VECTORISATION.md) montre
que le dict de buckets n'est PAS le goulot d'etranglement (5% du temps de
requete), contrairement a l'hypothese initiale. Le vrai cout etait la
reconstruction d'une vue non contigue de self._D_all a chaque rotation
(94% du temps de requete). Le corriger ne necessite ni structure de
donnees differente pour les buckets, ni changement de langage : juste
stocker les memes nombres dans un ordre memoire different.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def _next_pow2(d: int) -> int:
    """Plus petite puissance de deux >= d."""
    return 1 << (d - 1).bit_length()


def _fht_batch(X: np.ndarray) -> np.ndarray:
    """Transformee de Walsh-Hadamard rapide, en place, colonne par colonne.

    Layout transpose (d, n) : les papillons operent le long du premier axe
    (contigu), ce qui est ~3x plus rapide en NumPy que le layout (n, d)
    car l'acces par paires reste local en cache.
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


class CrossPolytope:
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

        self._D: List[List[List[np.ndarray]]] = []
        self._D_all: Optional[np.ndarray] = None       # (L*K, num_rotations, padded_d) — inchange, utilise au build
        self._D_all_query: Optional[np.ndarray] = None  # NOUVEAU : (num_rotations, padded_d, L*K) contigu
        self._tables: List[Dict[Tuple[int, ...], List[int]]] = []
        self._base: Optional[np.ndarray] = None
        self._base_T: Optional[np.ndarray] = None
        self._padded_d: int = 0
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
            Y = _fht_batch(Y)
        norms = np.linalg.norm(Y, axis=0, keepdims=True)
        return Y / np.where(norms == 0, 1.0, norms)

    def _rotate_query_all(self, x_pad: np.ndarray) -> np.ndarray:
        """Identique a l'original, sauf que D_all_query est deja contigu
        dans l'ordre d'acces (rotation, padded_d, L*K) — plus de .T a la volee."""
        assert self._D_all_query is not None
        Y = x_pad[:, np.newaxis] * self._D_all_query[0]
        for r in range(1, self.num_rotations):
            _fht_batch(Y)
            Y = Y * self._D_all_query[r]
        _fht_batch(Y)
        norms = np.linalg.norm(Y, axis=0, keepdims=True)
        return Y / np.where(norms == 0, 1.0, norms)

    def _hash_dim(self, j: int) -> int:
        if (
            self.last_cp_dimensions is not None
            and j == self.K - 1
            and self.last_cp_dimensions < self._padded_d
        ):
            return int(self.last_cp_dimensions)
        return self._padded_d

    def _hash_batch(self, X_T: np.ndarray, t: int) -> np.ndarray:
        n = X_T.shape[1]
        keys = np.zeros((n, self.K), dtype=np.int64)
        for j in range(self.K):
            Y_T = self._rotate_batch_T(X_T, t, j)
            dim = self._hash_dim(j)
            Y = Y_T[:dim, :].T
            idx = np.argmax(np.abs(Y), axis=1)
            sign_bit = (Y[np.arange(n), idx] < 0).astype(np.int64)
            keys[:, j] = idx * 2 + sign_bit
        return keys

    def build(self, X_base: np.ndarray) -> None:
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

        self._D_all = np.stack(
            [D for table_D in self._D for D in table_D], axis=0
        )  # (L*K, num_rotations, padded_d)

        # LE FIX : precalculer la version contigue utilisee a la requete,
        # une seule fois, au lieu de retransposer a chaque appel de query().
        self._D_all_query = np.ascontiguousarray(
            self._D_all.transpose(1, 2, 0)
        )  # (num_rotations, padded_d, L*K)

        X_pad = self._pad(self._base)
        self._base_T = X_pad.T.copy()
        for t in range(self.L):
            keys = self._hash_batch(self._base_T, t)
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
        assert self._base is not None, "Call build() before query()."
        x = np.asarray(x, dtype=np.float32)
        norm = np.linalg.norm(x)
        x = x / norm if norm > 0 else x
        x_pad = self._pad(x[np.newaxis, :])[0]

        candidates: set = set()
        n_buckets_visited = 0

        Y_all = self._rotate_query_all(x_pad)

        for t in range(self.L):
            primary: List[int] = []
            ys: List[np.ndarray] = []
            for j in range(self.K):
                dim = self._hash_dim(j)
                y = Y_all[:dim, t * self.K + j]
                ys.append(y)
                idx = int(np.argmax(np.abs(y)))
                primary.append(idx * 2 + (1 if y[idx] < 0 else 0))

            probes: Dict[Tuple[int, ...], None] = {tuple(primary): None}
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

    def estimate_memory_bytes(self) -> int:
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
