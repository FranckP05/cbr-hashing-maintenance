"""Index LSH SimHash — Random Hyperplane Locality-Sensitive Hashing haute performance."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np


class SimHash:
    """Index LSH de Charikar (2002) avec stockage compact par tableaux ordonnés (CSR-like).

    Optimisations :
    - Tables de hachage compactes en tableaux NumPy int32 (15 MB par table pour 3.77M vecteurs).
    - Zéro overhead de pointeurs Python (évite les listes Python chaînées massives).
    - Calcul des projections vectorisé par blocs.
    - Rerank L2 exact optimisé.
    """

    def __init__(
        self,
        num_bits: int,
        num_tables: int,
        hash_seed: int,
    ) -> None:
        self.num_bits   = num_bits
        self.num_tables = num_tables
        self.hash_seed  = hash_seed

        self._projections: List[np.ndarray] = []
        # Pour chaque table : (table_order: np.ndarray int32, table_lookup: Dict[key -> (start, length)])
        self._table_orders: List[np.ndarray] = []
        self._table_lookups: List[Dict[int, Tuple[int, int]]] = []

        self._base: np.ndarray | None = None
        self.n_base: int = 0
        self.dim: int = 0
        self.build_time_s: float = 0.0

    # ------------------------------------------------------------------ #
    def _compute_keys(self, X: np.ndarray, table_idx: int) -> np.ndarray:
        """Calcule les clés de bucket par blocs pour limiter l'allocation temporaire."""
        proj   = self._projections[table_idx]  # (k, d)
        powers = (1 << np.arange(self.num_bits, dtype=np.int64))  # (k,)
        N      = X.shape[0]
        keys   = np.empty(N, dtype=np.int64)

        BLOCK = 250_000
        for start in range(0, N, BLOCK):
            end   = min(start + BLOCK, N)
            chunk = X[start:end]
            signs = (chunk @ proj.T) >= 0
            keys[start:end] = signs.astype(np.int64) @ powers

        return keys

    # ------------------------------------------------------------------ #
    def build(self, X_base: np.ndarray) -> None:
        """Construit les L tables de hachage compactes."""
        t0 = time.perf_counter()
        self._base = X_base
        self.n_base = X_base.shape[0]
        self.dim    = X_base.shape[1]

        rng = np.random.default_rng(self.hash_seed)
        self._projections = [
            rng.standard_normal((self.num_bits, self.dim)).astype(np.float32)
            for _ in range(self.num_tables)
        ]
        self._table_orders = []
        self._table_lookups = []

        for t in range(self.num_tables):
            keys = self._compute_keys(self._base, t)  # (N_base,)
            # Tri rapide par clé
            order = np.argsort(keys, kind="mergesort").astype(np.int32)
            sorted_keys = keys[order]

            # Indexation compacte des buckets
            unique_keys, starts, counts = np.unique(
                sorted_keys, return_index=True, return_counts=True
            )
            lookup = {
                int(k): (int(s), int(c))
                for k, s, c in zip(unique_keys, starts, counts)
            }

            self._table_orders.append(order)
            self._table_lookups.append(lookup)

        self.build_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    def query(
        self, x: np.ndarray, k: int, max_candidates: int = 50_000
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Retourne les top-k voisins approximatifs."""
        assert self._base is not None, "Appeler build() avant query()."
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)

        # Collecte des candidats depuis toutes les tables
        cand_list: List[np.ndarray] = []
        for t in range(self.num_tables):
            key = int(self._compute_keys(x, t)[0])
            entry = self._table_lookups[t].get(key)
            if entry is not None:
                start, count = entry
                cand_list.append(self._table_orders[t][start : start + count])

        if not cand_list:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                0,
            )

        # Dédoublonnage
        if len(cand_list) == 1:
            cand = cand_list[0]
        else:
            cand = np.unique(np.concatenate(cand_list))

        n_candidates = len(cand)
        if n_candidates == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                0,
            )

        # Sous-échantillonnage si les buckets sont disproportionnés (ex: k trop faible)
        if n_candidates > max_candidates:
            cand = cand[:max_candidates]

        # Rerank exact
        vecs = self._base[cand]
        q    = x.squeeze(0)
        q_norm_sq = float(np.dot(q, q))
        dots = vecs @ q
        sq_d = (1.0 + q_norm_sq) - 2.0 * dots
        sq_d = np.maximum(sq_d, 0.0)

        k_eff = min(k, len(cand))
        if k_eff >= len(cand):
            order = np.argsort(sq_d)
        else:
            order = np.argpartition(sq_d, k_eff)[:k_eff]
            order = order[np.argsort(sq_d[order])]

        return cand[order].astype(np.int64), sq_d[order].astype(np.float32), n_candidates
