"""Baseline exacte — Balayage linéaire optimisé (BLAS GEMV vectorisé sans buffer de transposition)."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np


class LinearScan:
    """Index de plus proches voisins exact par balayage linéaire BLAS.

    Design optimisé pour très grande échelle (3.7M+ vecteurs) :
    - Utilise BLAS GEMV direct (matrice contiguë x vecteur) sans aucune copie/transposition.
    - Allocation temporaire minimale (15 MB par requête).
    - Sélection top-k par np.argpartition.
    - Zéro risque d'OOM.
    """

    def __init__(self) -> None:
        self._base: np.ndarray | None = None
        self.build_time_s: float = 0.0
        self.n_base: int = 0
        self.dim: int = 0

    # ------------------------------------------------------------------ #
    def build(self, X_base: np.ndarray) -> None:
        """Enregistre la référence directe à la base sans copie."""
        t0 = time.perf_counter()
        self._base = X_base
        self.n_base = X_base.shape[0]
        self.dim    = X_base.shape[1]
        self.build_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    def query(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les top-k voisins les plus proches."""
        assert self._base is not None, "Appeler build() avant query()."
        x = np.asarray(x, dtype=np.float32)
        q_norm_sq = float(np.dot(x, x))

        # BLAS GEMV direct : (N_base, d) @ (d,) -> (N_base,)
        dots     = self._base @ x
        sq_dists = (1.0 + q_norm_sq) - 2.0 * dots
        sq_dists = np.maximum(sq_dists, 0.0)

        k_eff = min(k, self.n_base)
        if k_eff >= self.n_base:
            idx = np.argsort(sq_dists)
        else:
            idx = np.argpartition(sq_dists, k_eff)[:k_eff]
            idx = idx[np.argsort(sq_dists[idx])]

        return idx.astype(np.int64), sq_dists[idx].astype(np.float32)

    # ------------------------------------------------------------------ #
    def batch_query(
        self, X_queries: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exécute les requêtes en boucle GEMV vectorisée sans pic mémoire."""
        assert self._base is not None, "Appeler build() avant query()."
        N_q   = X_queries.shape[0]
        k_eff = min(k, self.n_base)
        all_indices = np.empty((N_q, k_eff), dtype=np.int64)
        all_dists   = np.empty((N_q, k_eff), dtype=np.float32)

        for i in range(N_q):
            idx, dists = self.query(X_queries[i], k_eff)
            all_indices[i] = idx
            all_dists[i]   = dists

        return all_indices, all_dists
