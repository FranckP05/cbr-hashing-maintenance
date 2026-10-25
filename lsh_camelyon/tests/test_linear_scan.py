"""Tests de conformité pour la Baseline linéaire optimisée (LinearScan)."""

from __future__ import annotations

import numpy as np
import pytest


def random_l2(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


class TestLinearScan:
    """Tests de la classe LinearScan."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from indexes.linear_scan import LinearScan
        self.LinearScan = LinearScan
        self.N, self.d = 500, 32
        self.X_base = random_l2(self.N, self.d, seed=1)
        self.index  = LinearScan()
        self.index.build(self.X_base)

    # --- Test d'exactitude (conformité numérique) ---------------------------

    def test_exact_result_matches_naive(self):
        """Les résultats doivent correspondre à un calcul naïf de distance L2."""
        rng = np.random.default_rng(99)
        x = random_l2(1, self.d, seed=99).squeeze()
        k = 10
        idx_fast, dists_fast = self.index.query(x, k)

        # Calcul naïf (boucle référence)
        naive_dists = np.array([
            float(np.sum((self.X_base[i] - x) ** 2))
            for i in range(self.N)
        ])
        naive_order = np.argsort(naive_dists)[:k]

        assert set(idx_fast.tolist()) == set(naive_order.tolist()), \
            "Les top-k voisins diffèrent du calcul naïf."

    def test_distances_are_sorted(self):
        """Les distances retournées doivent être triées par ordre croissant."""
        x = random_l2(1, self.d, seed=77).squeeze()
        _, dists = self.index.query(x, k=20)
        assert np.all(dists[:-1] <= dists[1:]), "Distances non triées."

    def test_distances_non_negative(self):
        """Aucune distance au carré ne doit être négative."""
        x = random_l2(1, self.d, seed=55).squeeze()
        _, dists = self.index.query(x, k=10)
        assert np.all(dists >= 0.0), "Distance négative détectée."

    def test_output_shape(self):
        """Le nombre de voisins retournés doit être min(k, N_base)."""
        x = random_l2(1, self.d).squeeze()
        idx, dists = self.index.query(x, k=15)
        assert len(idx) == 15
        assert len(dists) == 15

    def test_k_larger_than_base(self):
        """Quand k > N_base, tous les voisins sont retournés sans erreur."""
        x = random_l2(1, self.d).squeeze()
        idx, dists = self.index.query(x, k=self.N + 100)
        assert len(idx) == self.N

    def test_batch_query_matches_single(self):
        """batch_query doit retourner les mêmes résultats que des requêtes individuelles."""
        k = 5
        X_q = random_l2(10, self.d, seed=7)
        batch_idx, batch_dists = self.index.batch_query(X_q, k)

        for i in range(len(X_q)):
            single_idx, _ = self.index.query(X_q[i], k)
            assert set(batch_idx[i].tolist()) == set(single_idx.tolist()), \
                f"Requête {i} : divergence batch vs. single."

    def test_build_time_recorded(self):
        """Le temps de construction doit être positif."""
        assert self.index.build_time_s > 0.0
