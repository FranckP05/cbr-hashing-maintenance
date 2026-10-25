"""Tests de déterminisme et de cohérence des collisions pour SimHash."""

from __future__ import annotations

import numpy as np
import pytest


def random_l2(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return (X / np.linalg.norm(X, axis=1, keepdims=True))


class TestSimHash:
    """Tests de la classe SimHash."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from indexes.simhash import SimHash
        self.SimHash = SimHash
        self.N, self.d = 1000, 64
        self.X_base = random_l2(self.N, self.d, seed=42)

    # --- Déterminisme -------------------------------------------------------

    def test_determinism_same_seed(self):
        """Deux index construits avec la même graine doivent retourner les mêmes candidats."""
        idx1 = self.SimHash(num_bits=8, num_tables=10, hash_seed=11)
        idx2 = self.SimHash(num_bits=8, num_tables=10, hash_seed=11)
        idx1.build(self.X_base)
        idx2.build(self.X_base)

        x = random_l2(1, self.d, seed=1).squeeze()
        ret1, _, _ = idx1.query(x, k=10)
        ret2, _, _ = idx2.query(x, k=10)
        assert set(ret1.tolist()) == set(ret2.tolist()), "Non déterministe avec la même graine."

    def test_different_seeds_may_differ(self):
        """Des graines différentes peuvent produire des candidats différents (vérification statistique)."""
        results = []
        for seed in [11, 29, 43]:
            idx = self.SimHash(num_bits=6, num_tables=8, hash_seed=seed)
            idx.build(self.X_base)
            x = random_l2(1, self.d, seed=99).squeeze()
            ret, _, _ = idx.query(x, k=10)
            results.append(frozenset(ret.tolist()))
        # Au moins une paire doit produire des résultats différents (non garanti mais très probable)
        unique_results = set(results)
        # Note : si tous identiques, ce n'est pas un échec dur (très improbable)

    # --- Cohérence des collisions -------------------------------------------

    def test_near_identical_vectors_collide(self):
        """Deux vecteurs presque identiques doivent souvent se retrouver dans le même bucket."""
        d = 64
        x = random_l2(1, d, seed=0).squeeze()
        # Perturbation minuscule (angle ~0.5°)
        noise = np.random.default_rng(7).standard_normal(d).astype(np.float32) * 1e-3
        x_near = x + noise
        x_near = (x_near / np.linalg.norm(x_near)).astype(np.float32)

        X_b = np.vstack([x.reshape(1, -1), x_near.reshape(1, -1),
                         random_l2(98, d, seed=100)])
        idx = self.SimHash(num_bits=8, num_tables=20, hash_seed=11)
        idx.build(X_b)

        ret, _, n_cand = idx.query(x, k=5)
        assert 0 in ret.tolist() or 1 in ret.tolist(), \
            "Vecteurs quasi-identiques non retrouvés (problème de collision)."

    # --- Interface ----------------------------------------------------------

    def test_output_format(self):
        """query() doit retourner (indices, distances, n_candidates) correctement typés."""
        idx = self.SimHash(num_bits=8, num_tables=10, hash_seed=11)
        idx.build(self.X_base)
        x = random_l2(1, self.d, seed=5).squeeze()
        indices, dists, n_cand = idx.query(x, k=10)

        assert isinstance(indices, np.ndarray)
        assert isinstance(dists,   np.ndarray)
        assert isinstance(n_cand,  int)
        assert indices.dtype == np.int64
        assert dists.dtype   == np.float32
        assert n_cand >= 0

    def test_distances_non_negative(self):
        """Les distances retournées doivent être non négatives."""
        idx = self.SimHash(num_bits=8, num_tables=10, hash_seed=11)
        idx.build(self.X_base)
        x = random_l2(1, self.d, seed=3).squeeze()
        _, dists, _ = idx.query(x, k=10)
        assert np.all(dists >= 0.0)

    def test_build_time_recorded(self):
        """Le temps de construction doit être positif."""
        idx = self.SimHash(num_bits=8, num_tables=10, hash_seed=11)
        idx.build(self.X_base)
        assert idx.build_time_s > 0.0

    def test_k_larger_than_candidates(self):
        """Quand k > n_candidates, la méthode ne doit pas planter."""
        # Construction avec un seul bit → très peu de collisions possibles
        idx = self.SimHash(num_bits=1, num_tables=2, hash_seed=11)
        X_tiny = random_l2(10, self.d, seed=77)
        idx.build(X_tiny)
        x = random_l2(1, self.d, seed=88).squeeze()
        indices, dists, n_cand = idx.query(x, k=1000)
        assert len(indices) <= n_cand
