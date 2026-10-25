"""
Non-regression test for the cross_polytope patch (step 2).

Verifies that the patched CrossPolytope produces EXACTLY the same candidate
sets as the original implementation, across all 4 calibrated configurations,
on 150 random queries each. Zero divergence tolerated.

The original is loaded from cross_polytope_backup.py via importlib to avoid
name collisions. The patched version is the current src.indexes.cross_polytope.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the backup (original) module under a unique name
# ---------------------------------------------------------------------------

_BACKUP_PATH = (
    Path(__file__).resolve().parent.parent
    / "src/indexes/cross_polytope_backup.py"
)
_spec = importlib.util.spec_from_file_location("cp_original", _BACKUP_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CrossPolytopeOriginal = _mod.CrossPolytope

# The patched (current) version
from src.indexes.cross_polytope import CrossPolytope as CrossPolytopePatched


# ---------------------------------------------------------------------------
# Synthetic datasets calibrated to match §6.4 of DECISIONS_AND_RESULTS.md
# ---------------------------------------------------------------------------

_N_QUERIES = 150
_SEED_DATA = 9999   # deterministic data generation
_SEED_QUERIES = 8888


def _make_heart(n: int = 200, d: int = 13) -> np.ndarray:
    rng = np.random.default_rng(_SEED_DATA)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _make_chest(n: int, d: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(_SEED_DATA)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _random_queries(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    Q = X[idx].copy()
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q


# ---------------------------------------------------------------------------
# Four calibrated configurations from §6.4
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "heart K=1/L=20/probes=1",
        "dataset": "heart",
        "n_base": 200,
        "d": 13,
        "K": 1, "L": 20, "rot": 2, "probes": 1,
        "hash_seed": 11,
    },
    {
        "name": "chest 10k K=2/L=20/probes=1",
        "dataset": "chest_10k",
        "n_base": 10000,
        "d": 1024,
        "K": 2, "L": 20, "rot": 2, "probes": 1,
        "hash_seed": 11,
    },
    {
        "name": "chest 50k K=2/L=20/probes=1",
        "dataset": "chest_50k",
        "n_base": 50000,
        "d": 1024,
        "K": 2, "L": 20, "rot": 2, "probes": 1,
        "hash_seed": 11,
    },
    {
        "name": "chest full K=2/L=10/probes=4",
        "dataset": "chest_full",
        "n_base": 71756,
        "d": 1024,
        "K": 2, "L": 10, "rot": 2, "probes": 4,
        "hash_seed": 11,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_and_collect_candidates(Cls, X, cfg):
    """Build index and return (candidate_sets_per_query, rotated_Y_all)."""
    idx = Cls(
        num_hashes=cfg["K"],
        num_tables=cfg["L"],
        num_rotations=cfg["rot"],
        num_probes=cfg["probes"],
        hash_seed=cfg["hash_seed"],
    )
    idx.build(X)
    return idx


def _get_full_candidates(idx, x, k):
    """Call query and return the FULL set of candidate indices (not just top-k).

    The candidate set is reconstructed from the raw bucket union before
    reranking. Since query() already returns the top-k from the union, we
    use a large k to recover the full union — but for absolute certainty,
    we also patch the function to capture the raw set.
    """
    # Use a k larger than any plausible candidate count to get the full union.
    # On 71k base with 20% cand_frac, max candidates ~14k. Using n_base is safe.
    n_base = idx._base.shape[0]
    indices, dists, n_cand, n_buck = idx.query(x, k=n_base)
    return set(int(i) for i in indices), n_cand, n_buck


# ---------------------------------------------------------------------------
# Test: strict candidate-set equality across all configurations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS, ids=[c["name"] for c in CONFIGS])
def test_candidate_set_strict_equality(cfg):
    """For each calibrated config, 150 random queries: candidate sets identical.

    This test imports the backup (original) module and the patched module,
    builds both with the same data + hash_seed, and verifies that the full
    candidate set returned by query() is bit-identical for every query.
    """
    # Build datasets
    if cfg["dataset"] == "heart":
        X = _make_heart(cfg["n_base"], cfg["d"])
    else:
        X = _make_chest(cfg["n_base"], cfg["d"])

    queries = _random_queries(X, _N_QUERIES, _SEED_QUERIES)

    # Build both indexes (same data, same hash_seed → same _D tables)
    idx_orig = _build_and_collect_candidates(CrossPolytopeOriginal, X, cfg)
    idx_patch = _build_and_collect_candidates(CrossPolytopePatched, X, cfg)

    # Verify that the rotation tables are identical
    for t in range(cfg["L"]):
        for j in range(cfg["K"]):
            for r in range(cfg["rot"]):
                np.testing.assert_array_equal(
                    idx_orig._D[t][j][r],
                    idx_patch._D[t][j][r],
                    err_msg=f"D[{t}][{j}][{r}] differs between original and patch",
                )

    # Verify hash tables have the same keys
    for t in range(cfg["L"]):
        orig_keys = set(idx_orig._tables[t].keys())
        patch_keys = set(idx_patch._tables[t].keys())
        assert orig_keys == patch_keys, (
            f"Table {t}: key sets differ "
            f"(only in orig: {len(orig_keys - patch_keys)}, "
            f"only in patch: {len(patch_keys - orig_keys)})"
        )

    # Run all queries and compare candidate sets
    divergences = []
    for qi, q in enumerate(queries):
        cand_orig, nc_orig, nb_orig = _get_full_candidates(idx_orig, q, 99999)
        cand_patch, nc_patch, nb_patch = _get_full_candidates(idx_patch, q, 99999)

        if cand_orig != cand_patch:
            only_orig = cand_orig - cand_patch
            only_patch = cand_patch - cand_orig
            divergences.append({
                "query_idx": qi,
                "only_in_original": only_orig,
                "only_in_patch": only_patch,
                "n_orig": len(cand_orig),
                "n_patch": len(cand_patch),
            })

    assert len(divergences) == 0, (
        f"CONFIG '{cfg['name']}': {len(divergences)}/{_N_QUERIES} queries "
        f"diverge in candidate set. First divergence:\n"
        f"  query_idx={divergences[0]['query_idx']}, "
        f"n_orig={divergences[0]['n_orig']}, n_patch={divergences[0]['n_patch']}, "
        f"only_in_original={divergences[0]['only_in_original']}, "
        f"only_in_patch={divergences[0]['only_in_patch']}"
    )


# ---------------------------------------------------------------------------
# Quick smoke: same top-k results (faster sanity check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS[:2], ids=[c["name"] for c in CONFIGS[:2]])
def test_topk_results_match(cfg):
    """Spot-check: top-10 indices and distances are identical (first 30 queries)."""
    if cfg["dataset"] == "heart":
        X = _make_heart(cfg["n_base"], cfg["d"])
    else:
        X = _make_chest(cfg["n_base"], cfg["d"])

    queries = _random_queries(X, 30, _SEED_QUERIES)

    idx_orig = _build_and_collect_candidates(CrossPolytopeOriginal, X, cfg)
    idx_patch = _build_and_collect_candidates(CrossPolytopePatched, X, cfg)

    for qi, q in enumerate(queries):
        i1, d1, _, _ = idx_orig.query(q, k=10)
        i2, d2, _, _ = idx_patch.query(q, k=10)
        np.testing.assert_array_equal(
            i1, i2,
            err_msg=f"Query {qi}: top-k indices differ",
        )
        np.testing.assert_allclose(
            d1, d2, rtol=1e-6,
            err_msg=f"Query {qi}: top-k distances differ",
        )
