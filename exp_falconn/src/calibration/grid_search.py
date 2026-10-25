"""
Calibration grid search for LSH hyperparameters (section 8).

Calibration is performed on the *validation* set only.
No test data is ever used for parameter selection.

For E2LSH, the calibrated parameters are:
    w       : bucket width
    K       : hash functions per table
    L       : number of tables
    probes  : multi-probe count

For SimHash:
    num_bits
    num_tables

Selection criterion:
    Minimise median full query latency
    subject to Recall@10 >= 0.95 and macro-F1 drop <= 0.01 vs. linear.

Tiebreaking:
    1. Minimum mean candidate fraction.
    2. Minimum memory footprint.
    3. Minimum build time.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.protocol import (
    MIN_RECALL_AT_10,
    MAX_F1_DROP,
    TIMING_WARMUP,
    TIMING_REPEATS,
    K_NEIGHBOURS,
)
from src.evaluation.metrics import recall_at_k, macro_f1, candidate_fraction
from src.timing.harness import timed_query


@dataclass
class CalibrationResult:
    params: Dict[str, Any]
    recall_at_10: float
    f1: float
    f1_drop: float
    median_latency_s: float
    mean_candidate_fraction: float
    build_time_s: float
    memory_bytes: int
    passed: bool  # True if quality constraints satisfied


def calibrate_e2lsh(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    linear_neighbours: np.ndarray,  # (n_val, k) exact neighbour indices
    linear_f1: float,               # macro-F1 of linear on validation
    hash_seed: int,
    param_grid: Dict[str, List[Any]],
) -> CalibrationResult:
    """Grid search over E2LSH hyperparameters on the validation set.

    Args:
        X_train:          Training base vectors (float32).
        y_train:          Training labels.
        X_val:            Validation queries (float32).
        y_val:            Validation labels.
        linear_neighbours: Exact neighbours from linear search, shape (n_val, k).
        linear_f1:        macro-F1 of linear on validation (reference).
        hash_seed:        LSH hash seed for this calibration run.
        param_grid:       Dict with keys 'w', 'K', 'L', 'probes' mapping to lists.

    Returns:
        Best CalibrationResult by protocol criterion, or the best relaxed
        result if no config passes quality constraints.
    """
    from src.indexes.e2lsh import E2LSH

    results: List[CalibrationResult] = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        p = dict(zip(keys, combo))
        index = E2LSH(
            w=p["w"], K=p["K"], L=p["L"],
            n_probes=p.get("probes", 0),
            hash_seed=hash_seed,
        )
        index.build(X_train)

        latencies, recalls, cand_fracs, preds = [], [], [], []
        for i, q in enumerate(X_val):
            lat, (idxs, _, n_cand, _) = timed_query(
                index.query, q, K_NEIGHBOURS,
                warmup=TIMING_WARMUP, repeats=TIMING_REPEATS,
            )
            latencies.append(lat)
            rc = recall_at_k(idxs, linear_neighbours[i], K_NEIGHBOURS)
            recalls.append(rc)
            cand_fracs.append(candidate_fraction(n_cand, len(X_train)))
            pred = _majority_vote(y_train[idxs]) if len(idxs) > 0 else -1
            preds.append(pred)

        r10 = float(np.mean(recalls))
        f1 = macro_f1(y_val, preds)
        drop = linear_f1 - f1
        passed = (r10 >= MIN_RECALL_AT_10) and (drop <= MAX_F1_DROP)

        results.append(CalibrationResult(
            params=p,
            recall_at_10=r10,
            f1=f1,
            f1_drop=drop,
            median_latency_s=float(np.median(latencies)),
            mean_candidate_fraction=float(np.mean(cand_fracs)),
            build_time_s=index.build_time_s,
            memory_bytes=_estimate_memory(index),
            passed=passed,
        ))

    return _select_best(results)


def calibrate_cross_polytope(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    linear_neighbours: np.ndarray,  # (n_val, k) exact neighbour indices
    linear_f1: float,               # macro-F1 of linear on validation
    hash_seed: int,
    param_grid: Dict[str, List[Any]],
) -> CalibrationResult:
    """Grid search over CrossPolytope hyperparameters on the validation set.

    Args:
        X_train:          Training base vectors (float32, L2-normalised).
        y_train:          Training labels.
        X_val:            Validation queries (float32, L2-normalised).
        y_val:            Validation labels.
        linear_neighbours: Exact neighbours from linear search, shape (n_val, k).
        linear_f1:        macro-F1 of linear on validation (reference).
        hash_seed:        LSH hash seed for this calibration run.
        param_grid:       Dict with keys 'num_hashes', 'num_tables',
                          'num_rotations', 'num_probes' mapping to lists.

    Returns:
        Best CalibrationResult by protocol criterion, or the best relaxed
        result if no config passes quality constraints.
    """
    from src.indexes.cross_polytope import CrossPolytope

    results: List[CalibrationResult] = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        p = dict(zip(keys, combo))
        index = CrossPolytope(
            num_hashes=p["num_hashes"],
            num_tables=p["num_tables"],
            num_rotations=p.get("num_rotations", 2),
            num_probes=p.get("num_probes", 1),
            hash_seed=hash_seed,
        )
        index.build(X_train)

        latencies, recalls, cand_fracs, preds = [], [], [], []
        for i, q in enumerate(X_val):
            lat, (idxs, _, n_cand, _) = timed_query(
                index.query, q, K_NEIGHBOURS,
                warmup=TIMING_WARMUP, repeats=TIMING_REPEATS,
            )
            latencies.append(lat)
            rc = recall_at_k(idxs, linear_neighbours[i], K_NEIGHBOURS)
            recalls.append(rc)
            cand_fracs.append(candidate_fraction(n_cand, len(X_train)))
            pred = _majority_vote(y_train[idxs]) if len(idxs) > 0 else -1
            preds.append(pred)

        r10 = float(np.mean(recalls))
        f1 = macro_f1(y_val, preds)
        drop = linear_f1 - f1
        passed = (r10 >= MIN_RECALL_AT_10) and (drop <= MAX_F1_DROP)

        results.append(CalibrationResult(
            params=p,
            recall_at_10=r10,
            f1=f1,
            f1_drop=drop,
            median_latency_s=float(np.median(latencies)),
            mean_candidate_fraction=float(np.mean(cand_fracs)),
            build_time_s=index.build_time_s,
            memory_bytes=index.estimate_memory_bytes(),
            passed=passed,
        ))

    return _select_best(results)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _majority_vote(labels: np.ndarray) -> Any:
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def _estimate_memory(index: Any) -> int:
    """Rough estimate of index memory usage in bytes."""
    total = 0
    if hasattr(index, '_tables'):
        for table in index._tables:
            for bucket in table.values():
                total += len(bucket) * 8  # int64
    if hasattr(index, '_a'):
        for a in index._a:
            total += a.nbytes
    if hasattr(index, '_b'):
        for b in index._b:
            total += b.nbytes
    return total


def _select_best(results: List[CalibrationResult]) -> CalibrationResult:
    """Select the best configuration per protocol criterion."""
    passing = [r for r in results if r.passed]
    pool = passing if passing else results  # relax if nothing passes
    # Primary: minimum median latency.
    pool.sort(key=lambda r: (
        r.median_latency_s,
        r.mean_candidate_fraction,
        r.memory_bytes,
        r.build_time_s,
    ))
    return pool[0]
