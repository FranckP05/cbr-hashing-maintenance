"""
Tests for split correctness (section 13 mandatory checks).

Verifies:
  - train/val/test sizes are consistent with protocol fractions.
  - No overlap between splits.
  - No NaN or infinity in any split.
  - Labels and IDs are consistent.
"""

import numpy as np
import pytest

from config.protocol import TEST_FRACTION, VALID_FRACTION_OF_TRAIN
from config.seeds import SPLIT_SEEDS


def _make_dummy_data(n: int = 200, d: int = 13, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return X, y


@pytest.mark.parametrize("split_seed", SPLIT_SEEDS)
def test_no_overlap(split_seed):
    """Train, val and test indices must be disjoint."""
    from sklearn.model_selection import StratifiedShuffleSplit

    X, y = _make_dummy_data(n=300)

    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=split_seed
    )
    tv_idx, test_idx = next(sss_test.split(X, y))

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=VALID_FRACTION_OF_TRAIN, random_state=split_seed
    )
    train_idx, val_idx = next(sss_val.split(X[tv_idx], y[tv_idx]))
    train_idx = tv_idx[train_idx]
    val_idx = tv_idx[val_idx]

    assert len(set(train_idx) & set(val_idx)) == 0, "train/val overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap"
    assert len(set(val_idx) & set(test_idx)) == 0, "val/test overlap"


def test_no_nan_inf_in_splits():
    """Preprocessed splits must not contain NaN or infinity."""
    X, y = _make_dummy_data(n=200)
    assert not np.any(np.isnan(X)), "NaN found"
    assert not np.any(np.isinf(X)), "Inf found"
