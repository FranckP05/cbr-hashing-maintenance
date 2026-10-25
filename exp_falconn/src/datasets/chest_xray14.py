"""
ChestX-ray14 embedding loader.

Preprocessing steps (section 3 of protocol):
  - Use embeddings directly (no extra normalisation in the L2 arm).
  - Verify absence of NaN and infinity.
  - Stratified split using split_seed.

For the optional cosine arm, L2-normalise all vectors AFTER splitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Optional, Tuple

from config.seeds import SPLIT_SEEDS, SEED_SAMPLING
from config.protocol import TEST_FRACTION, VALID_FRACTION_OF_TRAIN
from config.metrics import CHEST_COSINE_ARM_ENABLED


def _check_array(X: np.ndarray, name: str) -> None:
    """Assert no NaN or infinity values."""
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError(f"{name} contains NaN or infinity values.")


def load_and_split(
    embeddings_path: str,
    labels_path: str,
    split_seed: int,
    max_queries: int = 300,
    normalise_cosine: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray,  # X_train, y_train
    np.ndarray, np.ndarray,  # X_val,   y_val
    np.ndarray, np.ndarray,  # X_test,  y_test
]:
    """Load ChestX-ray14 embeddings, validate, split.

    Args:
        embeddings_path: Path to .npy file of shape (N, 1024).
        labels_path:     Path to .npy file of shape (N,).
        split_seed:      One of SPLIT_SEEDS.
        max_queries:     Maximum number of test queries to select (300).
        normalise_cosine: If True, L2-normalise for the optional cosine arm.

    Returns:
        Six arrays: X_train, y_train, X_val, y_val, X_test, y_test.
        X_test is already trimmed to max_queries rows (selected with
        SEED_SAMPLING for reproducibility).
    """
    assert split_seed in SPLIT_SEEDS, f"Unknown split seed: {split_seed}"
    if normalise_cosine and not CHEST_COSINE_ARM_ENABLED:
        raise RuntimeError(
            "Cosine normalisation requested but CHEST_COSINE_ARM_ENABLED is False."
        )

    if str(embeddings_path).endswith(".parquet"):
        import pandas as pd
        embeddings_df = pd.read_parquet(embeddings_path)
        labels_df = pd.read_parquet(labels_path)
        merged = embeddings_df.merge(labels_df, on="image_filename")
        pathology_cols = [c for c in labels_df.columns if c != "image_filename"]
        merged["target"] = (merged[pathology_cols].sum(axis=1) > 0).astype(np.int64)
        emb_cols = [c for c in merged.columns if c.startswith("emb_")]
        X = merged[emb_cols].to_numpy(dtype=np.float32)
        y = merged["target"].to_numpy(dtype=np.int64)
    else:
        X = np.load(embeddings_path).astype(np.float32)
        y = np.load(labels_path).astype(np.int64)
    _check_array(X, "ChestX-ray14 embeddings")

    # --- Test split ---------------------------------------------------------
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=split_seed
    )
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test_full, y_test_full = X[test_idx], y[test_idx]

    # Reproducible query sub-selection.
    rng = np.random.default_rng(SEED_SAMPLING)
    chosen = rng.choice(len(X_test_full), size=min(max_queries, len(X_test_full)), replace=False)
    chosen.sort()
    X_test, y_test = X_test_full[chosen], y_test_full[chosen]

    # --- Validation split ---------------------------------------------------
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALID_FRACTION_OF_TRAIN,
        random_state=split_seed,
    )
    train_idx, val_idx = next(sss_val.split(X_tv, y_tv))
    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]

    # --- Optional cosine normalisation --------------------------------------
    if normalise_cosine:
        for arr in [X_train, X_val, X_test]:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr /= np.where(norms == 0, 1.0, norms)

    return X_train, y_train, X_val, y_val, X_test, y_test


def subsample_base(
    X_train: np.ndarray,
    y_train: np.ndarray,
    size: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a reproducible sub-base of the training set.

    Args:
        X_train: Full training embeddings.
        y_train: Full training labels.
        size:    Number of samples; None means use the full base.

    Returns:
        (X_base, y_base) pair of requested size.
    """
    if size is None or size >= len(X_train):
        return X_train, y_train
    rng = np.random.default_rng(SEED_SAMPLING)
    idx = rng.choice(len(X_train), size=size, replace=False)
    idx.sort()
    return X_train[idx], y_train[idx]
