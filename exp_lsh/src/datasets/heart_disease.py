"""
Heart Disease dataset loader and preprocessor.

Preprocessing steps (section 3 of protocol):
  1. Drop rows with invalid / NaN values.
  2. Stratified split BEFORE fitting any preprocessor.
  3. Optional median imputation fitted on train only.
  4. StandardScaler fitted on train only, applied to val and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from config.seeds import SPLIT_SEEDS
from config.protocol import TEST_FRACTION, VALID_FRACTION_OF_TRAIN


def load_and_split(
    csv_path: str,
    split_seed: int,
    target_column: str = "target",
) -> Tuple[
    np.ndarray, np.ndarray,  # X_train, y_train
    np.ndarray, np.ndarray,  # X_val,   y_val
    np.ndarray, np.ndarray,  # X_test,  y_test
]:
    """Load Heart Disease CSV, clean, split and scale.

    Args:
        csv_path:      Path to the raw CSV file.
        split_seed:    One of SPLIT_SEEDS.
        target_column: Column name for the classification target.

    Returns:
        Six arrays: X_train, y_train, X_val, y_val, X_test, y_test.
        All X arrays are float32, scaled with a StandardScaler
        fitted exclusively on train.
    """
    assert split_seed in SPLIT_SEEDS, f"Unknown split seed: {split_seed}"

    # Check if file has header or is raw UCI data (e.g. processed.cleveland.data)
    with open(csv_path, "r") as f:
        first_line = f.readline()

    if "?" in first_line or first_line.split(",")[0].replace(".", "").isdigit():
        # Raw UCI format: 14 unnamed columns, '?' represents missing values
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        df = pd.read_csv(csv_path, header=None, names=cols, na_values="?")
        df = df.dropna()
        # Binary target: 0 vs >0
        df["target"] = (df["target"] > 0).astype(int)
    else:
        df = pd.read_csv(csv_path)
        # Drop rows that contain any NaN or infinite value.
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if target_column in df.columns and df[target_column].dtype != int:
            df[target_column] = (df[target_column] > 0).astype(int)

    X = df.drop(columns=[target_column]).values.astype(np.float32)
    y = df[target_column].values.astype(np.int64)

    # --- Test split ---------------------------------------------------------
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=split_seed
    )
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # --- Validation split ---------------------------------------------------
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VALID_FRACTION_OF_TRAIN,
        random_state=split_seed,
    )
    train_idx, val_idx = next(sss_val.split(X_tv, y_tv))
    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val, y_val = X_tv[val_idx], y_tv[val_idx]

    # --- Scaling (fit on train only) ----------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test
