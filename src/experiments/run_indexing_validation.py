"""Validate indexing methods (LinearScan, KD-Tree, LSH) on a given dataset.

Compares exact neighbor recall and basic retrieval correctness.
"""

import argparse
import importlib

import numpy as np

from src.indexing.linear_scan import LinearScanIndex
from src.indexing.kdtree_index import KDTreeIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex


DATASET_REGISTRY = {
    "heart_disease": ("src.datasets.heart_disease", "load_heart_disease", "build_feature_matrix"),
    "diabetes130":   ("src.datasets.diabetes130",   "load_diabetes130",   "build_feature_matrix"),
}

DEFAULT_DATA_PATHS = {
    "heart_disease": "data/raw/heart_disease/processed.cleveland.data",
}


def main():
    parser = argparse.ArgumentParser(description="Validate indexing methods on any registered dataset.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        default="heart_disease",
    )
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    module_path, load_name, build_name = DATASET_REGISTRY[args.dataset]
    module = importlib.import_module(module_path)
    load_func = getattr(module, load_name)
    build_func = getattr(module, build_name)

    data_path = args.data_path or DEFAULT_DATA_PATHS.get(args.dataset)
    dataframe = load_func(data_path) if data_path is not None else load_func()

    print(f"=== Indexing Validation on {args.dataset} ===\n")

    # 1. Load and vectorize
    print(f"Data shape after cleaning: {dataframe.shape}")
    target_counts = dataframe["target"].value_counts().to_dict()
    print(f"Target distribution: {target_counts}")

    result = build_func(dataframe)
    vectors, targets = result[0], result[1]
    print(f"Feature matrix shape: {vectors.shape}")

    # 2. Linear scan baseline: neighbors for case 0 in the full dataset, k=5
    linear_all = LinearScanIndex()
    linear_all.build(vectors)
    indices_0, dists_0 = linear_all.query(vectors[0], k=5)
    print(f"\nLinearScan neighbors for case 0, k=5: indices {indices_0.tolist()}")

    # Use the first 237 cases as train (80%), remaining 60 as test (20%)
    # Deterministic split matching the original seed=42 convention.
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(vectors))
    split = int(0.8 * len(vectors))
    train_vectors = vectors[shuffled[:split]]
    test_vectors = vectors[shuffled[split:]]
    train_targets = targets[shuffled[:split]]
    test_targets = targets[shuffled[split:]]

    # 3. Build indexes
    linear = LinearScanIndex()
    linear.build(train_vectors)

    kdtree = KDTreeIndex()
    kdtree.build(train_vectors)

    lsh = RandomProjectionLSHIndex(num_tables=16, num_bits=5, random_seed=42)
    lsh.build(train_vectors)

    # 4. Verify KDTree matches LinearScan exactly
    exact_match = True
    for i, vec in enumerate(test_vectors):
        lin_idx, _ = linear.query(vec, k=5)
        kd_idx, _ = kdtree.query(vec, k=5)
        if not np.array_equal(lin_idx, kd_idx):
            exact_match = False
            print(f"  Mismatch at query {i}: Linear {lin_idx} vs KDTree {kd_idx}")
            break
    print(f"KD-Tree identical to LinearScan: {exact_match}")

    # 5. LSH approximate recall (use first 20 queries to match calibration)
    lsh_recalls = []
    k = 5
    for i, vec in enumerate(test_vectors[:20]):
        lin_idx, _ = linear.query(vec, k)
        lsh_idx, _ = lsh.query(vec, k)
        overlap = len(np.intersect1d(lin_idx, lsh_idx))
        lsh_recalls.append(overlap / k)
    mean_recall = np.mean(lsh_recalls)
    print(f"LSH (num_bits=5, num_tables=16): mean exact-neighbor recall = {mean_recall:.4f} "
          f"over 20 queries")

    # 6. Classification accuracy with LSH k=5 weighted vote
    correct = 0
    for i, vec in enumerate(test_vectors):
        idx, dists = lsh.query(vec, k=5)
        if len(idx) == 0:
            continue
        weights = 1.0 / np.maximum(dists, np.finfo(float).eps)
        pred = float(np.average(train_targets[idx], weights=weights) >= 0.5)
        correct += (pred == test_targets[i])
    acc = correct / len(test_vectors)
    print(f"LSH classification accuracy (k=5 weighted vote): {acc:.4f}")

    print("\n=== Validation complete ===")


if __name__ == "__main__":
    main()
