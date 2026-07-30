"""Calibrate LSH hyperparameters (num_bits, num_tables) on any registered dataset.

Sweeps over num_bits in [8, 10, 12, 14, 16] and num_tables in [8, 16, 32],
measuring exact-neighbor recall@10 against a KD-Tree reference on 30 random
training queries.
"""

import argparse
import importlib
import time

import numpy as np
import pandas as pd

from src.indexing.kdtree_index import KDTreeIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex


DATASET_REGISTRY = {
    "heart_disease": ("src.datasets.heart_disease", "load_heart_disease", "build_feature_matrix"),
    "diabetes130":   ("src.datasets.diabetes130",   "load_diabetes130",   "build_feature_matrix"),
}

DEFAULT_DATA_PATHS = {
    "heart_disease": "data/raw/heart_disease/processed.cleveland.data",
}

NUM_BITS_VALUES = [8, 10, 12, 14, 16]
NUM_TABLES_VALUES = [8, 16, 32]
K = 10
N_QUERIES = 30
SEED = 42


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate LSH num_bits and num_tables on any registered dataset."
    )
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

    result = build_func(dataframe)
    vectors, targets = result[0], result[1]

    print(f"=== LSH Calibration on {args.dataset} ===\n")
    print(f"Dataset: {vectors.shape[0]} cases, {vectors.shape[1]} features\n")

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(vectors))
    split = int(0.8 * len(vectors))
    train_vectors = vectors[shuffled[:split]]

    print(f"Building KD-Tree reference on {len(train_vectors)} training cases ...")
    t0 = time.perf_counter()
    kdtree = KDTreeIndex()
    kdtree.build(train_vectors)
    print(f"  done in {time.perf_counter() - t0:.4f}s\n")

    query_rng = np.random.default_rng(SEED)
    query_indices = query_rng.choice(len(train_vectors), size=N_QUERIES, replace=False)
    query_vectors = train_vectors[query_indices]

    print(f"Computing KD-Tree ground truth for {N_QUERIES} queries (k={K}) ...")
    gt_neighbors = []
    for vec in query_vectors:
        idx, _ = kdtree.query(vec, K)
        gt_neighbors.append(set(idx))
    print("  done.\n")

    print(f"{'num_bits':>8} {'num_tables':>10} {'mean_recall':>11} {'min_recall':>10} {'build_time_ms':>13}")
    print("-" * 56)

    results = []
    for num_bits in NUM_BITS_VALUES:
        for num_tables in NUM_TABLES_VALUES:
            t0 = time.perf_counter()
            lsh = RandomProjectionLSHIndex(
                num_tables=num_tables, num_bits=num_bits, random_seed=SEED
            )
            lsh.build(train_vectors)
            build_time = (time.perf_counter() - t0) * 1000

            recalls = []
            for vec, gt in zip(query_vectors, gt_neighbors):
                idx, _ = lsh.query(vec, K)
                overlap = len(gt & set(idx))
                recalls.append(overlap / K)

            mean_recall = float(np.mean(recalls))
            min_recall = float(np.min(recalls))

            results.append({
                "num_bits": num_bits,
                "num_tables": num_tables,
                "mean_recall": mean_recall,
                "min_recall": min_recall,
                "build_time_ms": build_time,
            })

            print(
                f"{num_bits:>8} {num_tables:>10} {mean_recall:>11.4f} "
                f"{min_recall:>10.4f} {build_time:>13.4f}"
            )

    df = pd.DataFrame(results)

    candidates = df[df["mean_recall"] >= 0.95].copy()
    if not candidates.empty:
        candidates = candidates.sort_values(
            by=["num_tables", "num_bits"], ascending=[True, False]
        )
        best = candidates.iloc[0]
        print(
            f"\nRecommandation : num_bits={int(best['num_bits'])}, "
            f"num_tables={int(best['num_tables'])} "
            f"(mean recall={best['mean_recall']:.4f}, "
            f"min recall={best['min_recall']:.4f})"
        )
    else:
        best = df.loc[df["mean_recall"].idxmax()]
        print(
            f"\nAucune combinaison n'atteint 0.95 de rappel moyen. "
            f"Meilleure : num_bits={int(best['num_bits'])}, "
            f"num_tables={int(best['num_tables'])} "
            f"(mean recall={best['mean_recall']:.4f})"
        )

    print("\n=== Calibration complete ===")


if __name__ == "__main__":
    main()
