"""Calibrate LSH hyperparameters (num_bits, num_tables) on any registered dataset.

Sweeps over num_bits in [8, 10, 12, 14, 16] and num_tables in [8, 16, 32],
measuring exact-neighbor recall@10 against a KD-Tree reference on 30 random
training queries, plus query time and candidate-set size for each combo.
"""

import argparse
import importlib
import time

import numpy as np
import pandas as pd

from src.evaluation.timing import time_query
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

    n_train = len(train_vectors)

    print(f"Building KD-Tree reference on {n_train} training cases ...")
    t0 = time.perf_counter()
    kdtree = KDTreeIndex()
    kdtree.build(train_vectors)
    print(f"  done in {time.perf_counter() - t0:.4f}s\n")

    query_rng = np.random.default_rng(SEED)
    query_indices = query_rng.choice(n_train, size=N_QUERIES, replace=False)
    query_vectors = train_vectors[query_indices]

    print(f"Computing KD-Tree ground truth for {N_QUERIES} queries (k={K}) ...")
    gt_neighbors = []
    for vec in query_vectors:
        idx, _ = kdtree.query(vec, K)
        gt_neighbors.append(set(idx))
    print("  done.\n")

    header = (
        f"{'bits':>4} {'tables':>6} {'recall':>7} {'recall_min':>10} "
        f"{'qry_ms':>7} {'cand':>6} {'cand%':>7} {'build_ms':>8}"
    )
    print(header)
    print("-" * 67)

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
            query_times = []
            candidate_counts = []
            for vec, gt in zip(query_vectors, gt_neighbors):
                idx, dists, q_elapsed = time_query(lsh, vec, K)
                query_times.append(q_elapsed * 1000)
                overlap = len(gt & set(idx))
                recalls.append(overlap / K)
                candidate_counts.append(lsh.last_candidate_count)

            mean_recall = float(np.mean(recalls))
            min_recall = float(np.min(recalls))
            mean_qms = float(np.mean(query_times))
            mean_cand = float(np.mean(candidate_counts))
            cand_pct = mean_cand / n_train * 100

            results.append({
                "num_bits": num_bits,
                "num_tables": num_tables,
                "mean_recall": mean_recall,
                "min_recall": min_recall,
                "query_time_ms": mean_qms,
                "candidate_count": mean_cand,
                "candidate_pct": cand_pct,
                "build_time_ms": build_time,
            })

            print(
                f"{num_bits:>4} {num_tables:>6} {mean_recall:>7.4f} {min_recall:>10.4f} "
                f"{mean_qms:>7.4f} {mean_cand:>6.0f} {cand_pct:>6.2f} {build_time:>8.2f}"
            )

    df = pd.DataFrame(results)

    candidates = df[df["mean_recall"] >= 0.95].copy()
    if not candidates.empty:
        print()
        print("Combinaisons avec rappel moyen >= 0.95 :")
        print(
            f"{'bits':>4} {'tables':>6} {'recall':>7} {'recall_min':>10} "
            f"{'qry_ms':>7} {'cand':>6} {'cand%':>7} {'build_ms':>8}"
        )
        print("-" * 67)
        for _, row in candidates.iterrows():
            print(
                f"{int(row['num_bits']):>4} {int(row['num_tables']):>6} "
                f"{row['mean_recall']:>7.4f} {row['min_recall']:>10.4f} "
                f"{row['query_time_ms']:>7.4f} {row['candidate_count']:>6.0f} "
                f"{row['candidate_pct']:>6.2f} {row['build_time_ms']:>8.2f}"
            )

        candidates = candidates.sort_values(
            by=["num_tables", "num_bits"], ascending=[True, False]
        )
        best = candidates.iloc[0]
        print(
            f"\nRecommandation heuristique : num_bits={int(best['num_bits'])}, "
            f"num_tables={int(best['num_tables'])} "
            f"(mean recall={best['mean_recall']:.4f}, "
            f"query_time={best['query_time_ms']:.3f}ms, "
            f"candidates={best['candidate_count']:.0f} / {best['candidate_pct']:.2f}%)"
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
