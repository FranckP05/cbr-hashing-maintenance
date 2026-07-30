"""Reproducible retrieval benchmark for any registered dataset.

The dataset is intentionally small.  This script therefore validates the experimental
protocol and produces figures, but it is not used to draw conclusions about scaling.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.indexing.linear_scan import LinearScanIndex
from src.indexing.kdtree_index import KDTreeIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex
from src.evaluation.retrieval_metrics import ranking_metrics
from src.evaluation.classification_metrics import distance_weighted_prediction
from src.evaluation.timing import time_index_build, time_query


DATASET_REGISTRY = {
    "heart_disease": ("src.datasets.heart_disease", "load_heart_disease", "build_feature_matrix"),
    "diabetes130":   ("src.datasets.diabetes130",   "load_diabetes130",   "build_feature_matrix"),
}

DEFAULT_DATA_PATHS = {
    "heart_disease": "data/raw/heart_disease/processed.cleveland.data",
}

DEFAULT_SEEDS = (42, 7, 123, 2024, 99)
DEFAULT_K_VALUES = (1, 5, 10)
METHODS = ("Baseline", "KD-Tree", "LSH")


def make_index(method: str):
    if method == "Baseline":
        return LinearScanIndex()
    if method == "KD-Tree":
        return KDTreeIndex()
    if method == "LSH":
        return RandomProjectionLSHIndex(num_tables=16, num_bits=5, random_seed=42)
    raise ValueError(f"Unknown method: {method}")


def aggregate_seed_results(query_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "precision_at_k",
        "label_recall_at_k",
        "map_at_k",
        "ndcg_at_k",
        "hit_at_k",
        "mrr_at_k",
        "exact_neighbor_recall_at_k",
        "returned_neighbors",
        "query_time_ms",
    ]
    rows = []
    for (seed, method, k), group in query_results.groupby(["seed", "method", "k"], sort=True):
        valid_predictions = group.dropna(subset=["prediction"])
        row = {
            "seed": seed,
            "method": method,
            "k": k,
            **{column: group[column].mean() for column in metric_columns},
            "prediction_coverage": len(valid_predictions) / len(group),
            "accuracy": np.nan,
            "f1": np.nan,
            "auc": np.nan,
        }
        if not valid_predictions.empty:
            targets = valid_predictions["query_target"].to_numpy()
            predictions = valid_predictions["prediction"].to_numpy()
            scores = valid_predictions["positive_score"].to_numpy()
            row["accuracy"] = float(np.mean(predictions == targets))
            row["f1"] = float(f1_score(targets, predictions, zero_division=0))
            if len(np.unique(targets)) == 2:
                row["auc"] = float(roc_auc_score(targets, scores))
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(
    dataset_name: str,
    data_path: Path | None,
    output_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    module_path, load_name, build_name = DATASET_REGISTRY[dataset_name]
    module = importlib.import_module(module_path)
    load_func = getattr(module, load_name)
    build_func = getattr(module, build_name)

    dataframe = load_func(data_path) if data_path is not None else load_func()
    output_dir.mkdir(parents=True, exist_ok=True)

    needs_feature_columns = hasattr(module, "FEATURE_COLUMNS")
    if needs_feature_columns:
        feature_columns = module.FEATURE_COLUMNS
        raw_features = dataframe[feature_columns].to_numpy(dtype=float)
        targets = dataframe["target"].to_numpy(dtype=int)
    else:
        vectors_full, targets = build_func(dataframe)[:2]

    query_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        if needs_feature_columns:
            train_indices, test_indices = next(
                StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                    raw_features, targets
                )
            )
            scaler = StandardScaler().fit(raw_features[train_indices])
            train_vectors = scaler.transform(raw_features[train_indices])
            test_vectors = scaler.transform(raw_features[test_indices])
        else:
            train_indices, test_indices = next(
                StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                    vectors_full, targets
                )
            )
            train_vectors = vectors_full[train_indices]
            test_vectors = vectors_full[test_indices]
        train_targets = targets[train_indices]
        test_targets = targets[test_indices]

        exact_neighbors: dict[int, np.ndarray] = {}
        for k in k_values:
            reference = LinearScanIndex()
            reference.build(train_vectors)
            exact_neighbors[k] = np.vstack(
                [reference.query(vector, k)[0] for vector in test_vectors]
            )

        for method in METHODS:
            index, build_elapsed = time_index_build(lambda: make_index(method), train_vectors)
            build_time_ms = build_elapsed * 1_000

            for k in k_values:
                for query_position, (vector, target) in enumerate(zip(test_vectors, test_targets)):
                    retrieved_indices, distances, query_elapsed = time_query(index, vector, k)
                    query_time_ms = query_elapsed * 1_000
                    metrics = ranking_metrics(retrieved_indices, train_targets, target, k)
                    exact_recall = len(
                        np.intersect1d(retrieved_indices, exact_neighbors[k][query_position])
                    ) / k
                    prediction, positive_score = distance_weighted_prediction(
                        distances, train_targets[retrieved_indices]
                    )
                    query_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "k": k,
                            "query_position": query_position,
                            "query_target": target,
                            "returned_neighbors": len(retrieved_indices),
                            "exact_neighbor_recall_at_k": exact_recall,
                            "prediction": prediction,
                            "positive_score": positive_score,
                            "query_time_ms": query_time_ms,
                            "index_build_ms": build_time_ms,
                            **metrics,
                        }
                    )

    per_query = pd.DataFrame(query_rows)
    per_seed = aggregate_seed_results(per_query)
    summary_metrics = [column for column in per_seed.columns if column not in {"seed", "method", "k"}]
    summary = per_seed.groupby(["method", "k"], sort=True)[summary_metrics].agg(["mean", "std"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns
    ]
    build_times = (
        per_query.groupby(["seed", "method"], sort=True)["index_build_ms"].first().reset_index()
    )
    build_summary = build_times.groupby("method", sort=True)["index_build_ms"].agg(["mean", "std"])
    build_summary = build_summary.reset_index().rename(
        columns={"mean": "index_build_ms_mean", "std": "index_build_ms_std"}
    )

    prefix = f"{dataset_name}_retrieval"
    per_query.to_csv(output_dir / f"{prefix}_per_query.csv", index=False)
    per_seed.to_csv(output_dir / f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    build_summary.to_csv(output_dir / f"{prefix}_index_build_summary.csv", index=False)
    metadata = {
        "dataset": dataset_name,
        "instances_after_missing_value_removal": int(len(dataframe)),
        "seeds": list(seeds),
        "k_values": list(k_values),
        "split": "80/20 stratified train/test; scaler fitted on each training split only",
        "lsh": {"num_tables": 16, "num_bits": 5, "random_seed": 42},
    }
    if needs_feature_columns:
        metadata["features"] = int(raw_features.shape[1])
    else:
        metadata["features"] = int(vectors_full.shape[1])
    (output_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return per_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        default="heart_disease",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/benchmarks")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path or DEFAULT_DATA_PATHS.get(args.dataset)
    results = run_benchmark(args.dataset, data_path, args.output_dir)
    print(
        f"Benchmark complete — {args.dataset}: "
        f"{len(results)} method/seed/k aggregates in {args.output_dir}."
    )
