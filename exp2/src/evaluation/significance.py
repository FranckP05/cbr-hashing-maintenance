"""Tests statistiques appariés (T5.1) — Wilcoxon signed-rank + t-test apparié.

Comparaison des méthodes deux à deux, par dataset et par k (k=1,5,10, plus
l'agrégat 'agg' = moyenne des k par seed), avec les 6 seeds comme échantillons
appariés, sur les métriques de temps et de qualité de
benchmark_timing_quality.csv. Sortie CSV : dataset, k, metric, method_a,
method_b, mean_a, mean_b, mean_diff, wilcoxon_p, t_p, n.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["linear", "kdtree", "lsh", "hecbr"]
METRICS = {
    "query_ms": "query_ms",
    "exact_recall_at_k": "exact_recall_at_k",
    "ndcg_at_k": "ndcg_at_k",
    "macro_f1": "macro_f1",
}


def per_seed_means(df):
    df = df.copy()
    df["seed"] = df["seed"].astype(str)
    cols = list(METRICS.keys())
    rows = []
    for k in [1, 5, 10]:
        part = df[df["k"] == k].groupby(["dataset", "method", "seed"])[cols].mean().reset_index()
        part["k"] = k
        rows.append(part)
    part = df.groupby(["dataset", "method", "seed"])[cols].mean().reset_index()
    part["k"] = "agg"
    rows.append(part)
    return pd.concat(rows, ignore_index=True)


def test_pairs(frame):
    records = []
    for (dataset, k), group in frame.groupby(["dataset", "k"]):
        for a, b in itertools.combinations(METHODS, 2):
            pa = group[group["method"] == a].set_index("seed")
            pb = group[group["method"] == b].set_index("seed")
            if pa.empty or pb.empty:
                continue
            for metric, col in METRICS.items():
                xa = pa[col].reindex(pb.index).dropna()
                xb = pb[col].reindex(pb.index).dropna()
                if len(xa) < 2 or len(xa) != len(xb):
                    continue
                diff = xb.values - xa.values
                if np.allclose(diff, 0.0):
                    w_p, t_p = 1.0, 1.0
                else:
                    w_p = stats.wilcoxon(xa.values, xb.values, zero_method="wilcox").pvalue
                    t_p = stats.ttest_rel(xa.values, xb.values).pvalue
                records.append({
                    "dataset": dataset, "k": k, "metric": metric,
                    "method_a": a, "method_b": b,
                    "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
                    "mean_diff": float(diff.mean()),
                    "wilcoxon_p": float(w_p), "t_p": float(t_p), "n": int(len(xa)),
                })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Tests statistiques T5.1 (NB11).")
    parser.add_argument("--csv", required=True, help="benchmark_timing_quality.csv")
    parser.add_argument("--output", required=True, help="CSV de sortie")
    args = parser.parse_args()

    frame = per_seed_means(pd.read_csv(args.csv))
    result = test_pairs(frame)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby(["dataset", "metric"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
