"""Sonde mémoire par sous-processus (T1.6).

Chaque (dataset, méthode) est mesuré dans un interpréteur frais : le pic RSS
(ru_maxrss) après construction de l'index y est propre à la méthode (pas de
pollution cumulative), et les allocations Python d'une requête sont mesurées via
tracemalloc. Résultat imprimé en JSON sur stdout.

Deux modes :
  --method <linear|kdtree|lsh|hecbr> : charge les données, construit l'index et
    imprime le pic RSS absolu après build + les allocations de requête ;
  --method baseline : charge uniquement les données (pas de build) et imprime le
    pic RSS absolu — la contribution d'un index vaut pic(build) - pic(baseline),
    et le pic étant dominé par le chargement, le delta doit se calculer entre deux
    processus propres, pas contre un sous-ensemble du même processus.

Usage : python benchmark_memory.py --dataset heart --method linear --exp2 <racine exp2>
"""

import argparse
import json
import resource
import sys
import tracemalloc

sys.path.insert(0, sys.argv[sys.argv.index("--exp2") + 1])

import numpy as np  # noqa: E402

from config import benchmark as bench_cfg  # noqa: E402
from src.datasets import adapters  # noqa: E402
from src.indexing.hecbr_index import HeCBRIndex  # noqa: E402
from src.indexing.kdtree_index import KDTreeIndex  # noqa: E402
from src.indexing.linear_scan import LinearScanIndex  # noqa: E402
from src.indexing.lsh_index import RandomProjectionLSHIndex  # noqa: E402


def factory_for(method, key):
    if method == "linear":
        return LinearScanIndex()
    if method == "kdtree":
        return KDTreeIndex()
    if method == "lsh":
        return RandomProjectionLSHIndex(**bench_cfg.LSH_PARAMS_BY_DATASET[key], random_seed=42)
    if method == "hecbr":
        return HeCBRIndex(bench_cfg.HECBR_CODES_PATH)
    raise ValueError(method)


def _rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--exp2", required=True)
    args = parser.parse_args()

    X_train, _, X_test, _, _, _ = adapters.get_split(args.dataset, seed=42)
    if args.method == "baseline":
        print(json.dumps({"dataset": args.dataset, "method": "baseline",
                          "peak_after_data_kb": int(_rss_kb())}))
        return

    index = factory_for(args.method, args.dataset)
    if args.method == "hecbr":
        index.build(None, restrict_split="train")
    else:
        index.build(X_train)
    peak_after_build_kb = int(_rss_kb())

    query = index.codes[0] if args.method == "hecbr" else X_test[0]
    rss_before = _rss_kb()
    tracemalloc.start()
    index.query(query, 10)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_kb()

    print(json.dumps({
        "dataset": args.dataset,
        "method": args.method,
        "peak_after_build_kb": peak_after_build_kb,
        "query_tracemalloc_peak_kb": round(peak_bytes / 1024.0, 2),
        "query_rss_delta_kb": int(rss_after - rss_before),
    }))


if __name__ == "__main__":
    main()
