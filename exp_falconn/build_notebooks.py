"""Generate the exp_falconn reference notebooks (cross-polytope run).

The experiment re-uses the exp_lsh pipeline structure but evaluates ONLY
the linear baseline and the new CrossPolytope (FALCONN-style) index on the
normalised cosine arm. SimHash / E2LSH are not re-run in this folder.

Notebooks:
    00_data_checks.ipynb   — dataset sanity / normalisation checks
    01_calibration.ipynb   — cross-polytope grid search on validation only
    02_benchmark.ipynb     — test benchmark (linear + cross-polytope)
    03_scaling.ipynb       — scaling analysis over base sizes
    04_analysis.ipynb      — statistical analysis and protocol figures
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

HEADER = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10.0"},
}


def nb(cells, metadata=None):
    meta = dict(HEADER)
    if metadata:
        meta.update(metadata)
    return {
        "cells": cells,
        "metadata": meta,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(True),
    }


# --------------------------------------------------------------------------- #
# 00 — Data checks
# --------------------------------------------------------------------------- #
def notebook_00():
    cells = [
        md(
            "# 00 — Data Checks (exp_falconn)\n"
            "\n"
            "Sanity checks on the L2-normalised cosine arm for both datasets.\n"
        ),
        code(
            "import numpy as np\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "root_dir = pathlib.Path().resolve().parent\n"
            "if str(root_dir) not in sys.path:\n"
            "    sys.path.append(str(root_dir))\n"
        ),
        code(
            "# ============================================================\n"
            "# CONFIG\n"
            "# ============================================================\n"
            "SPLIT_SEED = 42\n"
            "HEART_CSV_PATH = ''\n"
            "CHEST_EMBEDDINGS_PATH = ''\n"
            "CHEST_LABELS_PATH = ''\n"
        ),
        code(
            "from src.datasets.heart_disease import load_and_split as heart_load\n"
            "from src.datasets.chest_xray14 import load_and_split as chest_load\n"
            "\n"
            "H_tr, H_ytr, H_va, H_yva, H_te, H_yte = heart_load(HEART_CSV_PATH, split_seed=SPLIT_SEED, normalise_cosine=True)\n"
            "C_tr, C_ytr, C_va, C_yva, C_te, C_yte = chest_load(CHEST_EMBEDDINGS_PATH, CHEST_LABELS_PATH, split_seed=SPLIT_SEED, normalise_cosine=True)\n"
            "\n"
            "for name, X in [('heart-train', H_tr), ('heart-val', H_va), ('heart-test', H_te),\n"
            "                ('chest-train', C_tr), ('chest-val', C_va), ('chest-test', C_te)]:\n"
            "    norms = np.linalg.norm(X, axis=1)\n"
            "    print(f'{name}: shape={X.shape} mean_norm={norms.mean():.4f} std_norm={norms.std():.4f} nan={np.isnan(X).any()} inf={np.isinf(X).any()}')\n"
        ),
        code(
            "assert np.allclose(np.linalg.norm(H_tr, axis=1), 1.0, atol=1e-4)\n"
            "assert np.allclose(np.linalg.norm(C_tr, axis=1), 1.0, atol=1e-4)\n"
            "print('OK: all vectors L2-normalised (cosine arm).')\n"
        ),
    ]
    return nb(cells)


# --------------------------------------------------------------------------- #
# 01 — Calibration (cross-polytope only)
# --------------------------------------------------------------------------- #
def notebook_01():
    cells = [
        md(
            "# 01 — Cross-Polytope Calibration\n"
            "\n"
            "Grid search over `num_hashes`, `num_tables`, `num_probes` on the\n"
            "**validation set only** (protocol section 8). The test set is never\n"
            "touched here. Parametric: one Kaggle session per split seed.\n"
            "\n"
            "Method evaluated: **CrossPolytope** (FALCONN-style) only.\n"
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import time\n"
            "import json\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "try:\n"
            "    from tqdm.notebook import tqdm\n"
            "except ImportError:\n"
            "    import tqdm\n"
            "\n"
            "root_dir = pathlib.Path().resolve().parent\n"
            "if str(root_dir) not in sys.path:\n"
            "    sys.path.append(str(root_dir))\n"
        ),
        code(
            "# ============================================================\n"
            "# CONFIG — one Kaggle session per (seed, dataset, base_size)\n"
            "# ============================================================\n"
            "SPLIT_SEED = 42          # one of: 42, 7, 123, 2024, 99\n"
            "DATASET = 'heart'        # 'heart' or 'chest'\n"
            "BASE_SIZE = None         # None=full, 10_000, 50_000 (chest only)\n"
            "\n"
            "HEART_CSV_PATH = ''\n"
            "CHEST_EMBEDDINGS_PATH = ''\n"
            "CHEST_LABELS_PATH = ''\n"
            "# ============================================================\n"
        ),
        code(
            "from config.seeds import SPLIT_SEEDS, HASH_SEEDS\n"
            "from config.metrics import K_NEIGHBOURS\n"
            "\n"
            "assert SPLIT_SEED in SPLIT_SEEDS, f'Seed {SPLIT_SEED} not in {SPLIT_SEEDS}'\n"
            "HASH_SEED_CALIB = 11     # calibration uses a single hash seed (protocol)\n"
        ),
        code(
            "if DATASET == 'heart':\n"
            "    from src.datasets.heart_disease import load_and_split\n"
            "    X_base, y_base, X_val, y_val, X_test, y_test = load_and_split(\n"
            "        HEART_CSV_PATH, split_seed=SPLIT_SEED, normalise_cosine=True\n"
            "    )\n"
            "else:\n"
            "    from src.datasets.chest_xray14 import load_and_split\n"
            "    X_base, y_base, X_val, y_val, X_test, y_test = load_and_split(\n"
            "        CHEST_EMBEDDINGS_PATH, CHEST_LABELS_PATH,\n"
            "        split_seed=SPLIT_SEED, normalise_cosine=True,\n"
            "    )\n"
            "    if BASE_SIZE is not None:\n"
            "        idx = np.random.default_rng(SPLIT_SEED).choice(\n"
            "            len(X_base), size=min(BASE_SIZE, len(X_base)), replace=False\n"
            "        )\n"
            "        X_base = X_base[idx]\n"
            "        y_base = y_base[idx]\n"
            "\n"
            "print(f'X_base: {X_base.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')\n"
        ),
        code(
            "from src.indexes.linear_exact import LinearExact\n"
            "t0 = time.time()\n"
            "linear = LinearExact()\n"
            "linear.build(X_base)\n"
            "gt_val = []\n"
            "for x in X_val:\n"
            "    idxs, _ = linear.query(x, k=K_NEIGHBOURS)\n"
            "    gt_val.append(idxs)\n"
            "gt_val = np.array(gt_val)\n"
            "print(f'Linear ground truth for {len(X_val)} val queries in {time.time()-t0:.1f}s')\n"
        ),
        code(
            "from sklearn.metrics import f1_score\n"
            "\n"
            "def majority_vote(labels):\n"
            "    if len(labels) == 0:\n"
            "        return -1\n"
            "    values, counts = np.unique(labels, return_counts=True)\n"
            "    return values[np.argmax(counts)]\n"
            "\n"
            "lin_preds = [majority_vote(y_base[gt]) for gt in gt_val]\n"
            "linear_f1 = f1_score(y_val, lin_preds, average='macro', zero_division=0)\n"
            "print(f'Linear macro-F1 on val: {linear_f1:.4f}')\n"
        ),
        code(
            "from src.indexes.cross_polytope import CrossPolytope\n"
            "from src.evaluation.metrics import recall_at_k, candidate_fraction\n"
            "\n"
            "grid = {\n"
            "    'heart': {'num_hashes': [1, 2], 'num_tables': [5, 10, 20], 'num_rotations': [2], 'num_probes': [1, 4, 8]},\n"
            "    'chest': {'num_hashes': [1, 2], 'num_tables': [10, 20, 30], 'num_rotations': [2], 'num_probes': [1, 4, 8]},\n"
            "}\n"
            "param_grid = grid[DATASET]\n"
            "keys, values = zip(*param_grid.items())\n"
            "combos = [dict(zip(keys, v)) for v in __import__('itertools').product(*values)]\n"
        ),
        code(
            "cp_results = []\n"
            "for p in tqdm(combos, desc='CrossPolytope combos'):\n"
            "    cp = CrossPolytope(\n"
            "        num_hashes=p['num_hashes'],\n"
            "        num_tables=p['num_tables'],\n"
            "        num_rotations=p['num_rotations'],\n"
            "        num_probes=p['num_probes'],\n"
            "        hash_seed=HASH_SEED_CALIB,\n"
            "    )\n"
            "    t0 = time.time()\n"
            "    cp.build(X_base)\n"
            "    build_time = time.time() - t0\n"
            "\n"
            "    recalls, latencies, cands, preds = [], [], [], []\n"
            "    for i, q in enumerate(X_val):\n"
            "        t0 = time.perf_counter()\n"
            "        idxs, _, n_cand, _ = cp.query(q, k=K_NEIGHBOURS)\n"
            "        latencies.append(time.perf_counter() - t0)\n"
            "        recalls.append(recall_at_k(idxs, gt_val[i], K_NEIGHBOURS))\n"
            "        cands.append(candidate_fraction(n_cand, len(X_base)))\n"
            "        preds.append(majority_vote(y_base[idxs]))\n"
            "\n"
            "    r10 = float(np.mean(recalls))\n"
            "    f1 = f1_score(y_val, preds, average='macro', zero_division=0)\n"
            "    f1_drop = linear_f1 - f1\n"
            "    passed = (r10 >= 0.95) and (f1_drop <= 0.01)\n"
            "    cp_results.append({\n"
            "        'num_hashes': p['num_hashes'], 'num_tables': p['num_tables'],\n"
            "        'num_rotations': p['num_rotations'], 'num_probes': p['num_probes'],\n"
            "        'hash_seed': HASH_SEED_CALIB,\n"
            "        'recall@10': r10, 'macro_f1': f1, 'f1_drop': f1_drop,\n"
            "        'median_latency': float(np.median(latencies)),\n"
            "        'mean_cand_fraction': float(np.mean(cands)),\n"
            "        'build_time': build_time, 'passed': passed,\n"
            "    })\n"
            "\n"
            "cp_df = pd.DataFrame(cp_results)\n"
            "cp_df['dataset'] = DATASET\n"
            "cp_df['base_size'] = str(BASE_SIZE)\n"
            "cp_df['split_seed'] = SPLIT_SEED\n"
            "print(cp_df.sort_values(['passed', 'median_latency'], ascending=[False, True]).head(10))\n"
        ),
        code(
            "passing = cp_df[cp_df['passed']] if cp_df['passed'].any() else cp_df\n"
            "best_cp = passing.sort_values(['median_latency', 'mean_cand_fraction']).iloc[0]\n"
            "print('=== Best CrossPolytope ===')\n"
            "print(best_cp[['num_hashes', 'num_tables', 'num_probes', 'num_rotations',\n"
            "              'recall@10', 'macro_f1', 'median_latency', 'mean_cand_fraction']].to_dict())\n"
        ),
        code(
            "OUTPUT_DIR = pathlib.Path('results')\n"
            "OUTPUT_DIR.mkdir(exist_ok=True)\n"
            "cp_df.to_csv(OUTPUT_DIR / f'lsh_calibration_seed{SPLIT_SEED}.csv', index=False)\n"
            "\n"
            "best_params = {\n"
            "    'dataset': DATASET,\n"
            "    'base_size': str(BASE_SIZE),\n"
            "    'split_seed': SPLIT_SEED,\n"
            "    'crosspolytope': {\n"
            "        'num_hashes': int(best_cp['num_hashes']),\n"
            "        'num_tables': int(best_cp['num_tables']),\n"
            "        'num_rotations': int(best_cp['num_rotations']),\n"
            "        'num_probes': int(best_cp['num_probes']),\n"
            "        'hash_seed': int(best_cp['hash_seed']),\n"
            "    },\n"
            "}\n"
            "with open(OUTPUT_DIR / f'best_params_seed{SPLIT_SEED}.json', 'w') as f:\n"
            "    json.dump(best_params, f, indent=2)\n"
            "print(f'Saved calibration to {OUTPUT_DIR}/lsh_calibration_seed{SPLIT_SEED}.csv')\n"
        ),
    ]
    return nb(cells)


# --------------------------------------------------------------------------- #
# 02 — Benchmark (linear + cross-polytope only)
# --------------------------------------------------------------------------- #
def notebook_02():
    cells = [
        md(
            "# 02 — Test Benchmark (Cross-Polytope)\n"
            "\n"
            "Loads calibrated parameters from `01_calibration` and runs the test\n"
            "benchmark for the **linear baseline** and **CrossPolytope** across\n"
            "the 3 hash seeds. Test queries only — no parameter selection here.\n"
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import time\n"
            "import json\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "try:\n"
            "    from tqdm.notebook import tqdm\n"
            "except ImportError:\n"
            "    import tqdm\n"
            "\n"
            "root_dir = pathlib.Path().resolve().parent\n"
            "if str(root_dir) not in sys.path:\n"
            "    sys.path.append(str(root_dir))\n"
        ),
        code(
            "# ============================================================\n"
            "# CONFIG\n"
            "# ============================================================\n"
            "SPLIT_SEED = 42\n"
            "DATASET = 'heart'\n"
            "BASE_SIZE = None\n"
            "\n"
            "HEART_CSV_PATH = ''\n"
            "CHEST_EMBEDDINGS_PATH = ''\n"
            "CHEST_LABELS_PATH = ''\n"
            "CALIB_DIR = 'results'   # contains best_params_seed{SPLIT_SEED}.json\n"
            "# ============================================================\n"
        ),
        code(
            "from config.seeds import SPLIT_SEEDS, HASH_SEEDS\n"
            "from config.protocol import TIMING_WARMUP, TIMING_REPEATS\n"
            "from config.metrics import K_NEIGHBOURS\n"
            "assert SPLIT_SEED in SPLIT_SEEDS\n"
        ),
        code(
            "with open(pathlib.Path(CALIB_DIR) / f'best_params_seed{SPLIT_SEED}.json') as f:\n"
            "    best_params = json.load(f)\n"
            "cp_params = best_params['crosspolytope']\n"
            "print('Calibrated CrossPolytope params:', cp_params)\n"
        ),
        code(
            "if DATASET == 'heart':\n"
            "    from src.datasets.heart_disease import load_and_split\n"
            "    X_base, y_base, X_val, y_val, X_test, y_test = load_and_split(\n"
            "        HEART_CSV_PATH, split_seed=SPLIT_SEED, normalise_cosine=True\n"
            "    )\n"
            "else:\n"
            "    from src.datasets.chest_xray14 import load_and_split\n"
            "    X_base, y_base, X_val, y_val, X_test, y_test = load_and_split(\n"
            "        CHEST_EMBEDDINGS_PATH, CHEST_LABELS_PATH,\n"
            "        split_seed=SPLIT_SEED, normalise_cosine=True,\n"
            "    )\n"
            "    if BASE_SIZE is not None:\n"
            "        idx = np.random.default_rng(SPLIT_SEED).choice(\n"
            "            len(X_base), size=min(BASE_SIZE, len(X_base)), replace=False\n"
            "        )\n"
            "        X_base = X_base[idx]\n"
            "        y_base = y_base[idx]\n"
            "\n"
            "print(f'X_base: {X_base.shape}, X_test: {X_test.shape}')\n"
        ),
        code(
            "from src.indexes.linear_exact import LinearExact\n"
            "from src.indexes.cross_polytope import CrossPolytope\n"
            "from src.evaluation.metrics import (\n"
            "    recall_at_k, ndcg_at_k, candidate_fraction,\n"
            "    macro_f1, balanced_accuracy,\n"
            ")\n"
            "\n"
            "def majority_vote(labels):\n"
            "    if len(labels) == 0:\n"
            "        return -1\n"
            "    values, counts = np.unique(labels, return_counts=True)\n"
            "    return values[np.argmax(counts)]\n"
        ),
        code(
            "gt_path = pathlib.Path('results') / f'linear_gt_{DATASET}_{BASE_SIZE}_seed{SPLIT_SEED}.npz'\n"
            "lin_idx = LinearExact()\n"
            "lin_idx.build(X_base)\n"
            "\n"
            "gt_indices, gt_distances = [], []\n"
            "lin_lat, lin_preds = [], []\n"
            "for q in X_test:\n"
            "    t0 = time.perf_counter()\n"
            "    for _ in range(TIMING_WARMUP):\n"
            "        lin_idx.query(q, k=K_NEIGHBOURS)\n"
            "    times = []\n"
            "    for _ in range(TIMING_REPEATS):\n"
            "        t1 = time.perf_counter()\n"
            "        idxs, dists = lin_idx.query(q, k=K_NEIGHBOURS)\n"
            "        times.append(time.perf_counter() - t1)\n"
            "    lin_lat.append(float(np.median(times)))\n"
            "    gt_indices.append(idxs)\n"
            "    gt_distances.append(dists)\n"
            "    lin_preds.append(majority_vote(y_base[idxs]))\n"
            "\n"
            "np.savez(gt_path, indices=np.array(gt_indices), distances=np.array(gt_distances))\n"
            "print(f'Linear GT + timing done: median {np.median(lin_lat)*1e3:.2f} ms')\n"
        ),
        code(
            "quality_records, timing_records = [], []\n"
            "candidate_records, memory_records = [], []\n"
            "\n"
            "def add_linear_records():\n"
            "    quality_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': 'exact', 'method': 'linear',\n"
            "        'recall@1': 1.0, 'recall@5': 1.0, 'recall@10': 1.0, 'ndcg@10': 1.0,\n"
            "        'query_coverage': 1.0, 'empty_rate': 0.0,\n"
            "        'accuracy': float(np.mean(y_test == lin_preds)),\n"
            "        'macro_f1': macro_f1(y_test, lin_preds),\n"
            "        'balanced_accuracy': balanced_accuracy(y_test, lin_preds),\n"
            "    })\n"
            "    timing_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': 'exact', 'method': 'linear',\n"
            "        'median_latency_s': float(np.median(lin_lat)),\n"
            "        'mean_latency_s': float(np.mean(lin_lat)),\n"
            "        'p95_latency_s': float(np.percentile(lin_lat, 95)),\n"
            "        'build_time_s': lin_idx.build_time_s,\n"
            "    })\n"
            "    candidate_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': 'exact', 'method': 'linear',\n"
            "        'mean_candidates': float(len(X_base)), 'mean_cand_fraction': 1.0, 'mean_buckets_visited': 1.0,\n"
            "    })\n"
            "    memory_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': 'exact', 'method': 'linear', 'index_memory_bytes': X_base.nbytes,\n"
            "    })\n"
            "\n"
            "add_linear_records()\n"
        ),
        code(
            "for h_seed in tqdm(HASH_SEEDS, desc='CrossPolytope hash seeds'):\n"
            "    cp = CrossPolytope(\n"
            "        num_hashes=cp_params['num_hashes'],\n"
            "        num_tables=cp_params['num_tables'],\n"
            "        num_rotations=cp_params['num_rotations'],\n"
            "        num_probes=cp_params['num_probes'],\n"
            "        hash_seed=h_seed,\n"
            "    )\n"
            "    cp.build(X_base)\n"
            "\n"
            "    lat, r1, r5, r10, ndcg, cand, bkt, cov, preds = [], [], [], [], [], [], [], [], []\n"
            "    for i, q in enumerate(X_test):\n"
            "        t0 = time.perf_counter()\n"
            "        idxs, _, n_cand, n_b = cp.query(q, k=K_NEIGHBOURS)\n"
            "        lat.append(time.perf_counter() - t0)\n"
            "        r1.append(recall_at_k(idxs, gt_indices[i], 1))\n"
            "        r5.append(recall_at_k(idxs, gt_indices[i], 5))\n"
            "        r10.append(recall_at_k(idxs, gt_indices[i], 10))\n"
            "        ndcg.append(ndcg_at_k(idxs, gt_indices[i], 10))\n"
            "        cand.append(n_cand)\n"
            "        bkt.append(n_b)\n"
            "        cov.append(float(len(idxs) >= K_NEIGHBOURS))\n"
            "        preds.append(majority_vote(y_base[idxs]))\n"
            "\n"
            "    quality_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': h_seed, 'method': 'crosspolytope',\n"
            "        'recall@1': float(np.mean(r1)), 'recall@5': float(np.mean(r5)),\n"
            "        'recall@10': float(np.mean(r10)), 'ndcg@10': float(np.mean(ndcg)),\n"
            "        'query_coverage': float(np.mean(cov)), 'empty_rate': float(np.mean(np.array(cand) == 0)),\n"
            "        'accuracy': float(np.mean(y_test == preds)),\n"
            "        'macro_f1': macro_f1(y_test, preds),\n"
            "        'balanced_accuracy': balanced_accuracy(y_test, preds),\n"
            "    })\n"
            "    timing_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': h_seed, 'method': 'crosspolytope',\n"
            "        'median_latency_s': float(np.median(lat)),\n"
            "        'mean_latency_s': float(np.mean(lat)),\n"
            "        'p95_latency_s': float(np.percentile(lat, 95)),\n"
            "        'build_time_s': cp.build_time_s,\n"
            "    })\n"
            "    candidate_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': h_seed, 'method': 'crosspolytope',\n"
            "        'mean_candidates': float(np.mean(cand)),\n"
            "        'mean_cand_fraction': float(np.mean(cand)) / len(X_base),\n"
            "        'mean_buckets_visited': float(np.mean(bkt)),\n"
            "    })\n"
            "    memory_records.append({\n"
            "        'dataset': DATASET, 'base_size': str(BASE_SIZE), 'split_seed': SPLIT_SEED,\n"
            "        'hash_seed': h_seed, 'method': 'crosspolytope',\n"
            "        'index_memory_bytes': cp.estimate_memory_bytes(),\n"
            "    })\n"
        ),
        code(
            "OUTPUT_DIR = pathlib.Path('results')\n"
            "OUTPUT_DIR.mkdir(exist_ok=True)\n"
            "tag = f'{DATASET}_{BASE_SIZE}_seed{SPLIT_SEED}'\n"
            "\n"
            "pd.DataFrame(quality_records).to_csv(OUTPUT_DIR / f'benchmark_quality_{tag}.csv', index=False)\n"
            "pd.DataFrame(timing_records).to_csv(OUTPUT_DIR / f'benchmark_timing_{tag}.csv', index=False)\n"
            "pd.DataFrame(candidate_records).to_csv(OUTPUT_DIR / f'benchmark_candidates_{tag}.csv', index=False)\n"
            "pd.DataFrame(memory_records).to_csv(OUTPUT_DIR / f'benchmark_memory_{tag}.csv', index=False)\n"
            "print('Saved benchmark CSVs.')\n"
        ),
        code(
            "summary = pd.DataFrame(quality_records).merge(\n"
            "    pd.DataFrame(timing_records),\n"
            "    on=['dataset', 'base_size', 'split_seed', 'hash_seed', 'method'],\n"
            ")\n"
            "print(summary.groupby(['dataset', 'base_size', 'method'])[['recall@10', 'median_latency_s', 'macro_f1']].mean())\n"
        ),
    ]
    return nb(cells)


# --------------------------------------------------------------------------- #
# 03 — Scaling analysis
# --------------------------------------------------------------------------- #
def notebook_03():
    cells = [
        md("# 03 — Scaling Analysis (ChestX-ray14, Cross-Polytope)\n"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import pathlib\n"
            "import glob\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "sns.set_theme(style='whitegrid')\n"
        ),
        code(
            "BENCHMARK_DIR = pathlib.Path('../results/benchmark')\n"
            "FIGURES_DIR = pathlib.Path('../results/figures')\n"
            "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        ),
        code(
            "def load_and_merge(prefix):\n"
            "    files = sorted(BENCHMARK_DIR.glob(f'{prefix}_chest_*_seed*.csv'))\n"
            "    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()\n"
            "\n"
            "df_qual = load_and_merge('benchmark_quality')\n"
            "df_time = load_and_merge('benchmark_timing')\n"
            "df_cand = load_and_merge('benchmark_candidates')\n"
            "df_mem = load_and_merge('benchmark_memory')\n"
            "print(df_qual['base_size'].unique() if not df_qual.empty else 'no data')\n"
        ),
        code(
            "if not df_qual.empty and not df_time.empty and not df_cand.empty:\n"
            "    merge_keys = ['dataset', 'base_size', 'split_seed', 'hash_seed', 'method']\n"
            "    df_merged = (df_qual.merge(df_time, on=merge_keys)\n"
            "                         .merge(df_cand, on=merge_keys)\n"
            "                         .merge(df_mem, on=merge_keys))\n"
            "    df_agg = df_merged.groupby(['base_size', 'method'])[\n"
            "        ['recall@10', 'median_latency_s', 'mean_cand_fraction', 'build_time_s']\n"
            "    ].mean().reset_index()\n"
            "    print(df_agg)\n"
        ),
        code(
            "if not df_agg.empty:\n"
            "    plt.figure(figsize=(9, 6))\n"
            "    order = sorted(df_agg['base_size'].astype(float).unique())\n"
            "    sns.lineplot(data=df_agg, x='base_size', y='median_latency_s', hue='method', marker='o')\n"
            "    plt.yscale('log')\n"
            "    plt.title('Median query latency vs base size (ChestX-ray14)')\n"
            "    plt.xlabel('Base size'); plt.ylabel('Median latency (s)')\n"
            "    plt.tight_layout(); plt.savefig(FIGURES_DIR / 'fig7_scaling_latency.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "if not df_agg.empty:\n"
            "    speedup = df_agg[df_agg['method'] != 'linear'].merge(\n"
            "        df_agg[df_agg['method'] == 'linear'][['base_size', 'median_latency_s']],\n"
            "        on='base_size', suffixes=('', '_lin')\n"
            "    )\n"
            "    speedup['speedup'] = speedup['median_latency_s_lin'] / speedup['median_latency_s']\n"
            "    print(speedup[['base_size', 'method', 'speedup', 'recall@10']])\n"
        ),
        md(
            "### Scaling Summary\n"
            "\n"
            "1. **Latency scaling**: CrossPolytope query cost grows with the candidate\n"
            "   set, which is a *fraction* of the base — sublinear vs the O(n·d) linear scan.\n"
            "2. **Recall**: must stay >= 0.95 (calibration constraint) across base sizes.\n"
            "3. **Candidate fraction**: the key trade-off — smaller candidates ⇒ faster queries.\n"
        ),
    ]
    return nb(cells)


# --------------------------------------------------------------------------- #
# 04 — Statistical analysis & figures
# --------------------------------------------------------------------------- #
def notebook_04():
    cells = [
        md(
            "# 04 — Statistical Analysis & Figures (Cross-Polytope)\n"
            "\n"
            "Combines the phase-1 (SimHash / E2LSH) results with the new\n"
            "cross-polytope run to produce the protocol figures and outcome\n"
            "classification (protocol section 12).\n"
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scipy.stats as stats\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import json\n"
            "from pathlib import Path\n"
            "sns.set_theme(style='whitegrid')\n"
        ),
        code(
            "BENCHMARK_DIR = Path('../results/benchmark')\n"
            "FIG_DIR = Path('../results/figures')\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "def load_csvs(pattern):\n"
            "    files = sorted(BENCHMARK_DIR.glob(pattern))\n"
            "    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()\n"
            "\n"
            "df_quality = load_csvs('benchmark_quality_*seed*.csv')\n"
            "df_timing = load_csvs('benchmark_timing_*seed*.csv')\n"
            "df_cand = load_csvs('benchmark_candidates_*seed*.csv')\n"
            "df_mem = load_csvs('benchmark_memory_*seed*.csv')\n"
        ),
        code(
            "merge_keys = ['dataset', 'base_size', 'split_seed', 'hash_seed', 'method']\n"
            "df_merged = (df_quality.merge(df_timing, on=merge_keys)\n"
            "                        .merge(df_cand, on=merge_keys)\n"
            "                        .merge(df_mem, on=merge_keys))\n"
            "df_agg = df_merged.groupby(['dataset', 'base_size', 'split_seed', 'method']).mean(numeric_only=True).reset_index()\n"
            "print(df_agg.groupby(['dataset', 'base_size', 'method'])[['recall@10', 'median_latency_s', 'macro_f1']].mean())\n"
        ),
        md(
            "### Statistical Tests\n"
            "With only 5 splits, Wilcoxon p-values are interpreted cautiously.\n"
            "We rely on absolute differences and 95% bootstrap CIs of the\n"
            "paired difference vs linear.\n"
        ),
        code(
            "def bootstrap_ci(differences, n_bootstrap=1000, alpha=0.05, seed=42):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    boot = [rng.choice(differences, size=len(differences), replace=True).mean()\n"
            "            for _ in range(n_bootstrap)]\n"
            "    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))\n"
        ),
        code(
            "sig_rows = []\n"
            "for ds in df_agg['dataset'].unique():\n"
            "    for bs in df_agg[df_agg['dataset'] == ds]['base_size'].unique():\n"
            "        sub = df_agg[(df_agg['dataset'] == ds) & (df_agg['base_size'] == bs)]\n"
            "        lin = sub[sub['method'] == 'linear']\n"
            "        for method in ['crosspolytope', 'SimHash', 'E2LSH']:\n"
            "            m = sub[sub['method'] == method]\n"
            "            if lin.empty or m.empty:\n"
            "                continue\n"
            "            diff = m['median_latency_s'].values - lin['median_latency_s'].values\n"
            "            lo, hi = bootstrap_ci(diff)\n"
            "            if len(diff) >= 5:\n"
            "                _, p = stats.wilcoxon(diff)\n"
            "            else:\n"
            "                p = float('nan')\n"
            "            sig_rows.append({'dataset': ds, 'base_size': bs, 'method': method,\n"
            "                             'mean_diff_latency_s': float(np.mean(diff)),\n"
            "                             'ci_low': lo, 'ci_high': hi, 'p_wilcoxon': p})\n"
            "sig_df = pd.DataFrame(sig_rows)\n"
            "if not sig_df.empty:\n"
            "    sig_df.to_csv(BENCHMARK_DIR / 'significance_tests.csv', index=False)\n"
            "    print(sig_df)\n"
        ),
        code(
            "# Figure 1: latency comparison\n"
            "if not df_agg.empty:\n"
            "    df_agg['ds_bs'] = df_agg['dataset'] + '_' + df_agg['base_size'].astype(str)\n"
            "    plt.figure(figsize=(11, 6))\n"
            "    sns.barplot(data=df_agg, x='ds_bs', y='median_latency_s', hue='method', errorbar=('ci', 95))\n"
            "    plt.yscale('log')\n"
            "    plt.title('Figure 1: Median query latency')\n"
            "    plt.xticks(rotation=30)\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig1_latency.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Figure 2: Recall@k\n"
            "if not df_agg.empty:\n"
            "    df_melt = df_agg.melt(id_vars=['ds_bs', 'method'], value_vars=['recall@1', 'recall@5', 'recall@10'])\n"
            "    plt.figure(figsize=(11, 6))\n"
            "    sns.boxplot(data=df_melt, x='ds_bs', y='value', hue='method')\n"
            "    plt.title('Figure 2: Recall@k')\n"
            "    plt.xticks(rotation=30)\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig2_recall.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Figure 3: candidate fraction\n"
            "if not df_cand.empty:\n"
            "    df_c = df_cand[df_cand['method'] != 'linear']\n"
            "    df_c = df_cand[df_cand['method'] == 'crosspolytope']\n"
            "    if not df_c.empty:\n"
            "        df_c['ds_bs'] = df_c['dataset'] + '_' + df_c['base_size'].astype(str)\n"
            "        plt.figure(figsize=(11, 6))\n"
            "        sns.boxplot(data=df_c, x='ds_bs', y='mean_cand_fraction')\n"
            "        plt.title('Figure 3: Candidate fraction (cross-polytope)')\n"
            "        plt.xticks(rotation=30)\n"
            "        plt.tight_layout(); plt.savefig(FIG_DIR / 'fig3_candidates.png', dpi=150)\n"
            "        plt.show()\n"
        ),
        code(
            "# Figure 4: build time\n"
            "if not df_timing.empty:\n"
            "    df_t = df_timing.copy()\n"
            "    df_t['ds_bs'] = df_t['dataset'] + '_' + df_t['base_size'].astype(str)\n"
            "    plt.figure(figsize=(11, 6))\n"
            "    sns.barplot(data=df_t, x='ds_bs', y='build_time_s', hue='method', errorbar=('ci', 95))\n"
            "    plt.title('Figure 4: Index build time')\n"
            "    plt.xticks(rotation=30)\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig4_build.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Figure 5: memory\n"
            "if not df_mem.empty:\n"
            "    df_m = df_mem.copy()\n"
            "    df_m['ds_bs'] = df_m['dataset'] + '_' + df_m['base_size'].astype(str)\n"
            "    df_m['mem_mb'] = df_m['index_memory_bytes'] / 1e6\n"
            "    plt.figure(figsize=(11, 6))\n"
            "    sns.barplot(data=df_m, x='ds_bs', y='mem_mb', hue='method', errorbar=('ci', 95))\n"
            "    plt.title('Figure 5: Index memory (MB)')\n"
            "    plt.xticks(rotation=30)\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig5_memory.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Figure 6: recall-latency trade-off\n"
            "if not df_agg.empty:\n"
            "    plt.figure(figsize=(9, 6))\n"
            "    sns.scatterplot(data=df_agg, x='median_latency_s', y='recall@10', hue='method', style='ds_bs', s=100)\n"
            "    plt.xscale('log')\n"
            "    plt.title('Figure 6: Recall@10 vs latency')\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig6_tradeoff.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Figure 7: scaling (chest)\n"
            "chest = df_agg[df_agg['dataset'] == 'chest']\n"
            "if not chest.empty:\n"
            "    plt.figure(figsize=(9, 6))\n"
            "    chest['bs'] = chest['base_size'].astype(float)\n"
            "    sns.lineplot(data=chest, x='bs', y='median_latency_s', hue='method', marker='o')\n"
            "    plt.xscale('log'); plt.yscale('log')\n"
            "    plt.title('Figure 7: Latency scaling (chest)')\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig7_scaling.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Figure 8: hash-seed variance (cross-polytope)\n"
            "df_cp = df_merged[df_merged['method'] == 'crosspolytope']\n"
            "if not df_cp.empty:\n"
            "    df_cp['ds_bs'] = df_cp['dataset'] + '_' + df_cp['base_size'].astype(str)\n"
            "    plt.figure(figsize=(11, 6))\n"
            "    sns.boxplot(data=df_cp, x='ds_bs', y='recall@10', hue='hash_seed')\n"
            "    plt.title('Figure 8: Recall@10 by hash seed (cross-polytope)')\n"
            "    plt.xticks(rotation=30)\n"
            "    plt.tight_layout(); plt.savefig(FIG_DIR / 'fig8_hashseed.png', dpi=150)\n"
            "    plt.show()\n"
        ),
        code(
            "# Outcome classification (protocol section 12)\n"
            "outcomes = []\n"
            "for ds in df_agg['dataset'].unique():\n"
            "    for bs in df_agg[df_agg['dataset'] == ds]['base_size'].unique():\n"
            "        sub = df_agg[(df_agg['dataset'] == ds) & (df_agg['base_size'] == bs)]\n"
            "        lin = sub[sub['method'] == 'linear']\n"
            "        if lin.empty:\n"
            "            continue\n"
            "        lin_lat = lin['median_latency_s'].mean()\n"
            "        lin_f1 = lin['macro_f1'].mean()\n"
            "        for method in ['crosspolytope']:\n"
            "            m = sub[sub['method'] == method]\n"
            "            if m.empty:\n"
            "                continue\n"
            "            r10 = m['recall@10'].mean()\n"
            "            f1 = m['macro_f1'].mean()\n"
            "            lat = m['median_latency_s'].mean()\n"
            "            if r10 >= 0.95 and (lin_f1 - f1) <= 0.01:\n"
            "                verdict = 'speedup' if lat < lin_lat * 0.9 else 'quality_ok_no_speedup'\n"
            "            else:\n"
            "                verdict = 'quality_fail'\n"
            "            outcomes.append({'dataset': ds, 'base_size': bs, 'method': method, 'verdict': verdict,\n"
            "                             'recall@10': r10, 'macro_f1': f1, 'latency_s': lat})\n"
            "out_df = pd.DataFrame(outcomes)\n"
            "out_df.to_csv(BENCHMARK_DIR / 'outcomes.csv', index=False)\n"
            "print(out_df)\n"
        ),
        md(
            "### Conclusion\n"
            "Based on the outcomes classification: whether CrossPolytope LSH\n"
            "provides a consistent speedup with recall >= 0.95, conditionally\n"
            "works, or fails in quality. Results are compared against phase-1\n"
            "(SimHash / E2LSH) in `lsh_sota_review.md` and `progress.md`.\n"
        ),
    ]
    return nb(cells)


def main():
    for name, builder in [
        ("00_data_checks", notebook_00),
        ("01_calibration", notebook_01),
        ("02_benchmark", notebook_02),
        ("03_scaling", notebook_03),
        ("04_analysis", notebook_04),
    ]:
        path = OUT / f"{name}.ipynb"
        with open(path, "w") as f:
            json.dump(builder(), f, indent=1)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
