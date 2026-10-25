"""E2 — Condensation « code-bucket » à l'échelle (HeCBR data-dependent).

Deux volets :
- TIER A (comparatif, n=10⁴) : mêmes stratégies que exp2 T3.2 mais en espace code :
  random / cbm (couverture relative exacte dans l'espace code) / code-bucket
  (représentant par code identique, puis boule Hamming r>0) / cnn (échelle exacte
  sur l'espace code). Taux cibles 0.2 / 0.5.
- TIER B (échelle, n=1M) : condensation code-bucket sur la base complète,
  courbe macro-F1 vs taux, latence et mémoire après maintenance vs brute.

Évaluation : toujours sur le test stratifié 50/50 (300 requêtes). Contrôle de
cohérence par F1 de référence hecbr_scan ~0.5402.

Sorties : results/raw/E2_condensation.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import HECBR_CODES_V4_NPZ, RESULTS_RAW_DIR
from config.seeds import SPLIT_SEED
from indexes.hecbr_index import _popcount64
from scripts.maintenance_common import (
    assemble_labels_full, to_u64, eval_f1_hamming, condense_code_bucket,
    iter_slide_arrays, K,
)

REF_F1_SCAN = 0.5401810979060555

TIER_A_SUBSAMPLE = 10_000
TIER_A_RATES = [0.2, 0.5]
TIER_B_RADII = [0, 1, 2]


# ---------------------------------------------------------------------------
# Stratégies de condensation (dans l'espace des codes)
# ---------------------------------------------------------------------------
def random_condense(codes_u64, y, size):
    idx = np.random.default_rng(SPLIT_SEED + 3).choice(len(codes_u64), size=size, replace=False)
    return codes_u64[idx], y[idx]


def cbm_condense(codes_u64, y, size, coverage_radius):
    """CBM exact (couverture relative) dans l'espace code — O(N²) borné (tier A)."""
    n = len(codes_u64)
    # matrice de couverture : case i couvre j si d_H(i,j) <= coverage_radius
    # couverture relative : nombre de cas couverts (reachability)
    coverage = np.zeros(n, dtype=np.int64)
    for i in range(n):
        hd = _popcount64(np.bitwise_xor(codes_u64, codes_u64[i]))
        coverage[i] = int(np.sum(hd <= coverage_radius)) - 1   # sans soi-même
    order = np.argsort(-coverage)
    return codes_u64[order[:size]], y[order[:size]]


def cnn_condense(codes_u64, y, size, radius_budget=None):
    """Condensation des PP les plus proches (Hart) sur espace code — greedy."""
    codes = list(codes_u64)
    keep = []
    rng = np.random.default_rng(SPLIT_SEED + 9)
    order = rng.permutation(len(codes))
    kept_u64 = []
    y_kept = []
    for j in order:
        c = codes_u64[j]
        if len(kept_u64) == 0:
            kept_u64.append(c); y_kept.append(y[j])
        else:
            hd = _popcount64(np.bitwise_xor(kept_u64, c))
            if np.min(hd) > 0:          # pas de doublon exact
                kept_u64.append(c); y_kept.append(y[j])
    arr = np.array(kept_u64, dtype=np.uint64)
    yy = np.array(y_kept)
    if len(arr) > size:
        idx = np.random.default_rng(SPLIT_SEED + 9).choice(len(arr), size=size, replace=False)
        return arr[idx], yy[idx]
    return arr, yy


def build_and_f1(codes, y, codes_eval_u64, y_eval, k=K):
    f1, lat_ms, _ = eval_f1_hamming(codes, y, codes_eval_u64, y_eval, k)
    return f1, lat_ms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier-a", action="store_true", help="Exécute le volet comparatif n=10⁴")
    p.add_argument("--tier-b", action="store_true", help="Exécute le volet échelle n=1M")
    p.add_argument("--no-cache-labels", action="store_true")
    args = p.parse_args()
    if not (args.tier_a or args.tier_b):
        args.tier_a = args.tier_b = True

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> Données/labels (sans X en RAM) ...", flush=True)
    meta = {sid: (path, label, n) for sid, path, label, n in iter_slide_arrays()}
    sids = sorted(meta.keys())
    slide_labels = {sid: t[1] for sid, t in meta.items()}
    sizes = {sid: t[2] for sid, t in meta.items()}
    y_base, y_eval, keep, chosen = assemble_labels_full(sids, slide_labels, sizes)

    print(">>> Codes v4 en cache ...", flush=True)
    data = np.load(HECBR_CODES_V4_NPZ)
    codes_base = to_u64(data["codes_base"])
    codes_eval = to_u64(data["codes_eval"])

    f1_ref, _, _ = eval_f1_hamming(codes_base, y_base, codes_eval, y_eval, K)
    ok = abs(f1_ref - REF_F1_SCAN) < 2e-3
    print(f"    F1 scan (réf.) = {f1_ref:.4f} -> {'OK' if ok else 'ÉCHEC'}", flush=True)
    if not ok:
        sys.exit("Arrêt : désalignement labels/codes.")

    records = []

    if args.tier_a:
        print(">>> TIER A : condensation comparative sur n=10⁴ ...", flush=True)
        rng = np.random.default_rng(SPLIT_SEED + 7)
        sub = rng.choice(len(codes_base), size=TIER_A_SUBSAMPLE, replace=False)
        codes_sub, y_sub = codes_base[sub], y_base[sub]

        # brut sur le sous-échantillon (référence)
        f1_sub, lat_sub = build_and_f1(codes_sub, y_sub, codes_eval, y_eval)
        records.append({"tier": "A", "method": "raw", "rate_target": 1.0,
                        "n_orig": TIER_A_SUBSAMPLE, "n_cond": TIER_A_SUBSAMPLE,
                        "achieved_rate": 1.0, "f1_macro": f1_sub,
                        "query_median_ms": lat_sub, "build_s": 0.0})

        # taux cibles
        for rate in TIER_A_RATES:
            size = max(1, int(rate * TIER_A_SUBSAMPLE))
            # random
            t0 = time.perf_counter()
            c_r, y_r = random_condense(codes_sub, y_sub, size)
            bld = time.perf_counter() - t0
            f1_r, lat_r = build_and_f1(c_r, y_r, codes_eval, y_eval)
            records.append({"tier": "A", "method": "random", "rate_target": rate,
                            "n_orig": TIER_A_SUBSAMPLE, "n_cond": len(c_r),
                            "achieved_rate": len(c_r) / TIER_A_SUBSAMPLE,
                            "f1_macro": f1_r, "query_median_ms": lat_r, "build_s": bld})
            # cbm (exact, O(N²) sur 10⁴)
            t0 = time.perf_counter()
            c_c, y_c = cbm_condense(codes_sub, y_sub, size, coverage_radius=2)
            bld = time.perf_counter() - t0
            f1_c, lat_c = build_and_f1(c_c, y_c, codes_eval, y_eval)
            records.append({"tier": "A", "method": "cbm", "rate_target": rate,
                            "n_orig": TIER_A_SUBSAMPLE, "n_cond": len(c_c),
                            "achieved_rate": len(c_c) / TIER_A_SUBSAMPLE,
                            "f1_macro": f1_c, "query_median_ms": lat_c, "build_s": bld})
            # cnn
            t0 = time.perf_counter()
            c_n, y_n = cnn_condense(codes_sub, y_sub, size)
            bld = time.perf_counter() - t0
            f1_n, lat_n = build_and_f1(c_n, y_n, codes_eval, y_eval)
            records.append({"tier": "A", "method": "cnn", "rate_target": rate,
                            "n_orig": TIER_A_SUBSAMPLE, "n_cond": len(c_n),
                            "achieved_rate": len(c_n) / TIER_A_SUBSAMPLE,
                            "f1_macro": f1_n, "query_median_ms": lat_n, "build_s": bld})
        # code-bucket à taux naturels (r=0, r=1, r=2)
        for r in [0, 1, 2]:
            t0 = time.perf_counter()
            c_b, y_b, _nret = condense_code_bucket(codes_sub, y_sub, radius=r)
            bld = time.perf_counter() - t0
            f1_b, lat_b = build_and_f1(c_b, y_b, codes_eval, y_eval)
            records.append({"tier": "A", "method": f"code_bucket_r{r}", "rate_target": None,
                            "n_orig": TIER_A_SUBSAMPLE, "n_cond": len(c_b),
                            "achieved_rate": len(c_b) / TIER_A_SUBSAMPLE,
                            "f1_macro": f1_b, "query_median_ms": lat_b, "build_s": bld})

    if args.tier_b:
        print(">>> TIER B : condensation sur base complète n=1M ...", flush=True)
        # brute référence
        f1_br, lat_br = build_and_f1(codes_base, y_base, codes_eval, y_eval)
        mem_bytes = codes_base.nbytes
        records.append({"tier": "B", "method": "raw", "rate_target": 1.0,
                        "n_orig": len(codes_base), "n_cond": len(codes_base),
                        "achieved_rate": 1.0, "f1_macro": f1_br,
                        "query_median_ms": lat_br, "build_s": 0.0,
                        "mem_MB": mem_bytes / 1e6})
        for r in TIER_B_RADII:
            t0 = time.perf_counter()
            c_b, y_b, nret = condense_code_bucket(codes_base, y_base, radius=r)
            bld = time.perf_counter() - t0
            f1_b, lat_b = build_and_f1(c_b, y_b, codes_eval, y_eval)
            records.append({"tier": "B", "method": f"code_bucket_r{r}", "rate_target": None,
                            "n_orig": len(codes_base), "n_cond": len(c_b),
                            "achieved_rate": len(c_b) / len(codes_base),
                            "f1_macro": f1_b, "query_median_ms": lat_b, "build_s": bld,
                            "mem_MB": c_b.nbytes / 1e6})
            print(f"    r={r} : n_cond={len(c_b)} ({100*len(c_b)/len(codes_base):.1f}%) "
                  f"F1={f1_b:.4f} lat={lat_b:.2f}ms build={bld:.2f}s", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(RESULTS_RAW_DIR / "E2_condensation.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()