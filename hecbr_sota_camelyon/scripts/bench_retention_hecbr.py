"""E1 — Rétention par distance Hamming à l'échelle (HeCBR scan exact).

Objectif : décider à l'arrivée de chaque nouveau cas s'il est REDONDANT (distance
Hamming au plus proche code <= theta_H) et ne pas le retenir. Mêmes métriques que
exp2 T3.1, mais dans l'espace des codes appris (64 bits) au lieu du LSH aléatoire.

Définition du flux : de VRAIS nouveaux cas — des patches de lames trainval NON
indexées (hors keep 1M), encodés lame par lame (mémoire bornée) avec le modèle v4
et un point d'ancrage mu/sd estimé sur un échantillon de la base.

Évaluation : requêtes de test stratifiées 50/50 (300). Contrôle de cohérence :
F1 de référence hecbr_scan ~0.5402 sur les codes en cache — si échec, arrêt
(chiffres faux refusés).

Sortie : results/raw/E1_retention_hecbr.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import HECBR_CODES_V4_NPZ, RESULTS_RAW_DIR, HECBR_MODEL_PATH
from config.seeds import SPLIT_SEED
from data.splitter import split_slides
from indexes.hecbr_index import _popcount64
from scripts.train_hecbr_v4 import HashNetV3
from scripts.maintenance_common import (
    assemble_labels_full, to_u64, eval_f1_hamming, iter_slide_arrays,
    load_slide_array, K,
)

REF_F1_SCAN = 0.5401810979060555
FLOW_N = 100_000
THRESH_SAMPLE = 2_000
PERCENTILE = 25
FLOW_SEEDS = [42, 7, 123]
EVAL_N = 300


def load_encode_model(device="cpu"):
    import torch
    ck = torch.load(HECBR_MODEL_PATH, map_location=device)
    model = HashNetV3(in_dim=ck.get("in_dim", 512), hidden=ck.get("hidden_dim", 512),
                      bit_length=ck.get("bit_length", 64))
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model


def encode_arr(model, X, mu, sd, device="cpu", bs=8192):
    import torch
    out = []
    for i in range(0, len(X), bs):
        x = torch.tensor((X[i:i + bs] - mu) / sd, dtype=torch.float32).to(device)
        with torch.no_grad():
            u = model(x)
            out.append((u >= 0).cpu().numpy())
        del x
    return np.concatenate(out).astype(np.uint8)


def encode_anchor(model, anchor_meta, bs=8192, device="cpu"):
    """mu/sd estimés sur un échantillon réparti de lames base (ancrage du flux)."""
    rng = np.random.default_rng(SPLIT_SEED + 5)
    chunks = []
    for path, _label, _n in anchor_meta:
        X = load_slide_array(path)
        sub = rng.choice(len(X), size=min(bs, len(X)), replace=False)
        chunks.append(X[sub].astype(np.float32))
        del X
    Xa = np.concatenate(chunks)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    del Xa
    return mu, sd


def _d_orig_slice(flow_u64, sl_start, sl_stop, base_u64):
    """Distance Hamming 1-NN de flow[sl_start:sl_stop] à la base (scan exact, blocs)."""
    FC, BC = 2048, 32768
    sl = slice(sl_start, sl_stop)
    flow = flow_u64[sl]
    d = np.full(flow.shape[0], 64, dtype=np.int32)
    for s in range(0, len(base_u64), BC):
        blk = base_u64[s:s + BC]
        for f0 in range(0, len(flow), FC):
            q = flow[f0:f0 + FC]
            hd = _popcount64(np.bitwise_xor(blk[None, :], q[:, None])).min(axis=1)
            d[f0:f0 + FC] = np.minimum(d[f0:f0 + FC], hd)
        del blk
    return sl_start, d


# Globals du pool (hérités par fork, pas de pickle des gros arrays).
_POOL_FLOW = None
_POOL_BASE = None


def _d_orig_pool_task(tup):
    start, stop = tup
    return _d_orig_slice(_POOL_FLOW, start, stop, _POOL_BASE)


def precompute_d_orig(flow_codes_u64, codes_base_u64, n_proc=None):
    """Précalcul parallèle de d_orig (distance à la base d'origine), flux découpé."""
    import multiprocessing as mp
    global _POOL_FLOW, _POOL_BASE
    n_flow = len(flow_codes_u64)
    n_workers = n_proc or mp.cpu_count()
    n_workers = max(1, min(n_workers, 8))
    flow = flow_codes_u64
    base = codes_base_u64
    if n_flow < n_workers * 2 or n_workers == 1:
        _, d = _d_orig_slice(flow, 0, n_flow, base)
        return d
    chunks = np.array_split(np.arange(n_flow), n_workers)
    ranges = [(int(c[0]), int(c[-1]) + 1) for c in chunks if len(c) > 0]
    ctx = mp.get_context("fork")
    _POOL_FLOW = flow
    _POOL_BASE = base
    # Pool déclaré en global pour garder l'état au-delà du fork.
    global _POOL
    _POOL = ctx.Pool(n_workers)
    try:
        results = _POOL.map(_d_orig_pool_task, ranges)
    finally:
        _POOL.close(); _POOL.join()
    d_orig = np.full(n_flow, 64, dtype=np.int32)
    for start, d in results:
        d_orig[start:start + len(d)] = d
    return d_orig


def simulate_flow(codes_base_u64, y_base, flow_codes_u64, flow_y, theta,
                  codes_eval_u64, y_eval, regime="retention",
                  k=K, n_proc=None, d_orig=None):
    """Décision séquentielle exacte de rétention pour un régime donné.

    Règle pour retention/ret_cond : retenir un cas ssi d_H(1-NN) > theta.
    Pour add_all : tout retenir (pas de filtrage).

    theta == 0 → détection exacte des doublons par set (O(N), quasi instantané).
    theta > 0 → via d_orig précalcul + min incrémental.
    """
    n_flow = len(flow_codes_u64)
    if regime == "add_all":
        # Tout retenir, aucun rejet
        final_codes = np.concatenate([codes_base_u64, flow_codes_u64])
        final_y = np.concatenate([y_base, flow_y])
        f1, lat_ms, _ = eval_f1_hamming(final_codes, final_y, codes_eval_u64, y_eval, k)
        return {"n_in": n_flow, "base_orig": len(codes_base_u64),
                "n_retained": n_flow, "n_rejected": 0,
                "reduction_pct": 0.0, "f1_macro": f1, "query_median_ms": lat_ms,
                "theta_h": 0}

    if theta <= 0:
        print(f"      [{regime}] theta=0 -> filtre doublons exacts (set) ...", flush=True)
        base_set = set(int(c) for c in codes_base_u64)
        retained_u64 = []
        retained_y = []
        n_rejected = 0
        for j in range(n_flow):
            c = int(flow_codes_u64[j])
            if c in base_set:
                n_rejected += 1
            else:
                base_set.add(c)
                retained_u64.append(c)
                retained_y.append(int(flow_y[j]))
    else:
        if d_orig is None:
            print(f"      [{regime}] précalc d_orig ({n_flow} × {len(codes_base_u64)}) ...", flush=True)
            d_orig = precompute_d_orig(flow_codes_u64, codes_base_u64, n_proc)
        print(f"      [{regime}] parcours séquentiel ({n_flow} cas) ...", flush=True)
        retained_u64 = []
        retained_y = []
        n_rejected = 0
        for j in range(n_flow):
            q = flow_codes_u64[j]
            d = int(d_orig[j])
            if len(retained_u64) > 0:
                hd_new = _popcount64(np.bitwise_xor(np.array(retained_u64, dtype=np.uint64), q))
                d = min(d, int(np.min(hd_new)))
            if d <= theta:
                n_rejected += 1
            else:
                retained_u64.append(q)
                retained_y.append(flow_y[j])
    final_codes = np.concatenate([codes_base_u64,
                                  np.array(retained_u64, dtype=np.uint64)])
    final_y = np.concatenate([y_base, np.array(retained_y, dtype=np.int64)])
    f1, lat_ms, _ = eval_f1_hamming(final_codes, final_y, codes_eval_u64, y_eval, k)
    n_ret = len(retained_u64)
    return {"n_in": n_flow, "base_orig": len(codes_base_u64),
            "n_retained": n_ret, "n_rejected": n_rejected,
            "reduction_pct": 100.0 * n_rejected / n_flow,
            "f1_macro": f1, "query_median_ms": lat_ms,
            "theta_h": theta}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-n", type=int, default=FLOW_N)
    p.add_argument("--theta-percentile", type=int, default=PERCENTILE)
    p.add_argument("--seeds", type=int, nargs="*", default=FLOW_SEEDS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-proc", type=int, default=None,
                   help="Processus pour le précalcul d_orig (défaut: cpu_count)")
    args = p.parse_args()

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> Métadonnées lames (listage + taille, mémoire bornée) ...", flush=True)
    meta = {sid: (path, label, n) for sid, path, label, n in iter_slide_arrays()}
    sids = sorted(meta.keys())
    slide_labels = {sid: t[1] for sid, t in meta.items()}
    sizes = {sid: t[2] for sid, t in meta.items()}
    base_ids, val_ids, test_ids = split_slides(sids, slide_labels, seed=SPLIT_SEED)
    print(f"    {len(sids)} lames | base={len(base_ids)} val={len(val_ids)} "
          f"test={len(test_ids)}", flush=True)

    print(">>> Labels de référence (même rng que benchmark) ...", flush=True)
    y_base, y_eval, keep, chosen = assemble_labels_full(sids, slide_labels, sizes)
    print(f"    y_base={y_base.shape} y_eval={y_eval.shape} "
          f"pos={int((y_eval==1).sum())} neg={int((y_eval==0).sum())}", flush=True)

    print(">>> Codes v4 en cache (base indexée 1M) ...", flush=True)
    data = np.load(HECBR_CODES_V4_NPZ)
    codes_base_bool = data["codes_base"]
    codes_eval_bool = data["codes_eval"]
    codes_base_u64 = to_u64(codes_base_bool)
    codes_eval_u64 = to_u64(codes_eval_bool)
    del data, codes_base_bool, codes_eval_bool

    print(">>> Contrôle de cohérence (F1 scan sur codes cache) ...", flush=True)
    f1_ref, lat_ref_ms, _ = eval_f1_hamming(codes_base_u64, y_base,
                                            codes_eval_u64, y_eval, K)
    ok = abs(f1_ref - REF_F1_SCAN) < 2e-3
    print(f"    F1 scan (réf.) = {f1_ref:.4f} vs {REF_F1_SCAN:.4f} "
          f"-> {'OK' if ok else 'ÉCHEC'}", flush=True)
    if not ok:
        sys.exit("!!! Désalignement labels/codes — arrêt. (Chiffres faux refusés.)")

    print(">>> Seuil theta_H (P%d, %d cas) ..." % (args.theta_percentile, THRESH_SAMPLE), flush=True)
    d1 = []
    rng_th = np.random.default_rng(SPLIT_SEED + 5)
    sample = rng_th.choice(len(codes_base_u64),
                           size=min(THRESH_SAMPLE, len(codes_base_u64)), replace=False)
    for s in sample:
        hd_all = _popcount64(np.bitwise_xor(codes_base_u64, codes_base_u64[s]))
        hd_all[s] = 2**30
        d1.append(int(np.min(hd_all)))
    theta = int(np.percentile(d1, args.theta_percentile))
    print(f"    theta_H = {theta} | d1 pcts:", np.percentile(d1, [25, 50, 75, 90, 99]).astype(int), flush=True)

    print(">>> Modèle v4 + ancrage mu/sd sur un échantillon base ...", flush=True)
    model = load_encode_model(args.device)
    mu, sd = encode_anchor(model, [meta[s] for s in base_ids], device=args.device)

    print(">>> Sélection des nouveaux cas du flux (positions globales HORS keep) ...", flush=True)
    # keep = indices globaux (trainval = base_ids + val_ids) de la base indexée 1M.
    # Le flux = patches des lames val NON inclus dans keep (strictement nouveaux).
    full_ids = base_ids + val_ids
    offs = np.cumsum([0] + [sizes[s] for s in full_ids])
    keep_arr = np.sort(keep).astype(np.int64)
    rng_flow = np.random.default_rng(7)
    flow_codes, flow_y = [], []
    budget = args.flow_n
    for sid in val_ids:
        if budget <= 0:
            break
        path, label, n = meta[sid]
        g0 = offs[full_ids.index(sid)] + np.arange(n)          # positions globales
        is_new = ~np.isin(g0, keep_arr)
        cand_local = np.flatnonzero(is_new)
        if len(cand_local) == 0:
            continue
        ne = min(budget, len(cand_local))
        sel = rng_flow.choice(len(cand_local), size=ne, replace=False)
        sel.sort()
        sel_local = cand_local[sel]
        X = load_slide_array(path)
        B = to_u64(encode_arr(model, X, mu, sd, device=args.device))
        flow_codes.append(B[sel_local]); flow_y.append(np.full(ne, label, dtype=np.int64))
        budget -= ne
        del X, B
    flow_codes_u64 = np.concatenate(flow_codes)
    flow_y = np.concatenate(flow_y)
    print(f"    flux : {flow_codes_u64.shape[0]} cas encodés "
          f"(pos={int((flow_y==1).sum())} neg={int((flow_y==0).sum())})", flush=True)

    print(">>> Simulation du flux (seeds d'ordre) ...", flush=True)
    # theta=0 -> pas de précalcul d_orig nécessaire (filtre par set, exact).
    # En annexe on peut estimer le profil de distances (échantillon) hors boucle seeds.
    if theta > 0:
        print("    précalcul d_orig unique (invariant à l'ordre) ...", flush=True)
        d_orig0 = precompute_d_orig(flow_codes_u64, codes_base_u64, args.n_proc)
    else:
        d_orig0 = None
    records = []
    for flow_seed in args.seeds:
        order = np.random.default_rng(flow_seed).permutation(len(flow_codes_u64))
        d_perm = d_orig0[order] if d_orig0 is not None else None
        rec = simulate_flow(codes_base_u64, y_base,
                            flow_codes_u64[order], flow_y[order], theta,
                            codes_eval_u64, y_eval, regime="retention",
                            k=K, n_proc=args.n_proc, d_orig=d_perm)
        rec["seed"] = flow_seed
        records.append(rec)
        print(f"  -- seed {flow_seed} --", flush=True)
        print(f"    {rec}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(RESULTS_RAW_DIR / "E1_retention_hecbr.csv", index=False)
    print(">>> CSV :", RESULTS_RAW_DIR / "E1_retention_hecbr.csv", flush=True)


if __name__ == "__main__":
    main()