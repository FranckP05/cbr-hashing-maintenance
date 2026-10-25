"""E3 — Obsolescence / dérive (la question qualité).

Scénario : la base initiale est construite sur un sous-ensemble de lames du train ;
un flux de nouvelles lames arrive par vagues (steady-state trainval, puis DÉRIVE :
lames du test, inter-participants). Trois régimes :
  1. SANS maintenance  (add-all)  : chaque cas du flux est ajouté ;
  2. RÉTENTION seule   : un cas n'est ajouté que si d_H(1-NN) > theta_H ;
  3. RÉTENTION + CONDENSATION : rétention puis, en fin de vague, footprint
     code-bucket (représentant par code identique).
À chaque vague on mesure macro-F1 et latence sur le test stratifié 50/50 (300).

Conception pour rester tractable sur le serveur CPU :
  * encodage lame par lame (mémoire bornée), ancrage mu/sd sur la base initiale ;
  * budget de flux global (100k cas) échantillonné déterministe par vague ;
  * décision de rétention vectorisée : d_initial précalculé une fois + min
    incrémental contre les cas déjà ajoutés (la base = init ∪ ajoutés, donc
    d(1-NN) = min sur l'union — exact) ;
  * contrôle de cohérence de référence sur les codes en cache (F1 ~0.5402),
    refus d'exécution si désalignement labels/codes.

Sortie : results/raw/E3_drift.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import HECBR_MODEL_PATH, HECBR_CODES_V4_NPZ, RESULTS_RAW_DIR
from config.seeds import SPLIT_SEED
from data.splitter import split_slides
from indexes.hecbr_index import _popcount64
from scripts.train_hecbr_v4 import HashNetV3
from scripts.maintenance_common import (
    to_u64, condense_code_bucket, eval_f1_hamming,
    assemble_labels_full, iter_slide_arrays, load_slide_array, K,
)

REF_F1_SCAN = 0.5401810979060555
N_WAVES = 6
FLOW_N = 100_000
THRESH_SAMPLE = 2_000
DRIFT_WAVE = N_WAVES - 1           # dernière vague = lames du test (dérive)


def load_encode_model(device="cpu"):
    import torch
    ck = torch.load(HECBR_MODEL_PATH, map_location=device)
    model = HashNetV3(in_dim=ck.get("in_dim", 512), hidden=ck.get("hidden_dim", 512),
                      bit_length=ck.get("bit_length", 64))
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model


def encode_arr(model, X, mu, sd, device="cpu", bs=8192, rng=None, n_keep=None):
    """Encode un array et, si n_keep < len(X), échantillonne n_keep codes."""
    import torch
    out = []
    for i in range(0, len(X), bs):
        x = torch.tensor((X[i:i + bs] - mu) / sd, dtype=torch.float32).to(device)
        with torch.no_grad():
            u = model(x)
            out.append((u >= 0).cpu().numpy())
        del x
    B = np.concatenate(out).astype(np.uint8)
    B_u64 = to_u64(B)
    if n_keep is not None and n_keep < len(B_u64):
        sel = rng.choice(len(B_u64), size=n_keep, replace=False)
        sel.sort()
        return B_u64[sel]
    return B_u64


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-waves", type=int, default=N_WAVES)
    p.add_argument("--flow-n", type=int, default=FLOW_N)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> Métadonnées lames (mémoire bornée) ...", flush=True)
    meta = {sid: (path, label, n) for sid, path, label, n in iter_slide_arrays()}
    sids = sorted(meta.keys())
    slide_labels = {sid: t[1] for sid, t in meta.items()}
    sizes = {sid: t[2] for sid, t in meta.items()}
    base_ids, val_ids, test_ids = split_slides(sids, slide_labels, seed=SPLIT_SEED)
    trainval_ids = base_ids + val_ids
    print(f"    {len(sids)} lames | base={len(base_ids)} val={len(val_ids)} "
          f"test={len(test_ids)}", flush=True)

    print(">>> Labels/requêtes de référence (même rng que benchmark) ...", flush=True)
    y_base, y_eval, keep, chosen = assemble_labels_full(sids, slide_labels, sizes)
    print(f"    y_base={y_base.shape} y_eval={y_eval.shape} "
          f"pos={int((y_eval==1).sum())} neg={int((y_eval==0).sum())}", flush=True)

    print(">>> Contrôle de cohérence (F1 scan sur codes cache) ...", flush=True)
    cache = np.load(HECBR_CODES_V4_NPZ)
    f1_gate, _, _ = eval_f1_hamming(to_u64(cache["codes_base"]), y_base,
                                    to_u64(cache["codes_eval"]), y_eval, K)
    ok = abs(f1_gate - REF_F1_SCAN) < 2e-3
    print(f"    F1 scan (réf.) = {f1_gate:.4f} vs {REF_F1_SCAN:.4f} "
          f"-> {'OK' if ok else 'ÉCHEC'}", flush=True)
    if not ok:
        sys.exit("!!! Désalignement labels/codes — arrêt. (Chiffres faux refusés.)")
    del cache

    # --- Vagues : ordre par taille croissante ; dernière vague = test (dérive) ---
    tv = sorted(range(len(trainval_ids)), key=lambda i: sizes[trainval_ids[i]])
    n_tv = len(tv)
    wave0 = tv[: n_tv // args.n_waves]                    # base initiale (plus petites lames)
    body = tv[n_tv // args.n_waves:]                      # à répartir en vagues 1..n_waves-2
    n_body = len(body)
    n_steady_waves = max(1, args.n_waves - 2)
    steady = [body[i * n_body // n_steady_waves:(i + 1) * n_body // n_steady_waves]
              for i in range(n_steady_waves)]
    waves_slide_idx = [wave0] + steady + [list(range(len(test_ids)))]  # dernière = dérive
    assert len(waves_slide_idx) == args.n_waves
    initial_ids = [trainval_ids[i] for i in waves_slide_idx[0]]
    print(f"    base initiale={len(initial_ids)} lames ; "
          f"{len(waves_slide_idx)-2} vagues steady + 1 dérive (test)", flush=True)

    # --- Ancrage mu/sd sur échantillon de la base initiale ---
    print(">>> Ancrage mu/sd (échantillon base initiale) ...", flush=True)
    rng_a = np.random.default_rng(SPLIT_SEED + 5)
    chunks = []
    for i in waves_slide_idx[0]:
        sid = trainval_ids[i]
        X = load_slide_array(meta[sid][0])
        sub = rng_a.choice(len(X), size=min(8192, len(X)), replace=False)
        chunks.append(X[sub].astype(np.float32))
        del X
    Xa = np.concatenate(chunks)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    del Xa

    print(">>> Encodage base initiale (lame par lame) ...", flush=True)
    model = load_encode_model(args.device)
    B_init, y_init = [], []
    for i in waves_slide_idx[0]:
        sid = trainval_ids[i]
        X = load_slide_array(meta[sid][0])
        B_init.append(encode_arr(model, X, mu, sd, args.device))
        y_init.append(np.full(sizes[sid], slide_labels[sid], dtype=np.int64))
        del X
    B_init = np.concatenate(B_init).astype(np.uint64)
    y_init = np.concatenate(y_init).astype(np.int64)
    print(f"    base initiale : {len(B_init)} cas "
          f"(pos={int((y_init==1).sum())} neg={int((y_init==0).sum())})", flush=True)

    print(">>> Encodage requêtes d'évaluation (test stratifié, choisi) ...", flush=True)
    y_test = np.concatenate([np.full(sizes[s], slide_labels[s], dtype=np.int64) for s in test_ids])
    B_test_all = []
    for sid in test_ids:
        X = load_slide_array(meta[sid][0])
        B_test_all.append(encode_arr(model, X, mu, sd, args.device))
        del X
    B_test_all = np.concatenate(B_test_all).astype(np.uint64)
    B_eval = B_test_all[chosen]
    y_eval = y_test[chosen]
    del B_test_all, y_test
    print(f"    éval : {len(chosen)} requêtes (pos={int((y_eval==1).sum())} "
          f"neg={int((y_eval==0).sum())})", flush=True)

    print(">>> Seuil theta_H = P25(d1 Hamming base initiale) ...", flush=True)
    rng_th = np.random.default_rng(SPLIT_SEED + 5)
    sample = rng_th.choice(len(B_init), size=min(THRESH_SAMPLE, len(B_init)), replace=False)
    d1 = []
    for s in sample:
        hd_all = _popcount64(np.bitwise_xor(B_init, B_init[s]))
        hd_all[s] = 2 ** 30
        d1.append(int(np.min(hd_all)))
    theta = int(np.percentile(d1, 25))
    print(f"    theta_H = {theta} | d1 pcts:", np.percentile(d1, [25, 50, 75, 90, 99]).astype(int), flush=True)

    # --- Budget de flux par vague et encodage des arrivées ---
    n_steady = len(waves_slide_idx) - 2
    budget_steady = max(1, args.flow_n // max(1, args.n_waves - 1) // 2)
    budget_drift = args.flow_n - budget_steady * n_steady
    budget_drift = max(budget_steady, budget_drift)
    rng_flow = np.random.default_rng(7)

    def encode_wave(idx_list, is_test, budget_wave) -> np.ndarray:
        """Encode les lames de la vague, échantillonne BUDGET total, retourne (codes, y)."""
        slides = []
        for i in idx_list:
            sid = test_ids[i] if is_test else trainval_ids[i]
            slides.append((meta[sid][0], meta[sid][1], meta[sid][2]))
        total_avail = sum(n for _, _, n in slides)
        codes, ys = [], []
        assigned = 0
        for path, label, n in slides:
            if assigned >= budget_wave:
                break
            share = min(n, max(1, int(n * budget_wave / max(1, total_avail))))
            share = min(share, budget_wave - assigned)
            X = load_slide_array(path)
            B = encode_arr(model, X, mu, sd, args.device,
                           rng=rng_flow, n_keep=share)
            codes.append(B)
            ys.append(np.full(len(B), label, dtype=np.int64))
            assigned += len(B)
            del X, B
        if len(codes) == 0:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
        return np.concatenate(codes).astype(np.uint64), np.concatenate(ys).astype(np.int64)

    # --- Trois régimes : état = tableaux uint64/int64 ---
    regimes = {
        "add_all":   (B_init.copy(), y_init.copy()),
        "retention": (B_init.copy(), y_init.copy()),
        "ret_cond":  (B_init.copy(), y_init.copy()),
    }
    for name in regimes:
        regimes[name] = [list(regimes[name][0]), list(regimes[name][1])]

    def decide_retain(base_codes: np.ndarray, new_codes: np.ndarray, theta: int) -> np.ndarray:
        """Masque vrai=retenir : d(1-NN) au RÉFÉRENTIEL base > theta.

        theta == 0 → règle = rejeter les doublons EXACTS (d > 0 ⟺ pas dans base).
        Détection exacte et quasi-instantanée par un set de codes uint64.
        theta > 0 → scan Hamming par blocs (cas théorique, non atteint ici).
        """
        if theta <= 0:
            base_set = set(int(c) for c in base_codes)
            return np.array([int(c) not in base_set for c in new_codes], dtype=bool)
        BLK = 8192
        keep = np.zeros(len(new_codes), dtype=bool)
        for s0 in range(0, len(new_codes), BLK):
            q = new_codes[s0:s0 + BLK]
            for b0 in range(0, len(base_codes), 65536):
                blk = base_codes[b0:b0 + 65536]
                drops = np.zeros(len(q), dtype=np.int32) | 64
                for q0 in range(0, len(q), 2048):
                    hd = _popcount64(np.bitwise_xor(blk[None, :], q[q0:q0 + 2048, None]))
                    drops[q0:q0 + 2048] = np.minimum(drops[q0:q0 + 2048], hd.min(axis=1))
                keep[s0:s0 + BLK] |= (drops > theta)
        return keep

    records = []
    from benchmark.metrics import cbr_predict
    prev_pred = {}   # regime -> y_pred de la vague précédente
    for w_idx in range(1, args.n_waves):
        is_drift = (w_idx == args.n_waves - 1)
        budget_wave = budget_drift if is_drift else budget_steady
        wave_codes, wave_y = encode_wave(waves_slide_idx[w_idx], is_drift, budget_wave)
        n_arr = len(wave_codes)
        print(f"--- vague {w_idx}/{args.n_waves-1} "
              f"({'DÉRIVE (test inter-participants)' if is_drift else 'steady'}) "
              f"| {n_arr} cas ---", flush=True)
        for reg, st in regimes.items():
            base_codes_arr = np.array(st[0], dtype=np.uint64)
            if reg == "add_all":
                keep = np.ones(n_arr, dtype=bool)
            else:
                keep = decide_retain(base_codes_arr, wave_codes, theta)
                if not keep.any():
                    keep[0] = True   # garder au moins un représentant par vague
            kept_codes = [int(c) for c, k in zip(wave_codes, keep) if k]
            kept_y = [int(y) for y, k in zip(wave_y, keep) if k]
            n_kept = len(kept_codes)
            if reg == "ret_cond":
                combined = np.concatenate([base_codes_arr, np.array(kept_codes, dtype=np.uint64)])
                cy = np.concatenate([np.array(st[1], dtype=np.int64), np.array(kept_y, dtype=np.int64)])
                cc, yy, _ret = condense_code_bucket(combined, cy, radius=0)
                st[0] = list(cc); st[1] = list(yy)
            else:
                st[0].extend(kept_codes); st[1].extend(kept_y)
            n_base = len(st[0])
            base_arr = np.array(st[0], dtype=np.uint64)
            y_arr = np.array(st[1], dtype=np.int64)
            f1, lat_ms, all_idx = eval_f1_hamming(base_arr, y_arr, B_eval, y_eval, K)
            y_pred = cbr_predict(y_arr, all_idx)
            n_changed = int((prev_pred[reg] != y_pred).sum()) if reg in prev_pred else None
            prev_pred[reg] = y_pred
            np.savez(RESULTS_RAW_DIR / f"E3_dbg_{w_idx}_{reg}.npz",
                     codes=base_arr, y=y_arr, Beval=B_eval, yeval=y_eval,
                     pred=y_pred)
            records.append({
                "wave": w_idx, "regime": reg, "drift": int(is_drift),
                "n_arrived": n_arr, "n_kept": n_kept, "n_base": n_base,
                "f1_macro": f1, "query_median_ms": lat_ms,
                "n_pred_changed": n_changed,
            })
            print(f"    | {reg:10s} | arrivés={n_arr} retenus={n_kept} base={n_base} "
                  f"F1={f1:.4f} lat={lat_ms:.2f}ms", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(RESULTS_RAW_DIR / "E3_drift.csv", index=False)
    print(">>> CSV :", RESULTS_RAW_DIR / "E3_drift.csv", flush=True)


if __name__ == "__main__":
    main()