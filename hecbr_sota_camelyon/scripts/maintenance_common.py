"""Fonctions communes aux benchmarks de maintenance HeCBR (E1/E2/E3).

Hypothèse de travail : toute la maintenance agit dans l'espace des codes binaires
appris (64 bits) — c'est la contribution du mémoire (maintenance guidée par le
hachage). La seule reconstruction mémoire lourde (labels y, échantillons L2 de
contrôle) reste bornée et déterministe.

Protocole de test anti-artefact : les requêtes d'évaluation sont TOUJOURS
échantillonnées stratifiées 50/50 (rng SPLIT_SEED+7, N=300) — jamais le test brut,
qui était mono-classe (cf. REPORT_SCAN_HECBR.md §2.2, F1=1.00 artefact).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.paths import CAMELYON_ROOT
from config.seeds import SPLIT_SEED
from data.splitter import split_slides
from indexes.hecbr_index import HeCBRIndex, _popcount64
from benchmark.timing import benchmark_queries, aggregate_latencies
from benchmark.metrics import cbr_predict, macro_f1

TEST_N = 300
BASE_N = 1_000_000          # plafond de la base d'indexation (SPLIT_SEED+11)
K = 10


def iter_slide_arrays():
    """Itère lame par lame — mémoire bornée.

    Yields (slide_id, path_abs, label, n_rows) où label ∈ {0,1} est déduit du
    répertoire (0-normal/, 1-tumor/) comme dans data/loader.py, et n_rows = nombre
    de patches (obtenu par comptage de lignes, sans parser les CSV : 1 passe disque
    légère ~17 Go, pas de pic RAM).
    """
    for dir_name, label in [("0-normal", 0), ("1-tumor", 1)]:
        d = CAMELYON_ROOT / dir_name
        if not d.exists():
            continue
        for p in sorted(d.glob("*.csv")):
            n_lines = 0
            with open(p, "rb") as fh:
                for _ in fh:
                    n_lines += 1
            yield p.stem, p, label, max(0, n_lines - 1)   # entête


def load_slide_array(path_abs) -> np.ndarray:
    """Lit un CSV de lames (features 512) en float32."""
    df = pd.read_csv(path_abs, header=0)
    return df.values[:, :512].astype(np.float32)


# ---------------------------------------------------------------------------
# Construction des étiquettes/requêtes sans charger les X en RAM
# ---------------------------------------------------------------------------
def assemble_labels_full(
    slide_ids: List[str],
    slide_labels: Dict[str, int],
    slide_sizes: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruit y du trainval complet, y_test, keep et chosen (déterministes).

    Retourne (y_base[keep], y_eval, keep, chosen) dans le MÊME ordre que
    `run_comparative_benchmark.py:main()` — ordre de build_split_matrices puis
    rng_cap(SPLIT_SEED+11) pour la base et rng(SPLIT_SEED+7) stratifié 50/50
    pour les 300 requêtes de test.
    """
    sids = sorted(slide_ids)
    base_ids, val_ids, test_ids = split_slides(sids, slide_labels, seed=SPLIT_SEED)

    full_ids = base_ids + val_ids
    y_full = np.concatenate([np.full(slide_sizes[s], slide_labels[s], dtype=np.int64) for s in full_ids])
    # NB: les tailles des lames trainval peuvent etre lues sans X (cache ou comptage).

    test_ids_sorted = list(test_ids)
    y_test = np.concatenate([np.full(slide_sizes[s], slide_labels[s], dtype=np.int64) for s in test_ids_sorted])

    rng_cap = np.random.default_rng(SPLIT_SEED + 11)
    keep = rng_cap.choice(y_full.shape[0], size=min(BASE_N, y_full.shape[0]), replace=False)
    y_base = y_full[keep]

    # Echantillonnage stratifie 50/50 des requetes (le point anti-artefact).
    N_req = min(TEST_N, y_test.shape[0])
    rng = np.random.default_rng(SPLIT_SEED + 7)
    idx_pos = np.flatnonzero(y_test == 1)
    idx_neg = np.flatnonzero(y_test == 0)
    n_pos = max(1, N_req // 2)
    if len(idx_pos) == 0:
        chosen = rng.choice(idx_neg, size=N_req, replace=len(idx_neg) < N_req)
    else:
        pos_sel = idx_pos[rng.choice(len(idx_pos), size=min(n_pos, len(idx_pos)), replace=len(idx_pos) < n_pos)]
        need_neg = N_req - len(pos_sel)
        neg_sel = idx_neg[rng.choice(len(idx_neg), size=min(need_neg, len(idx_neg)), replace=len(idx_neg) < need_neg)]
        chosen = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(chosen)
        chosen = chosen[:N_req]
    y_eval = y_test[chosen]

    return y_base, y_eval, keep, chosen


# ---------------------------------------------------------------------------
# Décision de rétention / distances Hamming
# ---------------------------------------------------------------------------
def hamming_d1_u64(codes_u64: np.ndarray, q_u64: np.ndarray) -> np.ndarray:
    """Distance Hamming au plus proche code (scan exact SWAR par requête)."""
    hd = _popcount64(np.bitwise_xor(codes_u64, q_u64))
    return np.min(hd)


def to_u64(codes_bool: np.ndarray) -> np.ndarray:
    """Convertit des codes bool (N, 64) en uint64 packés (ordinateur little-endian)."""
    packed = np.packbits(codes_bool.astype(np.uint8), axis=1)
    return np.ascontiguousarray(packed).view(np.uint64).reshape(-1)


def eval_f1_hamming(
    codes_base: np.ndarray,
    y_base: np.ndarray,
    codes_eval: np.ndarray,
    y_eval: np.ndarray,
    k: int = K,
) -> Tuple[float, float, List[np.ndarray]]:
    """F1 macro + latence médiane d'un CBR HeCBR scan Hamming exact."""
    index = HeCBRIndex(bit_length=64)
    index.build(codes_base)
    latencies, results = benchmark_queries(index.query, codes_eval, k)
    lat_stats = aggregate_latencies(latencies)
    all_idx = [res[0] for res in results]
    y_pred = cbr_predict(y_base, all_idx)
    return macro_f1(y_eval, y_pred), lat_stats["median_ms"], all_idx


# ---------------------------------------------------------------------------
# Condensation code-bucket
# ---------------------------------------------------------------------------
def condense_code_bucket(
    codes_u64: np.ndarray,
    y: np.ndarray,
    radius: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Représentant par code identique (r=0) ou par boule de Hamming (r>0).

    Greedy borné : pour r=0, un représentant par code unique. Pour r>0, on
    parcourt les codes (triés) et on couvre les voisins à distance <= r par le
    premier code pivot, sans retirer les autres (expressif, O(N)) — équivalent
    « footprint avant » du chapitre.
    Retourne (codes_condenses, y_condenses, n_retires).
    """
    if radius == 0:
        uniq, inv, counts = np.unique(codes_u64, return_inverse=True, return_counts=True)
        # représentant = premier cas de chaque code (ordre original)
        first = np.empty(len(uniq), dtype=np.int64)
        first.fill(-1)
        seen = {}
        for i, u in enumerate(inv):
            if u not in seen:
                seen[u] = i
                first[u] = i
        order = np.argsort(first)
        return codes_u64[first[order]], y[first[order]], int(len(codes_u64) - len(uniq))

    # rayon r>0 : footprint « premier pivot », parcours glouton.
    order = np.argsort(codes_u64)
    codes_sorted = codes_u64[order]
    y_sorted = y[order]
    pivots = []
    i = 0
    n = len(codes_sorted)
    while i < n:
        pivots.append(i)
        # couvrir les indices j>i avec dist(codes_sorted[j], code_pivot) <= r
        j = i + 1
        while j < n and _popcount64(np.bitwise_xor(codes_sorted[i], codes_sorted[j])) <= radius:
            j += 1
        i = j
    piv_idx = np.array(pivots, dtype=np.int64)
    return codes_sorted[piv_idx], y_sorted[piv_idx], int(n - len(pivots))


def reduction_rate(n_condensed: int, n_orig: int) -> float:
    return 1.0 - n_condensed / max(1, n_orig)