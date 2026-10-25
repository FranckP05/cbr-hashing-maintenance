"""Index binaire HeCBR — Recherche exacte ou approchee (multi-probe) par distance de Hamming."""

from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


_SWAR_M1 = np.uint64(0x5555555555555555)
_SWAR_M2 = np.uint64(0x3333333333333333)
_SWAR_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_SWAR_H01 = np.uint64(0x0101010101010101)


def _popcount64(v) -> np.ndarray:
    """Nombre de bits a 1 du XOR (SWAR vectorise sur uint64)."""
    v = np.atleast_1d(np.asarray(v, dtype=np.uint64))
    with np.errstate(over="ignore"):
        v = v - ((v >> 1) & _SWAR_M1)
        v = (v & _SWAR_M2) + ((v >> 2) & _SWAR_M2)
        v = (v + (v >> 4)) & _SWAR_M4
        return (v * _SWAR_H01) >> 56


def _code_to_q64(query_code: np.ndarray, bit_length: int) -> np.uint64:
    """Convertit un code (bool, uint8 packed, uint8(bit_length,) ou uint64) en uint64."""
    c = np.asarray(query_code)
    if c.dtype == bool or c.shape[0] == bit_length and c.ndim == 1:
        packed = np.packbits(c.astype(np.uint8))
        return np.uint64(int.from_bytes(packed.tobytes(), "little"))
    elif c.size == 1:
        return np.uint64(c.reshape(-1)[0])
    else:
        packed = np.ascontiguousarray(c, dtype=np.uint8)
        return np.uint64(int.from_bytes(packed.tobytes(), "little"))


class HeCBRMultiProbeIndex:
    """Index de cas binaires HeCBR avec tables de hachage binaires et multi-probe Hamming.

    Au lieu de scanner l'integralite de la base pour chaque requete (4.4M cas),
    divise les 64 bits en `L` sous-ensembles de `m` bits. Chaque table regroupe les
    cas par valeur entiere de ces `m` bits. A la requete, on sonde les buckets voisins
    (a distance Hamming <= radius sur le sous-ensemble) puis on re-rank exactement
    les candidats par distance de Hamming. Evite le scan lineaire complet.
    """

    def __init__(self, bit_length: int = 64) -> None:
        self.bit_length = bit_length
        self.num_bytes = bit_length // 8
        self._codes: Optional[np.ndarray] = None
        self.n_base: int = 0
        self._L: int = 0
        self._m: int = 0
        self._radius: int = 0
        self._bit_sets: List[np.ndarray] = []
        self._table_order: List[np.ndarray] = []
        self._table_lookup: List[Dict[int, Tuple[int, int]]] = []
        self._probe_offsets: List[np.ndarray] = []
        self.build_time_s: float = 0.0

    def _extract_keys(self, codes: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """Pour chaque code uint64, extrait la valeur entiere des bits de `bits`."""
        shots = ((codes[:, None] >> bits.astype(np.uint64)[None, :]) & np.uint64(1)).astype(np.int64)
        powers = np.left_shift(np.int64(1), np.arange(self._m, dtype=np.int64))
        return (shots * powers[None, :]).sum(axis=1)

    def build(
        self,
        binary_codes: np.ndarray,
        num_tables: int = 8,
        bits_per_table: int = 16,
        radius: int = 2,
        hash_seed: int = 42,
    ) -> None:
        """Construit l'index multi-probe a partir de codes binaires.

        Args:
            num_tables: nombre de tables de hachage L.
            bits_per_table: nombre de bits m par table (sous-ensemble).
            radius: nombre max de bits a basculer (distance Hamming) lors du multi-probe.
            hash_seed: graine pour l'echantillonnage reproductible des sous-ensembles de bits.
        """
        t0 = time.perf_counter()
        self.n_base = binary_codes.shape[0]
        if binary_codes.dtype == bool or (binary_codes.ndim == 2 and binary_codes.shape[1] == self.bit_length):
            packed = np.packbits(binary_codes.astype(np.uint8), axis=1)
            codes = packed.view(np.uint64).reshape(-1)
        elif binary_codes.ndim == 2 and binary_codes.shape[1] == self.num_bytes:
            codes = np.ascontiguousarray(binary_codes, dtype=np.uint8).view(np.uint64).reshape(-1)
        else:
            codes = np.ascontiguousarray(binary_codes, dtype=np.uint64)
        self._codes = codes

        self._L = int(num_tables)
        self._m = int(bits_per_table)
        self._radius = int(radius)
        all_bits = np.arange(self.bit_length, dtype=np.int64)
        rng = np.random.default_rng(hash_seed)
        self._bit_sets = [rng.choice(all_bits, size=self._m, replace=False) for _ in range(self._L)]

        # Precalcul des cles (valeurs entieres des m bits) + buckets pour chaque table
        self._table_order = []
        self._table_lookup = []
        self._probe_offsets = []

        # Masques de probe : pour radius r, toutes les combinaisons de {1..r} bits a basculer
        offsets = []
        for r in range(self._radius + 1):
            offsets.append(np.array([1 << i for i in range(r)], dtype=np.int64))
        self._probe_offsets = offsets  # en fait on genere les cles directement a la requete

        BLOCK = 500_000
        for t in range(self._L):
            keys = np.empty(self.n_base, dtype=np.int64)
            for start in range(0, self.n_base, BLOCK):
                end = min(start + BLOCK, self.n_base)
                keys[start:end] = self._extract_keys(codes[start:end], self._bit_sets[t])

            order = np.argsort(keys, kind="mergesort").astype(np.int64)
            sorted_keys = keys[order]
            unique_k, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
            lookup = {int(k): (int(s), int(c)) for k, s, c in zip(unique_k, starts, counts)}
            self._table_order.append(order)
            self._table_lookup.append(lookup)

        self.build_time_s = time.perf_counter() - t0

    def _probe_keys(self, q64: np.uint64, bits: np.ndarray, budget: int) -> np.ndarray:
        """Genere les cles a sonder : valeur des m bits de la requete + fluctuations <= radius."""
        vals = ((q64 >> bits.astype(np.uint64)) & np.uint64(1)).astype(np.int64)
        powers = np.left_shift(np.int64(1), np.arange(self._m, dtype=np.int64))
        base = int((vals * powers).sum())
        keys = [base]
        for r in range(1, self._radius + 1):
            for combo in combinations(range(self._m), r):
                flip = int((powers[np.array(combo)]).sum())
                keys.append(base ^ flip)
                if len(keys) >= budget:
                    return np.array(keys)
        return np.array(keys)

    def query(
        self, query_code: np.ndarray, k: int, max_buckets_per_table: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Retourne top-k voisins Hamming via multi-probe + rerank exact sur candidats.

        Returns:
            (indices triees, distances, nb_candidats_scannes)
        """
        assert self._codes is not None, "Appeler build() avant query()."
        q64 = _code_to_q64(query_code, self.bit_length)

        gained: List[np.ndarray] = []
        for t in range(self._L):
            probes = self._probe_keys(q64, self._bit_sets[t], max_buckets_per_table)
            look = self._table_lookup[t]
            ord_t = self._table_order[t]
            for pk in probes:
                pk_int = int(pk)
                entry = look.get(pk_int)
                if entry is not None:
                    s, c = entry
                    gained.append(ord_t[s : s + c])

        if not gained:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

        cand = np.unique(np.concatenate(gained))
        n_candidates = cand.shape[0]
        if n_candidates == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

        codes_cand = self._codes[cand]
        hd = _popcount64(np.bitwise_xor(codes_cand, q64)).astype(np.float32)

        k_eff = min(k, len(cand))
        if k_eff >= len(cand):
            order = np.argsort(hd)
        else:
            order = np.argpartition(hd, k_eff)[:k_eff]
            order = order[np.argsort(hd[order])]

        return cand[order].astype(np.int64), hd[order], int(n_candidates)

    def batch_query(
        self, query_codes: np.ndarray, k: int, max_buckets_per_table: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute un lot de requetes multi-probe."""
        assert self._codes is not None
        N_q = query_codes.shape[0]
        k_eff = min(k, self.n_base)
        all_indices = np.empty((N_q, k_eff), dtype=np.int64)
        all_dists = np.empty((N_q, k_eff), dtype=np.float32)
        all_cands = np.empty(N_q, dtype=np.int64)
        for i in range(N_q):
            idx, dists, nc = self.query(query_codes[i], k_eff, max_buckets_per_table)
            if idx.shape[0] == 0:
                all_indices[i] = -1
                all_dists[i] = self.bit_length + 1
            else:
                all_indices[i, : idx.shape[0]] = idx
                all_dists[i, : idx.shape[0]] = dists
            all_cands[i] = nc
        return all_indices, all_dists, all_cands


class HeCBRIndex:
    """Index de cas binaires HeCBR base sur la distance de Hamming (XOR + POPCNT).

    Stockage : codes compacts uint64 (8 octets par cas).
    Empreinte memoire : ~30 Mo pour 4.4M cas (divisee par ~250 vs float32).
    """

    def __init__(self, bit_length: int = 64) -> None:
        self.bit_length = bit_length
        self.num_bytes  = bit_length // 8
        self._codes: np.ndarray | None = None   # shape (N,) uint64
        self.n_base: int = 0
        self.build_time_s: float = 0.0

    def build(self, binary_codes: np.ndarray) -> None:
        """Construit l'index a partir de codes binaires uint8 ou booleens."""
        t0 = time.perf_counter()
        if binary_codes.dtype == bool or (binary_codes.ndim == 2 and binary_codes.shape[1] == self.bit_length):
            # Conversion (N, 64) bool -> (N, 8) uint8 via packbits puis vue uint64
            packed = np.packbits(binary_codes.astype(np.uint8), axis=1)
            self._codes = packed.view(np.uint64).reshape(-1)
        elif binary_codes.ndim == 2 and binary_codes.shape[1] == self.num_bytes:
            packed = np.ascontiguousarray(binary_codes, dtype=np.uint8)
            self._codes = packed.view(np.uint64).reshape(-1)
        else:
            # deja des uint64
            self._codes = np.ascontiguousarray(binary_codes, dtype=np.uint64)

        self.n_base = self._codes.shape[0]
        self.build_time_s = time.perf_counter() - t0

    def query(self, query_code: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les top-k plus proches voisins au sens de la distance de Hamming."""
        assert self._codes is not None, "Appeler build() avant query()."
        if query_code.ndim > 0 and (query_code.dtype == bool or query_code.shape[0] == self.bit_length):
            q_packed = np.packbits(np.asarray(query_code).astype(np.uint8))
            q64 = np.uint64(int.from_bytes(q_packed.tobytes(), "little"))
        elif query_code.size == 1:
            q64 = np.uint64(np.asarray(query_code).reshape(-1)[0])
        else:
            q_packed = np.ascontiguousarray(query_code, dtype=np.uint8)
            q64 = np.uint64(int.from_bytes(q_packed.tobytes(), "little"))

        hamming_dists = _popcount64(np.bitwise_xor(self._codes, q64))

        k_eff = min(k, self.n_base)
        if k_eff >= self.n_base:
            idx = np.argsort(hamming_dists)
        else:
            idx = np.argpartition(hamming_dists, k_eff)[:k_eff]
            idx = idx[np.argsort(hamming_dists[idx])]

        return idx.astype(np.int64), hamming_dists[idx].astype(np.float32)

    def batch_query(self, query_codes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Execute un lot de requetes en boucle vectorisee."""
        assert self._codes is not None
        N_q = query_codes.shape[0]
        k_eff = min(k, self.n_base)
        all_indices = np.empty((N_q, k_eff), dtype=np.int64)
        all_dists   = np.empty((N_q, k_eff), dtype=np.float32)

        for i in range(N_q):
            idx, dists = self.query(query_codes[i], k_eff)
            all_indices[i] = idx
            all_dists[i]   = dists

        return all_indices, all_dists
