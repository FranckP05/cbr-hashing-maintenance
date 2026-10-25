"""Partitionnement stratifié 90/10 par lame entière avec gestion mémoire optimisée."""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from config.protocol import TRAIN_RATIO, TEST_RATIO, VAL_RATIO
from config.seeds import SPLIT_SEED
from data.preprocessing import l2_normalize

logger = logging.getLogger(__name__)
EMBEDDING_DIM = 512


def split_slides(
    slide_ids: List[str],
    slide_labels: Dict[str, int],
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float  = TEST_RATIO,
    val_ratio: float   = VAL_RATIO,
    seed: int          = SPLIT_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """Partitionne les identifiants de lames entières (399 WSI) de façon stratifiée."""
    slide_ids_arr = np.array(slide_ids)
    labels_arr    = np.array([slide_labels[sid] for sid in slide_ids])

    # 1. Séparer 10% test vs 90% trainval
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(splitter1.split(slide_ids_arr, labels_arr))

    trainval_ids    = slide_ids_arr[trainval_idx].tolist()
    trainval_labels = labels_arr[trainval_idx]
    test_ids        = slide_ids_arr[test_idx].tolist()

    # 2. Séparer validation vs base
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + 1)
    base_idx, val_idx = next(splitter2.split(np.array(trainval_ids), trainval_labels))

    base_ids = [trainval_ids[i] for i in base_idx]
    val_ids  = [trainval_ids[i] for i in val_idx]

    logger.info(
        "Split WSI : base=%d, val=%d, test=%d (total=%d lames)",
        len(base_ids), len(val_ids), len(test_ids), len(slide_ids),
    )
    return base_ids, val_ids, test_ids


def _pop_matrices(
    slide_ids: List[str],
    slide_data: Dict[str, np.ndarray],
    slide_labels: Dict[str, int],
    normalize_l2: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construit une matrice contiguë en libérant immédiatement les tableaux du dictionnaire."""
    total_patches = sum(slide_data[sid].shape[0] for sid in slide_ids if sid in slide_data)
    X = np.empty((total_patches, EMBEDDING_DIM), dtype=np.float32)
    y = np.empty((total_patches,), dtype=np.int64)

    offset = 0
    for sid in slide_ids:
        if sid not in slide_data:
            continue
        arr = slide_data.pop(sid)
        n = arr.shape[0]
        X[offset : offset + n] = arr
        y[offset : offset + n] = slide_labels[sid]
        offset += n
        del arr

    if normalize_l2:
        l2_normalize(X, in_place=True)

    gc.collect()
    return X, y


def build_split_matrices(
    base_ids: List[str],
    val_ids:  List[str],
    test_ids: List[str] | None,
    slide_data:   dict,
    slide_labels: dict,
    normalize_l2: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Construit les matrices NumPy complètes sans pic mémoire OOM."""
    logger.info("Construction de la matrice base (%d lames)...", len(base_ids))
    X_base, y_base = _pop_matrices(base_ids, slide_data, slide_labels, normalize_l2=normalize_l2)

    logger.info("Construction de la matrice validation (%d lames)...", len(val_ids))
    X_val, y_val = _pop_matrices(val_ids, slide_data, slide_labels, normalize_l2=normalize_l2)

    if test_ids:
        logger.info("Construction de la matrice test (%d lames)...", len(test_ids))
        X_test, y_test = _pop_matrices(test_ids, slide_data, slide_labels, normalize_l2=normalize_l2)
    else:
        X_test, y_test = None, None

    return X_base, y_base, X_val, y_val, X_test, y_test
