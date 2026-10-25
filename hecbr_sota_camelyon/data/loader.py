"""Chargement des embeddings Camelyon16 avec réutilisation directe du cache."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.paths import (
    DSMIL_ROOT, CAMELYON_ROOT, CACHE_FILE,
)

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
LABEL_NORMAL  = 0
LABEL_TUMOR   = 1


def load_slides(
    use_cache: bool = True,
    cache_path: Path = CACHE_FILE,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Charge l'ensemble des 399 lames depuis le cache pickle."""
    if use_cache and cache_path.exists():
        logger.info("Chargement accéléré depuis le cache : %s ...", cache_path)
        with open(cache_path, "rb") as fh:
            slide_data, slide_labels = pickle.load(fh)
        total_patches = sum(arr.shape[0] for arr in slide_data.values())
        logger.info("Cache chargé : %d lames, %d patches.", len(slide_data), total_patches)
        return slide_data, slide_labels

    # Repli : reconstruction si cache absent
    normal_dir = CAMELYON_ROOT / "0-normal"
    tumor_dir  = CAMELYON_ROOT / "1-tumor"
    slide_data: Dict[str, np.ndarray] = {}
    slide_labels: Dict[str, int] = {}

    for dir_path, label in [(normal_dir, LABEL_NORMAL), (tumor_dir, LABEL_TUMOR)]:
        for p in sorted(dir_path.glob("*.csv")):
            df = pd.read_csv(p, header=0)
            arr = df.values[:, :EMBEDDING_DIM].astype(np.float32)
            slide_data[p.stem] = arr
            slide_labels[p.stem] = label

    return slide_data, slide_labels


def get_slide_ids(slide_labels: Dict[str, int]) -> Tuple[List[str], List[int]]:
    slide_ids = sorted(slide_labels.keys())
    labels    = [slide_labels[sid] for sid in slide_ids]
    return slide_ids, labels
