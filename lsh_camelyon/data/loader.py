"""Chargement des embeddings de patches Camelyon16 avec support direct dossier/CSV et mise en cache."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.paths import (
    DSMIL_ROOT, CAMELYON_ROOT, NORMAL_DIR, TUMOR_DIR,
    NORMAL_INDEX_CSV, TUMOR_INDEX_CSV, RESULTS_RAW_DIR,
)

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
LABEL_NORMAL  = 0
LABEL_TUMOR   = 1
CACHE_FILE    = RESULTS_RAW_DIR / "camelyon16_slides_cache.pkl"


def _load_slide_csv(csv_path: Path) -> np.ndarray:
    """Charge un CSV de lame individuel (512 colonnes, 1 ligne = 1 patch)."""
    df = pd.read_csv(csv_path, header=0)
    arr = df.values.astype(np.float32)
    if arr.shape[1] > EMBEDDING_DIM:
        arr = arr[:, :EMBEDDING_DIM]
    elif arr.shape[1] < EMBEDDING_DIM:
        raise ValueError(
            f"{csv_path} : {arr.shape[1]} colonnes au lieu de {EMBEDDING_DIM}."
        )
    return arr


def _load_from_directory(dir_path: Path, class_label: int) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Charge toutes les lames .csv présentes dans un répertoire."""
    slide_data:   Dict[str, np.ndarray] = {}
    slide_labels: Dict[str, int]        = {}
    csv_files = sorted(dir_path.glob("*.csv"))
    for p in csv_files:
        try:
            arr = _load_slide_csv(p)
            slide_id = p.stem
            slide_data[slide_id]   = arr
            slide_labels[slide_id] = class_label
        except Exception as exc:
            logger.warning("Erreur lors de la lecture de %s : %s", p, exc)
    logger.info("  [classe=%d] %d lames chargées depuis le dossier %s", class_label, len(slide_data), dir_path)
    return slide_data, slide_labels


def _load_from_index_csv(index_csv: Path, class_label: int, dsmil_root: Path = DSMIL_ROOT) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Charge les lames listées dans un fichier CSV d'index."""
    df = pd.read_csv(index_csv, header=0, names=["path", "label"])
    slide_data:   Dict[str, np.ndarray] = {}
    slide_labels: Dict[str, int]        = {}
    for _, row in df.iterrows():
        rel_path = row["path"]
        csv_path = dsmil_root / rel_path
        if not csv_path.exists():
            # Essai relatif à CAMELYON_ROOT
            csv_path = CAMELYON_ROOT / Path(rel_path).name
        if not csv_path.exists():
            continue
        try:
            arr = _load_slide_csv(csv_path)
            slide_id = Path(rel_path).stem
            slide_data[slide_id]   = arr
            slide_labels[slide_id] = class_label
        except Exception as exc:
            logger.warning("Erreur sur %s : %s", csv_path, exc)
    logger.info("  [classe=%d] %d lames chargées depuis l'index %s", class_label, len(slide_data), index_csv.name)
    return slide_data, slide_labels


def load_slides(
    normal_source: Path | None = None,
    tumor_source:  Path | None = None,
    dsmil_root:    Path = DSMIL_ROOT,
    use_cache:     bool = True,
    cache_path:    Path = CACHE_FILE,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Charge l'ensemble des lames Camelyon16.

    Supporte automatiquement :
    - Les chemins de répertoires (e.g. 0-normal/, 1-tumor/)
    - Les chemins d'index CSV (e.g. 0-normal.csv, 1-tumor.csv)
    - Le chargement accéléré par cache binaire (.pkl)

    Args:
        normal_source : dossier ou fichier d'index des lames normales (défaut auto).
        tumor_source  : dossier ou fichier d'index des lames tumorales (défaut auto).
        dsmil_root    : racine DSMIL pour la résolution des chemins relatifs.
        use_cache     : tenter le chargement depuis le cache disque.
        cache_path    : emplacement du fichier cache.

    Returns:
        slide_data   : Dict[slide_id -> (n_patches, 512)].
        slide_labels : Dict[slide_id -> int].
    """
    if use_cache and cache_path.exists():
        logger.info("Chargement accéléré depuis le cache : %s ...", cache_path)
        try:
            with open(cache_path, "rb") as fh:
                slide_data, slide_labels = pickle.load(fh)
            total_patches = sum(arr.shape[0] for arr in slide_data.values())
            logger.info("Cache chargé avec succès : %d lames, %d patches.", len(slide_data), total_patches)
            return slide_data, slide_labels
        except Exception as exc:
            logger.warning("Erreur de lecture du cache (%s), rechargement depuis les CSV.", exc)

    # Détermination des sources normales
    if normal_source is None:
        normal_source = NORMAL_INDEX_CSV if NORMAL_INDEX_CSV.exists() else NORMAL_DIR
    if tumor_source is None:
        tumor_source = TUMOR_INDEX_CSV if TUMOR_INDEX_CSV.exists() else TUMOR_DIR

    # Chargement normal
    logger.info("Chargement des lames normales depuis %s ...", normal_source)
    if normal_source.is_dir():
        normal_data, normal_labels = _load_from_directory(normal_source, LABEL_NORMAL)
    else:
        normal_data, normal_labels = _load_from_index_csv(normal_source, LABEL_NORMAL, dsmil_root)

    # Chargement tumor
    logger.info("Chargement des lames tumorales depuis %s ...", tumor_source)
    if tumor_source.is_dir():
        tumor_data, tumor_labels = _load_from_directory(tumor_source, LABEL_TUMOR)
    else:
        tumor_data, tumor_labels = _load_from_index_csv(tumor_source, LABEL_TUMOR, dsmil_root)

    slide_data   = {**normal_data,   **tumor_data}
    slide_labels = {**normal_labels, **tumor_labels}

    total_patches = sum(arr.shape[0] for arr in slide_data.values())
    logger.info(
        "Total chargé : %d lames, %d patches (dim=%d).",
        len(slide_data), total_patches, EMBEDDING_DIM,
    )

    # Sauvegarde dans le cache
    if use_cache and len(slide_data) > 0:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Sauvegarde dans le cache %s ...", cache_path)
            with open(cache_path, "wb") as fh:
                pickle.dump((slide_data, slide_labels), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Cache sauvegardé avec succès.")
        except Exception as exc:
            logger.warning("Impossible de sauvegarder le cache : %s", exc)

    return slide_data, slide_labels


def get_slide_ids(slide_labels: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """Retourne les identifiants de lames dans un ordre déterministe."""
    slide_ids = sorted(slide_labels.keys())
    labels    = [slide_labels[sid] for sid in slide_ids]
    return slide_ids, labels


def build_matrices(
    slide_ids: List[str],
    slide_data: Dict[str, np.ndarray],
    slide_labels: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatène les patches de plusieurs lames en une seule matrice."""
    arrays = []
    labels = []
    for sid in slide_ids:
        patches = slide_data[sid]
        arrays.append(patches)
        labels.append(
            np.full(patches.shape[0], slide_labels[sid], dtype=np.int64)
        )
    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y
