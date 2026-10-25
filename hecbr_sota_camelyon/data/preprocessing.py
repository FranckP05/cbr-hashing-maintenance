"""Normalisation L2 en place pour vecteurs 512-D."""

from __future__ import annotations

import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-10, in_place: bool = True) -> np.ndarray:
    """Normalise chaque ligne de X en norme L2 unitaire (in-place par blocs pour éviter l'OOM)."""
    if not in_place:
        X = X.copy()

    N = X.shape[0]
    BLOCK = 500_000
    for start in range(0, N, BLOCK):
        end = min(start + BLOCK, N)
        chunk = X[start:end]
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        np.maximum(norms, eps, out=norms)
        chunk /= norms

    return X
