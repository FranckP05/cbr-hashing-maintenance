"""Normalisation L2 des représentations vectorielles avec optimisation mémoire."""

from __future__ import annotations

import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-10, in_place: bool = True) -> np.ndarray:
    """Normalise chaque ligne de X en norme L2 unitaire.

    Optimisé pour les très grands volumes (4.88M+ vecteurs) en évitant
    les duplications mémoire inutiles.

    Après cette opération :
      - ||x_i||_2 == 1  pour tout i
      - ||p - q||_2^2 == 2 - 2*cos(p, q)  (équivalence cosinus <-> L2 sur la sphère)
      - SimHash est correctement aligné sur la métrique cosinus

    Args:
        X        : np.ndarray de forme (N, d), dtype float32.
        eps      : plancher numérique pour éviter la division par zéro.
        in_place : si True, modifie X sur place pour économiser la RAM.

    Returns:
        X_normalized : np.ndarray de forme (N, d), float32, ||x||_2 = 1.
    """
    if not in_place:
        X = X.copy()
    
    # Calcul par bloc si la matrice dépasse 1M lignes pour limiter les pics temporaires
    N = X.shape[0]
    BLOCK = 500_000
    for start in range(0, N, BLOCK):
        end = min(start + BLOCK, N)
        chunk = X[start:end]
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        np.maximum(norms, eps, out=norms)
        chunk /= norms

    return X


def check_normalized(X: np.ndarray, tol: float = 1e-4) -> bool:
    """Vérifie que toutes les lignes de X ont une norme proche de 1.

    Args:
        X   : np.ndarray de forme (N, d).
        tol : tolérance absolue autour de 1.0.

    Returns:
        True si toutes les normes sont dans [1-tol, 1+tol].
    """
    norms = np.linalg.norm(X, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))
