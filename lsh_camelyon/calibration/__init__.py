"""Module de calibration des hyperparamètres LSH.

Rôle :
    Fournit la logique de recherche sur grille (grid search) sur l'ensemble de validation
    afin d'identifier la configuration (k, L) optimale et la plus légère respectant
    le seuil de qualité cible.

Dépendances vis-à-vis des autres modules :
    - Fichiers amonts : config/, data/, indexes/, benchmark/.
    - Fichiers avals : scripts/run_calibration.py, scripts/run_all.py.
"""
