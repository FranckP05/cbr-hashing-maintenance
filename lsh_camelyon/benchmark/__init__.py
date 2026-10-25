"""Module de benchmarking et d'évaluation des performances.

Rôle :
    Fournit le harnais de mesure précise du temps d'exécution (warm-up, répétitions, agrégations)
    et les fonctions d'évaluation de la qualité de voisinage (Recall@k) et de classification (F1).

Dépendances vis-à-vis des autres modules :
    - Fichiers amonts : config/ (protocol, seeds, paths), indexes/ (linear_scan, simhash).
    - Fichiers avals : calibration/, visualization/, scripts/, tests/.
"""
