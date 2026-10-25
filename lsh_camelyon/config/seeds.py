"""Graines aléatoires pour la reproductibilité et la robustesse statistique."""

# Graine utilisée pour le partitionnement stratifié 90/10 des lames
SPLIT_SEED: int = 42

# Graine unique utilisée pendant la phase de calibration (grid search sur validation)
CALIBRATION_HASH_SEED: int = 11

# Liste de graines indépendantes pour le benchmark final
# (évalue la robustesse aux projections aléatoires de SimHash)
BENCHMARK_HASH_SEEDS = [11, 29, 43, 67, 97]
