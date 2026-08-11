"""Seeds et hyperparamètres fixes de l'expérience (procédure a priori).

Le plan (docs/plan_experimentation_reproductible.md, §1.6) impose 6 seeds :
5 seeds d'expérience + 1 seed dédié à l'échantillonnage (bases 10k/50k/100k,
PCA exploratoire, sous-échantillonnage des condensations).
"""

SEEDS = [42, 7, 123, 2024, 99]
SAMPLING_SEED = 1337
ALL_SEEDS = SEEDS + [SAMPLING_SEED]

TEST_FRACTION = 0.2
K_VALUES = [1, 5, 10]
