# LSH Camelyon16 — Évaluation SimHash vs Balayage Linéaire à l'Échelle 10^6+

## Vue d'ensemble
Ce projet implémente un protocole expérimental rigoureux comparant la recherche de plus proches voisins par hachage sensible à la localité (**LSH - famille SimHash à hyperplans aléatoires**) face à un **balayage linéaire optimisé (BLAS/NumPy)** sur une base de données histopathologique réelle de grande échelle (**Camelyon16**, ~4,9 millions de patches, dimension 512).

L'objectif scientifique est d'évaluer directement l'hypothèse selon laquelle l'avantage computationnel du LSH n'apparaît qu'au-delà d'une masse critique de données ($10^5$ à $10^8$ points), alors qu'il est dominé par le balayage linéaire vectorisé à des échelles inférieures ($\le 72k$).

## Structure du Projet
```text
lsh_camelyon/
├── README.md                 # Documentation et instructions d'exécution
├── DECISIONS_AND_RESULTS.md  # Journal de continuité scientifique et technique
├── requirements.txt          # Dépendances Python (NumPy, Scikit-learn, Matplotlib, etc.)
├── config/                   # Paramètres globaux, protocoles, seeds et chemins
├── data/                     # Chargement WSI par lame, split stratifié 90/10, normalisation L2
├── indexes/                  # Implémentations d'index (LinearScan BLAS, SimHash LSH)
├── calibration/              # Grid search (k, L) sur validation uniquement
├── benchmark/                # Mesure rigoureuse du temps (warm-up, médianes) et métriques (F1, Recall@k)
├── visualization/            # Thème centralisé Matplotlib et scripts de génération de figures
├── scripts/                  # Points d'entrée CLI pour calibration, benchmark et pipeline complet
├── results/                  # Dossiers de sortie pour les figures et données brutes
└── tests/                    # Tests unitaires de validation (intégrité des splits, index, baseline)
```

## Prérequis et Installation
```bash
pip install -r requirements.txt
```

## Exécution
Tout le pipeline s'exécute en scripts Python purs sur l'environnement Lightning AI :

1. **Calibration de SimHash sur l'ensemble de validation :**
   ```bash
   python scripts/run_calibration.py
   ```
2. **Benchmark comparatif final sur l'ensemble de test :**
   ```bash
   python scripts/run_benchmark.py
   ```
3. **Pipeline complet (chargement, split, calibration, benchmark, tracés) :**
   ```bash
   python scripts/run_all.py
   ```
4. **Lancement des tests unitaires :**
   ```bash
   pytest tests/
   ```
