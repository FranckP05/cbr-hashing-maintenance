# exp2 — Couplage hachage et maintenance en CBR

Expérience principale de ce travail. Elle vise à étudier le couplage entre l'indexation par hachage et la *case-base maintenance* (CBM), à la fois indépendamment et ensemble.

## Objectif

Comparer différents index pour la recherche par plus proches voisins : `LinearScan`, `KD-Tree`, `LSH` (SimHash/E2LSH selon les configurations), et une adaptation de `HeCBR` (hachage appris). Au-delà de la simple accélération de la recherche, on évalue comment réutiliser les structures de hachage pour :

- **Rétention en flux** (*streaming retention*) : décider d'insérer ou non un nouveau cas en s'appuyant sur les collisions/voisinage dans l'espace de hachage, sans balayage intégral systématique.
- **Condensation par buckets** (*bucket condensation*) : réduire la base en regroupant les cas partageant des signatures proches dans l'espace Hamming, en préservant la compétence.
- **Scaling** : observer le comportement aux tailles croissantes, jusqu'aux grands jeux (notamment ChestX-ray14).

## Jeux de données

| Jeu | Dimension | Usage |
|---|---|---|
| Heart Disease | ~13 | Contrôle, faible dimension |
| Diabetes-130 | variable selon preprocessing | Données tabulaires hétérogènes |
| ChestX-ray14 | 1024 (embeddings) | Haute dimension, mise en évidence des effets de concentration et du passage au multi-million |

## Pipeline (notebooks)

| Notebook | Rôle |
|---|---|
| `NB00_data_checks.ipynb` | Vérification des données, splits, cohérence |
| `NB01_metrics_protocol.ipynb` | Définition/mise au point des métriques et du protocole |
| `NB02_metrics.ipynb` | Métriques de base |
| `NB03_benchmark_index_seuls.ipynb` | Benchmark des index seuls (T1) |
| `NB04_figures_benchmark.ipynb` | Figures du benchmark |
| `NB05_scaling.ipynb` | Étude de scaling (T2) |
| `NB06_figures_scaling.ipynb` | Figures du scaling |
| `NB07_maintenance.ipynb` | Maintenance seule (rétention/condensation) (T3) |
| `NB08_figures_maintenance.ipynb` | Figures de la maintenance |
| `NB09_dimensionnalite.ipynb` | Étude dimensionnalité (T4) |
| `NB10_figures_dimensionnalite.ipynb` | Figures dimensionnalité |
| `NB11_statistique.ipynb` | Tests statistiques (T5) |
| `NB12_synthese_vainqueur.ipynb` | Synthèse et sélection (T6) |

Le code utilisé par ces notebooks est également généré/exporté dans `tools/cells/` pour faciliter la traçabilité.

## Structure du code

```
exp2/
├── config/        # Chemins, seeds, paramètres du protocole
├── notebooks/     # NB00-NB12
├── src/
│   ├── benchmark/      # Benchmarks index seuls, mémoire, scaling
│   ├── calibration/    # Calibration LSH/HeCBR
│   ├── datasets/       # Chargeurs Heart/Diabetes-130/ChestX-ray14
│   ├── dimensionality/ # Outils dimensionnalité
│   ├── evaluation/     # Métriques, tests statistiques, score vainqueur
│   ├── indexes/        # LinearScan, KD-Tree, LSH, HeCBR
│   ├── maintenance/    # Streaming retention, bucket condensation
│   ├── timing/         # Mesure temps/mémoire
│   └── visualization/  # Génération de figures
└── results/       # Résultats locaux/Kaggle
```

## Résultats

- **Run consolidé Kaggle** : `results/20260811_183845_kaggle/` (benchmarks, calibration, maintenance, `significance_tests.csv`, `synthesis_recap.csv`, etc.)
- **Bruts Kaggle** : `results/kaggle_raw/`
- **Synthèses tabulaires** : voir `../chapter3/RESULTATS_EXP2_INDEXATION.md` et `../chapter3/RESULTATS_EXP2_MAINTENANCE.md`

## Kaggle

L'exécution est pensée pour tourner aussi bien en local que sur Kaggle. `src/config/paths.py` résout les chemins selon le contexte (`/kaggle/input/...` ou chemins locaux). Les métadonnées pour le staging se trouvent dans `../tools/` et `../kaggle/` (non nécessaire à l'exécution). Les données ne sont pas toutes stockées dans le dépôt : sur Kaggle, elles proviennent des jeux attachés aux kernels.
