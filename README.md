# Couplage entre hachage et maintenance de la base de cas en CBR

Ce dépôt regroupe le code expérimental, les notebooks et les résultats relatifs à l'étude du couplage entre des mécanismes de hachage et la *case-base maintenance* (CBM) dans un système de *Case-Based Reasoning* (CBR).

## Idée directrice

Jusqu'ici, l'indexation par hachage et la maintenance de la base de cas sont traitées séparément. Ici, on fait le contraire : plutôt que d'utiliser les codes de hachage uniquement comme un index approché pour accélérer la recherche, on les réutilise directement comme opérateurs pour la maintenance.

Concrètement, les collisions dans les tables de hachage, ainsi que le regroupement dans des sphères de Hamming, permettent d'identifier des cas redondants, de décider de la rétention en flux (*streaming retention*) ou de condenser la base (*bucket condensation*), sans avoir à calculer systématiquement toutes les paires de distances en espace continu.

L'objectif est donc d'agir à la fois sur les deux goulots d'étranglement :

- **Recherche en haute dimension** : réduire le temps de recherche et l'empreinte mémoire face au balayage linéaire.
- **Croissance de la base** : limiter l'accumulation de cas redondants tout en préservant la compétence du système CBR.

## Organisation

| Dossier | Description |
| --- | --- |
| `exp1/` | Expérimentations préliminaires et premières validations. |
| `exp2/` | Expérience principale. Compare LinearScan, KD-Tree, LSH et HeCBR. Étudie le scaling, la maintenance (rétention/condensation) et la dimensionnalité. C'est le cœur de l'étude couplée. |
| `exp_lsh/` | Étude avec E2LSH à distributions p-stables et SimHash (branche L2). |
| `exp_falconn/` | Étude avec Cross-Polytope LSH sur l'espace normalisé / distance cosinus. |
| `lsh_camelyon/` | Variante sur le jeu Camelyon. |
| `hecbr_sota_camelyon/` | Variante HeCBR sur Camelyon. |
| `demo/` | Démo Flask s'appuyant sur `exp2/src/`. |
| `draft/` | Protocole expérimental détaillé. |
| `docs/` | Documentation annexe. |
| `chapter3/` | Résultats agrégés sous forme de tableaux (`RESULTATS_*.md`), figures et éléments du chapitre 3. |
| `tools/` | Outils annexes (génération de notebooks, staging Kaggle, etc.). |
| `data/` | Données locales (certaines données volumineuses sont exclues ; voir section Kaggle). |
| `configs/` | Configurations partagées. |
| `notebooks/` | Notebooks partagés. |

## Jeux de données

Les expérimentations couvrent des jeux de dimensions et de tailles variées : *Heart Disease* (faible dimension), *Diabetes-130* et *ChestX-ray14* (haute dimension). Des expériences à plus grande échelle sont également réalisées pour mettre en évidence l'effet du couplage hachage-maintenance.

## Schémas de hachage étudiés

- **Dépendant des données aléatoires / classiques** : SimHash, E2LSH, Cross-Polytope LSH
- **Dépendant des données appris** : adaptation de *HeCBR* (hachage supervisé) utilisé comme cas représentatif du hachage appris pour ce couplage

## Reproduction

- **Protocole** : `draft/protocol.md`
- **Pipeline principal (exp2)** : notebooks `NB00` à `NB12` dans `exp2/notebooks/`
- **Autres expériences** : notebooks dans `exp_lsh/notebooks/` et `exp_falconn/notebooks/`
- **Installation** : `pip install -r requirements.txt` (un environnement virtuel est recommandé). Certains sous-projets ont leur propre `requirements.txt`

## Résultats

| Emplacement | Contenu |
| --- | --- |
| `exp2/results/20260811_183845_kaggle/` | Run Kaggle consolidé (benchmarks, calibration, maintenance, tests statistiques, synthèse). |
| `exp2/results/kaggle_raw/` | Résultats bruts issus des kernels Kaggle. |
| `exp_lsh/results/` | Résultats E2LSH/SimHash. |
| `exp_falconn/results/` | Résultats Cross-Polytope LSH. |
| `chapter3/RESULTATS_*.md` | Synthèses tabulaires générées directement à partir des CSV. |

## Kaggle

Sur Kaggle, les jeux de données sont montés sous `/kaggle/input/`. La résolution des chemins est gérée par `exp2/config/paths.py` (local vs Kaggle). Le staging des *datasets/kernels* se fait via les scripts dans `tools/`. Les données brutes volumineuses ne sont pas nécessairement présentes dans ce dépôt, elles proviennent des jeux attachés aux kernels lors des exécutions Kaggle.
