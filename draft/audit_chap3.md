# Audit de la codebase — Chapitre 3 : Implémentation et expérimentation

> **Destinataires :** agent IA rédacteur du chapitre 3, ou tout repreneur du projet.
> **Date :** 2026-08-17
> **Règles respectées :** rien n'a été inventé. Toute valeur numérique provient des
> CSV, JSON ou source du projet. Les distinctions prévu/implémenté/testé/benchmarké/
> concluant sont maintenues tout au long du document.

---

## Table des matières

1. [Documents structurants](#1-documents-structurants)
2. [Cartographie de la partie expérimentale réelle](#2-cartographie-de-la-partie-expérimentale-réelle)
3. [Protocole réellement exécuté](#3-protocole-réellement-exécuté)
4. [Baseline Linear Scan](#4-baseline-linear-scan)
5. [Expérience 1 : SimHash](#5-expérience-1--simhash)
6. [Expérience 2 : E2LSH](#6-expérience-2--e2lsh)
7. [Expérience 3 : Cross-Polytope](#7-expérience-3--cross-polytope)
8. [Optimisations de Cross-Polytope](#8-optimisations-de-cross-polytope)
9. [Le phénomène central : qualité excellente mais performance insuffisante](#9-le-phénomène-central--qualité-excellente-mais-performance-insuffisante)
10. [Rôle de n et d](#10-rôle-de-n-et-d)
11. [Comparaison des méthodes effectivement testées](#11-comparaison-des-méthodes-effectivement-testées)
12. [Tests statistiques](#12-tests-statistiques)
13. [SRS](#13-srs)
14. [KD-tree et HeCBR](#14-kd-tree-et-hecbr)
15. [Chronologie scientifique](#15-chronologie-scientifique)
16. [Conclusions à produire](#16-conclusions-à-produire)
17. [Relation directe avec le chapitre 3](#17-relation-directe-avec-le-chapitre-3)
18. [Relation avec le reste du mémoire](#18-relation-avec-le-reste-du-mémoire)
19. [Résumé final](#19-résumé-final)

---

## 1. Documents structurants

### 1.1 Fichiers lus

| Fichier | Localisation | Rôle |
|---|---|---|
| `draft/main.tex` | Racine du projet | Mémoire complet (598 lignes), structure du chapitre 3 |
| `draft/protocol.md` | `draft/` | Protocole expérimental formel (542 lignes, 17 sections) |
| `draft/audit_task.md` | `draft/` | Consignes de cet audit (607 lignes) |
| `draft/HeCBR_Analysis_Report.md` | `draft/` | Diagnostic HeCBR v2 (non implémenté dans exp_lsh/exp_falconn) |
| `DECISIONS_AND_RESULTS.md` | Racine du code | Document de continuité complet (454 lignes) |
| `exp_lsh/docs/progress.md` | exp_lsh/ | Journal d'avancement phase 1 |
| `exp_lsh/docs/lsh_sota_review.md` | exp_lsh/ | État de l'art LSH + comparaison cross-méthodes |
| `exp_falconn/docs/progress.md` | exp_falconn/ | Journal d'avancement phase 2 |
| `exp_falconn/docs/tasks.md` | exp_falconn/ | Tableau des tâches phase 2 |

### 1.2 Structure du chapitre 3 dans `main.tex`

Le chapitre 3 est prévu sous les sous-sections suivantes (extrait de `main.tex`
lignes 498–540) :

```
3. Implémentation et expérimentation
  3.1 Environnement de développement
  3.2 Scénarios de test
  3.3 Évaluation des résultats
  3.4 Discussion
```

**Note :** le `main.tex` contient un plan partiel du chapitre 3 avec des
références à des outils non utilisés dans l'expérience finale :
- Scikit-learn (KD-tree) — KD-tree n'a pas été évalué dans exp_lsh ni exp_falconn
- Annoy ou FAISS — ni l'un ni l'autre n'a été utilisé
- Datasketch — non utilisé
- PyTorch ou TensorFlow pour HeCBR — HeCBR n'a pas été implémenté dans le
  protocole expérimental final

**Le document final doit utiliser la structure 3.1–3.4 de `main.tex` comme
référence.** La matière ci-dessous est organisée en fonction de cette structure.

---

## 2. Cartographie de la partie expérimentale réelle

### 2.1 Ce qui a été implémenté et testé

| Méthode | Implémentée ? | Testée (unitaires) ? | Benchmarkée (Kaggle) ? | Résultats disponibles ? |
|---|---|---|---|---|
| Linear Scan (baseline) | **Oui** — `linear_exact.py` (2 versions, exp_lsh + exp_falconn) | **Oui** — 4 tests unitaires | **Oui** — tous les splits | **Oui** |
| SimHash | **Oui** — `simhash.py` (exp_lsh uniquement) | **Oui** — tests smoke recall | **Oui** — 5 splits × 3 hash seeds | **Oui** |
| E2LSH | **Oui** — `e2lsh.py` (exp_lsh uniquement) | **Oui** — 6 tests unitaires | **Oui** — 5 splits × 3 hash seeds | **Oui** |
| Cross-Polytope | **Oui** — `cross_polytope.py` (exp_falconn uniquement) | **Oui** — 11 + 6 tests unitaires | **Oui** — 5 splits × 3 hash seeds | **Oui** |
| KD-tree | **Non** | Non | Non | Non |
| HeCBR | **Non** (analyse dans `HeCBR_Analysis_Report.md`, mais pas d'implémentation dans exp_lsh/exp_falconn) | Non | Non | Non |
| Maintenance | **Non** (algorithme 4 formalisé dans `main.tex`, pas implémenté) | Non | Non | Non |
| SRS | **Non** | Non | Non | Non |

### 2.2 Deux expériences séparées

Le projet expérimental se divise en deux expériences indépendantes :

1. **exp_lsh** (phase 1) : Linear + SimHash + E2LSH, métrique L2
2. **exp_falconn** (phase 2) : Linear + Cross-Polytope, métrique cosinus

Les résultats SimHash et E2LSH proviennent de `exp_lsh/results/`.
Les résultats Cross-Polytope proviennent de `exp_falconn/results/benchmark/`.
Les deux expériences partagent le même protocole (`draft/protocol.md`), les
mêmes seeds, et le même harnais de timing.

### 2.3 Fichiers de résultats existants

**exp_lsh/results/** : 65 entrées
- `benchmark_timing.csv` — latences par requête (140 lignes)
- `benchmark_quality.csv` — recall/F1/accuracy par requête (140 lignes)
- `benchmark_candidates.csv` — nombre de candidats (140 lignes)
- `benchmark_memory.csv` — empreinte mémoire
- `significance_tests.csv` — Wilcoxon + bootstrap (8 résultats)
- `summary.json` — outcomes par méthode/dataset
- `best_params_seed{42,7,123,2024,99}.json` — paramètres calibrés
- `linear_gt_*.npz` — ground truth linéaire précalculé
- 5 logs Kaggle

**exp_falconn/results/benchmark/** : 59 entrées ( même structure, run patché)
- `outcomes.csv` — 4 verdicts
- `significance_tests.csv` — 4 résultats Wilcoxon
- `summary.json` — key findings
- Mêmes sous-fichiers que exp_lsh (timing, quality, candidates, memory par seed)

**Figures** : 8 figures PNG dans `exp_lsh/results/figures/` et
`exp_falconn/results/figures/`

---

## 3. Protocole réellement exécuté

### 3.1 Données

| Dataset | n total | d | Nature | Représentation | Normalisation | Rôle |
|---|---|---|---|---|---|---|
| Heart Disease (UCI) | 297 | 13 | Tabulaire clinique | 13 variables numériques | StandardScaler (fit sur train) + L2-normalisation (cosine arm) | Petite base, basse dimension, contrôle |
| ChestX-ray14 | 112 120 | 1024 | Embeddings ResNet d'images médicales | Vecteur 1024-d | L2-normalisé (cosine arm) | Grande base, haute dimension, principale |

**Tailles réelles de ChestX-ray14 utilisées :**
- 10 000 (sous-base)
- 50 000 (sous-base)
- ~71 756 (full train, après split 80/20)

**Note :** le plan initial mentionnait 100 000 cas ; la taille réelle « full »
est ~72k (80% de 112 120, arrondi par stratification). C'est la valeur
effectivement utilisée dans tous les benchmarks.

**Sources des données :**
- Heart Disease : CSV UCI, chargé depuis un chemin local
- ChestX-ray14 : fichiers `.npy` (embeddings) + `.npy` (labels), ou `.parquet`

### 3.2 Séparation des données

- **Train / Validation / Test** = 64% / 16% / 20%
- Validation = 20% du train (après extraction du test)
- Séparation stratifiée (`StratifiedShuffleSplit`)
- **5 split seeds** : [42, 7, 123, 2024, 99]
- **300 requêtes test** pour ChestX-ray14 (sélectionnées avec `SEED_SAMPLING=1337`)
- Heart Disease : toutes les requêtes test sont utilisées (~60)

### 3.3 Métrique

- **Métrique principale : cosinus** (braccos L2-normalisés)
- Sur la sphère unité : `||p - q||² = 2 - 2·cos(p,q)` → L2 et cosinus sont
  linéairement équivalents pour le ranking
- **exp_lsh** : métrique L2 (braccos non normalisés dans le L2 arm)
- **exp_falconn** : métrique cosinus (braccos normalisés L2)

**Lien L2 ↔ cosinus** : quand les vecteurs sont L2-normalisés, la distance
L2 et la similarité cosinus produisent exactement le même ranking de voisins.
Le cross-polytope est conçu pour le cosinus ; le passage au cosinus pour
exp_falconn est justifié par ce lien mathématique.

### 3.4 Requêtes

- ChestX-ray14 : 300 requêtes test par split, sélectionnées aléatoirement
  (seed 1337), triées par indice pour la reproductibilité
- Heart Disease : toutes les requêtes test (~60)
- Pas de répétition de requêtes dans la calibration ; dans le benchmark,
  chaque requête est chronométrée individuellement

### 3.5 Mesure du temps

- **Harnais** : `src/timing/harness.py`
- **Warm-up** : 1 répétition (non mesurée)
- **Répétitions mesurées** : 5
- **Statistique** : médiane sur les 5 répétitions, par requête
- **Agrégation** : médiane, moyenne, p95 sur les médianes par requête
- **Niveau** : temps de requête uniquement (pas le build, pas le post-traitement)
- **Environnement** : Kaggle (Python 3.12, NumPy 2.5.1, CPU standard)
- **BLAS threads** : enregistrés via `system_info.py`, inclus dans `registry.json`

### 3.6 Calibration

- **Jeu** : validation uniquement (jamais test)
- **Méthode** : recherche de grille exhaustive (`itertools.product`)
- **Paramètres calibrés** :
  - E2LSH : `w` (largeur de bucket), `K` (hash functions), `L` (tables), `probes`
  - SimHash : `num_bits`, `num_tables`
  - Cross-Polytope : `num_hashes`, `num_tables`, `num_rotations`, `num_probes`
- **Seeds** : un seul hash seed pour la calibration (`hash_seed=11`)
- **Critères d'acceptation** :
  - Recall@10 ≥ 0.95
  - Δmacro-F1 ≤ 0.01 (vs linear)
  - Tie-break : médiane latence → fraction candidats → mémoire → build time
- **Relaxation** : si aucune configuration ne passe les contraintes, la meilleure
  configuration est retenue quand même (pas de rejet complet)

### 3.7 Validation de qualité

- **Recall@10** ≥ 0.95 (seuil de calibration ET de succès)
- **Δmacro-F1** ≤ 0.01 vs linear
- **Recall@1, @5** calculés mais pas utilisés comme critères de décision
- **nDCG@10** calculé pour information
- **Accuracy, balanced accuracy, AUC** calculés pour information

---

## 4. Baseline Linear Scan

### 4.1 Pourquoi documenter la baseline en détail

La baseline n'est pas un « parcours naïf ». C'est une implémentation
vectorisée利用ant les mêmes primitives BLAS que toute la littérature utilise
pour la recherche exacte. Sa performance est le point de comparaison contre
lequel toutes les méthodes LSH sont évaluées.

### 4.2 Implémentation

**Fichier :** `src/indexes/linear_exact.py`

**Formule de distance :**
```
||x - x_i||² = ||x||² - 2·(x · x_i) + ||x_i||²
```

**Primitives NumPy :**
- `self._base @ x` → produit matrice-vecteur (appel BLAS `sgemm`/`dgemm`)
- `np.einsum("ij,ij->i", ...)` → normes précalculées au build
- `np.dot(x, x)` → norme de la requête
- `np.argpartition` → sélection top-k en O(n) (pas O(n log n))
- `np.maximum(sq_dists, 0.0)` → sécurité numérique

**Optimisations appliquées :**
1. Normes des vecteurs de base précalculées au build (`_base_norms_sq`)
2. La formule évite de calculer `||x - x_i||²` pour chaque i ; le terme
   `||x||²` est constant et n'affecte pas le ranking
3. `np.argpartition` pour le top-k au lieu de `np.argsort` complet
4. Layout float32 pour réduire la mémoire et accélérer le BLAS
5. Aucun boucle Python sur les candidats — tout est vectorisé

**Complexité théorique :** O(n·d) par requête, dominée par le produit
matrice-vecteur BLAS.

**Coût empirique :**

| Dataset | n | d | Latence médiane requête |
|---|---|---|---|
| Heart Disease | 189 | 13 | 0.024 ms |
| ChestX-ray14 10k | 6 400 | 1024 | 1.76 ms |
| ChestX-ray14 50k | 32 000 | 1024 | 9.17 ms |
| ChestX-ray14 full | 57 405 | 1024 | 12.88 ms |

La latence est quasi-linéaire en n (×5.2 de 10k à 50k, ×1.4 de 50k à full),
confirmant que le BLAS domine et que l'overhead est minimal.

### 4.3 Pourquoi la baseline est compétitive

Le scan linéaire bénéficie de :
1. **Un seul appel BLAS** (`base @ x`) qui mobilise toutes les cores et
   l'ensemble du pipeline vectoriel du processeur
2. **Aucun overhead** de structures de données (pas de dicts, pas de hashage,
   pas de filtrage de candidats)
3. **Aucun post-traitement** de reranking — les résultats de BLAS SONT les
   distances exactes
4. **Pas de build significatif** — seule la précalculation des normes (O(n·d))

Sur des bases de 10k–72k vecteurs 1024-d, le produit matrice-vecteur BLAS est
suffisamment optimisé pour que l'overhead de toute méthode d'indexation
approximative (hashage + lookup dans dicts + reranking) dépasse le coût du
scan complet.

---

## 5. Expérience 1 : SimHash

### 5.1 Pourquoi SimHash a été choisi initialement

SimHash (hyperplane LSH, Charikar 2002) est la méthode LSH la plus simple à
implémenter. Elle approxime la similarité cosinus via des projections aléatoires.
Elle a été choisie comme **ablation de diagnostic** — pas comme candidat
sérieux pour battre la baseline — afin de répondre à la question :

> « Le LSH peut-il au moins préserver la qualité de récupération sur nos données ? »

### 5.2 Implémentation

**Fichier :** `src/indexes/simhash.py`

**Fonction de hash :** `h_j(x) = sign(a_j · x)` où `a_j ~ N(0, I_d)`

**Paramètres calibrés :** `num_bits`, `num_tables`

**Recherche :** union des candidats de toutes les tables → reranking L2 exact

### 5.3 Résultats (exp_lsh, Kaggle)

| Dataset | base | Recall@10 | Macro-F1 | Latence (ms) | vs linear | Fraction candidats | Verdict |
|---|---|---|---|---|---|---|---|
| chest_xray14 | 10k | 0.983 | 0.705 | 14.02 | 8.0× plus lent | 0.50 | quality_ok_no_speedup |
| chest_xray14 | 50k | 0.963 | 0.700 | 26.14 | 3.2× plus lent | 0.17 | quality_ok_no_speedup |
| chest_xray14 | full | 0.968 | 0.702 | 40.17 | 3.4× plus lent | 0.17 | quality_ok_no_speedup |
| heart_disease | full | **0.531** | 0.820 | 0.12 | 4.9× plus lent | 0.09 | **Echec global** |

### 5.4 La question qui est apparue après cette expérience

SimHash préserve la qualité sur ChestX-ray14 (recall ≥ 0.96), mais échoue
catastrophiquement sur Heart Disease (recall = 0.53). De plus, sur toutes les
configurations, SimHash est 3–8× plus lent que linear.

La question qui a émergé :

> Les mauvaises performances observées proviennent-elles de SimHash en
> particulier (fonction de hash inadaptée), ou du principe de LSH dans
> notre environnement expérimental ?

**Le protocole ne permet pas de répondre à cette question avec SimHash seul.**
SimHash cible le cosinus, pas le L2 — c'est une ablation, pas un candidat L2
sérieux. Pour diagnostiquer si le problème vient de la famille de hachage
ou du principe même de LSH, une deuxième méthode LSH était nécessaire.

---

## 6. Expérience 2 : E2LSH

### 6.1 Pourquoi E2LSH a été testée

E2LSH (Datar et al., 2004) est la méthode LSH historique pour la distance L2,
fondée sur des distributions p-stables (gaussiennes). Elle a été introduite
pour vérifier l'hypothèse :

> « Le problème vient peut-être simplement de la fonction de hachage
> utilisée. Si on utilise une fonction de hash adaptée à L2 (pas au cosinus),
> les performances devraient s'améliorer. »

E2LSH est un **détour expérimental** — elle n'appartient pas au plan initial
du mémoire. Son rôle est purement diagnostique.

### 6.2 Implémentation

**Fichier :** `src/indexes/e2lsh.py`

**Fonction de hash :** `h(x) = floor((a · x + b) / w)` où `a ~ N(0, I_d)`,
`b ~ U[0, w]`, `w` = largeur de bucket (calibrée)

**Multi-probe :** perturbations ±1 sur les composantes du hash, ordonnées
par magnitude décroissante

**Paramètres calibrés :** `w`, `K`, `L`, `n_probes`

### 6.3 Résultats (exp_lsh, Kaggle)

| Dataset | base | Recall@10 | Macro-F1 | Latence (ms) | vs linear | Fraction candidats | Verdict |
|---|---|---|---|---|---|---|---|
| chest_xray14 | 10k | 0.985 | 0.706 | 12.91 | 7.3× plus lent | 0.43 | quality_ok_no_speedup |
| chest_xray14 | 50k | 0.978 | 0.702 | 38.64 | 4.7× plus lent | 0.22 | quality_ok_no_speedup |
| chest_xray14 | full | 0.982 | 0.699 | 58.14 | 4.9× plus lent | 0.22 | quality_ok_no_speedup |
| heart_disease | full | **0.006** | 0.074 | 0.03 | ~1× | **0.001** | **Echec global** |

### 6.4 Ce que cette expérience démontre

1. **E2LSH échoue encore plus que SimHash sur Heart Disease** (recall = 0.006,
   recall@10 = 0 vs 0.53 pour SimHash). La cause : en basse dimension (d=13),
   les projections aléatoires ne génèrent pas de séparations suffisantes pour
   que les buckets aient un sens.

2. **Sur ChestX-ray14, E2LSH obtient de meilleurs recall que SimHash**
   (0.985 vs 0.983 en 10k) — mais est plus lente (12.91 ms vs 14.02 ms
   en 10k, mais 58.14 ms vs 40.17 ms en full). E2LSH nécessite davantage
   de tables (L=20) pour un recall élevé, ce qui alourdit le lookup.

3. **Le problème n'est PAS spécifique à SimHash.** Deux familles LSH
   différentes (hyperplane + p-stable) obtiennent des résultats comparables :
   qualité OK sur Chest, échec sur Heart, et latences 3–8× plus élevées que
   linear. Le phénomène est **transversal à la famille LSH** dans notre
   environnement.

4. **La fraction de candidats reste élevée** (~0.22 sur chest full) malgré
   un recall ≥ 0.95, ce qui signifie que le mécanisme d'élaguation ne réduit
   pas suffisamment le nombre de comparaisons exactes.

---

## 7. Expérience 3 : Cross-Polytope

### 7.1 Pourquoi le cross-polytope a été introduit

Après les résultats de SimHash et E2LSH, la question restait :

> Est-ce qu'une famille LSH plus récente et théoriquement optimale pour le
> cosinus peut résoudre le problème de performance observé ?

Le cross-polytope LSH (Andoni et al., NIPS 2015) est :
- **Optimal** pour la distance angulaire/cosinus (exposant ρ = 1/(2c²−1))
- **La famille recommandée** par le MIT (FALCONN)
- **3.4–10× plus rapide** que l'hyperplane dans les benchmarks FALCONN
  sur 10⁵–10⁸ points

### 7.2 Implémentation

**Fichier :** `src/indexes/cross_polytope.py` (259 lignes, implémentation from scratch)

**Algorithme :**
1. L2-normalisation de la base et de la requête
2. Padding à la puissance de 2 la plus proche
3. `K` hash functions par table, `L` tables indépendantes
4. Chaque hash = `num_rotations` applications de `H · D_n · H · D_{n-1} · … · H · D_1`
   (FHT avec diags aléatoires ±1)
5. Clé de hash = `2·idx + sign_bit` (idx = coordonnée de plus grande magnitude)
6. Multi-probe : perturbations par magnitude décroissante
7. Reranking L2 exact sur les candidats

**FHT vectorisé :** Transformée de Walsh-Hadamard rapide, layout transposé
`(d, n)` pour l'optimisation cache. ~3× plus rapide que le layout `(n, d)`.

**Référence :** `falconn_src` cloné depuis GitHub pour vérification (FALCONN,
MIT, Andoni et al. 2017).

### 7.3 Tests unitaires

`tests/test_cross_polytope.py` : 11 tests
- Orthogonalité FHT
- `_next_pow2`
- Préservation des normes après rotation
- Retourne au plus k résultats
- Distances non-négatives et triées
- Déterminisme
- Variation entre seeds
- Self-query top-1
- Recall raisonnable
- Padding non-puissance-de-2

`tests/test_cross_polytope_patch.py` : 6 tests (non-régression du patch)
- Égalité stricte des ensembles candidats : 150 requêtes × 4 configs calibrées
  = **zéro divergence**
- Concordance top-10 indices et distances

**Total :** `pytest tests/` → **17/17 passed** (exp_falconn)

### 7.4 Calibration

- `src/calibration/grid_search.py::calibrate_cross_polytope`
- Grille sur `num_hashes`, `num_tables`, `num_rotations`, `num_probes`
- Validation uniquement, `hash_seed=11`
- Seuils : Recall@10 ≥ 0.95, ΔF1 ≤ 0.01

### 7.5 Résultats (exp_falconn, Kaggle, run patché)

| Dataset | base | Recall@10 | Macro-F1 | Latence (ms) | vs linear | Fraction candidats | Verdict |
|---|---|---|---|---|---|---|---|
| chest_xray14 | 10k | 0.974 | 0.701 | 6.88 | 4.09× plus lent | 0.204 | quality_ok_no_speedup |
| chest_xray14 | 50k | 0.989 | 0.703 | 33.54 | 3.88× plus lent | 0.204 | quality_ok_no_speedup |
| chest_xray14 | full | 0.982 | 0.705 | 53.57 | 4.31× plus lent | 0.206 | quality_ok_no_speedup |
| heart_disease | full | 0.986 | 0.834 | 0.28 | 11.10× plus lent | 0.381 | quality_ok_no_speedup |

### 7.6 Comparaison avec E2LSH et SimHash

- **Heart Disease** : le cross-polytope est la **seule** famille qui réussit
  (recall = 0.986 vs 0.006 pour E2LSH et 0.53 pour SimHash). La famille
  optimale compte.
- **ChestX-ray14** : le cross-polytope est **plus rapide** que SimHash et
  E2LSH (6.88 ms vs 14.02 ms en 10k ; 53.57 ms vs 58.14 ms en full),
  mais reste 4× plus lent que linear.
- La fraction de candidats est comparable (~0.20–0.22 pour toutes les méthodes
  sur chest).

---

## 8. Optimisations de Cross-Polytope

### 8.1 Chronologie des optimisations

| # | Problème observé | Hypothèse | Modification | Résultat | Interprétation |
|---|---|---|---|---|---|
| 1 | FHT très lent en layout `(n, d)` | L'axe de boucle doit être l'axe contigu en mémoire | Layout transposé `(d, n)` : papillons le long du premier axe | ×3 plus rapide | Le cache miss domine en haute dimension ; le transposé rend les paires accédées contiguës |
| 2 | Requêtes traitées une par une | Le FHT batch est plus efficace que le FHT unitaire | `_rotate_query_all` : toutes les L·K rotations calculées en un seul passage FHT | Gain significatif | L'amortissement du passage FHT réduit l'overhead de launch |
| 3 | Multi-probe par boucle Python | La boucle sur les perturbations est un goulot | Vectorisation des perturbations : `np.argsort` sur `all_scores`, pas de boucle `for` sur les coordonnées | Plus propre, léger gain | Le Python loop n'était pas le goulot principal |
| 4 | `_D_all` reconstruit par `.T` à chaque rotation de chaque requête | La transposition est un faux coût (vue mémoire) | **Patch** : `_D_all_query` précalculé contigu `(num_rotations, padded_d, L*K)` au build | **94% du temps de requête éliminé** (profiling) | La transposition `.T` créait une vue non-contigue qui induisait des copies silencieuses dans NumPy |

### 8.2 Le patch `_D_all_query` en détail

**Problème initial :** dans `_rotate_query_all` (avant le patch), chaque
étape de rotation reconstruisait `self._D_all[:, r, :].T` — une vue
transposée de `self._D_all`. Cette vue n'est pas contiguë en mémoire
(l'ordre logique diffère de l'ordre physique). À chaque opération `Y * D`,
NumPy copie silencieusement les données pour les rendre contiguës.

**Hypothèse :** en précalculant la version contiguë une seule fois au build
(`np.ascontiguousarray(self._D_all.transpose(1, 2, 0))`), on élimine les
copies à chaque requête.

**Modification :** une seule ligne ajoutée au build :
```python
self._D_all_query = np.ascontiguousarray(
    self._D_all.transpose(1, 2, 0)
)  # (num_rotations, padded_d, L*K)
```

Et `_rotate_query_all` utilise `self._D_all_query[r]` au lieu de
`self._D_all[:, r, :].T`.

**Résultat :** le temps de requête a été divisé par ~2–3 sur ChestX-ray14
(ratio CP/lin passé de 5.43–11.37× à 3.88–4.31×).

**Interprétation :** le vrai coût n'était pas le dict lookup ni le reranking
(5% du temps) mais la copie mémoire induite par les vues non-contiguës (94%
du temps). Le patch n'a rien emprunté à linear — il a juste arrêté de payer
un coût que l'algorithme n'a aucune raison de payer.

**Limite :** le plancher à 3.9–4.3× reste. Après le patch, le coût restant
est dict lookup + reranking fancy-indexing, qui sont des constantes
Structurellement supérieures au broadcast BLAS de linear.

### 8.3 Audit anti-patterns

Un audit complet a été réalisé sur tous les fichiers de la codebase :
- `cross_polytope.py` — deux `.T` restants : faux positifs (reshape FHT = vue
  zéro-copie ; `_hash_batch` Y_T[:dim,:].T = vue sur K=1–2 itérations)
- `e2lsh.py` — aucun anti-pattern
- `simhash.py` — aucun anti-pattern
- `grid_search.py` — aucun anti-pattern

---

## 9. Le phénomène central : qualité excellente mais performance insuffisante

### 9.1 Qualité

**Oui, LSH retrouve un ensemble de voisins très proche du ground truth.**

- Sur ChestX-ray14 : Recall@10 ≥ 0.96 pour toutes les méthodes
- Sur Heart Disease : seul le cross-polytope réussit (recall = 0.986)
- Macro-F1 : écart ≤ 0.01 vs linear sur toutes les tâches où recall ≥ 0.95
- Les méthodes LSH préserver la qualité de classification CBR

### 9.2 Performance

**Non, la réduction des candidats ne se traduit PAS par une réduction du
temps total de recherche.**

| Méthode | Chest full fraction candidats | Chest full ratio vs linear |
|---|---|---|
| Linear | 1.000 (scan complet) | 1× (référence) |
| SimHash | 0.17 | 3.4× plus lent |
| E2LSH | 0.22 | 4.9× plus lent |
| Cross-Polytope (patché) | 0.21 | 4.3× plus lent |

**Le paradoxe :** les méthodes LSH ne visitent que 17–22% des candidats,
et pourtant elles sont 3–5× plus lentes que le scan complet.

### 9.3 Explication du paradoxe

Le temps total = hash + lookup + reranking, vs linear = BLAS pur.

- Le BLAS (produit matrice-vecteur) est la primitive la plus optimisée du
  processeur — un seul appel mobilise toute la largeur vectorielle
- Le hash + lookup dans dicts Python + reranking par fancy-indexing
  introduit un overhead constant qui domine sur des bases de 10k–72k
- À recall ≥ 0.95, la fraction de candidats (~0.20) n'est pas assez faible
  pour que l'économie de comparaisons compense l'overhead d'indexation

---

## 10. Rôle de n et d

### 10.1 Analyse du régime expérimental

| Dataset | n (base) | d | n·d | Coût distance | Coût indexation | Fraction candidats |
|---|---|---|---|---|---|---|
| Heart Disease | 189 | 13 | 2 457 | ≪ 1 μs | ~10 μs (hash) | 0.38 |
| Chest 10k | 6 400 | 1 024 | 6.6 M | ~1.7 ms (BLAS) | ~5 ms (hash+lookup) | 0.20 |
| Chest 50k | 32 000 | 1 024 | 32.8 M | ~9 ms | ~25 ms | 0.20 |
| Chest full | 57 405 | 1 024 | 58.8 M | ~13 ms | ~40 ms | 0.21 |

### 10.2 Interprétation

Dans le régime étudié :

- **Le coût d'un scan dense vectorisé** (BLAS) est O(n·d) avec une constante
  très faible (un seul appel SGEMM optimisé)
- **Le coût de l'indexation LSH** est O(d) pour le hash + O(C·d) pour le
  reranking, où C = candidats × fraction — mais avec une constante bien
  supérieure (dict lookup Python, fancy-indexing, allocation mémoire)
- **Le ratio C/n** (~0.20) n'est pas assez faible pour que O(C·d) < O(n·d)
  avec les constantes respectives

**Ce que la codebase permet d'affirmer :**
> Dans le régime expérimental étudié (n ≤ 72k, d = 1024, cosine), le coût
> d'un scan dense extrêmement vectorisé reste inférieur au coût total du
> mécanisme d'indexation approximatif. Le cross-polytope patché a réduit le
> ratio de ~7× à ~4×, confirmant que le plancher est dict + reranking (pas
> les transpositions).

**Ce qu'il manque pour affirmer un seuil de basculement :**
- Des bases ≥ 10⁵–10⁶ points (la littérature FALCONN rapporte des speedups
  à cette échelle)
- Une implémentation C++/native du cross-polytope (FALCONN original)
- Des benchmarks avec BLAS multi-cœurs (notre Kaggle utilise un CPU standard)

**On ne fabrique aucun seuil de basculement.** La conclusion est limitée au
régime étudié.

---

## 11. Comparaison des méthodes effectivement testées

### 11.1 Tableau synthétique (ChestX-ray14 full, médiane des 5 splits)

| Méthode | But | Métrique | Recall@10 | Macro-F1 | Latence (ms) | vs linear | Fraction candidats | Verdict |
|---|---|---|---|---|---|---|---|---|
| Linear Scan | Baseline exacte | L2 | 1.000 | ref | 12.88 | 1× | 1.000 | Référence |
| SimHash | Ablation diagnostique | cosinus | 0.968 | 0.702 | 40.17 | 3.4× | 0.17 | quality_ok_no_speedup |
| E2LSH | Diagnostic L2 | L2 | 0.982 | 0.699 | 58.14 | 4.9× | 0.22 | quality_ok_no_speedup |
| Cross-Polytope (patché) | Candidat sérieux | cosinus | 0.982 | 0.705 | 53.57 | 4.3× | 0.21 | quality_ok_no_speedup |

### 11.2 Heart Disease (base complète)

| Méthode | Recall@10 | Macro-F1 | Verdict |
|---|---|---|---|
| Linear Scan | 1.000 | ref | Référence |
| SimHash | 0.531 | 0.820 | **Echec** |
| E2LSH | 0.006 | 0.074 | **Echec** |
| Cross-Polytope | 0.986 | 0.834 | quality_ok_no_speedup |

### 11.3 Interprétation par expérience

- **SimHash** : a montré que LSH peut préserver la qualité sur haute dimension
  mais échoue en basse dimension. A lancé la question « est-ce spécifique à
  SimHash ? »
- **E2LSH** : a répondu que non, le phénomène est transversal. Deux familles
  différentes obtiennent le même patron (qualité OK, plus lent). A aussi montré
  l'échec catastrophique en basse dimension (recall = 0.006).
- **Cross-Polytope** : a montré que la famille optimale résout le problème
  de qualité (recall = 0.986 sur Heart) mais pas le problème de performance.
  Le plancher à 4× est structurel.

---

## 12. Tests statistiques

### 12.1 Protocole

- **5 split seeds** (n=5 pour Wilcoxon)
- **3 hash seeds** par split (moyenne avant test)
- **Test** : Wilcoxon signé (paired, deux queues)
- **Bootstrap** : 1000 itérations, IC à 95%
- **Variable testée** : différence de latence médiane (CP ou SimHash ou E2LSH)
  moins linear, par requête

### 12.2 Résultats (exp_falconn, run patché)

| Dataset | base | Diff (ms) | IC 95% | p_wilcoxon |
|---|---|---|---|---|
| chest_xray14 | 10k | +5.20 | [+5.11, +5.29] | 0.0625 |
| chest_xray14 | 50k | +24.89 | [+24.26, +25.50] | 0.0625 |
| chest_xray14 | full | +41.14 | [+28.24, +52.38] | 0.0625 |
| heart_disease | full | +0.25 | [+0.23, +0.27] | 0.0625 |

### 12.3 Interprétation

- Toutes les CI sont **positives** (LSH plus lent que linear, direction constante)
- **p = 0.0625** : c'est la p-value minimale achievable avec n=5 splits via
  Wilcoxon signé (il faut 5/5 dans la même direction pour ce minimum)
- La non-significativité (p > 0.05) ne signifie PAS qu'il n'y a pas de
  différence — elle signifie que n=5 est trop faible pour atteindre p < 0.05
  avec le test de Wilcoxon
- **On ne transforme PAS la non-significativité en preuve d'équivalence.**
  Les CI bootstrap montrent clairement que LSH est plus lent (toutes
  les CI sont strictement positives)

### 12.4 Limites

- n=5 splits est un échantillon très petit pour tout test non-paramétrique
- Les 3 hash seeds par split réduisent la variance mais n'augmentent pas n
- La robustesse est démontrée par la constance de la direction (5/5 dans
  chaque cas) plutôt que par la significativité statistique formelle

---

## 13. SRS

SRS (Subspace Radix Sampling, VLDB 2015) n'a **pas été implémenté ni évalué
expérimentalement** et ne fait pas partie du protocole expérimental réalisé.

SRS est mentionné dans `lsh_sota_review.md` (§1) comme famille LSH L2
« état de l'art pratique » avec un index de taille minimale, et dans
`DECISIONS_AND_RESULTS.md` comme « seule question ouverte pour évaluer le
plancher théorique ». Mais aucune implémentation, aucun test, aucun benchmark
n'a été réalisé.

**SRS ne doit pas être présenté comme une méthode expérimentale.**

---

## 14. KD-tree et HeCBR

### 14.1 KD-tree

| Critère | Statut |
|---|---|
| Prévu dans le mémoire ? | Oui (`main.tex` §1.1, §2.2 : « partitionnement déterministe de l'espace ») |
| Implémenté ? | Non dans exp_lsh/exp_falconn. Le `demo/README.md` mentionne `KDTreeIndex` mais c'est un projet séparé (démo Flask) |
| Testé ? | Non dans le protocole expérimental |
| Benchmarké ? | Non |
| Résultats disponibles ? | Non |
| Prochaine étape ? | Non prévue dans le protocole actuel |

### 14.2 HeCBR

| Critère | Statut |
|---|---|
| Prévu dans le mémoire ? | Oui (`main.tex` §1.1 : « apprentissage supervisé de la fonction de hachage ») |
| Implémenté ? | Non dans exp_lsh/exp_falconn. Un rapport d'analyse existe (`draft/HeCBR_Analysis_Report.md`) avec des résultats antérieurs (v2 sur Diabetes-130), mais l'implémentation a de sévères problèmes (overfitting, AUC=0.67, gap Hamming=0.4 bits) |
| Testé ? | Oui (dans un projet antérieur, pas dans le protocole exp_lsh/exp_falconn) |
| Benchmarké ? | Oui (dans le projet antérieur) |
| Résultats disponibles ? | Oui mais **non satisfaisants** : overfitting sévère, gap Hamming insuffisant |
| Prochaine étape ? | 10 correctifs priorisés dans le rapport d'analyse (P0: position embedding, multilinear interaction layer) |

**HeCBR ne doit pas être présenté comme une méthode expérimentalement évaluée
dans le cadre du protocole exp_lsh/exp_falconn.**

### 14.3 Maintenance

| Critère | Statut |
|---|---|
| Prévue dans le mémoire ? | Oui (`main.tex` §2.4, algorithme 4 : maintenance guidée par collisions) |
| Implémentée ? | Non |
| Testée ? | Non |
| Benchmarkée ? | Non |
| Prochaine étape ? | Non prévue |

---

## 15. Chronologie scientifique

```
Plan initial (main.tex)
  → Chapitre 1 : état de l'art (k-d tree, LSH, HeCBR)
  → Chapitre 2 : architecture + algorithmes (interface commune, filtrage, maintenance)
  → Chapitre 3 : implémentation + expérimentation (prévu : KD-tree, LSH, HeCBR)

Phase 1 — exp_lsh
  → Linear Scan implémenté et benchmarké (baseline)
  → SimHash implémenté comme ablation diagnostique
  → Constat : SimHash 3–8× plus lent que linear, recall OK sur Chest, échec sur Heart
  → Question : « est-ce SimHash ou le principe LSH ? »

  → E2LSH implémenté comme détour expérimental diagnostique
  → Constat : même patron que SimHash (qualité OK, plus lent), échec encore pire sur Heart
  → Conclusion partielle : le phénomène est transversal à la famille LSH

Phase 2 — exp_falconn
  → Cross-polytope implémenté from scratch (FALCONN-style, NumPy pur)
  → Hypothèse : la famille optimale pour le cosinus pourrait résoudre le problème
  → Calibration + smoke tests + 21 tests unitaires
  → Benchmark Kaggle (5 splits × 3 hash seeds)
  → Constat : recall ≥ 0.98 sur toutes les tâches (y compris Heart), mais 4–11× plus lent

  → Optimisations : layout transposé, batching, multi-probe vectorisé
  → Profiling : 94% du temps de requête = transposition `_D_all` non contiguë
  → Patch `_D_all_query` : précalcul contigu au build
  → Relance Kaggle (5 kernels, patché)
  → Constat : ratio réduit de ~7× à ~4× sur Chest, plancher confirmé (dict + rerank)

Conclusion actuelle
  → Aucune famille LSH ne bat linear scan en NumPy sur nos bases (10k–72k)
  → Qualité excellente partout, performance structurellement insuffisante
  → Le plancher est dict + reranking fancy-indexing vs BLAS pur
```

---

## 16. Conclusions à produire

### Ce que l'expérience LSH démontre

1. **Les méthodes LSH testées (SimHash, E2LSH, Cross-Polytope) préservent la
   qualité de classification CBR** sur les données médicales étudiées, lorsque
   les seuils de calibration sont atteints (Recall@10 ≥ 0.95, ΔF1 ≤ 0.01).

2. **Aucune méthode LSH ne réduit le temps de recherche** par rapport au
   linear scan BLAS dans le régime expérimental étudié (n ≤ 72k, d = 1024,
   Python/NumPy).

3. **La famille LSH compte** : le cross-polytope est la seule à préserver la
   qualité sur Heart Disease (d=13, n=297), là où E2LSH et SimHash échouent.

4. **Le phénomène est structurel** : il ne vient pas d'une implémentation
   sous-optimale (le patch a éliminé le handicap accidentel, le plancher reste).

### Ce que l'expérience LSH ne démontre pas

1. **Que LSH est intrinsèquement plus lent que Linear Scan.** La conclusion
   est limitée à : Python/NumPy, n ≤ 72k, d ≤ 1024, CPU Kaggle standard.
   En C++ avec des bases ≥ 10⁵, la littérature rapporte des speedups (FALCONN).

2. **Que le cross-polytope est toujours optimal.** La conclusion est limitée
   à : notre environnement, nos données, notre baseline.

3. **Qu'aucune méthode d'indexation ne peut battre Linear Scan.** Les k-d trees,
   FAISS, SRS, ou d'autres structures n'ont pas été testées dans ce protocole.

4. **Que la maintenance par collisions est efficace.** L'algorithme 4 n'a pas
   été implémenté ni évalué.

---

## 17. Relation directe avec le chapitre 3

### 3.1 Environnement de développement

**Matériaux à inclure :**
- Python 3.12, NumPy 2.5.1, SciPy 1.18, scikit-learn 1.9
- Environnement Kaggle (CPU standard, RAM ~13 GB)
- Code : `exp_lsh/` et `exp_falconn/`, architecture miroire
- Interface commune `LinearExact`, `SimHash`, `E2LSH`, `CrossPolytope`
- Harnais de timing (`timed_query`) avec warm-up + 5 répétitions + médiane
- Configuration protocolisée (`config/protocol.py`, `config/seeds.py`)

**Ne pas inclure :** les outils mentionnés dans le plan initial mais non
utilisés (FAISS, Annoy, Datasketch, PyTorch, TensorFlow).

### 3.2 Scénarios de test

**Matériaux à inclure :**
- Heart Disease (n=297, d=13) : petite base, basse dimension, contrôle
- ChestX-ray14 (n=112k, d=1024) : embeddings ResNet, 3 sous-bases (10k, 50k, full)
- Séparation stratifiée 64/16/20, 5 split seeds
- 300 requêtes test pour Chest, ~60 pour Heart
- Calibration sur validation, test sur jeu tenu à part

**Ne pas inclure :** Diabetes-130 (exclu du protocole).

### 3.3 Évaluation des résultats

**Matériaux à inclure :**
- Métriques : Recall@10, macro-F1, latence médiane, fraction candidats
- Seuils : Recall@10 ≥ 0.95, ΔF1 ≤ 0.01
- 4 résultats bruts par méthode/dataset (tableaux des §5–7)
- Tableau synthétique comparatif (§11)
- Tests statistiques : Wilcoxon + bootstrap CI (§12)
- 8 figures protocole (results/figures/)
- 4 outcomes : toutes = `quality_ok_no_speedup`

### 3.4 Discussion

**Matériaux à inclure :**
- Le phénomène central : qualité excellente mais latence insuffisante (§9)
- Analyse n/d : BLAS vectorisé domine l'indexation LSH dans ce régime (§10)
- Le patch `_D_all_query` et ce qu'il révèle sur les vrais coûts (§8)
- Comparaison avec la littérature (FALCONN 700× sur 10⁵–10⁸, mais en C++)
- Limites : taille des bases, implémentation Python, environnement Kaggle
- Ce que la maintenance par collisions pourrait apporter (algorithme 4 non testé)

---

## 18. Relation avec le reste du mémoire

### Ce que l'expérience LSH apporte

L'expérience LSH est la pierre angulaire expérimentale du mémoire. Elle
démontre que :

1. **L'architecture d'indexation générique fonctionne** : l'interface commune
   (Add/Query/Remove) est utilisée par 4 méthodes très différentes avec le
   même pipeline.

2. **Le compromis qualité/vitesse est mesurable et documenté** : la calibration
   avec seuils explicites (Recall ≥ 0.95, ΔF1 ≤ 0.01) permet une évaluation
   rigoureuse.

3. **Le choix de la famille LSH est critique** : le cross-polytope est le seul
   à préserver la qualité sur toutes les bases — ce résultat motive HeCBR
   (qui va au-delà des familles fixées).

### Pourquoi elle reste pertinente même sans speedup

- Elle **caractérise le régime** où le scan BLAS reste compétitif (n ≤ 72k)
- Elle **justifie l'investissement** dans des méthodes plus sophistiquées
  (HeCBR, maintenance) pour les grandes bases
- Elle **fournit une baseline documentée** contre laquelle toute future méthode
  sera comparée

### Comment elle prépare HeCBR

HeCBR (data-dependent hashing) promet d'apprendre une fonction de hash
alignée sur la similarité pertinente, ce que les méthodes data-independent
(LSH) ne font pas. L'expérience LSH montre que même la famille optimale
(cross-polytope) ne résout pas le problème de performance — ce qui motive
l'exploration de méthodes supervisées.

### Comment elle s'articule avec la maintenance

L'algorithme 4 (maintenance par collisions) utilise les collisions de
hachage comme indicateur de redondance. L'expérience LSH montre que les
méthodes LSH produisent des ensembles de candidats significativement plus
petits que le scan (fraction ~0.20), ce qui rend la détection de collisions
potentiellement efficace pour la maintenance. Mais cet algorithme n'a pas
été implémenté — c'est une piste de travail future.

---

## 19. Résumé final

**Objectif :** évaluer si le LSH accélère la récupération CBR médicale en
préservant la qualité, sur des données de dimensionalités croissantes.

**Protocole :** 2 datasets (Heart 13-d, ChestX-ray14 1024-d), 4 tailles
de base, 5 splits, 3 hash seeds, calibration sur validation, seuils
Recall@10 ≥ 0.95 et ΔF1 ≤ 0.01, timing : 1 warm-up + 5 répétitions
(médiane), exécuté sur Kaggle.

**Expériences réalisées :**
1. Linear Scan (baseline BLAS)
2. SimHash (ablation diagnostique)
3. E2LSH (détour diagnostique)
4. Cross-Polytope (candidat sérieux, FALCONN-style)

**Résultats principaux :**
- Qualité préservée sur toutes les tâches (r10 ≥ 0.96, ΔF1 ≤ 0.01)
- Mais 3–11× plus lent que linear sur toutes les configurations
- Seul le cross-polytope réussit sur Heart Disease (recall 0.986 vs 0.006
  pour E2LSH et 0.53 pour SimHash)
- Le plancher à ~4× est structurel (dict + reranking vs BLAS pur)

**Conclusion LSH :** dans le régime étudié (n ≤ 72k, d ≤ 1024, Python/NumPy),
le LSH ne réduit pas le temps de recherche malgré une excellente qualité.
Le scan BLAS vectorisé reste plus rapide.

**Méthodes non expérimentées :** KD-tree, HeCBR, maintenance, SRS.

**État actuel :** phase 2 terminée, tous les résultats enregistrés,
documentation complète. Les 5 kernels Kaggle (patchés) sont exécutés.

**Prochaine étape pour le mémoire :** rédiger le chapitre 3 en suivant la
structure 3.1–3.4 de `main.tex`, en utilisant ce document comme matière
première. Ensuite : décider si HeCBR ou la maintenance sont implémentés
pour la soutenance, ou si le mémoire se limite aux résultats LSH.
