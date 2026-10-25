# Plan d'Expérimentation Reproductible — Hachage + Maintenance pour le RàPC médical

> Référence : journal §25 (audit des compromis), waypoints.md, plan_de_travail_complet.pdf.
> Statut : DRAFT — document de référence pour la production des notebooks (Kaggle / Lightning AI).

---

## 1. Principes directeurs

1. **La procédure prime, le narratif suit.** Toutes les hypothèses de recherche sont
   formulées *a priori* (section 2) et testées telles quelles. Le narratif est rédigé
   *a posteriori*, à partir des résultats, sans direction imposée.
2. **Indépendance vis-à-vis des résultats passés.** Les anciens chiffres (journal §1-24,
   CSV, figures) sont archivés, pas un point d'ancrage. Le code et les interfaces
   (`CaseIndex`, index, chargeurs de datasets) sont réutilisés ; les valeurs, non.
3. **Reproductibilité totale.** Chaque notebook est exécutable de bout en bout ; toute
   sortie est écrite en CSV/JSON versionné avec métadonnées (seeds, paramètres, versions).
   Aucune valeur en dur dans les figures (lecture depuis les fichiers de résultats).
4. **Équité entre méthodes.** Implémentations alignées (LSH équitable einsum/BLAS),
   même famille de distance pour les méthodes comparées, mêmes données pour toutes les
   stratégies de maintenance, protocole de timing unique.
5. **Métriques robustes.** Déséquilibre de classes documenté ; utilisation de macro-F1,
   balanced accuracy et AUC en complément de l'accuracy brute (§25.1).
6. **6 seeds partout** (42, 7, 123, 2024, 99 + 1 seed d'échantillonnage dédié), split
   train/val/test stratifié, scaler ajusté sur train uniquement.
7. **Un vainqueur à la fin.** Les tâches T5/T6 désignent, par un critère objectif
   pré-déclaré (section 5), la méthode retenue pour construire un modèle CBR de démo.

---

## 2. Questions scientifiques (ouvertes, a priori)

### 2.1 Indexation (recherche seule)

- **QR1.** Le hachage (LSH) restitue-t-il les mêmes voisins que la recherche exacte, et à
  quel prix en rappel des voisins exacts ?
- **QR2.** La vitesse de recherche du hachage est-elle meilleure qu'un scan linéaire quand
  la base croît (10k/50k/100k) ? Dans quelles conditions (dimension, implémentation) ?
- **QR3.** Quel est le compromis temps/qualité (front de Pareto) de chaque méthode par
  niveau de dimensionnalité ?
- **QR4.** Le KD-tree perd-il son avantage en haute dimension (fléau de la dimensionnalité) ?
- **QR5.** Le hachage supervisé (HeCBR) bat-il LSH à longueur de code égale sur
  Diabetes-130 (data-dependent vs data-independent) ? — *HeCBR évalué uniquement sur
  Diabetes-130 (codes déjà entraînés), pas ré-entraîné sur d'autres datasets.*
- **QR6.** Quelle est l'empreinte mémoire (index + requête) de chaque structure ?

### 2.2 Maintenance

- **QM1.** La rétention par collisions supprime-t-elle correctement les cas redondants
  (vs référence exacte) et à quel taux de réduction ?
- **QM2.** Quelle stratégie de condensation — CNN (exacte), CBM (exacte), condensing par
  bucket (hachage) — préserve le mieux la précision à taux de réduction égal, sur les
  **mêmes données** ?
- **QM3.** La maintenance hachée (rétention par collisions + condensing par bucket) vaut-elle
  la maintenance exacte (CBM) ?
- **QM4.** La maintenance améliore-t-elle le temps de requête **sans perdre de qualité**
  (métriques post-maintenance vs base brute) ?
- **QM5.** Quel est le coût/bénéfice d'entraîner du hachage profond (HeCBR) vs LSH pour la
  boucle de maintenance sur Diabetes-130 ?

### 2.3 Dimensionnalité / scaling

- **QD1.** Comment les paramètres LSH (bits, tables) doivent-ils croître avec la dimension
  pour maintenir un rappel fixe (≥ 0.99), à taille de base fixée ?
- **QD2.** Comment le temps de requête croît-il avec la taille de base (10k/50k/100k) pour
  chaque méthode et chaque niveau de dimensionnalité ?
- **QD3.** Le filtrage du hachage (fraction de candidats) est-il réellement meilleur en
  haute dimension, ou confondu par la taille de base ? (tranche la confusion §23)
- **QD4.** Le déséquilibre des classes fausse-t-il les métriques de classification ?
  (macro-F1, balanced accuracy, AUC vs accuracy)

### 2.4 Statistique

- **QS1.** Les différences entre méthodes sont-elles statistiquement significatives
  (test apparié) plutôt que du bruit entre seeds ?

---

## 3. Tâches — liste exhaustive

### T0. Infrastructure commune (préalable à tout)

- **T0.1** Adapter générique de dataset : déclaration (scaler/encodage/cible), split
  train/val/test stratifié par seed.
- **T0.2** Registre de résultats : toute sortie en CSV/JSON versionné avec métadonnées.
- **T0.3** Politique de normalisation homogène, documentée, une seule règle par dataset.
  *Décision D130 (2026-08-11, implémentée dans `exp2/src/datasets/adapters.py`) : le
  vocabulaire catégoriel (one-hot) est figé globalement — cohérent avec le groupement
  topk global du loader —, seul le scaler numérique est ajusté sur le train. Ce choix
  garantit la parité dimensionnelle (390 dims, train et test) sans fuite de statistiques
  numériques.*
- **T0.4** Bibliothèque de métriques : Precision@k, Recall@k, MAP@k, nDCG@k, Hit@k, MRR,
  rappel des voisins exacts, accuracy, macro-F1, balanced accuracy, AUC.
- **T0.5** Protocole de timing unique (warm-up, médiane, répétitions, `perf_counter`).
- **T0.6** Mesure mémoire (pic RSS index + par requête, `tracemalloc`/`resource`).

### T1. Benchmark des index seuls (répond QR1-QR6)

- **T1.1** Baseline : scan linéaire BLAS (einsum/matvec, pas de batch amorti).
- **T1.2** KD-tree (sklearn).
- **T1.3** LSH équitable : reclassement einsum/BLAS (pas `np.linalg.norm`), calibration
  bits/tables.
- **T1.4** HeCBR : codes déjà entraînés (Diabetes-130 uniquement), distance cohérente
  avec LSH (Hamming sur codes binaires pour HeCBR ; famille de distance déclarée).
- **T1.5** Acquisition temps + qualité sur k={1,5,10} × 6 seeds.
- **T1.6** Empreinte mémoire (T0.6).
- **T1.7** Figure : métriques de qualité par méthode/dataset.
- **T1.8** Figure : temps de requête par méthode (protocole unique).

### T2. Scaling (répond QR2, QD2)

- **T2.1** Bases 10k/50k/100k par sous-échantillonnage du train (mêmes seeds).
- **T2.2** Temps de requête + construction vs taille de base, par méthode.
- **T2.3** Figure : courbes de scaling (log-log).
- **T2.4** Figure : temps de construction par méthode.

### T3. Maintenance (répond QM1-QM5, QD4)

- **T3.1** Rétention par collisions vs référence exacte (accord cas par cas, taux de
  réduction, seuil documenté).
- **T3.2** Condensation équitable : CNN, CBM, condensing par bucket — **même
  sous-échantillon pour les trois stratégies** (élimine le biais du §25.5).
- **T3.3** Métriques post-maintenance : temps de requête + qualité (macro-F1, acc, AUC)
  sur base condensée vs base brute.
- **T3.4** Temps de construction de chaque condensation.
- **T3.5** Figures : accord de rétention, qualité post-maintenance, temps post-maintenance.

### T4. Dimensionnalité (répond QD1, QD3)

- **T4.1** Calibration bits/tables pour rappel ≥ 0.99 à taille de base **fixée**, dimension
  croissante.
- **T4.2** Fraction de candidats réelle par dataset, taille de base contrôlée (tranche QD3).
- **T4.3** Figure : (bits, tables) → nombre de compartiments vs dimension.
- **T4.4** Figure : candidats vs dimension à n fixé.

### T5. Statistique (répond QS1)

- **T5.1** Tests appariés (Wilcoxon signed-rank, t-test apparié) entre méthodes, par
  dataset et par k, sur temps et qualité.
- **T5.2** Figure : matrice de significativité.

### T6. Synthèse et vainqueur (clôture)

- **T6.1** Tableau récapitulatif multi-datasets (métadonnées incluses).
- **T6.2** Figures du mémoire, lues depuis les CSV (aucune valeur en dur).
- **T6.3** Sélection du vainqueur par critère objectif pré-déclaré (section 5).
- **T6.4** Modèle CBR de démo basé sur le vainqueur (notebook applicatif).

> **Tâches écartées (décision utilisateur)** : ré-entraînement de HeCBR sur d'autres
> datasets que Diabetes-130 ; passe dédiée de correction du déséquilibre (les métriques
> robustes T0.4 restent utilisées, mais aucune expérience de ré-échantillonnage).

---

## 4. Jeux de données

| Dataset | Dimension | Type | n (total) | Ratio positif cible | Split | Scaling |
|---|---|---|---|---|---|---|---|
| Heart Disease (Cleveland) | 13 | Tabulaire | 297 | ~46 % | 80/20 stratifié × 6 seeds | Z-score sur train (§2) |
| Diabetes 130-US | 390 (8 num + 382 one-hot) | Mixte | 101 766 | **11,2 %** (`readmitted < 30 j`, mesuré) | idem | num scalées, one-hot non normalisé (§25.1) |
| ChestX-ray14 (embeddings DenseNet121) | 1024 | Multimodal | 112 120 | **46,2 %** (« ≥ 1 pathologie », mesuré) | idem | aucun (embeddings float32) (§25.1) |

> **Déséquilibre mesuré** (vérifié le 2026-08-10) : D130 est fortement déséquilibré
> (11,2 % de positifs), Chest quasi équilibré (46,2 %). Conséquences : métriques macro
> (T0.4) obligatoires pour D130 ; Chest est le choix retenu pour la démo CBR (section 6,
> NB13) précisément pour éviter ce biais dans les chiffres de la démo.

Décisions à documenter pour chaque dataset : gestion des valeurs manquantes, cardinalité
des catégories (HIGH_CARD_TOPK=50), politique de normalisation unique (T0.3), et cible
binaire (déséquilibre documenté).

---

## 5. Critère de sélection du vainqueur (à valider — pré-déclaré avant T1)

Proposition de critère objectif, à figer avant l'exécution pour éviter tout biais :

> **Score = moyenne normalisée de 4 composantes par dataset, pondérées :**
> 1. **Qualité post-maintenance** (macro-F1 ou balanced accuracy sur base maintenue) — 40 %
> 2. **Rapidité post-maintenance** (temps/requête sur base maintenue, log-normalisé) — 30 %
> 3. **Robustesse** (variance inter-seeds, significativité QS1) — 15 %
> 4. **Fidélité d'approximation** (rappel des voisins exacts) — 15 %
>
> Le vainqueur est la méthode maximisant le score **moyen sur les datasets** sur lesquels
> elle est applicable ; les scores par dataset sont rapportés dans tous les cas.
>
> **Vainqueur pour la démo (NB13)** : le modèle CBR de démo est construit sur
> **ChestX-ray14** avec la méthode qui gagne **sur Chest** (par le même score, appliqué
> à ce dataset). HeCBR n'étant pas entraîné sur Chest, la démo utilise la meilleure
> méthode disponible (LSH + maintenance attendue). Aucune extrapolation de HeCBR vers
> Chest n'est faite dans les conclusions (limites, section 8).

À ajuster/valider avec l'utilisateur **avant** la production de T1.

---

## 6. Production des notebooks — plan de travail

Notebooks exécutables (Kaggle / Lightning AI), chaque sortie CSV alimentant le suivant.
Un notebook de configuration (`00_config`) centralise T0.1-T0.3 et le registre.

| # | Notebook | Tâches | Produit |
|---|---|---|---|
| 00 | Configuration & registre | T0.1-T0.3 | config.yml, registre, adapter dataset |
| 01 | Exploration des données & figures | — (figures data) | figures §7.1, statistiques descriptives |
| 02 | Métriques & protocole | T0.4-T0.6 | lib metrics, timing, mémoire |
| 03 | Benchmark index seuls | T1.1-T1.6 | CSV temps/qualité/mémoire |
| 04 | Figures benchmark | T1.7-T1.8 | FIG-R1, FIG-R2 |
| 05 | Scaling | T2.1-T2.2 | CSV scaling |
| 06 | Figures scaling | T2.3-T2.4 | FIG-R3, FIG-R4 |
| 07 | Maintenance | T3.1-T3.4 | CSV rétention + condensation + post-maintenance |
| 08 | Figures maintenance | T3.5 | FIG-R5, FIG-R6, FIG-R7 |
| 09 | Dimensionnalité | T4.1-T4.2 | CSV bits/tables + candidats |
| 10 | Figures dimensionnalité | T4.3-T4.4 | FIG-R8, FIG-R9 |
| 11 | Statistique | T5.1-T5.2 | tests + matrice significativité |
| 12 | Synthèse & vainqueur | T6.1-T6.3 | tableau final + vainqueur |
| 13 | Démo CBR — **ChestX-ray14** | T6.4 | modèle CBR de démo (méthode gagnante sur Chest) |

Dépendances : 01 → 00 ; 02 → 00 ; 03 → 02 ; 04 → 03 ; 05 → 02 ; 06 → 05 ; 07 → 02 ;
08 → 07 ; 09 → 02 ; 10 → 09 ; 11 → 03,05,07 ; 12 → 04,06,08,10,11 ; 13 → 12.
Branches parallèles possibles après 02 : (03→04), (05→06), (07→08), (09→10).

### 6.1 Plan d'exécution — répartition Kaggle / local (budget 36 h)

Machine locale : Latitude 7400 (16 Go RAM, 8 threads) — **< 2 h par notebook local**.
Kaggle : 12 h par session, plusieurs sessions en parallèle.

| NB | Local | Kaggle | Estim. local | Estim. Kaggle | Justification |
|---|---|---|---|---|---|
| 00 | ✔ | — | < 5 min | — | config statique |
| 01 | ✔ | — | ~ 40 min | — | PCA randomisée Chest (112k×1024), distances NN subsamplées |
| 02 | ✔ | — | < 10 min | — | lib + sanity |
| 03 | ✘ | ✔ | > 2 h | ~ 2-4 h | KD-tree 1024-d ≈ 1 s/req × 6 seeds (fléau : c'est le résultat QR4) |
| 04 | ✔ | — | < 10 min | — | lecture CSV |
| 05 | ✘ | ✔ | > 2 h | ~ 3-5 h | scans BLAS 100k×1024 répétés × seeds × k |
| 06 | ✔ | — | < 10 min | — | lecture CSV |
| 07 | ✘ | ✔ | CNN/CBM O(n²) | ~ 4-6 h | condensation itérative sur D130 |
| 08 | ✔ | — | < 10 min | — | lecture CSV |
| 09 | ✔ | — | ~ 1 h | — | calibration bits/tables, n fixé |
| 10 | ✔ | — | < 10 min | — | lecture CSV |
| 11 | ✔ | — | < 10 min | — | tests sur CSV agrégés |
| 12 | ✔ | — | < 10 min | — | synthèse |
| 13 | ✔ | — | ~ 1 h | — | index LSH Chest + requêtes démo |

**Chronologie (chemin critique ≈ 19 h, marge 17 h) :**

1. **T0 → T2h (local)** : NB00, NB01, NB02 → config, registre, adapter datasets, lib
   métriques. Upload sur Kaggle.
2. **T2h → T14h (Kaggle, 3 sessions parallèles)** : NB03, NB05, NB07. **En parallèle
   (local)** : NB09. Téléchargement des CSV produits.
3. **T14h → T19h (local)** : figures NB04/06/08/10, statistique NB11, synthèse NB12,
   démo NB13.

> Risque principal : session Kaggle interrompue → checkpoint CSV par seed (un seed
> validé est conservé). Chaque notebook Kaggle écrit ses sorties incrémentalement.

---

## 7. Figures

### 7.1 Figures de présentation des données (NB01) — expliciter la donnée elle-même

- **FIG-D1** Répartition des classes par dataset (bar chart, proportions) — indispensable
  pour lire les métriques de classification (déséquilibre visible).
- **FIG-D2** Comparaison de dimensionnalité (bar chart log : 13 / 390 / 1024) avec labels
  du nombre de features.
- **FIG-D3** Composition des datasets (tableau/heatmap) : n, d, type (tabulaire/mixte/
  multimodal), taux de valeurs manquantes, nb de catégories par variable (esp. Diabetes).
- **FIG-D4** Projection PCA 2D par dataset, colorée par classe (structure/variabilité
  réelle des données).
- **FIG-D5** Histogrammes de la distance au plus proche voisin (distribution → justifie le
  choix du seuil de redondance T3.1).
- **FIG-D6** Répartition numériques vs one-hot / cardinalité des catégories (Diabetes).

### 7.2 Figures de résultats

- **FIG-R1** Qualité de récupération (rappel voisins exacts, nDCG@k, macro-F1) par
  méthode/dataset (T1.7).
- **FIG-R2** Temps de requête par méthode, échelle log, protocole unique (T1.8).
- **FIG-R3** Courbes de scaling temps vs taille de base (T2.3).
- **FIG-R4** Temps de construction d'index vs taille de base (T2.4).
- **FIG-R5** Accord rétention LSH vs exacte + taux de réduction (T3.5).
- **FIG-R6** Qualité post-maintenance (brut vs condensé, par stratégie) (T3.5).
- **FIG-R7** Temps de requête post-maintenance (brut vs condensé) (T3.5).
- **FIG-R8** Compartiments LSH (2^bits × tables) vs dimension (T4.3).
- **FIG-R9** Fraction de candidats vs dimension à n fixé (T4.4).
- **FIG-R10** Matrice de significativité (Wilcoxon) entre méthodes (T5.2).
- **FIG-R11** Compromis rappel / coût de la calibration LSH — **tous** les couples
  calibrés sur **ChestX-ray14**, abscisses ordonnées par permissivité croissante
  (tables ↑, bits ↓) ; rappel (moyen + min) à gauche, temps de construction à
  droite, couple retenu annoté (T1.3/NB03, `lsh_calibration.csv`).
  Message : la permissivité élève le rappel vers la saturation (≥ 0.95) mais
  alourdit la complexité temporelle (~3.7 s → ~17 s).

---

## 8. Limites assumées (transparentes dans le mémoire)

- HeCBR évalué uniquement sur Diabetes-130 (coût d'entraînement profond) — QR5/QM5
  limités à ce dataset. **Aucune extrapolation de HeCBR vers ChestX-ray14** : la démo
  utilise la meilleure méthode effectivement entraînée sur Chest (LSH attendu), et les
  conclusions ne spéculent pas sur le gain qu'aurait pu apporter HeCBR.
- La démo CBR est construite sur ChestX-ray14 (cible quasi équilibrée, 46,2 %) ; le
  déséquilibre de D130 (11,2 %) est traité au niveau des métriques (macro-F1, balanced
  accuracy, AUC) pour les réclamations de recherche, sans impact sur les chiffres de la démo.
- Les temps restent dépendants de la machine d'exécution ; comparer les méthodes entre
  elles, pas les valeurs absolues inter-plateformes.
- Le déséquilibre est documenté et traité au niveau des métriques (macro), pas des données.
- Les bases 10k/50k/100k sont des sous-échantillons du train disponible (pas de nouvelles
  données).
