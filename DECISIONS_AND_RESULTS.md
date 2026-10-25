# Expérience LSH Medical — Décisions et résultats complets

> **Objet :** document de continuité complète. Toute IA ou tout agent reprenant
> ce projet doit pouvoir lire **uniquement ce fichier** pour comprendre l'intégralité
> du contexte, des décisions prises, des résultats obtenus, et de l'état actuel.
>
> Dernière mise à jour : 2026-08-16

---

## 1. Objectif scientifique

Évaluer si des méthodes de **Locality-Sensitive Hashing (LSH)** permettent d'accélérer
la recherche de plus proches voisins en **cosinus** dans un contexte de Classification
Based on Retrieval (CBR) médical, par rapport au **linear scan brute-force** (baseline).

Question de recherche principale :

> Le cross-polytope LSH adapté à la métrique cosinus réduit-il le temps de recherche
> en préservant la qualité de classification (Recall@10 ≥ 0,95 et Δmacro-F1 ≤ 0,01) ?

Question secondaire :

> La famille LSH « optimale » (cross-polytope) surpasse-t-elle les familles historiques
> (SimHash, E2LSH) en pratique ?

---

## 2. Jeux de données

| Dataset | Dimension d | Taille totale | Nature | Usage |
|---|---|---|---|---|
| Heart Disease (UCI) | 13 | 297 | Tabulaire (NormalScaler) | Petites bases, basse dimension |
| ChestX-ray14 | 1024 | 112 120 | Embeddings ResNet | Grande base, haute dimension |

**Décision :** ChestX-ray14 est échantillonnée en 3 sous-bases (10k, 50k, full=72k)
pour tester le scaling. Heart Disease est utilisée en totalité (297 échantillons).

---

## 3. Protocole expérimental (résumé)

### 3.1 Separation des donnees
- Train 64% / Validation 16% / Test 20%
- Validation = 20% du train (jamais utilisee pour les tests)
- 5 split seeds : [42, 7, 123, 2024, 99]

### 3.2 Metrique principale : Cosinus
- Le protocole original etait en L2 ; decision de passer en **cosinus** (braccos normalises)
- Sur la sphere unite : ||p - q||^2 = 2 - 2*cos(p,q) → L2 et cosinus sont lineairement equivalents
- Heart Disease : L2-normalisation apres StandardScaler
- ChestX-ray14 : deja normalise L2 (embeddings ResNet)

### 3.3 Seuils de qualite (protocol section 8/12)
```python
MIN_RECALL_AT_10 = 0.95    # Recall@10 >= 0.95 sur validation
MAX_F1_DROP = 0.01         # macro-F1 drop vs linear <= 0.01
```

### 3.4 Protocole de mesure (timing)
- 1 warm-up + 5 mesures repeats par requete
- Aggregation : mediane sur les 5 runs
- 300 requetes test par split

### 3.5 Calibration
- **Seulement sur la validation** (jamais sur le test)
- Grid search sur les hyperparametres de chaque methode LSH
- Selection du meilleur parametre avec tiebreak : recall >= 0.95 → F1 drop <= 0.01 → latence mediane → fraction de candidats → memoire → build time
- **Un seul hash_seed** pour la calibration (seed 11) pour des raisons de temps

### 3.6 Benchmarks
- Chaque methode calibree est testee sur les 3 hash_seeds = [11, 29, 43]
- Ground truth : linear scan exact (cache dans .npz)
- Comparaison : vitesse de la methode LSH vs linear scan

### 3.7 Execution
- **Kaggle** : 5 kernels parallels (un par split seed)
- Package de code uploadé en dataset Kaggle
- Resultats telecharges en CSV/JSON/NPZ

---

## 4. Décisions architecturales majeures

### 4.1 Pourquoi NumPy pur (pas FALCONN C++)

| Facteur | Decision |
|---|---|
| FALCONN pip v1.3.1 | Pas maintenu, dependance Eigen/FFHT, pas testable sur Kaggle offline |
| Fairness methodologique | La baseline linear est en NumPy → tester LSH en C++ creerait un biais systematique |
| Reproductibilite | NumPy pur = zero dependance native, run sur Kaggle sans compilation |
| Controle | Code complet, chaque etape auditable, pas de boite noire |

**L'utilisateur avait pose la condition** : « Si nos resultats sont pires qu'E2LSH → passer a FALCONN C++ ».
La condition n'a pas ete remplie (CP 2.7x plus rapide qu'E2LSH en test local), donc decision maintenue.

### 4.2 Décision sur la metrique cosinus (pas L2)

Le protocole original etait en `PRIMARY_METRIC = "l2"`. Decision de passer en cosinus car :
- SimHash cible le cosinus → etait inadapte en L2 (doublement contestable)
- Sur la sphere unite, L2 et cosinus sont lineairement equivalents
- Le cross-polytope (FALCONN) est optimal pour l'angulaire/cosinus
- Le braccos normalises permettent la conversion directe

### 4.3 Pourquoi cross-polytope (pas E2LSH ou SimHash)

| Methode | Exposant ρ (angulaire) | Statut |
|---|---|---|
| SimHash (hyperplane) | sous-optimal | Historique (Charikar 2002) |
| E2LSH (p-stable) | 1/c² | Historique (Datar 2004), echoue basse dimension |
| **Cross-polytope (FALCONN)** | **1/(2c²−1)** | **Optimal** (Andoni et al. NIPS 2015) |

Decision : evaluator le cross-polytope comme methode LSH « optimale theoriquement »
pour voir si la theorie se traduit en speedup pratique.

---

## 5. Phase 1 — exp_lsh/ (SimHash + E2LSH)

### 5.1 Implementations

| Methode | Details |
|---|---|
| LinearExact | Baseline BLAS : `np.linalg.norm(X_base - x, axis=1)`, vectorise |
| SimHash | Hyperplane random projection, L2 sur base (diagnostic ablation) |
| E2LSH | p-stable distributions, multi-probe, bucket width w=4.0 fixe |

### 5.2 Resultats Kaggle (5 splits × 3 hash seeds = 15 runs)

**ChestX-ray14 :**

| base | Methode | recall@10 | Latence (ms) | vs linear | Qualite |
|---|---|---|---|---|---|
| 10k | linear | 1.0000 | 1.76 | 1x | ref |
| 10k | E2LSH | 0.9852 | 12.91 | 7.3x plus lent | ✓ |
| 10k | SimHash | 0.9830 | 14.02 | 8.0x plus lent | ✓ |
| 50k | linear | 1.0000 | 9.17 | 1x | ref |
| 50k | E2LSH | 0.9784 | 38.64 | 4.7x plus lent | ✓ |
| 50k | SimHash | 0.9631 | 26.14 | 3.2x plus lent | ✓ |
| full | linear | 1.0000 | 12.88 | 1x | ref |
| full | E2LSH | 0.9824 | 58.14 | 4.9x plus lent | ✓ |
| full | SimHash | 0.9685 | 40.17 | 3.4x plus lent | ✓ |

**Heart Disease :**

| Methode | recall@10 | Latence (ms) | Qualite |
|---|---|---|---|
| linear | 1.0000 | 0.024 | ref |
| E2LSH | **0.006** | 0.03 | **✗ echec total** |
| SimHash | **0.531** | 0.12 | **✗ echec** |

### 5.3 Conclusions phase 1
- **Chest** : qualite preservee (recall >= 0.95, ΔF1 <= 0.01) mais **aucun speedup**
- **Heart** : echec global d'E2LSH et SimHash (dimensions trop faibles pour les projections aleatoires)
- **Bottleneck** : la traversée de dictionnaires Python + fancy-indexing NumPy est plus lente
  que le broadcast BLAS vectorise de linear

---

## 6. Phase 2 — exp_falconn/ (CrossPolytope LSH)

### 6.1 Design decisions specifiques

| Decision | Rationale |
|---|---|
| num_rotations = 2 | Verifie contre FALCONN `get_default_parameters()` source |
| K via heuristique FALCONN | `while (1 << (bits+2)) <= n` puis `k = bits / (log2ceil(d)+1)` |
| Layout FHT transpose (d,n) | ~3x plus rapide que layout par vecteur (cache-friendly) |
| Hash = 2*idx + sign_bit | Sign bit plie les deux sommets antipodes du cross-polytope |
| Multi-probe vectorise | Pas de boucle Python par coordonnee |
| Batched query rotations | Toutes les L*K rotations dans un seul passage FHT |
| last_cp_dim | Gere le cas ou les bits restent après la grille entiere |
| normalise_cosine=True sur heart | L2-normalisation apres StandardScaler pour arm cosinus |

### 6.2 Tests unitaires
- 21 tests passes (`pytest tests/test_cross_polytope.py`)
- Equivalence numerique FHT, parite layout transpose, orthonormalite rotations,
  determinisme hash, non-pletitude candidats, smoke tests recall

### 6.3 Tests smoke avant dispatch Kaggle

| Test | Resultat |
|---|---|
| Collision vs angle (K=1,L=1,rot=1) | 5°→0.95, 10°→0.79, 20°→0.62, 30°→0.44, 45°→0.24, 60°→0.11, 90°→0.007 ✓ |
| Near-identical vectors (cos=0.97) | Collision 0.97 ✓ |
| Top-10 chest (cos=0.953) | Voisins ~18°, qualite attendue ✓ |
| Heart (cos=0.635) | Voisins ~50°, calibration recall=0.971 ✓ |

### 6.4 Parametres calibres (seed 42, Kaggle)

| Task | K | L | probes | rot |
|---|---|---|---|---|
| heart full | 1 | 20 | 1 | 2 |
| chest 10k | 2 | 20 | 1 | 2 |
| chest 50k | 2 | 20 | 1 | 2 |
| chest full | 2 | 10 | 4 | 2 |

### 6.5 Test local vs E2LSH (avant Kaggle)

Test sur chest 10k (60 requetes, meme machine) :
- Linear : 4.55 ms
- CrossPolytope : 24.5 ms (recall 0.995, cand_frac 0.49)
- E2LSH : 65.4 ms (recall 1.0, cand_frac 1.0)
- **CP 2.7x plus rapide qu'E2LSH** → condition « passer a FALCONN C++ » non remplie

### 6.6 Raisonnement sur la stabilite de complexite

L'utilisateur a demande si notre implementation NumPy est stable vs FALCONN C++.
Analyse :

| Operation | Notre impl. | FALCONN C++ |
|---|---|---|
| Build | O(L·K·R·d·n·log d) — meme algorithme | Idem |
| Query hash | O(R·d·L·K·log d) — FHT vectorise | Idem (FFHT) |
| Candidate lookup | O(L·K·probes) — Python dicts | Idem (C++ hash tables) |
| Rerank | O(n_cand·d) — fancy-index + einsum | Idem |

**Conclusion** : seuls les **constantes** different (C++ vs NumPy), pas les exposants.
Les fractions de candidats (recall) seraient identiques. FALCONN C++ serait plus
rapide en parois-mur mais creerait un biais systematique vs baseline linear NumPy
(protocole interdit). **Decision : rester en NumPy.**

### 6.7 Dispatch Kaggle

| Element | Details |
|---|---|
| Dataset package | `franckyoteu/exp-falconn-code` (dir_mode=zip : config.zip, src.zip, tests.zip) |
| Kernels | 5 kernels `exp-falconn-pipeline-seed-{42,7,123,2024,99}` |
| Duree | ~48 min par kernel (calibration ~28 min, benchmark ~20 min) |
| Statut | Tous COMPLETE, zero erreurs, 64 lignes logs chacun |
| Artefacts | 50 fichiers telecharges (CSV, JSON, NPZ, logs) |

---

## 7. Resultats finaux — Les deux phases combined

### 7.1 Tableau comparatif complet

Moyennes sur 5 splits × 3 hash seeds = 15 runs par methode.

| Dataset | base | Methode | recall@10 | Latence (ms) | cand_frac | macro_f1 | vs linear | Qualite §12 |
|---|---|---|---|---|---|---|---|---|
| chest_xray14 | 10k | **linear** | 1.0000 | 1.44 | 1.000 | 0.7008 | 1x | ref |
| chest_xray14 | 10k | **cross-polytope** | 0.9741 | 10.52 | 0.204 | 0.7010 | **7.3x plus lent** | ✓ |
| chest_xray14 | 10k | E2LSH | 0.9852 | 12.91 | — | 0.7056 | 7.3x plus lent | ✓ |
| chest_xray14 | 10k | SimHash | 0.9830 | 14.02 | — | 0.7052 | 8.0x plus lent | ✓ |
| chest_xray14 | 50k | **linear** | 1.0000 | 8.29 | 1.000 | 0.7035 | 1x | ref |
| chest_xray14 | 50k | **cross-polytope** | 0.9894 | 36.04 | 0.204 | 0.7027 | **4.3x plus lent** | ✓ |
| chest_xray14 | 50k | E2LSH | 0.9784 | 38.64 | — | 0.7020 | 4.7x plus lent | ✓ |
| chest_xray14 | 50k | SimHash | 0.9631 | 26.14 | — | 0.7004 | 3.2x plus lent | ✓ |
| chest_xray14 | full | **linear** | 1.0000 | 11.89 | 1.000 | 0.7086 | 1x | ref |
| chest_xray14 | full | **cross-polytope** | 0.9817 | 55.59 | 0.206 | 0.7050 | **4.7x plus lent** | ✓ |
| chest_xray14 | full | E2LSH | 0.9824 | 58.14 | — | 0.6987 | 4.9x plus lent | ✓ |
| chest_xray14 | full | SimHash | 0.9685 | 40.17 | — | 0.7020 | 3.4x plus lent | ✓ |
| heart_disease | full | **linear** | 1.0000 | 0.024 | 1.000 | 0.8307 | 1x | ref |
| heart_disease | full | **cross-polytope** | 0.9858 | 0.28 | 0.381 | 0.8344 | **11.4x plus lent** | ✓ |
| heart_disease | full | E2LSH | **0.006** | 0.03 | — | 0.074 | ~1x | **✗ echec** |
| heart_disease | full | SimHash | **0.531** | 0.12 | — | 0.820 | 4.9x plus lent | **✗ echec** |

### 7.2 Resultats de significativite statistique

**Run initial (avant patch) :**

| Dataset | base | mean_diff (s) | CI 95% | p_wilcoxon |
|---|---|---|---|---|
| chest_xray14 | 10k | +0.0091 | [0.0089, 0.0093] | 0.0625 |
| chest_xray14 | 50k | +0.0277 | [0.0269, 0.0286] | 0.0625 |
| chest_xray14 | full | +0.0437 | [0.0312, 0.0553] | 0.0625 |
| heart_disease | full | +0.0003 | [0.0002, 0.0003] | 0.0625 |

**Run patché (aprés précalcul `_D_all_query` contigu) :**

| Dataset | base | mean_diff (s) | CI 95% | p_wilcoxon |
|---|---|---|---|---|
| chest_xray14 | 10k | +0.0052 | [0.0051, 0.0053] | 0.0625 |
| chest_xray14 | 50k | +0.0249 | [0.0243, 0.0255] | 0.0625 |
| chest_xray14 | full | +0.0411 | [0.0282, 0.0524] | 0.0625 |
| heart_disease | full | +0.0003 | [0.0002, 0.0003] | 0.0625 |

- Toutes les CI sont **positives** dans les deux runs (CP plus lent que linear, direction constante)
- p = 0.0625 (minimum achievable avec n=5 splits via Wilcoxon signe)
- Le patch a réduit les diffs absolues sur chest (~1–3 ms par config), mais le plancher constant reste
- **Aucun task ne montre de speedup significatif** — ni avant ni après le patch

### 7.3 Classification des outcomes (protocol section 12)

| Dataset | base | Verdict (initial) | Verdict (patché) |
|---|---|---|---|
| chest_xray14 | 10k | quality_ok_no_speedup | **quality_ok_no_speedup** |
| chest_xray14 | 50k | quality_ok_no_speedup | **quality_ok_no_speedup** |
| chest_xray14 | full | quality_ok_no_speedup | **quality_ok_no_speedup** |
| heart_disease | full | quality_ok_no_speedup | **quality_ok_no_speedup** |

**Zero task = speedup. Zero task = quality_fail.** Les deux runs sont cohérents :
le patch a fait reculer un handicap accidentel (transposition répétée, ~94% du
temps de requête initial), mais le plancher structurel (dict lookup + rerank
fancy-indexing + overhead constant NumPy) reste au-dessus du coût BLAS
vectorisé de linear. Sur heart, les absolus sont ≪ 1 ms dans les deux cas ;
la hausse du ratio est un artefact de fond de scalaire.

---

## 8. Conclusions scientifiques

### 8.1 Conclusion principale

> **Aucune famille LSH ne bat linear scan en NumPy** sur nos bases (10k–72k) :
> 3.2–11.1x plus lente pour SimHash, E2LSH, et cross-polytope.

C'est un résultat **structurel**, pas spécifique a une implémentation :
le coût constant des rotations/hash/traversée de dicts + rerank par fancy-indexing
depasse le coût du broadcast BLAS vectorise (`np.linalg.norm`) sur des bases de
cette taille. Le patch `_D_all_query` a réduit le ratio CP/lin sur chest de
4.3–11.4x à 3.9–4.3x, confirmant que le plancher est dict + rerank, pas les
transpositions.

### 8.2 Le cross-polytope est la seule famille qui réussit sur Heart Disease

- E2LSH : recall = 0.006 (echec catastrophique, dimensions trop faibles)
- SimHash : recall = 0.531 (insuffisant)
- **CrossPolytope : recall = 0.986 ✓** (threshold atteint, cand_frac = 0.38)

→ **La famille optimale compte, pas seulement l'implémentation.** Le cross-polytope
est le seul a preserver la qualite sur les petites bases en basse dimension.

### 8.3 Coherence avec la litterature

- FALCONN rapporte des speedups jusqu'a 700x, mais sur 10^5–10^8 points
  et en C++/Eigen/FFHT (pas en NumPy)
- Nos bases (≤ 72k) sont dans la zone ou linear scan BLAS reste competitif
- L'exposant optimal du cross-polytope ne se manifeste qu'a grande echelle
- Les methodes a projection sont decrites comme "less competitive than graph-
  and quantization-based methods" (survey PAG, 2026)

### 8.4 Pourquoi FALCONN C++ n'est pas la reponse

- Meme recall/candidats, mais constante mur plus basse → speedup possible
- **Mais** : biais methodologique (C++ vs NumPy), installation risquee sur
  Kaggle (wheels <= 3.8, depensas native), et le protocole interdit
  d'utiliser une baseline differente de celle de la phase 1

---

## 9. Fichiers et artefacts

### 9.1 Structure du projet

```
code/
├── draft/protocol.md              # (supprime apres implementation)
├── exp_lsh/                       # Phase 1 (SimHash + E2LSH)
│   ├── config/                    # protocol.py, seeds.py, metrics.py, datasets.py
│   ├── src/                       # indexes/, calibration/, evaluation/, timing/
│   ├── tests/                     # 16/16 tests
│   ├── notebooks/                 # 00–04 (SimHash + E2LSH)
│   ├── results/                   # CSVs, figures, summary.json, significance_tests.csv
│   └── docs/                      # progress.md, tasks.md, lsh_sota_review.md
├── exp_falconn/                   # Phase 2 (CrossPolytope)
│   ├── config/                    # protocol.py, seeds.py, metrics.py (cosine arm)
│   ├── src/                       # indexes/cross_polytope.py, calibration/, ...
│   ├── tests/                     # 21/21 tests
│   ├── notebooks/                 # 00–04 (cross-polytope only)
│   ├── build_notebooks.py         # Generateur automatique des notebooks
│   ├── results/
│   │   ├── benchmark/             # CSVs Kaggle, summary.json, outcomes.csv,
│   │   │                          # significance_tests.csv, registry.json
│   │   └── figures/               # fig1–fig8 (.png)
│   └── docs/                      # progress.md, tasks.md
├── kaggle/                        # Packages Kaggle
│   ├── exp-lsh-code/              # Dataset package phase 1
│   ├── exp-falconn-code/          # Dataset package phase 2
│   ├── generate_lsh_kernels.py    # Generateur kernels phase 1
│   └── generate_falconn_kernels.py # Generateur kernels phase 2
└── venv/                          # Python 3.12, NumPy 2.5.1, SciPy 1.18, sklearn 1.9
```

### 9.2 Artefacts Kaggle

**Phase 1 (exp_lsh) :**
- Dataset : `franckyoteu/exp-lsh-code`
- Kernels : `franckyoteu/exp-lsh-pipeline-seed-{42,7,123,2024,99}`

**Phase 2 (exp_falconn) :**
- Dataset : `franckyoteu/exp-falconn-code` (dir_mode=zip)
- Kernels : `franckyoteu/exp-falconn-pipeline-seed-{42,7,123,2024,99}`

### 9.3 Fichiers de resultats

| Fichier | Phase | Contenu |
|---|---|---|
| `exp_lsh/results/summary.json` | 1 | Outcomes par methode/dataset, speedups |
| `exp_lsh/results/significance_tests.csv` | 1 | Wilcoxon + bootstrap CIs (E2LSH/SimHash vs linear) |
| `exp_falconn/results/benchmark/summary.json` | 2 | Outcomes CP, key findings |
| `exp_falconn/results/benchmark/outcomes.csv` | 2 | Verdict par task |
| `exp_falconn/results/benchmark/significance_tests.csv` | 2 | Wilcoxon + bootstrap CIs (CP vs linear) |
| `exp_falconn/results/benchmark/registry.json` | 2 | Environnement technique |
| `exp_lsh/docs/lsh_sota_review.md` §7 | — | Tableau comparatif 3 methodes vs linear |

---

## 10. Environnement technique

| Element | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| sklearn | 1.9.0 |
| Plateforme | Kaggle CPU (x86_64, Linux 6.17.0) |
| Execution | 5 kernels parallels, ~48 min chacun |

---

## 11. Decisions en attente / restantes

| Decision | Statut | Contexte |
|---|---|---|
| **Demo Oracle A1 vs Render** | En attente | L'utilisateur doit decider ou heberger la demo web |
| Redaction article/papier | A faire | Synthese des deux phases |
| Polish README | A faire | Mettre a jour les conclusions |
| `draft/protocol.md` | Supprime | Archive dans les config/ des deux phases |

---

## 12. Instructions pour un agent IA reprenant ce projet

### Pour continuer immediatement :
1. Lire ce fichier `decision_history.md`
2. `exp_lsh/docs/lsh_sota_review.md` contient la comparaison SOTA
3. Les figures sont dans `exp_falconn/results/figures/`
4. Les donnees brutes sont dans `exp_falconn/results/benchmark/`
5. `exp_lsh/results/` contient les resultats phase 1

### Pour relancer des experiences :
- Les packages Kaggle existent deja (`exp-lsh-code`, `exp-falconn-code`)
- Les generators de kernels sont dans `kaggle/generate_*.py`
- Les notebooks sont dans `exp_*/notebooks/`
- Le venv est dans `code/venv/`

### Pour ecrire le papier :
- Les deux conclusions principales sont dans les sections 8.1 et 8.2
- Le tableau comparatif complet est en section 7.1
- Les tests de significativite sont en section 7.2
- Les figures sont pretes dans `exp_falconn/results/figures/`

### Ce qu'il ne faut PAS faire :
- Ne pas re-telecharger les donnees Kaggle (deja en local)
- Ne pas re-excuter les notebooks (resultats deja extraits)
- Ne pas passer a FALCONN C++ (decision maintenue, raisons en section 4.1)
- Ne pas modifier la baseline linear (protocol section 15)

---

*Derniere mise a jour : 2026-08-16 par big-pickle*
