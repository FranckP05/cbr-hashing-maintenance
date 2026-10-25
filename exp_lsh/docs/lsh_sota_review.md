# État de l'art de la famille LSH — E2LSH et SimHash sont-ils optimaux ?

> **Objectif de ce document :** situer E2LSH et SimHash au sein de la famille LSH,
> identifier les familles réellement optimales (théorie et pratique) et guider la
> suite de l'expérience `exp_lsh`.
>
> Date : 2026-08-16 · Rédigé à partir d'une recherche web ciblée (sources en fin de document).

---

## 1. Verdict

**Non.** E2LSH et SimHash sont les méthodes de *référence historique*
(SimHash/hyperplane : Charikar, 2002 ; E2LSH : Datar et al., 2004), pas l'état de
l'art de la famille LSH. La page officielle du MIT (Andoni) qualifie E2LSH d'
*« early practical LSH algorithm »* et oriente vers **FALCONN** pour les idées
récentes. De plus, notre SimHash cible le cosinus alors que le protocole est en
L2 — c'est un *ablation* de diagnostic, pas un candidat L2 (cf. `progress.md`).

Les familles LSH réellement optimales sont :

| Famille | Métrique | Exposant ρ | Statut |
|---|---|---|---|
| E2LSH (p-stable) | L2 | 1/c² (optimal indépt) | historique, pratique lourde |
| **Cross-polytope (FALCONN)** | cosinus / angulaire | **1/(2c²−1)** | optimal **et** pratique |
| Data-dependent LSH (Andoni–Razenshteyn, STOC'15) | L2 | **1/(2c²−1)** | optimal théorique, pas d'impl. large |
| **SRS** (VLDB 2015) | L2 | — | état de l'art L2 pratique, index minuscule |
| **PM-LSH** (2020) | L2 | — | +70–240 % vs E2LSH/QALSH/Multi-Probe |
| C2LSH / QALSH | L2 | — | collision counting, w automatique |

---

## 2. Pourquoi E2LSH n'est pas optimal

- **Survey Jafari et al. (ACM CSUR, 2021)** : « E2LSH has several drawbacks » —
  c'est l'argument d'ouverture du survey, qui recense les alternatives.
- **FLSH (Pattern Recognition, 2025)** : E2LSH « suffers from excessive storage
  requirements for hash buckets and inadequacies in capturing the precise
  positional information of samples » ; le FLSH proposé (fuzzy rough sets) le
  surpasse en temps de requête et en espace.
- **Défauts documentés** :
  - Besoin de **L très grand** (souvent > 100 tables, parfois plusieurs centaines)
    pour un bon recall → index volumineux (~12 nL octets vs 4 nL pour FBLSH) ;
    mauvaise performance si L < 20.
  - Largeur de bucket `w = 4.0` **fixée manuellement** (critique explicite de
    QALSH, VLDB 2016).
  - Coût d'indexation : g(v) concaténé + deux hashages supplémentaires par table.
  - Échec en basse dimension (partitions inefficaces) — exactement ce que notre
    expérience observe sur Heart Disease (recall@10 = 0.0058).
  - Variantes récentes : CS-E2LSH / HCS-E2LSH (count sketch, TCS 2025),
    CP-E2LSH / TT-E2LSH (tensorisé, 2025), FLSH (2025).

## 3. Pourquoi SimHash (hyperplane/SRP) est sous-optimal

- Charikar (2002) est la famille « standard » mais **théoriquement non optimale**.
- **« Practical and Optimal LSH for Angular Distance » (NIPS 2015, Andoni, Indyk,
  Laarhoven, Razenshteyn, Schmidt)** : le **cross-polytope LSH** atteint l'exposant
  optimal ρ = 1/(2c²−1) et *« improves upon the well-studied hyperplane LSH in
  practice »* — **3.4× sur NYT, 4× sur pubmed, jusqu'à 10× vs hyperplane et 700×
  vs linear scan** sur 10⁵–10⁸ points.
- **FALCONN** (lib C++ + wrapper Python, MIT, falconn-lib.org) implémente
  hyperplane **et** cross-polytope avec multi-probe ; cross-polytope recommandé.
  Variante plus récente : **Falconn++** (Pham, 2022), filtrage par locality
  sensitive filtering, « competitive with HNSW » sur les hauts recalls.

## 4. Pourquoi SimHash est doublement contestable dans notre protocole

1. **Famille sous-optimale** : l'hyperplane a un exposant ρ plus mauvais que le
   cross-polytope pour l'angulaire.
2. **Métrique inadaptée** : SimHash approxime le cosinus ; notre protocole est en
   **L2** (`PRIMARY_METRIC = "l2"`). Notre SimHash est explicitement un ablation
   de diagnostic (§4.2 du protocole), pas un candidat L2.

## 5. FALCONN / cross-polytope : pourquoi c'est l'option à retenir

- L'angulaire et le L2 sur la sphère unité sont équivalents :
  ‖p − q‖² = 2 − 2·cos(p, q) pour des vecteurs normalisés. Le papier NIPS 2015
  l'affirme explicitement : *« angular distance can be expressed via the Euclidean
  distance between normalized vectors »*.
- Le cross-polytope (rotation aléatoire + `argmax |coordonnée|`) :
  - est **asymptotiquement optimal** pour l'angulaire ;
  - se combine avec des **rotations pseudo-aléatoires** (Hadamard, O(d log d))
    au lieu d'une matrice dense O(d²) ;
  - dispose d'un **multi-probe** efficace (classer les coordonnées par |valeur|) ;
  - est implémenté dans FALCONN (C++/Eigen/FFHT + wrapper Python).
- Attention : l'exposant optimal n'est atteint qu'à **grande échelle**
  (10⁵–10⁸ points). Sur nos bases (10k–72k), la littérature et nos propres
  résultats indiquent que le linear scan BLAS peut rester compétitif — c'est un
  résultat scientifique en soi.

## 6. Au-delà du LSH : mise en garde plus large

- **ANN-Benchmarks + surveys 2025/2026** : HNSW domine le compromis recall/latence ;
  IVF-PQ domine en mémoire. Les méthodes à projection (E2LSH, FALCONN, CEOs) sont
  décrites comme « less competitive than graph- and quantization-based methods »
  (survey PAG, 2026).
- Les speedups LSH publiés (FALCONN, 700×) ne tiennent qu'à **grande échelle**.
- La valeur scientifique d'un LSH dans notre cadre reste : garanties théoriques,
  index dynamique (insertions O(1)), et comparaison stricte avec linear scan.

---

## Sources

- LSH Home Page — https://www.mit.edu/~andoni/LSH/ (E2LSH = early implementation ; FALCONN = newer ideas)
- Survey Jafari, Maurya, Nagarkar et al., *A Survey on LSH Algorithms and their Applications*, arXiv:2102.08942
- Andoni, Indyk, Laarhoven, Razenshteyn, Schmidt, *Practical and Optimal LSH for Angular Distance*, NeurIPS 2015 — arXiv:1509.02897
- Andoni, Razenshteyn, *Optimal Data-Dependent Hashing for Approximate Near Neighbors*, STOC 2015 — arXiv:1501.01062
- Kennedy, Ward, *Fast Cross-Polytope LSH*, arXiv:1602.06922
- FALCONN — https://github.com/FALCONN-LIB/FALCONN · https://pypi.org/project/FALCONN/ · falconn-lib.org
- Pham, Liu, *Falconn++: A Locality-sensitive Filtering Approach for ANN Search*, arXiv:2206.01382
- SRS, *Fast ANN Search in High-Dimensional Euclidean Space with a Tiny Index*, VLDB 2015 — https://github.com/DBAIWangGroup/SRS
- Zheng et al., *PM-LSH*, arXiv:2107.05537
- Huang, Feng, Zhang, Wang, Qin, *Query-Aware LSH (QALSH)*, VLDB 2016 — vldb.org/pvldb/vol9/p1-huang.pdf
- Lv et al., *Multi-Probe LSH*, VLDB 2007
- Gan et al., *C2LSH*, SIGMOD 2012
- Verma, Pratap, *Faster and space efficient indexing for LSH (CS-E2LSH/HCS-E2LSH)*, TCS 2025
- FLSH, *A storage-efficient LSH scheme for high-dimensional ANN search*, Pattern Recognition 2025
- Verma, Pratap, *Improving LSH via Tensorized Random Projection (CP/TT-E2LSH)*, Acta Informatica 2025
- ANN-Benchmarks — https://ann-benchmarks.com/
- *Approximate Nearest Neighbor Search for Modern AI: A Projection-Augmented Graph Approach (PAG)*, arXiv:2603.06660
- Survey *Learning to Hash for Indexing Big Data*, Wang et al., Proc. IEEE 2016
- Survey *Deep Hashing*, ACM TKDD 2023

---

## 7. Comparaison empirique des trois méthodes (phase 1 + phase 2, Kaggle)

> Moyennes sur 5 splits × 3 hash seeds (15 runs par cellule), `recall@10` sur
> baseline de référence, qualité au sens du protocole §12 :
> recall@10 ≥ 0.95 **et** Δmacro-F1 ≤ 0.01. Latence = médiane par requête.
> **CP = run patché** (aprés précalcul `_D_all_query` contigu).

| Dataset | base | Méthode | recall@10 | Latence (ms) | vs linear | Qualité §12 |
|---|---|---|---|---|---|---|
| chest_xray14 | 10k | **linear** | 1.0000 | 1.68 | 1× | référence |
| chest_xray14 | 10k | **cross-polytope** | 0.9741 | 6.88 | **4.1× plus lent** | ✓ (pass) |
| chest_xray14 | 10k | E2LSH | 0.9852 | 12.91 | 7.3× plus lent | ✓ |
| chest_xray14 | 10k | SimHash | 0.9830 | 14.02 | 8.0× plus lent | ✓ |
| chest_xray14 | 50k | **linear** | 1.0000 | 8.65 | 1× | référence |
| chest_xray14 | 50k | **cross-polytope** | 0.9894 | 33.54 | **3.9× plus lent** | ✓ |
| chest_xray14 | 50k | E2LSH | 0.9784 | 38.64 | 4.7× plus lent | ✓ |
| chest_xray14 | 50k | SimHash | 0.9631 | 26.14 | 3.2× plus lent | ✓ |
| chest_xray14 | full | **linear** | 1.0000 | 12.43 | 1× | référence |
| chest_xray14 | full | **cross-polytope** | 0.9817 | 53.57 | **4.3× plus lent** | ✓ |
| chest_xray14 | full | E2LSH | 0.9824 | 58.14 | 4.9× plus lent | ✓ |
| chest_xray14 | full | SimHash | 0.9685 | 40.17 | 3.4× plus lent | ✓ |
| heart_disease | full | **linear** | 1.0000 | 0.02 | 1× | référence |
| heart_disease | full | **cross-polytope** | 0.9858 | 0.28 | **11.1× plus lent** | ✓ |
| heart_disease | full | E2LSH | 0.0058 | 0.03 | ~1× | ✗ (recall catastrophique) |
| heart_disease | full | SimHash | 0.5314 | 0.12 | 4.9× plus lent | ✗ (recall 0.53) |

### Lectures du tableau

1. **Aucune méthode LSH ne bat linear scan en NumPy, quelle que soit la famille.**
   La phase 2 (cross-polytope, famille *théoriquement optimale*) confirme la
   conclusion de la phase 1 : qualité préservée, **jamais** de speedup
   (3.9–11.1× plus lent, patch appliqué). Les trois familles sont du même ordre de grandeur
   (3–8× plus lentes que linear sur chest).
2. **Le cross-polytope est la seule famille qui réussit sur Heart Disease.**
   E2LSH échoue totalement (recall@10 = 0.006, échec de la stabilisation en
   basse dimension, critique connue des p-stable) et SimHash plafonne à 0.53.
   Le cross-polytope atteint 0.986 (threshold ✓) en visitant 38 % des candidats —
   preuve empirique que la *famille* optimale compte, pas seulement
   l'implémentation.
3. **Le bottleneck est structurel, pas spécifique à une famille** : le coût
   constant des hashs/rotations + traversée de dictionnaires Python + le
   re-rankinage par fancy-indexing vs un unique broadcast BLAS
   (`L2_norm` vectorisé) chez linear. Le patch `_D_all_query` a éliminé la
   transposition répétée (~94% du temps initial), réduisant le ratio de ~7× à
   ~4× sur chest full — confirmant que le plancher restant est dict + rerank.
4. **Échelle : cohérent avec la littérature** (cf. §5) — l'exposant optimal du
   cross-polytope ne se manifeste qu'à 10⁵–10⁸ points. Nos bases (≤ 72k) sont
   dans la zone où linear scan BLAS reste compétitif : c'est le résultat
   scientifique principal des deux phases.
5. **Méthodologiquement comparable** : mêmes seeds, mêmes bases, même linear
   baseline NumPy, même harnais Kaggle. Seule différence : le cross-polytope
   normalise L2 (protocole §5, arm cosinus) — linéairement équivalent au L2
   (‖p−q‖² = 2−2cos(p,q) sur la sphère unité).

### Conclusion SOTA révisée

Le cross-polytope FALCONN-style est bien la famille optimale (théorie + pratique,
seul passage du seuil qualité sur Heart), mais dans notre cadre NumPy il ne
procure pas de speedup face au linear scan. Le patch `_D_all_query` a réduit le
ratio de ~7× à ~4× sur chest, mais le plancher constant (dict + rerank) reste
insurmontable. Les speedups publiés (FALCONN C++, jusqu'à 700×) requièrent une
implémentation native **et** de très grandes bases.
