# Journal de Continuité — Protocole LSH Camelyon16 (10^6+)

> **Objet :** Document de continuité scientifique et technique unique.
> Toute IA ou chercheur reprenant ce projet doit pouvoir lire **uniquement ce fichier** pour comprendre l'intégralité du contexte, des résultats obtenus, des choix de conception et des conclusions empiriques.
>
> Dernière mise à jour : 2026-08-30

---

## 1. Synthèse Exécutive des Résultats

Sur la base médicale réelle **Camelyon16** (**4 886 434 patches réels**, 512 dimensions SimCLR, 399 lames entières, partition 90/10 stratifiée par lame = **3 772 657 vecteurs en base**), ce protocole a évalué la transition d'échelle entre le balayage linéaire optimisé (BLAS) et le LSH (SimHash à hyperplans aléatoires) :

| Méthode | Graine | Latence médiane (ms) | Speedup vs Linéaire | Recall@10 | Macro-F1 (CBR) | Fraction candidats | Build Time (s) |
|---|---|---|---|---|---|---|---|
| **Baseline Linéaire (BLAS)** | — | **403.91 ms** | **1.00x** (ref) | **1.0000** | **0.4424** | 1.0000 | 0.00 s |
| **SimHash (k=12, L=5)** | seed 11 | 289.67 ms | 1.39x plus rapide | 0.0110 | 0.2063 | 0.9559 | 8.94 s |
| **SimHash (k=12, L=5)** | seed 29 | 253.45 ms | 1.59x plus rapide | 0.0100 | 0.2042 | 0.9486 | 9.75 s |
| **SimHash (k=12, L=5)** | seed 43 | 314.86 ms | 1.28x plus rapide | 0.0093 | 0.2208 | 0.9966 | 9.14 s |
| **SimHash (k=12, L=5)** | seed 67 | 298.01 ms | 1.36x plus rapide | 0.0097 | 0.2000 | 0.9599 | 9.80 s |
| **SimHash (k=12, L=5)** | seed 97 | 270.73 ms | 1.49x plus rapide | 0.0097 | 0.2084 | 0.9668 | 8.56 s |
| **SimHash (Moyenne inter-seeds)** | — | **285.34 ms** | **1.42x plus rapide** | **0.0099** (0.99%) | **0.2079** | **0.9656** | **9.24 s** |

---

## 2. Conclusions Scientifiques Majeures

### 2.1 Inversion de la Tendance en Vitesse (Hypothèse d'Échelle Validée)
- **À petite échelle ($\le 72k$ points, cf. ChestX-ray14)** : Le balayage linéaire BLAS battait toutes les méthodes LSH (SimHash était 3.2x à 8.0x plus lent que le linéaire).
- **À grande échelle ($3.77 \times 10^6$ points)** : La tendance **s'est effectivement inversée**. SimHash est devenu **1.42x plus rapide** que le balayage linéaire BLAS ($285\text{ ms}$ vs $404\text{ ms}$).
- **Validation empirique** : Cela confirme directement l'hypothèse de la littérature (Andoni & Indyk, FALCONN) situant l'apparition de l'avantage de complexité sous-linéaire du LSH à partir de $10^5-10^8$ points.

### 2.2 Effondrement de la Qualité de Voisinage (Dilemme Fondamental de SimHash en Haute Dimension)
- Le rappel au rang 10 s'effondre à **$\approx 1.0\%$** (Recall@10 = 0.0099), entraînant une chute du Macro-F1 de 0.4424 (linéaire) à 0.2079.
- **Explication géométrique ($d=512$)** :
  - En dimension 512, les angles entre voisins restent modérément élevés ($\theta \approx 80^\circ \approx 1.4\text{ rad}$).
  - La probabilité élémentaire de collision par hyperplan est $P[h(p) = h(q)] = 1 - \frac{\theta}{\pi} \approx 0.555$.
  - Pour $k=12$ bits, la probabilité de collision dans un bucket d'une table est $(0.555)^{12} \approx 0.00085$.
  - Pour $L=5$ tables, la probabilité d'au moins une collision est $1 - (1 - 0.00085)^5 \approx \mathbf{0.42\%}$.
  - Le vrai plus proche voisin a moins de 1 chance sur 200 d'être capturé dans les tables.

### 2.3 Comparaison avec les Résultats Précédents
- **SimHash pur** (hyperplans aléatoires) sans multi-probing est insuffisant en haute dimension ($d=512$) à grande échelle : soit il ne filtre rien ($k \le 8$), soit il manque la quasi-totalité des voisins ($k \ge 12$).
- Cela démontre la nécessité absolue des techniques modernes avancées telles que **Cross-Polytope LSH** et le **multi-probing sphérique** (FALCONN) pour préserver la qualité de voisinage tout en conservant le gain de vitesse à grande échelle.

---

## 3. Données et Environnement d'Exécution

- **Dataset :** Camelyon16 (embeddings SimCLR 512-D, 399 WSI).
  - Normal : 2 637 729 patches (239 lames).
  - Tumeur : 2 249 104 patches (160 lames).
  - Total : 4 886 434 patches.
- **Split :** 90% base (305 lames, 3 772 657 patches) / 10% test (40 lames) / validation (54 lames).
- **Métrique :** Similarité Cosinus / Distance Euclidienne sur vecteurs normalisés L2.
- **Machine :** Instance Lightning AI Ubuntu 6.8.0, 14 GB RAM.

---

## 4. Figures Générées (300 DPI)

1. **`results/figures/fig1_latency_comparison.png`** : Barres de latence médiane (BLAS vs SimHash multi-seeds, moustaches p25-p75 et p95).
2. **`results/figures/fig2_speed_quality.png`** : Compromis Vitesse / Rappel@10.
3. **`results/figures/fig3_latency_distribution.png`** : Dispersion des latences inter-seeds.
4. **`results/figures/fig4_calibration_surface.png`** : Surface de latence selon $(k, L)$ sur la validation.
