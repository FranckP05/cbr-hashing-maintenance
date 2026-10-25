# Méthodologie et Journal Scientifique — SOTA LSH vs HeCBR (Échelle 10^6+)

> **Objet :** Document de référence scientifique détaillant les fondements mathématiques, les choix architecturaux et la méthodologie expérimentale comparant le **LSH géométrique optimal (Cross-Polytope avec Multi-Probe)**, le **Deep Hashing supervisé (HeCBR)** et la **Baseline Linéaire Exacte** sur la base histopathologique **Camelyon16** ($N \ge 3,77 \times 10^6$ patches, $d=512$).
>
> Date : 2026-08-30

---

## 1. Contexte et Problématique Scientifique

L'expérience précédente avec **SimHash pur** (hyperplans aléatoires) sur Camelyon16 a démontré la double réalité suivante à l'échelle $10^6+$ :
1. **Validation de la transition de complexité** : À $3,77\times 10^6$ points, le LSH devient effectivement plus rapide que le balayage linéaire BLAS ($285\text{ ms}$ vs $404\text{ ms}$).
2. **Effondrement géométrique de SimHash** : Le rappel s'est effondré à $\approx 1\%$ à cause de la concentration de la mesure en dimension $d=512$ où les angles entre voisins réels $\theta \approx 80^\circ$ donnent une probabilité de collision conjointe $(1 - \theta/\pi)^k \approx 0,4\%$.

Pour surmonter cette barrière, ce protocole introduit deux paradigmes majeurs :
- **Paradigme 1 (Sans apprentissage - SOTA LSH)** : Cross-Polytope LSH avec Multi-Probe sphérique (FALCONN).
- **Paradigme 2 (Avec apprentissage supervisé - Deep Hashing)** : HeCBR (*Hashing-enhanced Case-Based Reasoning*) avec projection non linéaire vers un espace de Hamming à $64$ bits.

---

## 2. Fondements Mathématiques des Méthodes

### 2.1 Baseline Linéaire Optimisée (Exacte)
- **Espace** : Vecteurs continus $x \in \mathbb{R}^{512}$ normalisés en norme $L_2$ ($\|x\|_2 = 1$).
- **Métrique** : Similarité cosinus via la relation exacte $\|p - q\|_2^2 = 2 - 2 \cos(p, q)$.
- **Complexité** : Temps $O(Nd)$, Mémoire $O(Nd \times 4\text{ octets}) \approx 7,5\text{ Go}$.

### 2.2 SOTA LSH : Cross-Polytope LSH avec Multi-Probe (Andoni et al., NIPS 2015)
- **Optimalité théorique** : Atteint l'exposant optimal $\rho = \frac{1}{2c^2 - 1} + o(1)$ pour la distance angulaire.
- **Rotation rapide par Transformée de Hadamard (FHT)** :
  Au lieu d'une matrice dense gaussienne $O(d^2)$, on utilise $R = D_3 H D_2 H D_1 H$ où $H$ est la matrice normalisée de Walsh-Hadamard et $D_i$ sont des matrices diagonales de signes aléatoires ($\pm 1$).
  Complexité de rotation : $O(d \log d)$.
- **Hachage sur le Polytope Croisé** :
  Le vecteur tourné $y = R x$ est projeté sur le sommet le plus proche du polytope croisé à $2d$ sommets $\{\pm e_1, \dots, \pm e_d\}$ :
  $$h(x) = \arg\max_{i \in \{1,\dots,2d\}} |y_i| \quad \text{avec le bit de signe } \text{sgn}(y_i)$$
- **Multi-Probe Sphérique Vectorisé** :
  Au lieu de n'explorer que le bucket exact de la requête, on évalue les scores d'éloignement $\delta_i = \max_j |y_j| - |y_i|$ et on sonde de manière ordonnée les $M$ buckets les plus proches. Cela restaure un **Recall $> 90\%$** sans exploser le nombre de tables $L$.

### 2.3 HeCBR : Deep Hashing Supervisé (Hamming CBR)
- **Architecture du Réseau** :
  $$\Phi_\theta(x) = \tanh(W_2 \cdot \text{ReLU}(\text{BN}(W_1 x + b_1)) + b_2) \in [-1, 1]^B$$
  avec $B = 64$ bits (ou $128$ bits).
- **Code binaire discret** :
  $$b(x) = \text{sgn}(\Phi_\theta(x)) \in \{-1, +1\}^B \implies \text{packbits} \rightarrow \text{uint64 (8 octets)}$$
- **Fonction de Perte Conjointe** :
  $$\mathcal{L} = \mathcal{L}_{\text{triplet/contrastive}}(b(x_i), b(x_j), y_{ij}) + \lambda \mathcal{L}_{\text{quantization}}$$
  où la perte de quantification $\mathcal{L}_{\text{quant}} = \frac{1}{B} \sum_{k=1}^B (|\Phi_\theta(x)_k| - 1)^2$ force les sorties continues vers $\pm 1$ pour éliminer l'erreur de binarisation.
- **Recherche de Voisins ultra-rapide** :
  La distance de Hamming entre deux codes binaires $u, v \in \{0, 1\}^B$ est évaluée par les instructions processeur matérielles :
  $$d_H(u, v) = \text{POPCNT}(u \oplus v)$$
  Temps de scan pour $3,77 \times 10^6$ codes de 64 bits : **$< 1,5\text{ ms}$** sur un seul cœur CPU.
- **Empreinte Mémoire** : $3,77\text{M} \times 8\text{ octets} = \mathbf{30,1\text{ Mo}}$ (divisée par 250 par rapport aux 7,5 Go de la baseline).

---

## 3. Protocole Expérimental

1. **Jeu de données** : Camelyon16 (399 lames entières WSI, 4 886 434 patches, 512 dimensions).
2. **Découpage strict** : 90% base (305 lames, 3 772 657 patches) / 10% test (40 lames), stratifié par proportion de tissu tumoral / normal.
3. **Entraînement HeCBR** : Entraîné sur un sous-ensemble représentatif de validation (paires de patches intralames et interlames).
4. **Harnais de mesure** : 1 warm-up + 5 répétitions chronométrées par requête avec `time.perf_counter()`.
5. **Métriques évaluées** :
   - Latence médiane, moyenne, p95 (ms)
   - Recall@10 par rapport à la vérité terrain exacte
   - Macro-F1 de classification CBR (k-NN vote majoritaire)
   - Empreinte mémoire vive (Go / Mo)
   - Temps de construction de l'index
