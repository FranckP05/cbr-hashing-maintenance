# HeCBR & SOTA LSH (Cross-Polytope Multi-Probe) sur Camelyon16 (10^6+)

## Vue d'ensemble du projet
Ce projet compare trois paradigmes fondamentaux de recherche de plus proches voisins à l'échelle industrielle ($N \ge 3,77 \times 10^6$ points, dimension $d=512$, classification binaire histopathologique sur Camelyon16) :

1. **Baseline Exacte (Balayage linéaire BLAS)** : Référence exacte en $O(Nd)$, complexité linéaire en temps, vecteurs `float32`.
2. **SOTA LSH Géométrique (Cross-Polytope avec Multi-Probe — FALCONN)** : Hachage sans apprentissage à exposant optimal $\rho = \frac{1}{2c^2 - 1}$, rotations rapides par Transformée de Hadamard (FHT en $O(d \log d)$) et exploration multi-cônes.
3. **Deep Hashing Supervisé (HeCBR — Hashing-enhanced CBR)** : Réseau de neurones profond apprenant une projection non linéaire vers un espace binaire compact ($B = 64$ bits), recherche par distance de Hamming vectorisée (`XOR` + `POPCNT`).

```text
hecbr_sota_camelyon/
├── README.md                 # Vue d'ensemble et guide d'exécution
├── DECISIONS_AND_RESULTS.md  # Journal de méthodologie, équations et résultats
├── PROGRESS.md               # Suivi d'avancement étape par étape
├── requirements.txt          # Dépendances Python
├── config/                   # Configuration du protocole, seeds, chemins et hyperparamètres
├── data/                     # Chargement et split par lame (réutilisation du cache 4.88M)
├── indexes/                  # Implémentations d'index (LinearScan, CrossPolytope LSH, HeCBR Hamming)
├── models/                   # Réseau HeCBR (PyTorch), loss contrastive/triplet et binarisation
├── calibration/              # Calibration des hyperparamètres SOTA LSH et HeCBR
├── benchmark/                # Harnais de mesure (warm-up, répétitions, Recall@k, F1, mémoire)
├── visualization/            # Thème Matplotlib centralisé et génération de figures (300 DPI)
├── scripts/                  # Points d'entrée d'entraînement et d'évaluation
└── tests/                    # Tests unitaires de validation
```
