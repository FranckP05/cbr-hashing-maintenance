# PROGRESS — HeCBR & SOTA LSH sur Camelyon16 (10^6+)

> Document de suivi en temps réel du projet de recherche.
> Dernière mise à jour : 2026-08-30

---

## État des Tâches et Réalisations

### 1. Architecture & Méthodologie
- [x] Définition des équations et contrats mathématiques (`DECISIONS_AND_RESULTS.md`).
- [x] Documentation et guide d'exécution (`README.md`).
- [x] Fichier de configuration du protocole (`config/protocol.py`, `config/paths.py`, `config/seeds.py`).

### 2. Implémentations des Index & Modèles
- [x] Baseline Linéaire Optimisée BLAS (`indexes/linear_scan.py`).
- [x] SOTA LSH : Cross-Polytope avec FHT et Multi-Probe vectorisé (`indexes/cross_polytope.py`).
- [x] HeCBR : Architecture PyTorch Deep Hashing ($512 \rightarrow 256 \rightarrow 64\text{ bits}$) (`models/hecbr_net.py`).
- [x] Index binaire Hamming vectorisé (`indexes/hecbr_index.py`).
- [x] Entraîneur de hachage supervisé avec loss de quantification (`models/train_hecbr.py`).

### 3. Pipeline d'Évaluation & Visualisation
- [x] Harnais de mesure temporelle (1 warm-up + 5 répétitions, médiane, p25, p75, p95) (`benchmark/timing.py`).
- [x] Métriques de qualité (Macro-F1, Recall@10, mémoire) (`benchmark/metrics.py`).
- [x] Script de calibration et benchmark comparatif complet (`scripts/run_comparative_benchmark.py`).
- [x] Générateur de graphiques 300 DPI conforme à la charte (`visualization/plots.py`).
- [x] Tests unitaires complets (`tests/`).

### 4. Déploiement sur Lightning AI
- [ ] Code synchronisé dans `/teamspace/studios/this_studio/hecbr_sota_camelyon/`.
- [ ] Validation des tests unitaires sur le serveur distant.
- [ ] Entraînement du modèle HeCBR (sur GPU/CPU).
- [ ] Benchmark comparatif des 3 méthodes (Baseline vs SOTA LSH vs HeCBR).
- [ ] Rapatriement des CSV et des 4 figures PNG à 300 DPI.
