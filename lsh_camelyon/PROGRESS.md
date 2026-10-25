# PROGRESS — LSH Camelyon16 (10^6+)

> Document de suivi en temps réel du projet de recherche.
> Dernière mise à jour : 2026-08-30

---

## État des Tâches et Réalisations

### ✅ TOUTES LES ÉTAPES TERMINÉES ET VALIDÉES

1. **Arborescence & Code Métier** :
   - Pipeline modulaire pur Python (sans notebook Jupyter).
   - 21/21 tests unitaires validés avec succès (`pytest tests/`).

2. **Exécution sur Lightning AI (Échelle 10^6+)** :
   - Données Camelyon16 réelles chargées : 399 lames entières WSI (**4 886 434 patches**, 512-D SimCLR).
   - Base d'indexation finale : **3 772 657 patches**.
   - Calibration de la grille $(k, L)$ réalisée sur l'ensemble de validation.
   - Benchmark comparatif complet multi-seeds (seeds 11, 29, 43, 67, 97) exécuté avec harnais temporel (1 warm-up + 5 répétitions).

3. **Résultats Récupérés et Disponibles en Local** :
   - Fichiers CSV/JSON dans `lsh_camelyon/results/raw/` :
     - `benchmark_timing.csv` : Latences détaillées (médianes, moyennes, p25, p75, p95, p99).
     - `benchmark_quality.csv` : Recall@10, Macro-F1, fractions de candidats.
     - `best_params_simhash.json` : Paramètres optimaux de calibration ($k=12, L=5$).
     - `calibration_results.csv` : Tableau complet des 49 points de la grille de validation.
   - Figures générées dans `lsh_camelyon/results/figures/` (format 8x4.5 pouces, 300 DPI) :
     - `fig1_latency_comparison.png`
     - `fig2_speed_quality.png`
     - `fig3_latency_distribution.png`
     - `fig4_calibration_surface.png`

4. **Journal de Continuité** :
   - `DECISIONS_AND_RESULTS.md` entièrement mis à jour avec les tableaux comparatifs et les conclusions scientifiques.
