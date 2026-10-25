# Demo — CBR + Hachage + Maintenance

Démo interactive Flask (live) pour la soutenance. Réutilise exclusivement le code validé
dans `exp2/src/` et les constantes calibrées exp2 (voir `../docs/demo.md`).

## Lancement

```bash
# depuis la racine du projet : ne pas utiliser le Python système
./venv/bin/python demo/app.py  # http://127.0.0.1:5000
```

Le Python système ne contient pas `pyarrow`; les jeux Diabetes-130 (codes HeCBR)
et ChestX-ray14 (embeddings) ne peuvent donc pas y être lus.

Les actifs HTMX et Chart.js nécessaires à la présentation sont versionnés dans
`static/vendor/` : la démo ne dépend pas d'un CDN au moment de son exécution.
Le détail de l'architecture et des critères de fiabilité est dans
[`../docs/demo_stabilization_plan.md`](../docs/demo_stabilization_plan.md).

## Avertissement important

La **première sélection de ChestX-ray14 prend ~50 s** (construction KD-Tree + LSH sur
~90k cas en 1024-d). À tester **avant** la présentation, pas découvert en direct devant le
public. Les sélections suivantes sont instantanées (état en mémoire).

## Flux

1. Choisir un niveau de dimensionnalité (heart / diabetes130 / chestxray14).
2. Générer un cas automatiquement (jeu de test, réellement nouveau) ou le saisir à la main
   (Heart Disease uniquement — standardisé par le scaler du train).
3. Lancer l'analyse : les trois index (`LinearScanIndex`, `KDTreeIndex`,
   `RandomProjectionLSHIndex`) + `HeCBRIndex` (Diabetes-130) exécutent une vraie requête
   chronométrée ; décision de rétention (distance 1-NN < seuil θ) ; accord des voisins
   top-5 avec le balayage linéaire (référence déterministe).
4. Consulter le panneau maintenance **précalculé** (CBM vs CNN, badge explicite).

Le bouton **Réinitialiser** reconstruit les index du dataset actif depuis les vecteurs
bruts (base ramenée à sa taille initiale).

## Régénération du panneau maintenance

Le fichier `static/data/maintenance_snapshot.json` est généré depuis
`exp2/results/<run_id>/condensation_quality.csv` :

```bash
python -c "import sys; sys.path.insert(0,'.'); from services.maintenance_snapshot import generate_snapshot; generate_snapshot('../exp2/results/20260811_183845_kaggle/condensation_quality.csv')"
```

## Vérification rapide

```bash
cd ..
python -m unittest demo/tests/test_app.py
node --check demo/static/js/charts.js
```
