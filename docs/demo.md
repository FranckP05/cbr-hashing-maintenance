# Spécification — Démo interactive CBR + Hachage + Maintenance (exp2)

> **Version exp2** — cette spécification remplace la version d'origine (`draft/demo.md`),
> écrite pour le contexte exp1 et **périmée** : tous les seuils, paramètres LSH et
> résultats cités y provenaient du journal exp1 (`docs/journal_decisions_methodologiques.md`)
> et ne correspondent pas à nos résultats exp2. **Ne pas se référer au journal exp1.**
> Références valides : `docs/plan_experimentation_reproductible.md` (protocole) et les
> sorties exp2 (`exp2/results/<run_id>/*.csv`).

## Objectif

Une application web Flask qui exécute **EN DIRECT** (pas de simulation, pas de valeurs
précalculées sauf pour le panneau maintenance, explicitement identifié comme tel) les
méthodes d'indexation déjà implémentées et validées dans **exp2** sur un cas nouveau
choisi ou saisi par le public, pour une démonstration lors de la soutenance :

- **Balayage linéaire**, **KD-Tree**, **LSH** — les trois datasets (live).
- **HeCBR** (recherche par codes binaires 64 bits, distance de Hamming) — **Diabetes-130
  uniquement** (les codes pré-entraînés n'existent que pour ce dataset), en plus des trois
  méthodes (décision actée : « toutes les méthodes »).

RÈGLE ABSOLUE : la démo **ne réimplémente aucun algorithme**. Elle importe et réutilise
exclusivement le code validé dans `exp2/src/` (`datasets/`, `indexing/`, `evaluation/`,
`maintenance/`). Si une fonctionnalité manque pour la démo, on l'ajoute comme petite
extension au module concerné — on ne duplique jamais la logique dans `demo/`.

## Emplacement

Tout le code de la démo vit dans `demo/`, à la racine du projet `code/`, à côté de `exp2/`.
La démo importe depuis `exp2/src/` via un chemin relatif (voir section Import).

## Palette et thème visuel (distincte du thème Matplotlib du mémoire — NE PAS confondre)

Thème du **wireframe utilisateur** (artefacts distincts : figures du mémoire en thème
Nordic Research Space Grotesk/Manrope obsidienne/orange brûlé ; démo en thème wireframe).

- Violet principal : `#6D28D9` (violet-700) — accent fort, titre « Demo », barre LSH.
- Jaune : `#FACC15` (yellow-400) — boutons secondaires, barre K-D Tree, puces d'étapes.
- Noir : `#18181B` — texte principal, fond des cartes de dimensionnalité, barre Baseline.
- Blanc : fond de page.
- Jaune très clair : `#FEF3C7` — halo décoratif des icônes du hero.

Polices (wireframe) : **Space Grotesk** pour les titres et les boutons, **Manrope** pour le
corps de texte. La même palette que les figures du mémoire peut réutiliser ces polices — ce
qui distingue la démo du mémoire est le **thème couleur** (wireframe), pas la police.

## Architecture

```
demo/
├── app.py                    # point d'entrée Flask
├── requirements.txt          # flask (+ pandas, numpy, scikit-learn déjà dans venv/)
├── config.py                 # constantes exp2 : datasets, seuils, paramètres LSH
├── state.py                  # état mono-session en mémoire (section État)
├── services/
│   ├── __init__.py
│   ├── dataset_bridge.py     # adapters exp2 : get_split (80/20 stratifié), scaler, ids
│   ├── index_manager.py      # construit/reconstruit les index, cache, reset
│   ├── live_analysis.py      # query() réels chronométrés + rétention + accord
│   └── maintenance_snapshot.py  # panneau précalculé (CBM vs CNN), valeurs exp2
├── templates/
│   ├── base.html             # layout Tailwind + CDN htmx + CDN Alpine.js
│   ├── index.html            # page complète, étape 1 visible au chargement
│   └── fragments/
│       ├── dataset_selected.html   # étape 2 (génération / saisie)
│       ├── manual_case_form.html   # saisie manuelle (Heart Disease uniquement)
│       ├── case_ready.html         # étape 3 (cas prêt, bouton « Lancer l'analyse »)
│       ├── analysis_results.html   # étape 4 (cartes de résultats)
│       └── maintenance_panel.html  # panneau maintenance précalculé
├── static/
│   ├── css/                   # styles (Tailwind CDN acceptable : démo locale)
│   └── data/
│       └── maintenance_snapshot.json  # CBM vs CNN, valeurs exp2 (une fois généré)
└── README.md                 # lancement + avertissement Chest (~50 s)
```

## Import du code existant (exp2)

En tout début d'`app.py` :

```python
import sys
from pathlib import Path

# La démo réutilise le code validé d'exp2 (src/, config/) — on ajoute exp2/ au chemin
# d'import pour faire "from src.indexing.linear_scan import LinearScanIndex" ou
# "from config.datasets import DATASETS" exactement comme les notebooks exp2.
EXP2 = Path(__file__).resolve().parent.parent / "exp2"
sys.path.insert(0, str(EXP2))
```

## Config (config.py) — constantes exp2

Les constantes sont les **valeurs calibrées d'exp2** (et non celles du journal exp1).
Elles sont figées ici pour éviter tout temps de démarrage inutile et garantir la cohérence
avec les résultats du mémoire. **Ne pas les recalculer au démarrage.**

| Dataset (clé exp2) | Label | n | dim | LSH (NB03) | Seuil rétention (NB07, seed 42) |
|---|---|---|---|---|---|
| `heart` | Faible dimensionnalité | 297 | 13 | bits 3 / tables 8 | 1.641387 |
| `diabetes130` | Moyenne dimensionnalité | 101 766 | 390 | bits 8 / tables 16 | 2.630000 |
| `chestxray14` | Haute dimensionnalité | 112 120 | 1024 | bits 10 / tables 8 | 0.972312 |

Provenance :
- **LSH** : `exp2/results/<run_id>/lsh_calibration.csv` + `nb03_summary.json`
  (`lsh_params_chosen`, calibré base complète, 30 requêtes, rappel ≥ 0.95). NB03.
- **Seuil de rétention** θ : 25e percentile des distances 1-NN du train
  (échantillon de 2 000 cas), seed 42 — `retention_agreement.csv` (colonne `threshold`).
  NB07 (protocole `config/maintenance.py` : `RETENTION_NN_PERCENTILE=25`,
  `RETENTION_THRESHOLD_SAMPLE=2000`).
- **n / dim** : chargés par les loaders exp2 (heart 297, diabetes 101 766, chest 112 120 ;
  dims 13 / 390 / 1024).

```python
K = 5  # nombre de voisins pour l'analyse live (cohérent avec exp2 K_VALUES)

DATASETS = {
    "heart": {
        "label": "Faible dimensionnalité",
        "n_features": 13,
        "n_total": 297,
        "allow_manual_entry": True,          # seul dataset avec saisie manuelle
        "retention_threshold": 1.641387,     # NB07 seed 42
        "lsh_params": {"num_bits": 3, "num_tables": 8},   # NB03
    },
    "diabetes130": {
        "label": "Moyenne dimensionnalité",
        "n_features": 390,
        "n_total": 101766,
        "allow_manual_entry": False,
        "retention_threshold": 2.630000,     # NB07 seed 42
        "lsh_params": {"num_bits": 8, "num_tables": 16},  # NB03
    },
    "chestxray14": {
        "label": "Haute dimensionnalité",
        "n_features": 1024,
        "n_total": 112120,
        "allow_manual_entry": False,
        "retention_threshold": 0.972312,     # NB07 seed 42
        "lsh_params": {"num_bits": 10, "num_tables": 8},  # NB03
    },
}
```

Les chemins de données et la convention de clés sont repris de `exp2/config/datasets.py`
(fonction `source(key)`) — la démo ne duplique pas les chemins.

## État mono-session (state.py)

État global en mémoire partagé par tous les visiteurs (mono-session assumée — un poste,
une présentation, pas un déploiement public). Par dataset déjà sélectionné : vecteurs
bruts du train (immuables), trois index actifs (mutables via `add_case()`), taille de base.
`current_dataset` pointe le dataset actif ; `current_case` contient le dernier cas en
attente d'analyse.

```python
class DemoState:
    def __init__(self):
        self.pristine_train_vectors = {}   # key -> np.ndarray (jamais modifié)
        self.pristine_test_vectors = {}     # key -> np.ndarray (génération auto)
        self.train_encounter_ids = {}       # key -> np.ndarray (ids alignés X_train ; utile HeCBR)
        self.active_indexes = {}            # key -> {"linear", "kdtree", "lsh", "hecbr"?(D130)}
        self.base_size = {}                 # key -> int, évolue avec add_case()
        self.current_dataset = None
        self.current_case = None
        self.current_case_meta = {}         # source, encounter_id, label (affichage)
        self.scaler = {}                    # key -> scaler ajusté sur train (saisie manuelle heart)

demo_state = DemoState()
```

Chargement **paresseux** : vecteurs et index d'un dataset ne sont chargés qu'à sa première
sélection (la construction KD-Tree/LSH sur ChestX-ray14 prend ~50 s — à tester avant la
soutenance, pas en direct devant le public).

## Reset

Bouton « Réinitialiser » (visible dès qu'un dataset est sélectionné) : reconstruit les
index du dataset actif depuis `pristine_train_vectors` (déjà en mémoire), remet
`base_size` à sa valeur initiale, vide `current_case`. Indicateur de chargement pendant la
reconstruction (jusqu'à ~50 s sur Chest — prévenir dans l'interface).

## Flux détaillé et routes Flask

### Étape 1 — GET `/`
`index.html` : trois cartes de dimensionnalité (chiffres réels du config), un bouton par carte.

### Étape 1→2 — POST `/demo/select-dataset`
Corps : `dataset` (clé parmi DATASETS).
- Si jamais chargé : `dataset_bridge` charge le dataset via
  `src.datasets.adapters.get_split(key)` (split **stratifié 80/20, seed 42,
  TEST_FRACTION=0.2** — décision actée), construit les index (`LinearScanIndex`,
  `KDTreeIndex`, `RandomProjectionLSHIndex(lsh_params)` + `HeCBRIndex` pour D130 via
  `exp2/src/indexing/hecbr_index.py`, codes `data/processed/hecbr_diabetes130_codes_final.parquet`,
  `build(restrict_split="train")`), stocke dans `demo_state`.
- Si déjà chargé : réutilise l'état existant.
- Positionne `demo_state.current_dataset`, retourne `dataset_selected.html`
  (bouton saisie manuelle visible **seulement** si `allow_manual_entry`).

### Étape 2 — POST `/demo/generate-case`
Corps : `mode` (`auto` | `manual`).
- `auto` : index aléatoire dans `pristine_test_vectors[key]` (le jeu de TEST, jamais
  utilisé pour construire les index → cas réellement nouveau) ; stocke le cas + ses métadonnées.
- `manual` : retourne `manual_case_form.html` (un champ par feature de Heart Disease —
  colonnes réelles `heart_disease.FEATURE_COLUMNS` : age, sex, cp, trestbps, chol, fbs,
  restecg, thalach, exang, oldpeak, slope, ca, thal).

### Étape 2 (saisie manuelle) — POST `/demo/submit-manual-case`
Les 13 valeurs. Applique le **scaler ajusté sur le train** (`adapters.fit_transform_train(
"heart", train_df)` → `demo_state.scaler["heart"]`, pas un scaler ajusté sur le seul cas —
ce serait mathématiquement incorrect) pour standardiser la saisie. Stocke le vecteur dans
`demo_state.current_case`.

> Pas de modification de signature nécessaire : `adapters.fit_transform_train` expose déjà
> le scaler ajusté (contrairement à la version exp1 de la spec qui proposait de modifier
> `build_feature_matrix`).

### Étape 3 — fragment `case_ready.html`
Résumé du cas (13 valeurs pour Heart ; « cas #N du jeu de test » pour les autres). Bouton
« Lancer l'analyse ».

### Étape 3→4 — POST `/demo/analyze`
Logique dans `live_analysis.py` :

```python
def run_live_analysis(dataset_name, case_vector, case_meta):
    """Exécute les méthodes EN DIRECT : chronométrage time.perf_counter() (aucune valeur
    précalculée), décision de rétention (dist. 1-NN < seuil calibré), accord KD-Tree/LSH/
    HeCBR avec le balayage linéaire (référence déterministe) sur les k=5 voisins — même
    principe que neighbors_match() déjà validé dans exp2. Pas de métrique F1/AUC (pas de
    cible connue en live)."""
    indexes = demo_state.active_indexes[dataset_name]
    threshold = DATASETS[dataset_name]["retention_threshold"]
    methods = ["linear", "kdtree", "lsh"] + (["hecbr"] if dataset_name == "diabetes130" else [])

    linear_indices, linear_distances, linear_ms = time_query(indexes["linear"], case_vector, k=K)
    linear_redundant = linear_distances[0] < threshold

    for m in methods:
        indices, distances, ms = time_query(indexes[m], case_vector, k=K)
        if m == "linear":
            agreement = K
        else:
            # HeCBR : indices en espace codes → comparer par encounter_id ;
            # euclidien : positions base (ordonnancement commun).
            agreement = overlap(reference=linear_indices, other=indices, method=m)
        results[m] = {
            "query_time_ms": ms,
            "nearest_distance": float(distances[0]),
            "redundant": distances[0] < threshold,
            "agreement_with_reference": agreement,
            "decision_matches_reference": (distances[0] < threshold) == linear_redundant,
        }
        if m != "hecbr" and not results[m]["redundant"]:   # HeCBR : index figé (codes pré-entraînés)
            indexes[m].add_case(case_vector)
    if not linear_redundant:
        demo_state.base_size[dataset_name] += 1
    return results, demo_state.base_size[dataset_name]
```

Retourne `analysis_results.html` : cartes Baseline / KD-Tree / LSH (+ HeCBR sur D130),
chacune avec temps réel, décision de rétention (distance mesurée vs seuil), badge
« X/5 voisins identiques à la référence » et « décision identique » (vert) / « sur-rétention »
ou « sous-rétention » (orange/rouge avec explication courte), taille de base avant → après.
HeCBR : distance de Hamming (espace codes), pas de seuil euclidien — badge « index figé »
et décision de rétention non applicable, uniquement temps + accord des voisins.

### Panneau maintenance — GET `/demo/maintenance`
Charge `static/data/maintenance_snapshot.json` (CBM vs CNN, **précalculé** — condensation
complète de plusieurs minutes, pas une exécution en direct). Badge explicite
« Résultat précalculé — pas une exécution en direct ». Fichier généré une seule fois depuis
`exp2/results/<run_id>/condensation_quality.csv` (moyennes inter-seeds) — ne relance aucun
calcul de condensation. Valeurs exp2 :

| Dataset | CBM (moy.) | CNN (moy.) | Réduction |
|---|---|---|---|
| heart | 0.7681 | 0.7167 | 65.2 % |
| diabetes130 | 0.8208 | 0.7162 | 65.0 % |
| chestxray14 | 0.6275 | 0.5631 | 65.0 % |

## Lancement

```bash
cd demo
source ../venv/bin/activate   # réutilise le venv du projet principal, pas un nouveau
python app.py                 # http://127.0.0.1:5000
```

Le `README.md` de `demo/` documente cette commande + avertissement : la première sélection
de ChestX-ray14 prend ~50 s (construction KD-Tree + LSH) — à tester AVANT la présentation.

## Vérification avant de considérer la démo terminée

1. Les trois datasets se sélectionnent et se chargent sans erreur (lazy).
2. Un cas auto sur chacun des trois datasets produit une analyse complète (cartes
   Baseline/KD-Tree/LSH) sans crash.
3. Diabetes-130 affiche en plus la carte HeCBR (temps réel en espace de Hamming, accord des
   voisins par encounter_id) sans erreur.
4. La saisie manuelle Heart Disease fonctionne et produit un résultat cohérent avec le
   scaler du train (pas de valeurs aberrantes).
5. Le bouton Réinitialiser ramène `base_size` à sa valeur initiale et reconstruit les index
   (un cas ajouté avant reset n'est plus retrouvable après).
6. Le panneau maintenance affiche le badge « précalculé » et les chiffres ci-dessus.
