"""Cellules du notebook NB05 (scaling — T2.1 à T2.2)."""

CELLS = [
    ("md", """\
# NB05 — Scaling : temps & qualité vs taille de base

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

- **T2.1** Bases 10k / 50k / max(train) par sous-échantillonnage déterministe du train
  (premiers n cas du split du seed).
- **T2.2** Temps de requête + construction vs taille de base, par méthode
  (linear / kdtree / lsh), qualité sur k=10.

**Protocole (a priori)** : requêtes = 60 (heart) / 500 (diabetes130) / 300 (chestxray14)
échantillonnées par seed ; build mesuré 1× par (dataset, taille) ; warm-up=1, répétitions=3
(T0.5). Heart (237 cas) ne permet pas de scaling : un seul point à taille pleine (référence).
HeCBR exclu (codes fixes). Paramètres LSH = valeurs validées (`config.benchmark`), pas de
re-calibration ici (NB09 pour rappel ≥ 0.99).

Sorties : `scaling_timing_quality.csv`, `nb05_summary.json`. Variables de réduction locale :
`NB05_SEEDS`, `NB05_QUERY_CAP`.
"""),

    ("code", """\
import sys
from pathlib import Path

_candidates = [Path.cwd() / "exp2"]
if Path("/kaggle/input").exists():
    _candidates += sorted(Path("/kaggle/input").glob("*/"))
EXP2 = next((p for p in _candidates if (p / "src").exists() and (p / "config").exists()), None)
assert EXP2 is not None, "Dossier exp2 introuvable (local: cwd/exp2, Kaggle: /kaggle/input/*)"
sys.path.insert(0, str(EXP2))
print("EXP2 =", EXP2)
"""),

    ("code", """\
import json

import numpy as np
import pandas as pd

from config import datasets as ds_cfg
from config import run as run_cfg
from config import scaling as scaling_cfg
from config import seeds as seed_cfg

run_dir = run_cfg.get_run_dir()
print("Run courant :", run_dir)
print()
print("Tailles de base :", scaling_cfg.BASE_SIZES_BY_DATASET)
print("k :", scaling_cfg.SCALING_K)
print("Méthodes :", scaling_cfg.SCALING_METHODS)
print("Seeds :", seed_cfg.ALL_SEEDS)
"""),

    ("md", """\
## T2.1 / T2.2 — Acquisition temps + qualité vs taille de base

Pour chaque dataset × seed × taille de base : construction (mesurée 1× par
(dataset, taille)), requêtes, puis métriques de reclassement + rappel des voisins
exacts + classification. Checkpoint CSV incrémental (reprise après interruption).
"""),

    ("code", """\
from src.scaling import scaling_harness as sh

for key in ds_cfg.ORDER:
    print(f"\\n########## SCALING {key} ##########")
    sh.benchmark_scaling(key, run_dir, progress=print)

print("\\nScaling terminé.")
print("Aperçu :", run_dir / "scaling_timing_quality.csv")
pd.read_csv(run_dir / "scaling_timing_quality.csv").head(8)
"""),

    ("code", """\
from datetime import datetime

res = pd.read_csv(run_dir / "scaling_timing_quality.csv")
summary = {
    "notebook": "NB05",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "protocol": {
        "base_sizes_by_dataset": scaling_cfg.BASE_SIZES_BY_DATASET,
        "k": scaling_cfg.SCALING_K,
        "methods": scaling_cfg.SCALING_METHODS,
        "seeds": seed_cfg.ALL_SEEDS,
        "lsh_params": "config.benchmark.LSH_PARAMS_BY_DATASET (validés exp1)",
    },
    "result_rows": int(len(res)),
    "datasets_done": sorted(res["dataset"].unique().tolist()),
    "methods_done": sorted(res["method"].unique().tolist()),
    "seeds_done": sorted(res["seed"].astype(str).unique().tolist()),
}
(run_dir / "nb05_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb05_summary.json")
print("Lignes de résultats :", len(res))
print("Méthodes :", summary["methods_done"], "| seeds :", summary["seeds_done"])
"""),

    ("md", """\
## Validation

- [x] T2.1 — bases 10k / 50k / max(train), sous-échantillonnage déterministe du train
- [x] T2.2 — temps de requête + construction vs taille de base, par méthode (k=10)
- [x] Checkpoint CSV incrémental (reprise après interruption, pas de doublon)
- [x] Sorties : `scaling_timing_quality.csv`, `nb05_summary.json`

**Limites** : heart (237 cas) sans scaling (point de référence unique) ; HeCBR exclu
(codes fixes) ; temps dépendants de la machine (comparaison entre méthodes uniquement).

**Étape suivante** : NB06 — figures scaling (FIG-R3, FIG-R4) depuis ces CSV.""")
]
