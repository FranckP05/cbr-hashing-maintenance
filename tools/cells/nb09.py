"""Cellules du notebook NB09 (dimensionnalité — T4.1 à T4.2)."""

CELLS = [
    ("md", """\
# NB09 — Dimensionnalité (T4.1-T4.2)

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Questions adressées :

- **QD1** (T4.1) : comment les paramètres LSH (bits, tables) doivent-ils croître avec
  la dimension pour maintenir un rappel fixe (≥ 0.99), à taille de base **fixée** ?
- **QD3** (T4.2) : la fraction de candidats du filtrage LSH est-elle réellement
  meilleure en haute dimension, ou confondue par la taille de base ? → mesurée à n fixé.

Protocole : base fixée (10k, ou tout le train si plus petit), requêtes du **test**
échantillonnées par seed 42 (60), grille bits × tables {8,16,32,64}, rappel des voisins
exacts (k=10) et fraction de candidats (compartiments / base) par cellule.

Sorties : `dim_calibration.csv` + `nb09_summary.json`. Notebook local (~1 h) ;
variables de réduction sanity : `NB09_QUERIES`, `NB09_BASE_SIZE`, `NB09_TABLES`.
"""),

    ("code", """\
import sys
from pathlib import Path

_candidates = [Path.cwd(), Path.cwd() / "exp2"]
if Path("/kaggle/input").exists():
    _candidates += sorted(Path("/kaggle/input").glob("*/"))
EXP2 = next((p for p in _candidates if (p / "src").exists() and (p / "config").exists()), None)
assert EXP2 is not None, "Dossier exp2 introuvable (local: cwd/exp2, Kaggle: /kaggle/input/*)"
sys.path.insert(0, str(EXP2))
print("EXP2 =", EXP2)
"""),

    ("code", """\
import os

from config import run as run_cfg
from config.datasets import ORDER

run_dir = run_cfg.get_run_dir()
print("Run courant :", run_dir)
print("Datasets  :", ORDER)
print("Réduction sanity : NB09_QUERIES=%s NB09_BASE_SIZE=%s NB09_TABLES=%s" % (
    os.environ.get("NB09_QUERIES", "(full)"),
    os.environ.get("NB09_BASE_SIZE", "(10000)"),
    os.environ.get("NB09_TABLES", "(8,16,32,64)"),
))
"""),

    ("md", """\
## T4.1 — Calibration bits × tables, rappel ≥ 0.99, base fixée

Pour chaque dataset, on balaie la grille bits × tables sur la base fixée et on mesure
le rappel des voisins exacts (k=10) ainsi que la fraction de candidats du filtrage.
La configuration retenue minimise les compartiments (2^bits × tables) sous la
contrainte rappel ≥ 0.99 (sinon : meilleur rappel atteignable).
"""),

    ("code", """\
import pandas as pd

from src.dimensionality import dim_harness as dh

records = []
for key in ORDER:
    print(f"\\n########## CALIBRATION {key} ##########")
    df = dh.calibrate_dataset(key, progress=print)
    records.append(df)

calib = pd.concat(records, ignore_index=True)
calib.to_csv(run_dir / "dim_calibration.csv", index=False)
print("\\nÉcrit :", run_dir / "dim_calibration.csv", f"({len(calib)} cellules)")
"""),

    ("md", """\
## T4.2 — Configuration retenue + fraction de candidats à n fixé

Lecture directe de `dim_calibration.csv` : pour chaque dataset, la configuration
retenue (rappel ≥ 0.99, compartiments minimaux) et la fraction de candidats qui en
résulte à taille de base fixée (tranche QD3).
"""),

    ("code", """\
import json
from datetime import datetime

summary = {
    "notebook": "NB09",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "protocol": {
        "base_size": int(calib["base_size"].max()),
        "n_queries": int(calib["query_ms"].size),
        "k": dh.K,
        "recall_target": dh.RECALL_TARGET,
        "tables_grid": [8, 16, 32, 64],
        "seed": 42,
    },
    "chosen": {},
}
for key in ORDER:
    sub = calib[calib["dataset"] == key]
    chosen = dh.choose_best(sub)
    summary["chosen"][key] = chosen
    print(f"=== {key} (dim={int(sub['dim'].iloc[0])}) ===")
    print(f"  retenu : bits={chosen['num_bits']} tables={chosen['num_tables']} "
          f"compartiments={chosen['compartments']:,} "
          f"rappel={chosen['mean_recall']:.3f} ({'>=0.99' if chosen['reached_target'] else 'MAX'})")
    print(f"  candidats : {chosen['mean_cand_fraction']:.1%} de la base")

(run_dir / "nb09_summary.json").write_text(json.dumps(summary, indent=2))
print("\\nÉcrit :", run_dir / "nb09_summary.json")
"""),

    ("md", """\
## Validation

- [x] T4.1 — calibration bits × tables pour rappel ≥ 0.99 à taille de base fixée,
      dimension croissante (13 / 390 / 1024)
- [x] T4.2 — fraction de candidats réelle par dataset à n fixé (tranche QD3)
- [x] Sorties : `dim_calibration.csv`, `nb09_summary.json`

**Étape suivante** : NB10 — figures dimensionnalité (FIG-R8, FIG-R9)."""),
]
