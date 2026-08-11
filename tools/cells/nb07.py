"""Cellules du notebook NB07 (maintenance — T3.1 à T3.4)."""

CELLS = [
    ("md", """\
# NB07 — Maintenance : rétention + condensation

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

- **T3.1** Rétention par collisions vs référence exacte : seuil de redondance
  θ = P25 des distances 1-NN du train ; flux de cas (test, par seed) ; accord cas par
  cas, taux de réduction, coût du flux, taille du bassin de candidats LSH.
- **T3.2** Condensation équitable : random / CBM / bucket (hachage) / CNN sur le **même
  sous-échantillon** du train, aux taux cibles **20 % / 50 %**.
- **T3.3** Métriques post-maintenance : qualité 1-NN + temps de requête (k=10) sur base
  condensée vs brute (même sous-échantillon).
- **T3.4** Temps de construction de chaque condensation.

**Protocole (a priori)** : θ = P25 des distances 1-NN sur 2000 cas du train ; rétention
en base croissante (add_case) ; sous-échantillon de condensation = premiers 10 000 cas du
train du seed ; bucket : num_bits a priori par dataset, 1 table, représentant = cas le
plus proche du centroïde ; CNN : taille naturelle (si > cible, premiers pivots de l'ordre
de condensation). Paramètres LSH = valeurs validées (`config.benchmark`).

Sorties : `retention_agreement.csv`, `condensation_quality.csv`, `nb07_summary.json`.
Variables de réduction locale : `NB07_SEEDS`, `NB07_QUERY_CAP`, `NB07_TRAIN_CAP`,
`NB07_SUBSAMPLE`.
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

from config import benchmark as bench_cfg
from config import datasets as ds_cfg
from config import maintenance as mnt_cfg
from config import run as run_cfg, seeds

run_dir = run_cfg.get_run_dir()
print("Run courant :", run_dir)
print()
print("Seuil rétention : P", mnt_cfg.RETENTION_NN_PERCENTILE,
      " sur", mnt_cfg.RETENTION_THRESHOLD_SAMPLE, "cas du train")
print("Sous-échantillon condensation :", mnt_cfg.COND_SUBSAMPLE)
print("Taux cibles :", mnt_cfg.COND_RATES, "| stratégies :", mnt_cfg.COND_STRATEGIES)
print("Bucket num_bits :", mnt_cfg.BUCKET_NUM_BITS_BY_DATASET)
print("Seeds :", seeds.ALL_SEEDS)
"""),

    ("md", """\
## T3.1 — Rétention par collisions vs référence exacte

Flux de cas du test (échantillonnés par seed) ; à chaque cas : voisin 1-NN LSH et exact
sur la base courante, redondant si distance < θ ; les cas retenus sont ajoutés à la base
(add_case). Mesures : accord cas par cas, taux de réduction, coût par cas, bassin de
candidats LSH (collisions).
"""),

    ("code", """\
from src.maintenance import benchmark_maintenance as mh

for key in ds_cfg.ORDER:
    print(f"\\n=== RÉTENTION {key} ===")
    mh.retention(key, run_dir, progress=print)

print("\\nÉcrit :", run_dir / "retention_agreement.csv")
pd.read_csv(run_dir / "retention_agreement.csv")
"""),

    ("md", """\
## T3.2 / T3.3 / T3.4 — Condensation (20 % / 50 %) + post-maintenance

Chaque stratégie condense le **même sous-échantillon** du train. Qualité 1-NN
(accuracy, macro-F1, balanced accuracy) et temps de requête (k=10, linear/kdtree/lsh)
sur la base condensée vs brute ; build mesuré pour chaque condensation.
"""),

    ("code", """\
for key in ds_cfg.ORDER:
    print(f"\\n=== CONDENSATION {key} ===")
    mh.condensation(key, run_dir, progress=print)

print("\\nÉcrit :", run_dir / "condensation_quality.csv")
pd.read_csv(run_dir / "condensation_quality.csv").head(10)
"""),

    ("code", """\
from datetime import datetime

ret = pd.read_csv(run_dir / "retention_agreement.csv")
con = pd.read_csv(run_dir / "condensation_quality.csv")
summary = {
    "notebook": "NB07",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "protocol": {
        "retention_percentile": mnt_cfg.RETENTION_NN_PERCENTILE,
        "retention_threshold_sample": mnt_cfg.RETENTION_THRESHOLD_SAMPLE,
        "cond_subample": mnt_cfg.COND_SUBSAMPLE,
        "cond_rates": mnt_cfg.COND_RATES,
        "cond_strategies": mnt_cfg.COND_STRATEGIES,
        "bucket_num_bits": mnt_cfg.BUCKET_NUM_BITS_BY_DATASET,
        "seeds": seeds.ALL_SEEDS,
        "lsh_params": "config.benchmark.LSH_PARAMS_BY_DATASET (validés exp1)",
    },
    "retention_rows": int(len(ret)),
    "condensation_rows": int(len(con)),
    "datasets_done": sorted(con["dataset"].unique().tolist()),
    "strategies_done": sorted(con["strategy"].unique().tolist()),
    "seeds_done": sorted(con["seed"].astype(str).unique().tolist()),
}
(run_dir / "nb07_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb07_summary.json")
print("Rétention :", len(ret), "lignes | Condensation :", len(con), "lignes")
print("Stratégies :", summary["strategies_done"], "| seeds :", summary["seeds_done"])
"""),

    ("md", """\
## Validation

- [x] T3.1 — rétention LSH vs exacte : accord cas par cas, taux de réduction, coût,
      bassin de candidats (base croissante par add_case)
- [x] T3.2 — condensation équitable (même sous-échantillon) : random / CBM / bucket / CNN
      aux taux 20 % / 50 %
- [x] T3.3 — qualité 1-NN + temps de requête (k=10) post-maintenance vs base brute
- [x] T3.4 — build mesuré pour chaque condensation
- [x] Sorties : `retention_agreement.csv`, `condensation_quality.csv`, `nb07_summary.json`

**Limites** : CNN condense à sa taille naturelle (si < cible, la réduction est plus forte
— taux réel rapporté) ; bucket dépend des num_bits a priori (assez de compartiments pour
atteindre 20 %) ; temps dépendants de la machine (comparaison entre méthodes uniquement).

**Étape suivante** : NB08 — figures maintenance (FIG-R5, FIG-R6, FIG-R7) depuis ces CSV.""")
]
