"""Cellules du notebook NB04 (figures benchmark — T1.7 à T1.8)."""

CELLS = [
    ("md", """\
# NB04 — Figures benchmark

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Figures de résultats lues depuis `benchmark_timing_quality.csv` (aucune valeur en dur) :

- **T1.7** → **FIG-R1** : qualité de récupération (rappel des voisins exacts, nDCG@k,
  macro-F1) par méthode/dataset.
- **T1.8** → **FIG-R2** : temps de requête par méthode, échelle log, protocole unique.

Agrégation : moyenne sur les seeds et k ∈ {1, 5, 10} ; barres d'erreur = écart type
inter-seeds. Notebook local (< 10 min), lecture seule des CSV du run courant.
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
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
print("EXP2 =", EXP2)
"""),

    ("code", """\
from config import run as run_cfg

run_dir = run_cfg.get_run_dir()
csv = run_dir / "benchmark_timing_quality.csv"
figdir = run_dir / "figures"
figdir.mkdir(parents=True, exist_ok=True)
print("Run courant :", run_dir)
assert csv.exists(), f"CSV manquant : {csv}"
print("Lecture :", csv)
"""),

    ("md", """\
## T1.7 — Qualité de récupération (FIG-R1)

Rappel des voisins exacts, nDCG@k et macro-F1 par méthode et dataset (moyenne 6 seeds,
k = 1, 5, 10 ; HeCBR figé, 1 seed).
"""),

    ("code", """\
import pandas as pd

from exp2.src.visualization import fig_benchmark_quality as fq

per_seed = fq.aggregate(pd.read_csv(csv))
fq.fig_r1(per_seed, figdir / "FIG-R1_qualite_recuperation.png")
fq.fig_r2(per_seed, figdir / "FIG-R2_temps_requete.png")
print("Écrit :")
print("  ", figdir / "FIG-R1_qualite_recuperation.png")
print("  ", figdir / "FIG-R2_temps_requete.png")
"""),

    ("code", """\
try:
    from IPython.display import Image, display

    for name in ["FIG-R1_qualite_recuperation.png", "FIG-R2_temps_requete.png"]:
        display(Image(filename=str(figdir / name)))
except ImportError:
    print("IPython indisponible (exécution hors notebook) — affichage ignoré.")
"""),

    ("code", """\
from datetime import datetime
import json

figs = sorted(p.name for p in figdir.glob("FIG-*.png"))
summary = {
    "notebook": "NB04",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "tasks": {"T1.7": "FIG-R1_qualite_recuperation.png", "T1.8": "FIG-R2_temps_requete.png"},
    "figures": figs,
}
(run_dir / "nb04_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb04_summary.json")
print("Figures du run :", figs)
"""),

    ("md", """\
## Validation

- [x] T1.7 — FIG-R1 : rappel voisins exacts, nDCG@k, macro-F1 par méthode/dataset
- [x] T1.8 — FIG-R2 : temps de requête par méthode (échelle log, protocole unique)
- [x] Aucune valeur en dur (lecture unique de `benchmark_timing_quality.csv`)

**Étape suivante** : NB05 (scaling, déjà exécuté) → NB06 figures scaling (FIG-R3, FIG-R4)."""),
]
