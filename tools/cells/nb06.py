"""Cellules du notebook NB06 (figures scaling — T2.3 à T2.4)."""

CELLS = [
    ("md", """\
# NB06 — Figures scaling

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Figures de résultats lues depuis `scaling_timing_quality.csv` (aucune valeur en dur) :

- **T2.3** → **FIG-R3** : temps de requête vs taille de base (10k / 50k / full), échelle
  log-log, par méthode.
- **T2.4** → **FIG-R4** : temps de construction d'index vs taille de base.

Panneaux Diabetes-130 / ChestX-ray14 (Heart exclu : n=237, pas de scaling). Agrégation :
moyenne sur 6 seeds et k ; barres d'erreur = écart type inter-seeds. Notebook local
(< 10 min), lecture seule des CSV du run courant.
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
csv = run_dir / "scaling_timing_quality.csv"
figdir = run_dir / "figures"
figdir.mkdir(parents=True, exist_ok=True)
print("Run courant :", run_dir)
assert csv.exists(), f"CSV manquant : {csv}"
print("Lecture :", csv)
"""),

    ("md", """\
## T2.3 / T2.4 — Temps de requête et de construction vs taille de base (FIG-R3, FIG-R4)

Courbes log-log sur les bases 10k / 50k / full (sous-échantillons du train, mêmes seeds).
"""),

    ("code", """\
import pandas as pd

from exp2.src.visualization import fig_scaling as fs

per_seed = fs.aggregate(pd.read_csv(csv))
fs.build_figure(
    per_seed, "query_ms", "Temps de requête (ms)",
    "FIGURE FIG-R3 — Temps de requête vs taille de base, échelle log-log (moyenne sur 6 seeds).",
    figdir / "FIG-R3_scaling_temps_requete.png",
)
fs.build_figure(
    per_seed, "build_ms", "Temps de construction (s)",
    "FIGURE FIG-R4 — Temps de construction d'index vs taille de base, échelle log-log.",
    figdir / "FIG-R4_scaling_temps_construction.png",
)
print("Écrit :")
print("  ", figdir / "FIG-R3_scaling_temps_requete.png")
print("  ", figdir / "FIG-R4_scaling_temps_construction.png")
"""),

    ("code", """\
try:
    from IPython.display import Image, display

    for name in ["FIG-R3_scaling_temps_requete.png", "FIG-R4_scaling_temps_construction.png"]:
        display(Image(filename=str(figdir / name)))
except ImportError:
    print("IPython indisponible (exécution hors notebook) — affichage ignoré.")
"""),

    ("code", """\
from datetime import datetime
import json

figs = sorted(p.name for p in figdir.glob("FIG-*.png"))
summary = {
    "notebook": "NB06",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "tasks": {"T2.3": "FIG-R3_scaling_temps_requete.png", "T2.4": "FIG-R4_scaling_temps_construction.png"},
    "figures": figs,
}
(run_dir / "nb06_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb06_summary.json")
print("Figures du run :", figs)
"""),

    ("md", """\
## Validation

- [x] T2.3 — FIG-R3 : temps de requête vs taille de base (log-log, 10k/50k/full)
- [x] T2.4 — FIG-R4 : temps de construction d'index vs taille de base
- [x] Aucune valeur en dur (lecture unique de `scaling_timing_quality.csv`)

**Étape suivante** : NB07 (maintenance, déjà exécuté) → NB08 figures maintenance
(FIG-R5, FIG-R6, FIG-R7)."""),
]
