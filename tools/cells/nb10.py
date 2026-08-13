"""Cellules du notebook NB10 (figures dimensionnalité — T4.3 à T4.4)."""

CELLS = [
    ("md", """\
# NB10 — Figures dimensionnalité (T4.3-T4.4)

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Figures de résultats lues depuis `dim_calibration.csv` + `nb09_summary.json`
(aucune valeur en dur) :

- **T4.3** → **FIG-R8** : compartiments (2^bits × tables) de la configuration retenue
  (rappel ≥ 0.99) vs dimensionnalité.
- **T4.4** → **FIG-R9** : fraction de candidats vs dimensionnalité à taille de base fixée.

3 points (13 / 390 / 1024) : le message est le sens de croissance des paramètres avec
la dimension. Notebook local (< 10 min), lecture seule des sorties de NB09.
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
csv = run_dir / "dim_calibration.csv"
summary_json = run_dir / "nb09_summary.json"
figdir = run_dir / "figures"
figdir.mkdir(parents=True, exist_ok=True)
print("Run courant :", run_dir)
for p in (csv, summary_json):
    assert p.exists(), f"fichier manquant : {p}"
    print("Lecture :", p)
"""),

    ("md", """\
## T4.3 / T4.4 — Compartiments et candidats vs dimensionnalité

Les configurations retenues par NB09 (rappel ≥ 0.99 à compartiments minimaux) sont
lues depuis `nb09_summary.json` ; les compartiments et fractions de candidats
correspondants sont pris dans `dim_calibration.csv`.
"""),

    ("code", """\
from exp2.src.visualization import fig_dimensionality as fd

data = fd.load(csv, summary_json)
fd.fig_r8(data, figdir / "FIG-R8_compartiments_dimension.png")
fd.fig_r9(data, figdir / "FIG-R9_candidats_dimension.png")
print("Écrit :")
print("  ", figdir / "FIG-R8_compartiments_dimension.png")
print("  ", figdir / "FIG-R9_candidats_dimension.png")
"""),

    ("code", """\
try:
    from IPython.display import Image, display

    for name in ["FIG-R8_compartiments_dimension.png", "FIG-R9_candidats_dimension.png"]:
        display(Image(filename=str(figdir / name)))
except ImportError:
    print("IPython indisponible (exécution hors notebook) — affichage ignoré.")
"""),

    ("code", """\
from datetime import datetime
import json

figs = sorted(p.name for p in figdir.glob("FIG-*.png"))
summary = {
    "notebook": "NB10",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "tasks": {
        "T4.3": "FIG-R8_compartiments_dimension.png",
        "T4.4": "FIG-R9_candidats_dimension.png",
    },
    "figures": figs,
}
(run_dir / "nb10_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb10_summary.json")
print("Figures du run :", figs)
"""),

    ("md", """\
## Validation

- [x] T4.3 — FIG-R8 : compartiments (bits × tables) vs dimensionnalité
- [x] T4.4 — FIG-R9 : fraction de candidats vs dimensionnalité à n fixé
- [x] Aucune valeur en dur (lecture de `dim_calibration.csv` + `nb09_summary.json`)

**Étape suivante** : NB11 (statistique, exécuté) → NB12 — synthèse & vainqueur
(T6.1-T6.3)."""),
]
