"""Cellules du notebook NB08 (figures maintenance — T3.5)."""

CELLS = [
    ("md", """\
# NB08 — Figures maintenance

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Figures de résultats lues depuis `retention_agreement.csv` et `condensation_quality.csv`
(aucune valeur en dur) :

- **T3.5a** → **FIG-R5** : accord rétention LSH vs exacte (cas par cas) + taux de réduction.
- **T3.5b** → **FIG-R6** : qualité post-maintenance (macro-F1) brut vs condensé, par
  stratégie, à taux de réduction égal (0.2 / 0.5).
- **T3.5c** → **FIG-R7** : temps de requête post-maintenance (brut vs condensé).

Agrégation : moyenne sur 6 seeds ; barres d'erreur = écart type inter-seeds. Notebook
local (< 10 min), lecture seule des CSV du run courant.
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
ret_csv = run_dir / "retention_agreement.csv"
cond_csv = run_dir / "condensation_quality.csv"
figdir = run_dir / "figures"
figdir.mkdir(parents=True, exist_ok=True)
print("Run courant :", run_dir)
for csv in (ret_csv, cond_csv):
    assert csv.exists(), f"CSV manquant : {csv}"
    print("Lecture :", csv)
"""),

    ("md", """\
## T3.5 — Figures post-maintenance

FIG-R5 : l'accord cas par cas entre la rétention LSH et la référence exacte (même seuil)
et le taux de réduction qui en découle. FIG-R6 / FIG-R7 : qualité (macro-F1) et temps de
requête du scan linéaire sur la base condensée, à taux de réduction égal (0.2 / 0.5),
comparés à la base brute (`raw`).
"""),

    ("code", """\
import pandas as pd

from exp2.src.visualization import fig_maintenance as fm

ret = fm.aggregate_retention(pd.read_csv(ret_csv))
cond = fm.aggregate_condensation(pd.read_csv(cond_csv))
fm.fig_r5(ret, figdir / "FIG-R5_accord_retention.png")
fm.fig_r6(cond, figdir / "FIG-R6_qualite_post_maintenance.png")
fm.fig_r7(cond, figdir / "FIG-R7_temps_post_maintenance.png")
print("Écrit :")
for name in ["FIG-R5_accord_retention.png", "FIG-R6_qualite_post_maintenance.png",
             "FIG-R7_temps_post_maintenance.png"]:
    print("  ", figdir / name)
"""),

    ("code", """\
try:
    from IPython.display import Image, display

    for name in ["FIG-R5_accord_retention.png", "FIG-R6_qualite_post_maintenance.png",
                 "FIG-R7_temps_post_maintenance.png"]:
        display(Image(filename=str(figdir / name)))
except ImportError:
    print("IPython indisponible (exécution hors notebook) — affichage ignoré.")
"""),

    ("code", """\
from datetime import datetime
import json

figs = sorted(p.name for p in figdir.glob("FIG-*.png"))
summary = {
    "notebook": "NB08",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "tasks": {
        "T3.5a": "FIG-R5_accord_retention.png",
        "T3.5b": "FIG-R6_qualite_post_maintenance.png",
        "T3.5c": "FIG-R7_temps_post_maintenance.png",
    },
    "figures": figs,
}
(run_dir / "nb08_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb08_summary.json")
print("Figures du run :", figs)
"""),

    ("md", """\
## Validation

- [x] T3.5a — FIG-R5 : accord rétention LSH vs exacte + taux de réduction
- [x] T3.5b — FIG-R6 : qualité post-maintenance (macro-F1) brut vs condensé
- [x] T3.5c — FIG-R7 : temps de requête post-maintenance (brut vs condensé)
- [x] Aucune valeur en dur (lecture de `retention_agreement.csv` + `condensation_quality.csv`)

**Étape suivante** : NB09 — dimensionnalité (T4.1-T4.2), puis NB10 figures (FIG-R8, FIG-R9)."""),
]
