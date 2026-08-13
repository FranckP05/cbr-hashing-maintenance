"""Cellules du notebook NB11 (statistique — T5.1 à T5.2)."""

CELLS = [
    ("md", """\
# NB11 — Statistique (T5.1-T5.2)

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Tests statistiques appariés entre méthodes, par dataset et par k, avec les 6 seeds comme
échantillons appariés :

- **T5.1** — Wilcoxon signed-rank + t-test apparié (moyennes par seed) sur le temps de
  requête et la qualité de `benchmark_timing_quality.csv` → `significance_tests.csv`.
- **T5.2** → **FIG-R10** : matrice de significativité (Wilcoxon, k agrégé).

Notebook local (< 10 min). La lecture unique est `benchmark_timing_quality.csv`
(aucune valeur en dur) ; `significance_tests.csv` est régénéré s'il manque.
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
out_csv = run_dir / "significance_tests.csv"
figdir = run_dir / "figures"
figdir.mkdir(parents=True, exist_ok=True)
print("Run courant :", run_dir)
assert csv.exists(), f"CSV manquant : {csv}"
print("Lecture :", csv)
"""),

    ("md", """\
## T5.1 — Tests appariés (Wilcoxon + t-test)

Moyennes par seed et par k ∈ {1, 5, 10} + agrégat (`agg`) ; comparaison deux à deux des
méthodes (Linear, KD-Tree, LSH) sur les métriques de temps et de qualité.
"""),

    ("code", """\
import pandas as pd

from exp2.src.evaluation import significance as sig

frame = sig.per_seed_means(pd.read_csv(csv))
tests = sig.test_pairs(frame)
out_csv.write_text("")
tests.to_csv(out_csv, index=False)
print("Écrit :", out_csv, f"({len(tests)} lignes)")
print()
print(tests.groupby(["dataset", "metric"]).size().unstack(fill_value=0))
"""),

    ("md", """\
## T5.2 — Matrice de significativité (FIG-R10)

Une matrice méthode × méthode par (dataset, métrique), Wilcoxon sur le k agrégé :
orange = p < 0.05, gris = non significatif. HeCBR exclu (1 seed).
"""),

    ("code", """\
from exp2.src.visualization import fig_significance as fg

rows = fg.load(out_csv)
fg.fig_r10(rows, figdir / "FIG-R10_matrice_significativite.png")
print("Écrit :", figdir / "FIG-R10_matrice_significativite.png")
"""),

    ("code", """\
try:
    from IPython.display import Image, display

    display(Image(filename=str(figdir / "FIG-R10_matrice_significativite.png")))
except ImportError:
    print("IPython indisponible (exécution hors notebook) — affichage ignoré.")
"""),

    ("code", """\
from datetime import datetime
import json

nsig = len(tests)
sig_pairs = tests[(tests["k"] == "agg") & (tests["wilcoxon_p"] < 0.05)]
summary = {
    "notebook": "NB11",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "tasks": {"T5.1": "significance_tests.csv", "T5.2": "FIG-R10_matrice_significativite.png"},
    "test_rows": int(nsig),
    "sig_pairs_agg_005": int(len(sig_pairs)),
    "method_pairs": ["linear-vs-kdtree", "linear-vs-lsh", "kdtree-vs-lsh"],
}
(run_dir / "nb11_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb11_summary.json")
print("Lignes de tests :", nsig, "| paires significatives (k agg, p<0.05) :", len(sig_pairs))
"""),

    ("md", """\
## Validation

- [x] T5.1 — Wilcoxon signed-rank + t-test apparié (6 seeds) par dataset/k/métrique
- [x] T5.2 — FIG-R10 : matrice de significativité (orange p<0.05, gris ns, HeCBR exclu)
- [x] Aucune valeur en dur (lecture unique de `benchmark_timing_quality.csv`)

**Étape suivante** : NB12 — synthèse & vainqueur (T6.1-T6.3)."""),
]
