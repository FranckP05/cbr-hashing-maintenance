"""Cellules du notebook NB12 (synthèse & vainqueur — T6.1 à T6.3)."""

CELLS = [
    ("md", """\
# NB12 — Synthèse & vainqueur (T6.1-T6.3)

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Clôture de l'expérience :

- **T6.1** → tableau récapitulatif multi-datasets (`synthesis_recap.csv`) : méthodes ×
  datasets (qualité, rappel voisins exacts, temps requête/construction) + résultats de
  maintenance (macro-F1 et temps post-maintenance, meilleure stratégie par dataset).
- **T6.2** → inventaire des figures du run (FIG-R1..R11), lues depuis les CSV.
- **T6.3** → score §5 (0.4·qualité + 0.3·rapidité + 0.2... cf. `winner_score.py`) :
  vainqueur global (score moyen sur datasets applicables) et score par dataset.

Notebook local (< 10 min), lecture seule des CSV du run courant.
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
bench_csv = run_dir / "benchmark_timing_quality.csv"
cond_csv = run_dir / "condensation_quality.csv"
figdir = run_dir / "figures"
print("Run courant :", run_dir)
for p in (bench_csv, cond_csv):
    assert p.exists(), f"CSV manquant : {p}"
    print("Lecture :", p)
"""),

    ("md", """\
## T6.1 — Tableau récapitulatif multi-datasets

Moyennes sur les seeds et k ∈ {1,5,10} par (dataset, méthode). Lignes maintenance :
meilleure stratégie de condensation par dataset (taux 0.5), macro-F1 et temps du scan
linéaire sur la base maintenue (base-level, commun aux 3 méthodes).
"""),

    ("code", """\
import numpy as np
import pandas as pd

from config.datasets import ORDER

bench = pd.read_csv(bench_csv)
bench["seed"] = bench["seed"].astype(str)
per = bench.groupby(["dataset", "method", "seed"]).agg(
    macro_f1=("macro_f1", "mean"), exact_recall_at_k=("exact_recall_at_k", "mean"),
    query_ms=("query_ms", "mean"), build_ms=("build_ms", "mean"),
).reset_index()

rows = []
for ds in ORDER:
    for _, r in per[per["dataset"] == ds].groupby("method").agg(
        macro_f1=("macro_f1", "mean"), exact_recall_at_k=("exact_recall_at_k", "mean"),
        query_ms=("query_ms", "mean"), build_ms=("build_ms", "mean"),
    ).iterrows():
        rows.append({
            "dataset": ds, "method": r.name, "kind": "index_seul",
            "macro_f1": round(r["macro_f1"], 4), "exact_recall_at_k": round(r["exact_recall_at_k"], 4),
            "query_ms": round(r["query_ms"], 4), "build_ms": round(r["build_ms"], 4),
            "note": "",
        })

cond = pd.read_csv(cond_csv)
cond["seed"] = cond["seed"].astype(str)
for ds in ORDER:
    sub = cond[(cond["dataset"] == ds) & (cond["rate"] == 0.5) & (cond["strategy"] != "raw")]
    strat_mean = sub.groupby("strategy").agg(
        macro_f1=("nn_macro_f1", "mean"), query_ms=("query_ms_linear", "mean"),
    )
    best_strategy = strat_mean["macro_f1"].idxmax()
    best = strat_mean.loc[best_strategy]
    rows.append({
        "dataset": ds, "method": f"maint.{best_strategy}", "kind": "maintenance",
        "macro_f1": round(float(best["macro_f1"]), 4), "exact_recall_at_k": np.nan,
        "query_ms": round(float(best["query_ms"]), 4), "build_ms": np.nan,
        "note": f"taux 0.5 (moyenne 6 seeds)",
    })

recap = pd.DataFrame(rows)
recap.to_csv(run_dir / "synthesis_recap.csv", index=False)
print("Écrit :", run_dir / "synthesis_recap.csv")
print()
print(recap[["dataset", "method", "kind", "macro_f1", "exact_recall_at_k", "query_ms"]].to_string(index=False))
"""),

    ("md", """\
## T6.2 — Inventaire des figures du run

Toutes les figures du mémoire sont produites par les notebooks figures (NB04, NB06,
NB08, NB10, NB11) et lues depuis les CSV : aucune valeur en dur.
"""),

    ("code", """\
figs = sorted(p.name for p in figdir.glob("FIG-*.png"))
print(f"{len(figs)} figures dans {figdir} :")
for name in figs:
    print("  -", name)
"""),

    ("md", """\
## T6.3 — Vainqueur par le score §5 (pré-déclaré)

Score = 0.4·qualité + 0.3·rapidité (log-normalisée) + 0.15·robustesse (variance
inter-seeds) + 0.15·fidélité (rappel voisins exacts), normalisé par dataset.
HeCBR exclu du classement (1 seed, 1 dataset) ; scores par dataset rapportés.
"""),

    ("code", """\
import json
from datetime import datetime

from src.evaluation import winner_score as ws

per_seed = ws.per_seed_agg(bench)
scores = {ds: ws.score_dataset(per_seed, ds) for ds in ORDER}
winner, means = ws.select_winner(scores)
winner_on_chest = max(scores["chestxray14"], key=lambda m: scores["chestxray14"][m]["score"])

for ds in ORDER:
    print(f"=== {ds} ===")
    for m, info in scores[ds].items():
        print(f"  {m:7s} score={info['score']:.3f} "
              f"[qualité={info['quality']:.2f} rapidité={info['speed']:.2f} "
              f"robustesse={info['robustness']:.2f} fidélité={info['fidelity']:.2f}]")
print()
print("Score moyen (datasets applicables) :", {m: round(v, 3) for m, v in means.items()})
print("VAINQUEUR global :", winner)
print("VAINQUEUR sur ChestX-ray14 (démo NB13) :", winner_on_chest)
"""),

    ("code", """\
summary = {
    "notebook": "NB12",
    "run_dir": str(run_dir),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "criteria_weights": ws.WEIGHTS,
    "scores_by_dataset": scores,
    "mean_scores": means,
    "winner_global": winner,
    "winner_on_chest": winner_on_chest,
}
(run_dir / "nb12_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("Écrit :", run_dir / "nb12_summary.json")
"""),

    ("md", """\
## Validation

- [x] T6.1 — `synthesis_recap.csv` : tableau multi-datasets (index seuls + maintenance)
- [x] T6.2 — inventaire des figures FIG-R1..R11
- [x] T6.3 — score §5 pré-déclaré : vainqueur global + vainqueur sur Chest (démo)

**Étape suivante** : NB13 — démo CBR (T6.4) sur ChestX-ray14 avec la méthode gagnante
sur Chest.""")
]
