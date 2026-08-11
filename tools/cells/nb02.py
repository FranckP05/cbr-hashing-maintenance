"""Cellules du notebook NB02 (métriques & protocole de mesure — T0.4 à T0.6)."""

CELLS = [
    ("md", """\
# NB02 — Métriques & protocole de mesure

**Expérience 2 (reproductible)** — référence : `docs/plan_experimentation_reproductible.md`.

Ce notebook valide, sur données connues et sur un petit dataset (heart, 297×13) :

1. **T0.4 — métriques** : accuracy, macro-F1, balanced accuracy, AUC (classification) et
   Precision@k, Recall@k, MAP@k, nDCG@k, Hit@k, MRR (rangement) — vérifiées par recoupement
   avec `sklearn` et sur des cas construits à valeurs connues ;
2. **T0.5 — timing** : protocole unique (warm-up, médiane sur répétitions, `perf_counter`) ;
3. **T0.6 — mémoire** : pic RSS de la construction d'index + pic d'allocations par requête
   (`tracemalloc`).

Sorties : `metrics_sanity.csv`, `timing_sanity.csv`, `memory_sanity.csv` et `nb02_summary.json`
dans le run courant. Les notebooks lourds (NB03+) réutilisent ces bibliothèques.
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

from config import run as run_cfg, seeds

run_dir = run_cfg.get_run_dir()
print("Run courant :", run_dir)
"""),

    ("md", """\
## T0.4 — Vérification des métriques

### Cas construits (valeurs connues à la main)
"""),

    ("code", """\
from src.evaluation import classification_metrics as cm, retrieval_metrics as rm

checks = {}

# — Classification : cas construit ———————————
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
checks["acc_manual"] = cm.accuracy(y_true, y_pred) == 0.8
checks["macro_f1_manual"] = np.isclose(cm.macro_f1(y_true, y_pred), 0.8)
checks["balanced_acc_manual"] = np.isclose(cm.balanced_accuracy(y_true, y_pred), 0.8)

# — Classification : cas construit AUC (monotone) ———
y_score = np.array([0.9, 0.1, 0.7, 0.8, 0.2, 0.6, 0.3, 0.4, 0.95, 0.05])
auc_manual = cm.auc(y_true, y_score)
# AUC attendue : courbe ROC parfaite sécable — vérifiée contre sklearn plus bas.
checks["auc_finite"] = 0.0 <= auc_manual <= 1.0

# — distance_weighted_prediction : cas connus ———
pred, score = cm.distance_weighted_prediction(
    np.array([1.0, 1.0, 1.0]), np.array([1, 0, 1])
)
checks["dwp_flat"] = (pred == 1.0) and np.isclose(score, 2.0 / 3.0)
pred2, score2 = cm.distance_weighted_prediction(
    np.array([1.0, 100.0]), np.array([1, 0])
)
checks["dwp_weighted"] = (pred2 == 1.0) and np.isclose(score2, 1.0 / 1.01)

# — Rangement : cas construit ———
# train_targets : sept positifs (idx 0,2,3,5,7,8,9) ; retrieved = [2,5,1,8] à k=4
train_targets = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
metrics = rm.ranking_metrics(np.array([2, 5, 1, 8]), train_targets, query_target=1, k=4)
dcg_expected = 1.0 + 1.0 / np.log2(3) + 0.0 + 1.0 / np.log2(5)
idcg_expected = 1.0 + 1.0 / np.log2(3) + 0.5 + 1.0 / np.log2(5)
expected = {
    "precision_at_k": 0.75,
    "label_recall_at_k": 3.0 / 7.0,
    "map_at_k": (1.0 + 1.0 + 0.75) / 4.0,
    "ndcg_at_k": dcg_expected / idcg_expected,
    "hit_at_k": 1.0,
    "mrr_at_k": 1.0,
}
for key, val in expected.items():
    checks[f"ranking_{key}"] = np.isclose(metrics[key], val)

print("Checks construits :", sum(checks.values()), "/", len(checks))
for key, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {key}")
assert all(checks.values()), "Échec d'un check construit"
"""),

    ("code", """\
import sklearn.metrics as skm

# — Recoupement sklearn : classification ———
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
y_score = np.array([0.9, 0.1, 0.7, 0.8, 0.2, 0.6, 0.3, 0.4, 0.95, 0.05])

sk_cross = {
    "acc": (cm.accuracy(y_true, y_pred), skm.accuracy_score(y_true, y_pred)),
    "macro_f1": (cm.macro_f1(y_true, y_pred), skm.f1_score(y_true, y_pred, average="macro")),
    "balanced_acc": (cm.balanced_accuracy(y_true, y_pred),
                     skm.balanced_accuracy_score(y_true, y_pred)),
    "auc": (cm.auc(y_true, y_score), skm.roc_auc_score(y_true, y_score)),
}
for key, (mine, ref) in sk_cross.items():
    ok = np.isclose(mine, ref, atol=1e-9)
    print(f"  [{'OK' if ok else 'FAIL'}] {key:12s} mine={mine:.10f}  sklearn={ref:.10f}")
    assert ok, key

# — Recoupement : Balanced accuracy = moyenne des rappels (sklearn) ———
print("\\nRecoupement sklearn : OK")
"""),

    ("code", """\
from src.datasets import adapters
from src.indexing.linear_scan import LinearScanIndex

X_train, y_train, X_test, y_test, _, _ = adapters.get_split("heart")
print(f"heart : X_train={X_train.shape}  X_test={X_test.shape}  "
      f"pos_ratio_train={np.mean(y_train):.4f}")

index = LinearScanIndex()
index.build(X_train)

# Auto-vérification : requête sur soi-même (k=1) => rang 1, distance ≈ 0
# (précision float32 : résidu ~1e-3, seuil 1e-2)
self_index = LinearScanIndex()
self_index.build(X_test)
n_self, self_ok = 0, True
for q in X_test[:50]:
    idx_q, dist_q = self_index.query(q, 1)
    self_ok = self_ok and (dist_q[0] < 1e-2) and (y_test[idx_q[0]] == y_test[n_self])
    n_self += 1
print(f"Auto-match sur {n_self} requêtes : {'OK' if self_ok else 'FAIL'}")
assert self_ok

# Agrégats moyens des métriques de rangement sur des requêtes de test
rows = []
for k in seeds.K_VALUES:
    agg = {}
    for q_idx in range(50):
        idx_q, _ = index.query(X_test[q_idx], k)
        for key, val in rm.ranking_metrics(
            idx_q, y_train, query_target=int(y_test[q_idx]), k=k
        ).items():
            agg[key] = agg.get(key, 0.0) + val
    agg = {key: val / 50 for key, val in agg.items()}
    rows.append({"k": k, **agg})
    print(f"k={k:2d}  " + "  ".join(f"{key}={val:.4f}" for key, val in agg.items()))

ranking_sanity = pd.DataFrame(rows)
ranking_sanity.to_csv(run_dir / "metrics_sanity.csv", index=False)
print("\\nÉcrit :", run_dir / "metrics_sanity.csv")
"""),

    ("md", """\
## T0.5 — Protocole de timing (warm-up, médiane, répétitions)
"""),

    ("code", """\
from src.evaluation import timing
from src.indexing.linear_scan import LinearScanIndex

factory = lambda: LinearScanIndex()

index, med_build, times_build = timing.time_index_build(factory, X_train)
print(f"build    : médiane={med_build*1e3:.3f} ms  ({len(times_build)} répétitions)  "
      f"min={min(times_build)*1e3:.3f}  max={max(times_build)*1e3:.3f}")
assert len(times_build) == timing.DEFAULT_REPETITIONS
assert min(times_build) <= med_build <= max(times_build)

med_query, per_query = timing.time_queries(index, X_test[:50], k=5)
print(f"requêtes : médiane_agrégée={med_query*1e6:.1f} µs/requête  "
      f"(médiane des médianes sur {len(per_query)} requêtes)")
assert len(per_query) == 50

timing_rows = [{
    "operation": "build",
    "repetitions": len(times_build),
    "median_ms": round(med_build * 1e3, 4),
    "min_ms": round(min(times_build) * 1e3, 4),
    "max_ms": round(max(times_build) * 1e3, 4),
}, {
    "operation": "query_k5",
    "repetitions": len(per_query),
    "median_ms": round(med_query * 1e3, 4),
    "min_ms": round(min(per_query) * 1e3, 4),
    "max_ms": round(max(per_query) * 1e3, 4),
}]
pd.DataFrame(timing_rows).to_csv(run_dir / "timing_sanity.csv", index=False)
print("\\nÉcrit :", run_dir / "timing_sanity.csv")
"""),

    ("md", """\
## T0.6 — Mémoire (pic RSS construction + allocations par requête)
"""),

    ("code", """\
from src.evaluation import memory

index, mem_idx = memory.measure_index_memory(factory, X_train)
print(f"index build : pic RSS +{mem_idx['index_peak_rss_kb']:.0f} Ko")

_, _, mem_q = memory.measure_query_memory(index, X_test[0], k=5)
print(f"requête     : tracemalloc pic={mem_q['query_tracemalloc_peak_kb']:.0f} Ko, "
      f"RSS delta={mem_q['query_rss_delta_kb']:.0f} Ko")

pd.DataFrame([{"operation": "index_build", "peak_rss_kb": mem_idx["index_peak_rss_kb"]},
              {"operation": "query_k5", "tracemalloc_peak_kb": mem_q["query_tracemalloc_peak_kb"],
               "rss_delta_kb": mem_q["query_rss_delta_kb"]}])\
    .to_csv(run_dir / "memory_sanity.csv", index=False)
print("\\nÉcrit :", run_dir / "memory_sanity.csv")
"""),

    ("code", """\
from datetime import datetime

summary = {
    "notebook": "NB02",
    "run_dir": str(run_dir),
    "validated_at": datetime.now().isoformat(timespec="seconds"),
    "classification": {key: bool(ok) for key, ok in checks.items()},
    "sklearn_cross_check": {key: bool(ok) for key, ok in sk_cross.items()},
    "retrieval_self_match": bool(self_ok),
    "rank_sanity_samples": int(n_self),
    "timing": {r["operation"]: r["median_ms"] for r in timing_rows},
    "memory": {
        "index_peak_rss_kb": mem_idx["index_peak_rss_kb"],
        "query_tracemalloc_peak_kb": mem_q["query_tracemalloc_peak_kb"],
    },
}
(run_dir / "nb02_summary.json").write_text(json.dumps(summary, indent=2))
print("Écrit :", run_dir / "nb02_summary.json")
print("\\nProtocoles T0.4/T0.5/T0.6 validés — NB03/05/07/09 peuvent démarrer.")
"""),

    ("md", """\
## Validation

- [x] T0.4 — métriques de classification exactes (cas construits + recoupement sklearn)
- [x] T0.4 — métriques de rangement exactes (cas construit à valeurs connues)
- [x] T0.4 — rangement end-to-end sur heart (auto-match k=1 + agrégats k=1/5/10)
- [x] T0.5 — protocole de timing : warm-up + médiane sur répétitions + `perf_counter`
- [x] T0.6 — mémoire : pic RSS index + pic tracemalloc par requête
- [x] Sorties versionnées : `metrics_sanity.csv`, `timing_sanity.csv`,
      `memory_sanity.csv`, `nb02_summary.json`

**Étape suivante** : NB03 — baseline LSH & facteur d'approximation (puis NB05 k-d-tree,
NB07 condensation, NB09 multi-index, exécutables en parallèle sur Kaggle)."""),
]
