"""Démo interactive CBR + Hachage + Maintenance — point d'entrée Flask.

Réutilise exclusivement le code exp2 (src/, config/) via l'insertion de exp2/ au
chemin d'import. Mono-session (state.DemoState). Voir docs/demo.md.
"""

import os
import sys
from pathlib import Path

# La démo réutilise le code validé d'exp2 (src/, config/) — on ajoute exp2/ au chemin
# d'import, exactement comme le font les notebooks exp2.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP2 = PROJECT_ROOT / "exp2"
sys.path.insert(0, str(EXP2))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Les chemins de données d'exp2 (data/raw/..., data/processed/...) sont relatifs à la
# racine du projet : on s'y place quelle que soit la façon dont le serveur est lancé.
os.chdir(PROJECT_ROOT)

import numpy as np  # noqa: E402
from flask import Flask, abort, render_template, request  # noqa: E402

from demo_config import DATASETS, K, ORDER  # noqa: E402
from services import dataset_bridge, index_manager, live_analysis, maintenance_snapshot  # noqa: E402
from state import demo_state  # noqa: E402

app = Flask(__name__)


@app.template_filter("fmt_ms")
def fmt_ms(value):
    """Formatage lisible d'un temps de requête réel (µs en dessous de 1 ms)."""
    if value is None:
        return "—"
    if value < 1.0:
        return f"{value * 1000.0:.0f} µs"
    if value < 100.0:
        return f"{value:.2f} ms"
    return f"{value:.1f} ms"


def _benchmark_chart_data(results):
    """Données sérialisables du benchmark pour le module Chart.js statique."""
    if not results:
        return None

    definitions = (
        ("lsh", "LSH", "#6D28D9"),
        ("kdtree", "K-D Tree", "#FACC15"),
        ("hecbr", "HeCBR", "#4338CA"),
        ("linear", "Balayage linéaire", "#18181B"),
    )
    points = [
        (label, result["query_time_ms"], color)
        for key, label, color in definitions
        if (result := results.get(key))
        and not result.get("error")
        and result.get("query_time_ms") is not None
    ]
    if not points:
        return None

    labels, values, colors = zip(*points)
    return {"labels": labels, "values": values, "colors": colors}


def _base_ctx():
    """Contexte complet pour le flux continu de la démo (étapes 1 à 5)."""
    key = demo_state.current_dataset
    has_dataset = key in DATASETS
    has_case = demo_state.current_case is not None
    analyzed = demo_state.last_results is not None
    cfg = DATASETS[key] if has_dataset else None

    # Métriques de maintenance CBM globales pour la section 5
    maint_info = None
    cbm_applied = False
    if has_dataset:
        cbm_applied = demo_state.cbm_applied.get(key, False)
        snap_list = maintenance_snapshot.load_snapshot()
        snap_dict = {item["dataset"]: item for item in snap_list}
        snap = snap_dict.get(key, {"proportion_reduced": 0.65, "cbm_precision": 0.75})
        n_total = cfg["n_total"]
        pct_reduced = round(snap["proportion_reduced"] * 100.0, 1)
        n_reduced = int(round(n_total * (1.0 - snap["proportion_reduced"])))
        maint_info = {
            "n_total": n_total,
            "n_reduced": n_reduced,
            "pct_reduced": pct_reduced,
            "pct_kept": round(100.0 - pct_reduced, 1),
            "precision": round(snap["cbm_precision"] * 100.0, 1),
        }

    return {
        "key": key,
        "cfg": cfg,
        "has_dataset": has_dataset,
        "has_case": has_case,
        "analyzed": analyzed,
        "cbm_applied": cbm_applied,
        "maint_info": maint_info,
        "state": demo_state,
        "k": K,
        "results": demo_state.last_results if analyzed else None,
        "benchmark_chart": _benchmark_chart_data(demo_state.last_results) if analyzed else None,
        "before": demo_state.last_before,
        "after": demo_state.last_after,
        "errors": demo_state.manual_errors or {},
        "prev_values": demo_state.manual_values or {},
    }


def _dispatch():
    """Renvoie le flux continu accumulé (étapes 2, 3, 4 visibles simultanément)."""
    if demo_state.current_dataset in DATASETS:
        return render_template("fragments/flow.html", **_base_ctx())
    return ("", 204)


def _reset_analysis_state():
    """Revenir avant l'étape 4 (nouveau cas, reset) invalide les résultats affichés."""
    demo_state.last_results = None
    demo_state.last_before = None
    demo_state.last_after = None


@app.get("/")
def index():
    return render_template("index.html", datasets=DATASETS, order=ORDER)


@app.post("/demo/select-dataset")
def select_dataset():
    key = request.form.get("dataset")
    if key not in DATASETS:
        abort(400)
    try:
        if key not in demo_state.active_indexes:
            if key not in demo_state.pristine_train_vectors:
                dataset_bridge.load_dataset(key)
            demo_state.active_indexes[key] = index_manager.build_dataset_indexes(
                key, demo_state.pristine_train_vectors[key]
            )
    except ImportError as exc:
        return render_template(
            "fragments/load_error.html",
            dataset=DATASETS[key],
            dependency_error=str(exc),
        )
    demo_state.current_dataset = key
    demo_state.current_case = None
    demo_state.current_case_meta = {}
    demo_state.manual_errors = {}
    demo_state.manual_values = {}
    demo_state.manual_mode = False
    _reset_analysis_state()
    return _dispatch()


@app.post("/demo/generate-case")
def generate_case():
    key = demo_state.current_dataset
    if key not in DATASETS:
        abort(400)
    mode = request.form.get("mode", "auto")
    _reset_analysis_state()

    if mode == "manual":
        if not DATASETS[key]["allow_manual_entry"]:
            abort(400)
        demo_state.manual_mode = True
        demo_state.current_case = None
        return _dispatch()

    # auto : cas réellement nouveau tiré du jeu de TEST
    demo_state.manual_mode = False
    n_test = len(demo_state.pristine_test_vectors[key])
    idx = int(np.random.default_rng().integers(0, n_test))
    demo_state.current_case = demo_state.pristine_test_vectors[key][idx]
    meta = {"source": "auto", "case_number": idx}
    if key in demo_state.pristine_test_targets:
        meta["target"] = int(demo_state.pristine_test_targets[key][idx])
    if key in demo_state.test_encounter_ids:
        meta["encounter_id"] = int(demo_state.test_encounter_ids[key][idx])
    
    # Extraire les valeurs lisibles des caractéristiques du cas généré
    try:
        meta["values"] = dataset_bridge.get_case_feature_values(key, idx)
    except Exception:
        meta["values"] = {}

    demo_state.current_case_meta = meta
    demo_state.manual_errors = {}
    demo_state.manual_values = {}
    return _dispatch()


@app.post("/demo/submit-manual-case")
def submit_manual_case():
    key = demo_state.current_dataset
    if key not in DATASETS or not DATASETS[key]["allow_manual_entry"]:
        abort(400)
    cols = DATASETS[key]["feature_columns"]
    ranges = demo_state.feature_ranges.get(key, {})
    raw = {}
    errors = {}
    for c in cols:
        try:
            raw[c] = float(request.form[c])
        except (KeyError, ValueError):
            errors[c] = "valeur numérique requise"
            continue
        lo, hi = ranges[c]
        if raw[c] < lo - 1e-6 or raw[c] > hi + 1e-6:
            errors[c] = f"hors de l'intervalle du train [{lo:g}–{hi:g}]"
    if errors:
        demo_state.manual_errors = errors
        demo_state.manual_values = raw
        demo_state.manual_mode = True
        return _dispatch(), 422

    demo_state.current_case = dataset_bridge.manual_case_vector(key, raw, cols)
    demo_state.current_case_meta = {
        "source": "manual",
        "values": {c: round(raw[c], 4) for c in cols},
    }
    demo_state.manual_errors = {}
    demo_state.manual_values = {}
    demo_state.manual_mode = False
    _reset_analysis_state()
    return _dispatch()


@app.post("/demo/analyze")
def analyze():
    key = demo_state.current_dataset
    if key not in DATASETS or demo_state.current_case is None:
        abort(400)  # étape 3 verrouillée tant qu'aucun cas n'est prêt
    results, before, after = live_analysis.run_live_analysis(
        key, demo_state.current_case, demo_state.current_case_meta
    )
    demo_state.last_results = results
    demo_state.last_before = before
    demo_state.last_after = after
    return _dispatch()


@app.post("/demo/reset")
def reset():
    key = demo_state.current_dataset
    if key not in DATASETS:
        abort(400)
    index_manager.reset_dataset(key)
    demo_state.manual_errors = {}
    demo_state.manual_values = {}
    demo_state.manual_mode = False
    _reset_analysis_state()
    return _dispatch()


@app.post("/demo/apply-maintenance")
def apply_maintenance():
    key = demo_state.current_dataset
    if key not in DATASETS:
        abort(400)
    demo_state.cbm_applied[key] = True
    return _dispatch()


@app.post("/demo/reset-maintenance")
def reset_maintenance():
    key = demo_state.current_dataset
    if key not in DATASETS:
        abort(400)
    demo_state.cbm_applied[key] = False
    return _dispatch()


@app.get("/demo/maintenance")
def maintenance():
    """Compatibilité : la maintenance est désormais incluse dans le flux courant."""
    return _dispatch()


if __name__ == "__main__":
    app.run(debug=False)
