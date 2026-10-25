"""Analyse live d'un cas — exécution réelle des index exp2, aucun résultat précalculé.

Chronométrage via time.perf_counter() autour de query(k=5). Décision de rétention par
comparaison de la distance au plus proche voisin au seuil calibré (config). Accord des
voisins KD-Tree/LSH/HeCBR avec la référence déterministe (balayage linéaire).
"""

import time

from demo_config import DATASETS, K


def time_query(index, vector, k=K):
    """Une seule requête réelle, chronométrée (perf_counter)."""
    start = time.perf_counter()
    indices, distances = index.query(vector, k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return indices, distances, elapsed_ms


def _agreement_euclidean(indices, linear_indices):
    return len(set(indices.tolist()) & set(linear_indices.tolist()))


def _agreement_hecbr(hecbr_index, indices, linear_indices, train_encounter_ids):
    """Accord par encounter_id (les espaces d'index diffèrent : codes vs vecteurs)."""
    hecbr_ids = set(hecbr_index.encounter_ids[indices].tolist())
    linear_ids = set(train_encounter_ids[linear_indices].tolist())
    return len(hecbr_ids & linear_ids)


def _hecbr_query(dataset_name, case_meta):
    """Code binaire 64 bits du cas pour la requête HeCBR (espace de Hamming)."""
    from services import dataset_bridge
    from state import demo_state

    encounter_id = case_meta.get("encounter_id")
    if encounter_id is None:
        return None
    return dataset_bridge.hecbr_code_for_encounter(demo_state, encounter_id)


def run_live_analysis(dataset_name, case_vector, case_meta):
    """Exécute les méthodes en direct ; retourne (results, base_size_before, base_size_after)."""
    from state import demo_state

    indexes = demo_state.active_indexes[dataset_name]
    threshold = DATASETS[dataset_name]["retention_threshold"]
    base_before = demo_state.base_size[dataset_name]

    methods = ["linear", "kdtree", "lsh"]
    if dataset_name == "diabetes130" and "hecbr" in indexes:
        methods.append("hecbr")

    linear_indices, linear_distances, linear_ms = time_query(indexes["linear"], case_vector)
    linear_redundant = len(linear_distances) > 0 and linear_distances[0] < threshold

    results = {}

    for m in methods:
        if m == "hecbr":
            code = _hecbr_query(dataset_name, case_meta)
            if code is None:
                results[m] = {
                    "error": "code HeCBR indisponible pour ce cas",
                    "query_time_ms": None,
                    "nearest_distance": None,
                    "redundant": None,
                    "agreement_with_reference": None,
                    "decision_matches_reference": None,
                }
                continue
            indices, distances, elapsed_ms = time_query(indexes["hecbr"], code)
            agreement = _agreement_hecbr(
                indexes["hecbr"], indices, linear_indices,
                demo_state.train_encounter_ids[dataset_name],
            )
            results[m] = {
                "query_time_ms": round(elapsed_ms, 3),
                "nearest_distance": float(distances[0]) if len(distances) else None,
                "redundant": None,  # espace de Hamming : seuil euclidien non applicable
                "agreement_with_reference": agreement,
                "decision_matches_reference": None,
                "index_frozen": True,  # codes pré-entraînés, pas d'add_case
            }
            continue

        indices, distances, elapsed_ms = time_query(indexes[m], case_vector)
        if len(distances) == 0:  # LSH sans aucun candidat
            redundant = False
            agreement = 0 if m != "linear" else K
        else:
            redundant = bool(distances[0] < threshold)
            agreement = _agreement_euclidean(indices, linear_indices) if m != "linear" else K

        results[m] = {
            "query_time_ms": round(elapsed_ms, 3),
            "nearest_distance": float(distances[0]) if len(distances) else None,
            "redundant": redundant,
            "agreement_with_reference": agreement,
            "decision_matches_reference": redundant == linear_redundant,
        }

        if not redundant:
            indexes[m].add_case(case_vector)

    if not linear_redundant:
        demo_state.base_size[dataset_name] += 1
    base_after = demo_state.base_size[dataset_name]

    results["_reference"] = {
        "query_time_ms": round(linear_ms, 3),
        "redundant": linear_redundant,
    }
    results["_threshold"] = threshold
    results["_after_added"] = not linear_redundant

    # Verdict honnête : le plus rapide PARMI TOUTES les méthodes (linear inclus) qui ont
    # répondu. Sur petite base, le balayage linéaire gagne souvent (surcoût LSH non amorti) —
    # le template le formule en conséquence.
    candidates = [
        m for m in ("linear", "kdtree", "lsh", "hecbr")
        if m in results and not results[m].get("error")
        and results[m]["query_time_ms"] is not None
    ]
    results["_fastest"] = min(candidates, key=lambda m: results[m]["query_time_ms"]) \
        if candidates else None

    return results, base_before, base_after
