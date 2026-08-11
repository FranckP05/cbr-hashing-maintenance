"""Validate indexing methods (LinearScan, KD-Tree, LSH, HeCBR) on a given dataset.

Compares exact neighbor recall and basic retrieval correctness.
"""

import argparse
import importlib
import time

import numpy as np
from tqdm import tqdm

from src.indexing.linear_scan import LinearScanIndex
from src.indexing.kdtree_index import KDTreeIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex
from src.indexing.hecbr_index import HeCBRIndex
from src.evaluation.timing import time_index_build


DATASET_REGISTRY = {
    "heart_disease": ("src.datasets.heart_disease", "load_heart_disease", "build_feature_matrix"),
    "diabetes130":   ("src.datasets.diabetes130",   "load_diabetes130",   "build_feature_matrix"),
    "chestxray14":   ("src.datasets.chestxray14",   "load_chestxray14",   "build_feature_matrix"),
}

DEFAULT_DATA_PATHS = {
    "heart_disease": "data/raw/heart_disease/processed.cleveland.data",
}

LSH_PARAMS_BY_DATASET = {
    "heart_disease": {"num_bits": 5, "num_tables": 16},
    "diabetes130":   {"num_bits": 12, "num_tables": 32},
    "chestxray14":   {"num_bits": 16, "num_tables": 32},
}

# Codes binaires HeCBR pré-calculés (64 bits par cas). Seul Diabetes-130 dispose d'un modèle
# HeCBR entraîné ; Heart Disease n'a jamais été entraîné et ChestX-ray14 n'a que les embeddings
# (entraînement HeCBR non fait). L'ordre des lignes des codes est identique à celui du loader
# du dataset (vérifié par encounter_id) : le code d'une requête s'obtient par index de ligne.
HECBR_CODES_PATHS = {
    "diabetes130": "data/processed/hecbr_diabetes130_codes_final.parquet",
}


def neighbors_match(linear_indices, linear_distances, kdtree_indices, kdtree_distances,
                     tol=1e-5):
    """
    Compare deux résultats de recherche de k plus proches voisins en tolérant les égalités
    de distance (ties) : deux méthodes peuvent légitimement choisir un ordre différent parmi
    des candidats à distance strictement identique, sans que ce soit une erreur
    d'implémentation. On vérifie donc que les DISTANCES renvoyées correspondent (à une
    tolérance flottante près), pas que les indices sont dans le même ordre.
    """
    if not np.allclose(sorted(linear_distances), sorted(kdtree_distances), atol=tol):
        return False, "distances_differ"

    linear_set = set(linear_indices)
    kdtree_set = set(kdtree_indices)
    if linear_set == kdtree_set:
        return True, "exact_match"

    only_in_linear = linear_set - kdtree_set
    only_in_kdtree = kdtree_set - linear_set
    boundary_distance = linear_distances[-1]
    tie_at_boundary = all(
        abs(linear_distances[list(linear_indices).index(i)] - boundary_distance) < tol
        for i in only_in_linear
    )
    return tie_at_boundary, "tie_at_boundary" if tie_at_boundary else "real_mismatch"


def main():
    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Validate indexing methods on any registered dataset.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        default="heart_disease",
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of test queries (for quick validation on large datasets)")
    args = parser.parse_args()

    module_path, load_name, build_name = DATASET_REGISTRY[args.dataset]
    module = importlib.import_module(module_path)
    load_func = getattr(module, load_name)
    build_func = getattr(module, build_name)

    data_path = args.data_path or DEFAULT_DATA_PATHS.get(args.dataset)
    dataframe = load_func(data_path) if data_path is not None else load_func()

    print(f"=== Indexing Validation on {args.dataset} ===\n")

    # 1. Load and vectorize
    print(f"Data shape after cleaning: {dataframe.shape}")
    target_counts = dataframe["target"].value_counts().to_dict()
    print(f"Target distribution: {target_counts}")

    result = build_func(dataframe)
    vectors, targets = result[0], result[1]
    print(f"Feature matrix shape: {vectors.shape}")

    # 2. Linear scan baseline: neighbors for case 0 in the full dataset, k=5
    linear_all = LinearScanIndex()
    linear_all.build(vectors)
    indices_0, dists_0 = linear_all.query(vectors[0], k=5)
    print(f"\nLinearScan neighbors for case 0, k=5: indices {indices_0.tolist()}")

    # Use the first 237 cases as train (80%), remaining 60 as test (20%)
    # Deterministic split matching the original seed=42 convention.
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(vectors))
    split = int(0.8 * len(vectors))
    train_vectors = vectors[shuffled[:split]]
    test_vectors = vectors[shuffled[split:]]
    train_targets = targets[shuffled[:split]]
    test_targets = targets[shuffled[split:]]

    # Optional: random subset of test queries (seed 42), avoiding positional bias
    test_orig_rows = shuffled[split:]
    sample_indices = None
    if args.max_queries is not None and args.max_queries < len(test_vectors):
        sample_rng = np.random.default_rng(42)
        sample_indices = sample_rng.choice(len(test_vectors), size=args.max_queries, replace=False)
        full_test_count = len(test_vectors)
        test_vectors = test_vectors[sample_indices]
        test_targets = test_targets[sample_indices]
        test_orig_rows = test_orig_rows[sample_indices]
        print(f"Using random sample of {len(test_vectors)} / {full_test_count} test queries (seed=42)")

    # 3. Build indexes (avec mesure des temps de construction)
    linear, linear_build = time_index_build(LinearScanIndex, train_vectors)

    kdtree, kdtree_build = time_index_build(KDTreeIndex, train_vectors)

    lsh_params = LSH_PARAMS_BY_DATASET[args.dataset]
    lsh, lsh_build = time_index_build(
        lambda: RandomProjectionLSHIndex(**lsh_params, random_seed=42), train_vectors
    )

    # HeCBR : charge les codes binaires pré-calculés (build() ignore train_vectors).
    # L'ordre des lignes des codes == celui de `vectors` (vérifié par encounter_id).
    hecbr_codes_path = HECBR_CODES_PATHS.get(args.dataset)
    hecbr, hecbr_build = (time_index_build(lambda: HeCBRIndex(hecbr_codes_path), train_vectors)
                          if hecbr_codes_path is not None else (None, 0.0))
    print(f"\nTemps de construction — LinearScan: {linear_build * 1000:.1f} ms | "
          f"KD-Tree: {kdtree_build:.2f} s | LSH {lsh_params}: {lsh_build:.2f} s"
          + (f" | HeCBR: {hecbr_build:.2f} s" if hecbr is not None else " | HeCBR: non disponible"))

    # 4-5. Boucle principale unique : le balayage linéaire sert de référence UNE seule fois
    # par requête — le même résultat alimente la comparaison KD-tree ET le rappel LSH.
    k = 5
    exact_count = 0
    tie_count = 0
    mismatch_count = 0
    lsh_recalls = []
    correct = 0
    hecbr_correct = 0
    hecbr_answered = 0
    n_queries = len(test_vectors)
    for i, (vec, orig_row) in enumerate(tqdm(zip(test_vectors, test_orig_rows),
                                             total=n_queries,
                                             desc="Validation KD-Tree + LSH + HeCBR",
                                             unit="q")):
        lin_idx, lin_dists = linear.query(vec, k=k)

        kd_idx, kd_dists = kdtree.query(vec, k=k)
        ok, reason = neighbors_match(lin_idx, lin_dists, kd_idx, kd_dists)
        if reason == "exact_match":
            exact_count += 1
        elif reason == "tie_at_boundary":
            tie_count += 1
        else:
            if mismatch_count == 0:
                tqdm.write(f"  Real mismatch at query {i}: Linear {lin_idx} vs KDTree {kd_idx}")
            mismatch_count += 1

        lsh_idx, lsh_dists = lsh.query(vec, k=k)
        overlap = len(np.intersect1d(lin_idx, lsh_idx))
        lsh_recalls.append(overlap / k)

        if len(lsh_idx) > 0:
            weights = 1.0 / np.maximum(lsh_dists, np.finfo(float).eps)
            pred = float(np.average(train_targets[lsh_idx], weights=weights) >= 0.5)
            correct += (pred == test_targets[i])

        # HeCBR : recherche par distance de Hamming dans l'espace des codes binaires. Le
        # rappel des voisins exacts n'a pas de sens ici (HeCBR n'approxime pas la même
        # distance) ; on ne mesure que la précision de classification. Le code de la requête
        # est cherché dans le parquet par index de ligne (aligné à `vectors`). Seuls les
        # voisins de l'ensemble train votent (cohérent avec les autres méthodes, qui
        # n'indexent que la base d'entraînement).
        if hecbr is not None:
            query_code = hecbr.codes[orig_row]
            h_idx, h_d = hecbr.query(query_code, k=k)
            train_mask = hecbr.split[h_idx] == "train"
            if np.any(train_mask):
                h_idx_t, h_d_t = h_idx[train_mask], h_d[train_mask]
                weights = 1.0 / np.maximum(h_d_t, np.finfo(float).eps)
                pred = float(np.average(hecbr.labels[h_idx_t], weights=weights) >= 0.5)
                hecbr_correct += (pred == test_targets[i])
                hecbr_answered += 1
        if (i + 1) % 500 == 0:
            hecbr_prog = (f" | acc HeCBR cumulée={hecbr_correct / max(hecbr_answered, 1):.4f}"
                          if hecbr is not None else "")
            tqdm.write(
                f"  [étape {i + 1}/{n_queries}] exact={exact_count} tie={tie_count} "
                f"mismatch={mismatch_count} | rappel cumulé={np.mean(lsh_recalls):.4f} "
                f"| acc cumulée={correct / (i + 1):.4f}{hecbr_prog} | "
                f"{time.perf_counter() - t_start:.0f}s écoulés"
            )
    kd_total = n_queries
    print(f"KD-Tree vs LinearScan: {exact_count} exact, {tie_count} tie, "
          f"{mismatch_count} mismatch / {kd_total} queries")

    mean_recall = np.mean(lsh_recalls)
    n_q_lsh = len(lsh_recalls)
    acc = correct / n_q_lsh
    print(f"LSH {lsh_params}: mean exact-neighbor recall = {mean_recall:.4f} "
          f"({n_q_lsh} queries) | classification accuracy = {acc:.4f}")

    if hecbr is not None:
        print(f"HeCBR (64 bits, distance de Hamming) : classification accuracy = "
              f"{hecbr_correct / hecbr_answered:.4f} ({hecbr_answered} requêtes, voisins train "
              f"uniquement — le rappel des voisins exacts n'a pas de sens pour HeCBR)")

    t_elapsed = time.perf_counter() - t_start
    print(f"\nTotal execution time: {t_elapsed:.1f}s")
    print("=== Validation complete ===")


if __name__ == "__main__":
    main()
