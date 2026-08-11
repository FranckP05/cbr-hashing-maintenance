"""Evaluate retention filter and condensation (CNN vs CBM) on a given dataset.

Replicates the full pipeline from data_loading.py __main__:
  - Load and vectorize
  - Split base / incoming stream
  - Calibrate redundancy threshold
  - LSH retention simulation vs exact reference
  - Condensation (CNN) over multiple seeds
  - CBM comparison at equal condensed size

Grands datasets (Diabetes-130, ChestX-ray14) : protocole adapté (CNN sur sous-échantillon 10%,
CBM à proportion équivalente sur la base complète), voir run_condensation_large_dataset().
"""

import argparse
import importlib
import time

import numpy as np

from src.indexing.linear_scan import LinearScanIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex
from src.maintenance.retention_lsh import (
    compute_baseline_nn_distance_threshold,
    simulate_case_stream,
)
from src.maintenance.condensation_cnn import (
    condense_dataset,
    evaluate_1nn_accuracy,
    evaluate_condensation_over_seeds,
)
from src.maintenance.condensation_cbm import (
    compute_relative_coverage,
    condense_by_competence,
    evaluate_cbm_over_seeds,
)


DATASET_REGISTRY = {
    "heart_disease": ("src.datasets.heart_disease", "load_heart_disease", "build_feature_matrix"),
    "diabetes130":   ("src.datasets.diabetes130",   "load_diabetes130",   "build_feature_matrix"),
    "chestxray14":   ("src.datasets.chestxray14",   "load_chestxray14",   "build_feature_matrix"),
}

DEFAULT_DATA_PATHS = {
    "heart_disease": "data/raw/heart_disease/processed.cleveland.data",
}

SEEDS = [42, 7, 123, 2024, 99]

LSH_PARAMS_BY_DATASET = {
    "heart_disease": {"num_bits": 5, "num_tables": 16},
    "diabetes130":   {"num_bits": 12, "num_tables": 32},
    "chestxray14":   {"num_bits": 16, "num_tables": 32},
}


def run_condensation_large_dataset(vectors, targets, dataset_name):
    """Protocole CNN vs CBM adapté aux grands datasets (Diabetes-130, ChestX-ray14).

    CNN est exécuté sur un sous-échantillon aléatoire de 10 % du train car son
    algorithme (Hart, 1968) est intrinsèquement itératif et infaisable en O(n²) sur la base
    complète. CBM, vectorisé (NearestNeighbors), peut traiter la base complète.

    La comparaison se fait donc par PROPORTION conservée plutôt que par taille absolue :
    cbm_target_size = round(proportion_kept * n_train) garantit que CBM et CNN retiennent une
    fraction équivalente de la base, seule comparaison équitable possible ici.
    """
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(vectors))
    split = int(0.8 * len(vectors))
    train_vectors, train_targets = vectors[shuffled[:split]], targets[shuffled[:split]]
    test_vectors, test_targets = vectors[shuffled[split:]], targets[shuffled[split:]]
    n_train, n_test = len(train_vectors), len(test_vectors)
    sample_size = int(0.1 * n_train)
    print(f"Train / Test : {n_train} / {n_test} | échantillon CNN : {sample_size} cas")

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        timings = {}

        t0 = time.perf_counter()
        sample_rng = np.random.default_rng(seed)
        sample_indices = sample_rng.choice(n_train, size=sample_size, replace=False)
        cnn_sample_vectors = train_vectors[sample_indices]
        cnn_sample_targets = train_targets[sample_indices]
        timings["sample_10pct"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cnn_condensed_indices = condense_dataset(
            cnn_sample_vectors, cnn_sample_targets, random_seed=seed
        )
        timings["cnn_condense"] = time.perf_counter() - t0
        cnn_condensed_size = len(cnn_condensed_indices)
        proportion_kept = cnn_condensed_size / sample_size

        t0 = time.perf_counter()
        solve_threshold = compute_baseline_nn_distance_threshold(train_vectors, percentile=50)
        timings["cbm_threshold"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        relative_coverage = compute_relative_coverage(
            train_vectors, train_targets, solve_threshold
        )
        timings["cbm_coverage"] = time.perf_counter() - t0

        # adaptation nécessaire : comparaison par PROPORTION (voir docstring du protocole)
        cbm_target_size = round(proportion_kept * n_train)
        t0 = time.perf_counter()
        cbm_indices = condense_by_competence(
            train_vectors, train_targets, cbm_target_size, solve_threshold
        )
        timings["cbm_select"] = time.perf_counter() - t0
        cbm_vectors = train_vectors[cbm_indices]
        cbm_targets_c = train_targets[cbm_indices]

        t0 = time.perf_counter()
        cnn_precision = evaluate_1nn_accuracy(
            cnn_sample_vectors[cnn_condensed_indices],
            cnn_sample_targets[cnn_condensed_indices],
            test_vectors,
            test_targets,
        )
        timings["eval_cnn"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cbm_precision = evaluate_1nn_accuracy(
            cbm_vectors, cbm_targets_c, test_vectors, test_targets
        )
        timings["eval_cbm"] = time.perf_counter() - t0

        print(f"CNN condensé (sur {sample_size}) : {cnn_condensed_size} cas "
              f"| proportion conservée : {proportion_kept:.4f}")
        print(f"CBM cible (sur {n_train}) : {cbm_target_size} cas "
              f"| précision CNN : {cnn_precision:.4f} | précision CBM : {cbm_precision:.4f}")
        for step, elapsed in timings.items():
            print(f"  {step}: {elapsed:.1f}s")
        print(f"  total seed : {sum(timings.values()):.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retention filter and condensation on any registered dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        default="heart_disease",
    )
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    module_path, load_name, build_name = DATASET_REGISTRY[args.dataset]
    module = importlib.import_module(module_path)
    load_func = getattr(module, load_name)
    build_func = getattr(module, build_name)

    data_path = args.data_path or DEFAULT_DATA_PATHS.get(args.dataset)
    dataframe = load_func(data_path) if data_path is not None else load_func()

    print(f"Étape 1 : chargement des données ({args.dataset})")
    print(dataframe.shape)

    print("\nÉtape 2 : vectorisation")
    result = build_func(dataframe)
    vectors, targets = result[0], result[1]
    print("Forme de la matrice de vecteurs :", vectors.shape)

    print("\nÉtape 3 : découpage base initiale / flux entrant (80% / 20%)")
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(len(vectors))
    split_point = int(0.8 * len(vectors))
    base_vectors = vectors[shuffled_indices[:split_point]]
    incoming_vectors = vectors[shuffled_indices[split_point:]]
    print("Taille de la base initiale :", len(base_vectors))
    print("Taille du flux entrant :", len(incoming_vectors))

    print("\nÉtape 4 : calibration du seuil de redondance")
    distance_threshold = compute_baseline_nn_distance_threshold(base_vectors)
    print("Seuil de distance retenu (25e percentile) :", distance_threshold)

    lsh_params = LSH_PARAMS_BY_DATASET[args.dataset]

    print("\nÉtape 5 : simulation avec LSH (approximatif) — insertion incrémentale")
    lsh_final_base, lsh_retained, lsh_redundant, lsh_decisions, lsh_elapsed = simulate_case_stream(
        base_vectors, incoming_vectors, distance_threshold,
        index_factory=lambda: RandomProjectionLSHIndex(**lsh_params),
        desc=f"Simulation LSH ({args.dataset})",
    )
    print(f"Cas retenus : {lsh_retained} | Cas écartés : {lsh_redundant} "
          f"| Temps : {lsh_elapsed:.1f}s")

    print("\nÉtape 6 : simulation avec balayage linéaire (référence exacte)")
    exact_final_base, exact_retained, exact_redundant, exact_decisions, exact_elapsed = simulate_case_stream(
        base_vectors, incoming_vectors, distance_threshold,
        index_factory=lambda: LinearScanIndex(),
        desc=f"Simulation exacte ({args.dataset})",
    )
    print(f"Cas retenus : {exact_retained} | Cas écartés : {exact_redundant} "
          f"| Temps : {exact_elapsed:.1f}s")

    print("\nÉtape 7 : accord cas par cas entre LSH et la décision exacte")
    agreements = [a == b for a, b in zip(lsh_decisions, exact_decisions)]
    print("Nombre de décisions identiques :", sum(agreements), "sur", len(agreements))
    disagreement_indices = [i for i, agree in enumerate(agreements) if not agree]
    print("Indices en désaccord :", disagreement_indices)

    if args.dataset in ("diabetes130", "chestxray14"):
        print(f"\nÉtape 8 : condensation CNN vs CBM — protocole adapté grand dataset ({args.dataset})")
        run_condensation_large_dataset(vectors, targets, args.dataset)
        print("\n=== Maintenance eval complete ===")
        return

    print("\nÉtape 8 : validation de la condensation sur plusieurs seeds")
    condensation_results = evaluate_condensation_over_seeds(vectors, targets, SEEDS)

    for result in condensation_results:
        print(result)

    precisions_full = [r["precision_full"] for r in condensation_results]
    precisions_condensed = [r["precision_condensed"] for r in condensation_results]
    proportions_kept = [r["proportion_kept"] for r in condensation_results]

    print("\nRésumé sur", len(SEEDS), "seeds :")
    print("Précision base complète — moyenne :", np.mean(precisions_full),
          "| écart-type :", np.std(precisions_full))
    print("Précision base condensée — moyenne :", np.mean(precisions_condensed),
          "| écart-type :", np.std(precisions_condensed))
    print("Proportion conservée — moyenne :", np.mean(proportions_kept),
          "| écart-type :", np.std(proportions_kept))

    print("\nÉtape 9 : comparaison CNN vs CBM, à taille égale, sur les mêmes seeds")
    cbm_results = evaluate_cbm_over_seeds(vectors, targets, SEEDS, condensation_results)
    for result in cbm_results:
        print(result)

    precisions_cnn = [r["precision_cnn"] for r in cbm_results]
    precisions_cbm = [r["precision_cbm"] for r in cbm_results]
    print("\nPrécision CNN — moyenne :", np.mean(precisions_cnn),
          "| écart-type :", np.std(precisions_cnn))
    print("Précision CBM — moyenne :", np.mean(precisions_cbm),
          "| écart-type :", np.std(precisions_cbm))


if __name__ == "__main__":
    main()
