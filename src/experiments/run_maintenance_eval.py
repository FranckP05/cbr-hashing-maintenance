"""Evaluate retention filter and condensation (CNN vs CBM) on a given dataset.

Replicates the full pipeline from data_loading.py __main__:
  - Load and vectorize
  - Split base / incoming stream
  - Calibrate redundancy threshold
  - LSH retention simulation vs exact reference
  - Condensation (CNN) over multiple seeds
  - CBM comparison at equal condensed size
"""

import argparse
import importlib

import numpy as np

from src.indexing.linear_scan import LinearScanIndex
from src.indexing.lsh_index import RandomProjectionLSHIndex
from src.maintenance.retention_lsh import (
    compute_baseline_nn_distance_threshold,
    simulate_case_stream,
)
from src.maintenance.condensation_cnn import evaluate_condensation_over_seeds
from src.maintenance.condensation_cbm import evaluate_cbm_over_seeds


DATASET_REGISTRY = {
    "heart_disease": ("src.datasets.heart_disease", "load_heart_disease", "build_feature_matrix"),
    "diabetes130":   ("src.datasets.diabetes130",   "load_diabetes130",   "build_feature_matrix"),
}

DEFAULT_DATA_PATHS = {
    "heart_disease": "data/raw/heart_disease/processed.cleveland.data",
}

SEEDS = [42, 7, 123, 2024, 99]


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

    print("\nÉtape 5 : simulation avec LSH (approximatif)")
    lsh_final_base, lsh_retained, lsh_redundant, lsh_decisions = simulate_case_stream(
        base_vectors, incoming_vectors, distance_threshold,
        index_factory=lambda: RandomProjectionLSHIndex(num_tables=16, num_bits=5),
    )
    print("Cas retenus :", lsh_retained, "| Cas écartés :", lsh_redundant)

    print("\nÉtape 6 : simulation avec balayage linéaire (référence exacte)")
    exact_final_base, exact_retained, exact_redundant, exact_decisions = simulate_case_stream(
        base_vectors, incoming_vectors, distance_threshold,
        index_factory=lambda: LinearScanIndex(),
    )
    print("Cas retenus :", exact_retained, "| Cas écartés :", exact_redundant)

    print("\nÉtape 7 : accord cas par cas entre LSH et la décision exacte")
    agreements = [a == b for a, b in zip(lsh_decisions, exact_decisions)]
    print("Nombre de décisions identiques :", sum(agreements), "sur", len(agreements))
    disagreement_indices = [i for i, agree in enumerate(agreements) if not agree]
    print("Indices en désaccord :", disagreement_indices)

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
