import numpy as np

from .retention_lsh import compute_baseline_nn_distance_threshold


def compute_relative_coverage(vectors, targets, threshold):
    num_cases = len(vectors)
    coverage_sets = []
    for i in range(num_cases):
        distances = np.linalg.norm(vectors - vectors[i], axis=1)
        solved = np.where((distances <= threshold) & (targets == targets[i]))[0]
        coverage_sets.append(solved)

    reachability_sizes = np.array([len(s) for s in coverage_sets])

    relative_coverage = np.zeros(num_cases)
    for i in range(num_cases):
        relative_coverage[i] = sum(1.0 / reachability_sizes[j] for j in coverage_sets[i])

    return relative_coverage


def condense_by_competence(vectors, targets, target_size, threshold):
    relative_coverage = compute_relative_coverage(vectors, targets, threshold)
    order = np.argsort(relative_coverage)
    kept_indices = np.sort(order[-target_size:])
    return kept_indices


def evaluate_cbm_over_seeds(vectors, targets, seeds, cnn_results):
    results = []
    for seed, cnn_result in zip(seeds, cnn_results):
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(vectors))
        split_point = int(0.8 * len(vectors))
        train_indices = shuffled_indices[:split_point]
        test_indices = shuffled_indices[split_point:]
        train_vectors, train_targets = vectors[train_indices], targets[train_indices]
        test_vectors, test_targets = vectors[test_indices], targets[test_indices]

        solve_threshold = compute_baseline_nn_distance_threshold(train_vectors, percentile=50)

        cbm_indices = condense_by_competence(
            train_vectors, train_targets, cnn_result["condensed_size"], solve_threshold
        )
        cbm_vectors = train_vectors[cbm_indices]
        cbm_targets = train_targets[cbm_indices]

        correct_cbm = 0
        for i in range(len(test_vectors)):
            distances = np.linalg.norm(cbm_vectors - test_vectors[i], axis=1)
            correct_cbm += (cbm_targets[np.argmin(distances)] == test_targets[i])

        results.append({
            "seed": seed,
            "condensed_size": cnn_result["condensed_size"],
            "precision_cnn": cnn_result["precision_condensed"],
            "precision_cbm": correct_cbm / len(test_vectors),
        })
    return results
