import numpy as np
from sklearn.neighbors import NearestNeighbors

from .condensation_cnn import evaluate_1nn_accuracy
from .retention_lsh import compute_baseline_nn_distance_threshold


def compute_relative_coverage(vectors, targets, threshold):
    """
    Calcule pour chaque cas son score de couverture relative RC(c), tel que défini par
    Smyth & McKenna (1998). Réécrite avec NearestNeighbors.radius_neighbors() plutôt qu'une
    boucle exhaustive par paires : la version originale était en O(n²), infaisable au-delà
    de quelques milliers de cas — nécessaire dès Diabetes-130 (81 412 cas). Le résultat
    mathématique est identique, seule l'implémentation change (même principe que la
    réécriture de compute_baseline_nn_distance_threshold).
    """
    finder = NearestNeighbors(radius=threshold)
    finder.fit(vectors)
    neighbors_per_case = finder.radius_neighbors(vectors, return_distance=False)

    num_cases = len(vectors)
    solved_by = [
        indices[targets[indices] == targets[i]]
        for i, indices in enumerate(neighbors_per_case)
    ]
    reachability_sizes = np.array([len(s) for s in solved_by])

    relative_coverage = np.zeros(num_cases)
    for i in range(num_cases):
        relative_coverage[i] = np.sum(1.0 / reachability_sizes[solved_by[i]])
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

        precision_cbm = evaluate_1nn_accuracy(
            cbm_vectors, cbm_targets, test_vectors, test_targets
        )

        results.append({
            "seed": seed,
            "condensed_size": cnn_result["condensed_size"],
            "precision_cnn": cnn_result["precision_condensed"],
            "precision_cbm": precision_cbm,
        })
    return results
