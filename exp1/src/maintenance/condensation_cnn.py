import numpy as np
from sklearn.neighbors import NearestNeighbors


def evaluate_1nn_accuracy(reference_vectors, reference_targets, query_vectors, query_targets):
    """
    Évalue la précision d'une classification 1-NN par un seul appel vectorisé plutôt qu'une
    boucle Python sur chaque cas de requête — même principe que les autres réécritures :
    résultat mathématiquement identique, calcul par lot au lieu d'une boucle explicite.
    """
    finder = NearestNeighbors(n_neighbors=1)
    finder.fit(reference_vectors)
    _, nearest_indices = finder.kneighbors(query_vectors)
    predicted = reference_targets[nearest_indices[:, 0]]
    return float((predicted == query_targets).mean())


def condense_dataset(vectors, targets, random_seed=42):
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(vectors))

    condensed_indices = [order[0]]

    changed = True
    while changed:
        changed = False
        for i in order:
            if i in condensed_indices:
                continue

            condensed_vectors = vectors[condensed_indices]
            distances = np.linalg.norm(condensed_vectors - vectors[i], axis=1)
            nearest_pivot = condensed_indices[np.argmin(distances)]

            if targets[nearest_pivot] != targets[i]:
                condensed_indices.append(i)
                changed = True

    return np.array(condensed_indices)


def evaluate_condensation_over_seeds(vectors, targets, seeds):
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(vectors))
        split_point = int(0.8 * len(vectors))
        train_indices = shuffled_indices[:split_point]
        test_indices = shuffled_indices[split_point:]
        train_vectors, train_targets = vectors[train_indices], targets[train_indices]
        test_vectors, test_targets = vectors[test_indices], targets[test_indices]

        condensed_indices = condense_dataset(train_vectors, train_targets, random_seed=seed)
        condensed_vectors = train_vectors[condensed_indices]
        condensed_targets = train_targets[condensed_indices]

        precision_full = evaluate_1nn_accuracy(
            train_vectors, train_targets, test_vectors, test_targets
        )
        precision_condensed = evaluate_1nn_accuracy(
            condensed_vectors, condensed_targets, test_vectors, test_targets
        )

        results.append({
            "seed": seed,
            "train_size": len(train_vectors),
            "condensed_size": len(condensed_indices),
            "proportion_kept": len(condensed_indices) / len(train_vectors),
            "precision_full": precision_full,
            "precision_condensed": precision_condensed,
        })
    return results
