import numpy as np


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

        correct_full = 0
        correct_condensed = 0
        for i in range(len(test_vectors)):
            distances_full = np.linalg.norm(train_vectors - test_vectors[i], axis=1)
            correct_full += (train_targets[np.argmin(distances_full)] == test_targets[i])

            distances_condensed = np.linalg.norm(condensed_vectors - test_vectors[i], axis=1)
            correct_condensed += (condensed_targets[np.argmin(distances_condensed)] == test_targets[i])

        results.append({
            "seed": seed,
            "train_size": len(train_vectors),
            "condensed_size": len(condensed_indices),
            "proportion_kept": len(condensed_indices) / len(train_vectors),
            "precision_full": correct_full / len(test_vectors),
            "precision_condensed": correct_condensed / len(test_vectors),
        })
    return results
