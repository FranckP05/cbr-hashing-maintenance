import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_baseline_nn_distance_threshold(vectors, percentile=25):
    finder = NearestNeighbors(n_neighbors=2)
    finder.fit(vectors)
    distances, _ = finder.kneighbors(vectors)
    nearest_neighbor_distances = distances[:, 1]
    return np.percentile(nearest_neighbor_distances, percentile)


def is_redundant(new_vector, index, distance_threshold):
    candidate_indices, candidate_distances = index.query(new_vector, k=1)
    if len(candidate_indices) == 0:
        return False
    return candidate_distances[0] < distance_threshold


def simulate_case_stream(base_vectors, incoming_vectors, distance_threshold, index_factory):
    current_base = base_vectors.copy()
    retained_count = 0
    redundant_count = 0
    decisions = []

    for new_vector in incoming_vectors:
        index = index_factory()
        index.build(current_base)

        redundant = is_redundant(new_vector, index, distance_threshold)
        decisions.append(redundant)

        if redundant:
            redundant_count += 1
        else:
            current_base = np.vstack([current_base, new_vector])
            retained_count += 1

    return current_base, retained_count, redundant_count, decisions
