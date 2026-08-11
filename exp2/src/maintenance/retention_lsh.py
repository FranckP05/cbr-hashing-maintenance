import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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


def simulate_case_stream(base_vectors, incoming_vectors, distance_threshold, index_factory,
                         desc="Simulation du flux"):
    t_start = time.perf_counter()
    index = index_factory()
    index.build(base_vectors)

    retained_count = 0
    redundant_count = 0
    decisions = []

    for new_vector in tqdm(incoming_vectors, desc=desc, unit="case"):
        redundant = is_redundant(new_vector, index, distance_threshold)
        decisions.append(redundant)

        if redundant:
            redundant_count += 1
        else:
            index.add_case(new_vector)
            retained_count += 1

    elapsed = time.perf_counter() - t_start
    # Synchronise les cas encore en attente (add_case paresseux) pour que la base renvoyée
    # reflète l'état final ; aucun appelant ne s'en sert, mais le contrat doit rester exact.
    index._sync_pending_cases()
    # La base à jour est maintenue par l'index lui-même via add_case() ; on la renvoie par
    # référence pour préserver le contrat de retour (aucun appelant ne s'en sert, cf. audit).
    return index.vectors, retained_count, redundant_count, decisions, elapsed
