import numpy as np

from .case_index import CaseIndex


class RandomProjectionLSHIndex(CaseIndex):
    def __init__(self, num_tables=16, num_bits=5, random_seed=42):
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.random_seed = random_seed
        self.vectors = None
        self.hyperplanes = None
        self.buckets = None

    def build(self, vectors):
        self.vectors = vectors
        num_dimensions = vectors.shape[1]
        rng = np.random.default_rng(self.random_seed)
        self.hyperplanes = [
            rng.normal(size=(num_dimensions, self.num_bits))
            for _ in range(self.num_tables)
        ]
        self.buckets = []
        for table_hyperplanes in self.hyperplanes:
            table_buckets = {}
            hash_keys = self._hash_vectors(vectors, table_hyperplanes)
            for case_index, key in enumerate(hash_keys):
                table_buckets.setdefault(key, []).append(case_index)
            self.buckets.append(table_buckets)

    def query(self, vector, k):
        candidate_indices = set()
        for table_hyperplanes, table_buckets in zip(self.hyperplanes, self.buckets):
            key = self._hash_vectors(vector[np.newaxis, :], table_hyperplanes)[0]
            candidate_indices.update(table_buckets.get(key, []))

        candidate_indices = np.array(sorted(candidate_indices), dtype=int)
        if len(candidate_indices) == 0:
            return np.array([], dtype=int), np.array([])

        candidate_vectors = self.vectors[candidate_indices]
        distances = np.linalg.norm(candidate_vectors - vector, axis=1)
        order = np.argsort(distances)[:k]
        return candidate_indices[order], distances[order]

    def _hash_vectors(self, vectors, hyperplanes):
        projections = vectors @ hyperplanes
        bits = (projections > 0).astype(int)
        bit_weights = 2 ** np.arange(self.num_bits)
        keys = bits @ bit_weights
        return keys
