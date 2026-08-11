import numpy as np

from .case_index import CaseIndex


class RandomProjectionLSHIndex(CaseIndex):
    def __init__(self, num_tables=16, num_bits=5, random_seed=42):
        super().__init__()
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

    def add_case(self, vector):
        """
        Ajoute un seul cas à l'index LSH sans rehacher toute la base : le nouveau vecteur est
        mis en attente et ne sera haché qu'à la requête suivante (voir _sync_pending_cases).
        Le hachage utilise les hyperplans déjà générés (déterministes, ils ne dépendent que de
        la dimensionnalité). Équivalent exact à l'ancienne version eager (vstack immédiat +
        insertion dans les compartiments) : les buckets finaux sont identiques, seuls le moment
        de la copie et la taille des temporaires changent.
        """
        self._pending_cases.append(vector)

    def _sync_pending_cases(self):
        if not self._pending_cases:
            return
        pending = np.array(self._pending_cases)
        new_start = len(self.vectors)
        self.vectors = np.vstack([self.vectors] + self._pending_cases)
        for table_hyperplanes, table_buckets in zip(self.hyperplanes, self.buckets):
            keys = self._hash_vectors(pending, table_hyperplanes)
            for offset, key in enumerate(keys):
                table_buckets.setdefault(key, []).append(new_start + offset)
        self._pending_cases = []

    def query(self, vector, k):
        self._sync_pending_cases()
        candidate_indices = set()
        for table_hyperplanes, table_buckets in zip(self.hyperplanes, self.buckets):
            key = self._hash_vectors(vector[np.newaxis, :], table_hyperplanes)[0]
            candidate_indices.update(table_buckets.get(key, []))

        self.last_candidate_count = len(candidate_indices)
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
