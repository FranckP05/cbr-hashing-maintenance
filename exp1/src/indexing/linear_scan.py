import numpy as np

from .case_index import CaseIndex


class LinearScanIndex(CaseIndex):
    def __init__(self):
        super().__init__()
        self.vectors = None

    def build(self, vectors):
        self.vectors = vectors

    def query(self, vector, k):
        self._sync_pending_cases()
        # Même identité que le batch Baseline du k-sweep (NearestNeighbors/euclidean_distances) :
        # dist² = ||x||² - 2·x·v + ||v||², par produit matrice-vecteur BLAS et réductions sur
        # (n,) — évite les temporaires (n, d) de np.linalg.norm(vectors - vector, axis=1), qui
        # allouaient ~370 Mo par requête en 1024-d. Résultat mathématiquement identique.
        vectors_sq = np.einsum("ij,ij->i", self.vectors, self.vectors)
        vector_sq = float(np.dot(vector, vector))
        distances_sq = -2.0 * (self.vectors @ vector)
        distances_sq += vectors_sq
        distances_sq += vector_sq
        np.maximum(distances_sq, 0, out=distances_sq)
        distances = np.sqrt(distances_sq)
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices, distances[nearest_indices]
