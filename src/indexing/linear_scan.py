import numpy as np

from .case_index import CaseIndex


class LinearScanIndex(CaseIndex):
    def __init__(self):
        self.vectors = None

    def build(self, vectors):
        self.vectors = vectors

    def query(self, vector, k):
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices, distances[nearest_indices]
