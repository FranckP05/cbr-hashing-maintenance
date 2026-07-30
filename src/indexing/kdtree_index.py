from sklearn.neighbors import KDTree

from .case_index import CaseIndex


class KDTreeIndex(CaseIndex):
    def __init__(self):
        self.tree = None

    def build(self, vectors):
        self.tree = KDTree(vectors)

    def query(self, vector, k):
        distances, indices = self.tree.query([vector], k=k)
        return indices[0], distances[0]
