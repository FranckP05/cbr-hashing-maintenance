from sklearn.neighbors import KDTree

from .case_index import CaseIndex


class KDTreeIndex(CaseIndex):
    def __init__(self):
        super().__init__()
        self.tree = None
        self.vectors = None

    def build(self, vectors):
        self.vectors = vectors
        self.tree = KDTree(vectors)

    def _sync_pending_cases(self):
        # Contrairement au comportement par défaut, une fusion des cas en attente exige ici de
        # reconstruire l'arbre : scikit-learn ne permet pas l'insertion incrémentale d'un KDTree.
        if self._pending_cases:
            super()._sync_pending_cases()
            self.tree = KDTree(self.vectors)

    def query(self, vector, k):
        self._sync_pending_cases()
        distances, indices = self.tree.query([vector], k=k)
        return indices[0], distances[0]
