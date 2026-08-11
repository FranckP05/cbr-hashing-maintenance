from abc import ABC, abstractmethod

import numpy as np


class CaseIndex(ABC):
    def __init__(self):
        self._pending_cases: list[np.ndarray] = []

    @abstractmethod
    def build(self, vectors):
        pass

    @abstractmethod
    def query(self, vector, k):
        pass

    def add_case(self, vector):
        """
        Accumule les nouveaux cas dans une liste Python plutôt que de recopier l'intégralité de
        la base à chaque insertion (l'ancienne version faisait un np.vstack complet par cas
        retenu, soit une copie de plusieurs centaines de Mo répétée des milliers de fois). La
        liste est fusionnée dans self.vectors seulement au moment où une requête en a besoin
        (voir _sync_pending_cases), pas à chaque ajout.
        """
        self._pending_cases.append(vector)

    def _sync_pending_cases(self):
        if self._pending_cases:
            if self.vectors is None:
                self.vectors = np.vstack(self._pending_cases)
            else:
                self.vectors = np.vstack([self.vectors] + self._pending_cases)
            self._pending_cases = []
