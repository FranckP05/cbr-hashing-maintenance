from abc import ABC, abstractmethod


class CaseIndex(ABC):
    @abstractmethod
    def build(self, vectors):
        pass

    @abstractmethod
    def query(self, vector, k):
        pass
