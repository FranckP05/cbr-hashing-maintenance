import numpy as np
import pandas as pd

from .case_index import CaseIndex


class HeCBRIndex(CaseIndex):
    """
    Enveloppe les codes binaires HeCBR déjà entraînés (64 bits, un par cas) et effectue la
    recherche des plus proches voisins par distance de Hamming plutôt que par distance
    euclidienne — cohérent avec la nature du hachage produit par HeCBR. Ne réentraîne rien :
    charge simplement les codes déjà calculés (hecbr_diabetes130_codes_final.parquet) et les
    indexe pour la requête.
    """

    def __init__(self, codes_path):
        super().__init__()
        self.codes_path = codes_path
        self.codes = None
        self.labels = None
        self.encounter_ids = None
        self.split = None

    def build(self, vectors, restrict_split=None):
        # vectors est ignoré ici : les codes binaires viennent du fichier pré-calculé, pas
        # d'un nouveau calcul à partir des vecteurs bruts (contrairement aux autres index).
        df = pd.read_parquet(self.codes_path)
        bit_cols = [c for c in df.columns if c.startswith("bit")]
        self.codes = df[bit_cols].to_numpy(dtype=np.int8)
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.encounter_ids = df["encounter_id"].to_numpy(dtype=np.int64)
        self.split = df["split"].to_numpy(dtype=object)
        # restreindre la recherche à un split (ex. "train") exclut les autres cas de la
        # base de voisinage ; les indices renvoyés sont alors relatifs à ce sous-ensemble.
        if restrict_split is not None:
            mask = self.split == restrict_split
            self.codes = self.codes[mask]
            self.labels = self.labels[mask]
            self.encounter_ids = self.encounter_ids[mask]
            self.split = self.split[mask]

    def query(self, vector, k):
        vector = np.asarray(vector).astype(np.int8)
        if vector.ndim == 1 and self.codes.shape[1] == vector.shape[0]:
            hamming_distances = (self.codes != vector).sum(axis=1)
        else:
            raise ValueError(
                f"Requête HeCBR attendue comme code binaire 1-D de {self.codes.shape[1]} bits, "
                f"reçu shape={np.shape(vector)}. Le vecteur de requête doit être le code HeCBR "
                f"du cas (64 bits), pas la matrice de features brute."
            )
        nearest_indices = np.argsort(hamming_distances)[:k]
        return nearest_indices, hamming_distances[nearest_indices]
