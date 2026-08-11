import numpy as np
import pandas as pd


def load_chestxray14(data_dir="data/processed"):
    """
    Charge et fusionne les embeddings ChestX-ray14 (1024 dims, DenseNet121) et leurs labels
    de pathologies, en joignant les deux fichiers parquet sur image_filename. data_dir
    permet de cibler explicitement le dossier (local ou Kaggle). La cible
    binaire est définie comme : 0 si aucune des 14 pathologies n'est présente ("No Finding"),
    1 si au moins une pathologie est présente — choix qui reste cohérent avec le format
    binaire utilisé pour Heart Disease et Diabetes-130 dans le reste du pipeline.
    """
    embeddings_df = pd.read_parquet(f"{data_dir}/chestxray14_embeddings.parquet")
    labels_df = pd.read_parquet(f"{data_dir}/chestxray14_labels.parquet")
    merged = embeddings_df.merge(labels_df, on="image_filename")

    pathology_cols = [c for c in labels_df.columns if c != "image_filename"]
    merged["target"] = (merged[pathology_cols].sum(axis=1) > 0).astype(int)
    return merged


def build_feature_matrix(dataframe):
    """
    Extrait la matrice de features (les 1024 colonnes emb_0..emb_1023, déjà standardisées
    implicitement par DenseNet121, pas besoin de StandardScaler ici contrairement aux deux
    autres datasets) et la cible binaire.
    """
    embedding_cols = [c for c in dataframe.columns if c.startswith("emb_")]
    X = dataframe[embedding_cols].to_numpy(dtype=np.float32)
    y = dataframe["target"].to_numpy()
    return X, y
