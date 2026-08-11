"""Adapter générique de dataset (T0.1).

Responsabilités :
  1. Chargement des dataframes bruts via les loaders (chemins local/Kaggle).
  2. Split train/test stratifié par seed.
  3. Prétraitement ajusté sur le TRAIN uniquement (scaler / one-hot), transformé
     ensuite sur le test — élimine toute fuite d'information (règle §1.3 du plan).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import datasets as ds_cfg
from config import seeds
from . import chestxray14, diabetes130, heart_disease


def load_dataframe(key):
    """Charge le dataframe brut (features + cible 'target')."""
    src = ds_cfg.source(key)
    if key == "heart":
        return heart_disease.load_heart_disease(src)
    if key == "diabetes130":
        return diabetes130.load_diabetes130(csv_path=src)
    if key == "chestxray14":
        return chestxray14.load_chestxray14(data_dir=src)
    raise ValueError(f"dataset inconnu: {key}")


def _embedding_cols(df):
    return [c for c in df.columns if c.startswith("emb_")]


def _diabetes_cat_cols(df):
    """Colonnes catégorielles (liste fixe + colonnes object détectées dynamiquement)."""
    cat_cols = sorted(c for c in diabetes130.CATEGORICAL_COLS if c in df.columns)
    num_cols = [c for c in diabetes130.NUMERIC_COLS if c in df.columns]
    other_cat = sorted(c for c in df.columns
                       if c not in cat_cols and c not in num_cols
                       and c not in {"target", "encounter_id", "patient_nbr"}
                       and df[c].dtype == object)
    return sorted(set(cat_cols + other_cat))


def fit_transform_train(key, train_df, cat_vocab=None):
    """Prétraite le train (fit + transform). Retourne (X_train, fitted).

    Politique documentée (T0.3) : le vocabulaire catégoriel D130 est fixé
    globalement (cohérent avec le groupement topk global du loader) ; seul le
    scaler numérique est ajusté sur le train.
    """
    if key == "heart":
        scaler = StandardScaler()
        X = scaler.fit_transform(train_df[heart_disease.FEATURE_COLUMNS]).astype(np.float32)
        return X, {"scaler": scaler}

    if key == "diabetes130":
        cat_cols = _diabetes_cat_cols(train_df)
        num_cols = [c for c in diabetes130.NUMERIC_COLS if c in train_df.columns]
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                            categories=cat_vocab if cat_vocab is not None else "auto")
        cat_arr = ohe.fit_transform(train_df[cat_cols].astype(str))
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(train_df[num_cols].astype(np.float32)) \
            if num_cols else np.zeros((len(train_df), 0), dtype=np.float32)
        X = np.hstack([num_arr, cat_arr]).astype(np.float32)
        return X, {"ohe": ohe, "scaler": scaler, "cat_cols": cat_cols, "num_cols": num_cols}

    if key == "chestxray14":
        X = train_df[_embedding_cols(train_df)].to_numpy(dtype=np.float32)
        return X, {}

    raise ValueError(key)


def transform_test(key, test_df, fitted):
    """Transforme le test avec les objets ajustés sur le train."""
    if key == "heart":
        X = fitted["scaler"].transform(test_df[heart_disease.FEATURE_COLUMNS]).astype(np.float32)
        return X

    if key == "diabetes130":
        cat_arr = fitted["ohe"].transform(test_df[fitted["cat_cols"]].astype(str))
        num_cols = fitted["num_cols"]
        num_arr = fitted["scaler"].transform(test_df[num_cols].astype(np.float32)) \
            if num_cols else np.zeros((len(test_df), 0), dtype=np.float32)
        return np.hstack([num_arr, cat_arr]).astype(np.float32)

    if key == "chestxray14":
        return test_df[_embedding_cols(test_df)].to_numpy(dtype=np.float32)

    raise ValueError(key)


def get_split(key, seed=None, test_fraction=None):
    """Split stratifié train/test (par seed) + matrices prétraitées.

    Retourne (X_train, y_train, X_test, y_test, train_idx, test_idx).
    """
    seed = seeds.SEEDS[0] if seed is None else seed
    test_fraction = seeds.TEST_FRACTION if test_fraction is None else test_fraction

    df = load_dataframe(key)
    y = df["target"].to_numpy()
    stratify = y if len(np.unique(y)) >= 2 else None

    cat_vocab = None
    if key == "diabetes130":
        cat_cols = _diabetes_cat_cols(df)
        cat_vocab = [sorted(df[c].astype(str).dropna().unique().tolist()) for c in cat_cols]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train, fitted = fit_transform_train(key, train_df, cat_vocab)
    X_test = transform_test(key, test_df, fitted)

    return X_train, train_df["target"].to_numpy(), X_test, test_df["target"].to_numpy(), train_idx, test_idx
