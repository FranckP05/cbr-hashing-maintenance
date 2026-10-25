"""Pont vers les datasets exp2 — réutilise exclusivement exp2/src/datasets/adapters.py.

Ne réécrit aucune logique de chargement/prétraitement : c'est adapters.get_split() qui
fournit le split stratifié 80/20 (seed 42, TEST_FRACTION=0.2) et les matrices prétraitées.
"""

import numpy as np

from src.datasets import adapters

from demo_config import DATASETS


def load_dataset(key):
    """Charge un dataset exp2 et remplit demo_state (lazy, appelé à la 1re sélection).

    Retourne un dict de description pour l'interface (n_train, n_test, dim).
    """
    from state import demo_state

    X_train, y_train, X_test, y_test, train_idx, test_idx = adapters.get_split(key)

    desc = {
        "key": key,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
    }

    # Rechargement du dataframe brut uniquement quand les métadonnées l'exigent :
    # scaler ajusté sur train (saisie manuelle heart) et encounter_ids (HeCBR D130).
    df = None
    if key in ("heart", "diabetes130"):
        df = adapters.load_dataframe(key)
    scaler = None
    feature_ranges = None
    hecbr_codes = None
    if key == "heart":
        train_df = df.iloc[train_idx].reset_index(drop=True)
        _, fitted = adapters.fit_transform_train("heart", train_df)
        scaler = fitted["scaler"]
        feature_ranges = {
            col: (float(train_df[col].min()), float(train_df[col].max()))
            for col in DATASETS[key]["feature_columns"]
        }
    if key == "diabetes130":
        hecbr_codes = _read_hecbr_codes(demo_state)

    # Le chargement est atomique : une dépendance manquante ne laisse pas un dataset
    # partiellement prêt dans l'état global de la démo.
    demo_state.pristine_train_vectors[key] = X_train
    demo_state.pristine_test_vectors[key] = X_test
    demo_state.pristine_test_targets[key] = y_test
    demo_state.base_size[key] = X_train.shape[0]
    if key == "heart":
        demo_state.scaler[key] = scaler
        demo_state.feature_ranges[key] = feature_ranges
    if key == "diabetes130":
        demo_state.train_encounter_ids[key] = df["encounter_id"].to_numpy()[train_idx]
        demo_state.test_encounter_ids[key] = df["encounter_id"].to_numpy()[test_idx]
        demo_state.hecbr_codes = hecbr_codes

    return desc


def _read_hecbr_codes(demo_state):
    """Lit une fois les codes binaires HeCBR sans modifier l'état en cas d'échec."""
    if demo_state.hecbr_codes is not None:
        return demo_state.hecbr_codes
    import pandas as pd
    from demo_config import HECBR_CODES_PATH

    return pd.read_parquet(HECBR_CODES_PATH)


def hecbr_code_for_encounter(demo_state, encounter_id):
    """Retourne le code binaire 64 bits (np.int8) du cas, ou None si introuvable."""
    row = demo_state.hecbr_codes[
        demo_state.hecbr_codes["encounter_id"] == int(encounter_id)
    ]
    if row.empty:
        return None
    bit_cols = [c for c in row.columns if c.startswith("bit")]
    return row[bit_cols].to_numpy(dtype=np.int8)[0]


def manual_case_vector(key, raw_values, raw_column_names):
    """Standardise une saisie manuelle avec le scaler ajusté sur le train (heart)."""
    from state import demo_state

    scaler = demo_state.scaler[key]
    values = np.array([float(raw_values[c]) for c in raw_column_names]).reshape(1, -1)
    return scaler.transform(values).astype(np.float32)[0]


def get_case_feature_values(key, idx):
    """Retourne les valeurs réelles lisibles du cas n°idx du test set."""
    from state import demo_state

    df = adapters.load_dataframe(key)
    _, _, _, _, _, test_idx = adapters.get_split(key)
    row = df.iloc[test_idx[idx]]

    if key == "heart":
        cols = DATASETS[key]["feature_columns"]
        return {c: round(float(row[c]), 2) if isinstance(row[c], (int, float, np.number)) else str(row[c]) for c in cols if c in row}
    elif key == "diabetes130":
        main_cols = [
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses", "age", "race", "gender"
        ]
        res = {}
        for c in main_cols:
            if c in row:
                res[c] = int(row[c]) if isinstance(row[c], (int, np.integer)) else str(row[c])
        return res
    else:
        # chestxray14 : vecteur d'embedding 1024D
        vec = demo_state.pristine_test_vectors[key][idx]
        return {
            "dim_total": len(vec),
            "l2_norm": round(float(np.linalg.norm(vec)), 4),
            "mean": round(float(np.mean(vec)), 4),
            "std": round(float(np.std(vec)), 4),
            "min": round(float(np.min(vec)), 4),
            "max": round(float(np.max(vec)), 4),
            "sample_dim_0_4": [round(float(v), 4) for v in vec[:5]],
        }
