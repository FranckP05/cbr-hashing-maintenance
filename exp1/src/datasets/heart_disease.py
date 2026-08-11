import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


HEART_DISEASE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

FEATURE_COLUMNS = [c for c in HEART_DISEASE_COLUMNS if c != "target"]


def load_heart_disease(file_path):
    data = pd.read_csv(file_path, header=None, names=HEART_DISEASE_COLUMNS, na_values="?")
    data = data.dropna()
    data["target"] = (data["target"] > 0).astype(int)
    return data


def build_feature_matrix(dataframe):
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(dataframe[FEATURE_COLUMNS])
    targets = dataframe["target"].to_numpy()
    return feature_matrix, targets
