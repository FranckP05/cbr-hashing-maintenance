"""Score de sélection du vainqueur (T6.3) — critère §5 du plan.

> Score = moyenne normalisée de 4 composantes par dataset, pondérées :
>   1. Qualité post-maintenance (macro-F1)              — 40 %
>   2. Rapidité post-maintenance (temps/req, log-normalisé) — 30 %
>   3. Robustesse (variance inter-seeds, significativité)  — 15 %
>   4. Fidélité d'approximation (rappel des voisins exacts) — 15 %
> Le vainqueur maximise le score MOYEN sur les datasets applicables.

Composantes telles qu'opérationnalisées depuis les CSV du run (aucune valeur en dur) :

- Qualité : macro-F1 du k-NN par méthode sur la base brute (benchmark). La qualité
  post-maintenance mesurée (nn_macro_f1, base-level, condensation_quality.csv) est
  identique pour les 3 méthodes d'un même dataset (même base maintenue) : elle ne
  différencie pas les méthodes et est rapportée dans le tableau T6.1.
- Rapidité : query_ms (médiane) log-normalisée inversée par dataset (plus rapide → 1).
- Robustesse : 1 − std_inter_seeds(macro_F1) / max_std_du_dataset (variance faible → 1).
- Fidélité : rappel des voisins exacts (exact_recall_at_k), déjà en [0,1].

HeCBR est exclu du classement (1 seed figé, un seul dataset) ; ses scores par dataset
sont rapportés dans le tableau sans participer à la moyenne.
"""

import numpy as np
import pandas as pd

WEIGHTS = {"quality": 0.40, "speed": 0.30, "robustness": 0.15, "fidelity": 0.15}
METHODS = ["linear", "kdtree", "lsh"]


def per_seed_agg(df):
    """Moyennes par (dataset, method, seed) sur k ∈ {1,5,10}."""
    df = df.copy()
    df["seed"] = df["seed"].astype(str)
    return df.groupby(["dataset", "method", "seed"]).agg(
        macro_f1=("macro_f1", "mean"),
        exact_recall_at_k=("exact_recall_at_k", "mean"),
        query_ms=("query_ms", "mean"),
    ).reset_index()


def _normalize(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=series.index)
    return (series - lo) / (hi - lo)


def score_dataset(per_seed, dataset):
    """Score §5 des méthodes pour un dataset (dict {method: (score, components)})."""
    part = per_seed[per_seed["dataset"] == dataset]
    agg = part.groupby("method").agg(
        quality=("macro_f1", "mean"),
        fidelity=("exact_recall_at_k", "mean"),
        speed=("query_ms", "mean"),
        robust=("macro_f1", "std"),
    )
    present = [m for m in METHODS if m in agg.index]
    agg = agg.loc[present]

    comp = pd.DataFrame(index=agg.index)
    comp["quality"] = _normalize(agg["quality"])
    comp["fidelity"] = _normalize(agg["fidelity"])
    log_q = np.log(agg["speed"])
    comp["speed"] = _normalize(-log_q)  # plus rapide → log plus petit → 1
    max_std = agg["robust"].max()
    comp["robustness"] = 1.0 - (agg["robust"].fillna(0.0) / max_std if max_std > 0 else 0.0)

    score = (
        WEIGHTS["quality"] * comp["quality"]
        + WEIGHTS["speed"] * comp["speed"]
        + WEIGHTS["robustness"] * comp["robustness"]
        + WEIGHTS["fidelity"] * comp["fidelity"]
    )
    out = {}
    for m in present:
        out[m] = {
            "score": float(score[m]),
            **{k: round(float(comp.loc[m, k]), 4) for k in comp.columns},
            **{f"raw_{k}": round(float(agg.loc[m, k]), 4) for k in agg.columns},
        }
    return out


def select_winner(scores_by_dataset):
    """Vainqueur = score moyen maximal sur les datasets applicables."""
    methods = set().union(*[s.keys() for s in scores_by_dataset.values()])
    means = {}
    for m in methods:
        vals = [s[m]["score"] for s in scores_by_dataset.values() if m in s]
        means[m] = float(np.mean(vals))
    winner = max(means, key=means.get)
    return winner, means
