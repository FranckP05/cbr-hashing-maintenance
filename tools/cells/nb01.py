"""Cellules du notebook NB01 (exploration des données + figures de présentation).

Charte : Space Grotesk (titres) + Manrope (corps), couleurs conservées, légende
« Figure : … » en bas, échelles revisitées (D1 par panneau, D2/D6 cassées).
"""

CELLS = [
    ("md", """\
# NB01 — Exploration des données & figures de présentation

Produit les **FIG-D1..D6** (plan §7.1) et le profil des datasets (`data_profile.csv`).

- FIG-D1 : répartition des classes (échelle propre à chaque panneau)
- FIG-D2 : dimensionnalité comparative (échelle cassée : 13 / 390 / 1024)
- FIG-D3 : composition des datasets (tableau)
- FIG-D4 : projection PCA 2D colorée par classe
- FIG-D5 : distribution de la distance au plus proche voisin
- FIG-D6 : Diabetes-130, numériques vs one-hot (échelle cassée)

Charte : Space Grotesk (titres) / Manrope (corps), légende « Figure : … » en bas.
"""),

    ("code", """\
import sys
from pathlib import Path

_candidates = [Path.cwd() / "exp2"]
if Path("/kaggle/input").exists():
    _candidates += sorted(Path("/kaggle/input").glob("*/"))
EXP2 = next((p for p in _candidates if (p / "src").exists() and (p / "config").exists()), None)
assert EXP2 is not None, "Dossier exp2 introuvable"
sys.path.insert(0, str(EXP2))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import datasets as ds_cfg, run as run_cfg, seeds
from src.datasets import adapters, diabetes130
from src.visualization.theme import (CLASS_COLORS, DIM_COLOR, apply_theme,
                                     broken_axes, finalize, title)

apply_theme()
run_dir = run_cfg.get_run_dir()
(run_dir / "figures").mkdir(exist_ok=True)
print("Run courant :", run_dir)
"""),

    ("code", """\
# Profil brut des datasets (n, cible, valeurs manquantes)
rows = []
for key in ds_cfg.ORDER:
    df = adapters.load_dataframe(key)
    y = df["target"]
    missing = float(df.isna().sum().sum()) / (df.shape[0] * df.shape[1])
    rows.append({
        "dataset": key,
        "label": ds_cfg.DATASETS[key]["label"],
        "n": int(len(df)),
        "n_raw_features": int(df.shape[1]),
        "n_positifs": int(y.sum()),
        "pos_ratio": round(float(y.mean()), 4),
        "missing_ratio": round(missing, 6),
    })
    print(rows[-1])

raw_profile = pd.DataFrame(rows)
raw_profile.to_csv(run_dir / "data_profile_raw.csv", index=False)
"""),

    ("code", """\
# FIG-D1 — Répartition des classes (échelle propre à chaque panneau)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, key in zip(axes, ds_cfg.ORDER):
    df = adapters.load_dataframe(key)
    counts = df["target"].value_counts().sort_index()
    bars = ax.bar(["0", "1"], counts.values, color=CLASS_COLORS, width=0.55)
    title(ax, ds_cfg.DATASETS[key]["label"])
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre de cas")
    ax.set_ylim(0, counts.max() * 1.18)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

finalize(
    fig,
    caption="Répartition des classes par dataset",
    output_path=run_dir / "figures" / "FIG-D1_repartition_classes.png",
)
plt.close(fig)
print("FIG-D1 enregistrée")
"""),

    ("code", """\
# FIG-D2 — Dimensionnalité comparative (échelle cassée : petites + grandes valeurs)
keys = ds_cfg.ORDER
labels = ["Heart Disease", "Diabetes 130", "ChestX-ray14"]
dims = [ds_cfg.DATASETS[k]["n_features"] for k in keys]

fig, ax_top, ax_bottom = broken_axes(figsize=(8, 4.5), hspace=0.16)
for ax in (ax_top, ax_bottom):
    ax.bar(labels, dims, color=DIM_COLOR, width=0.5)
    ax.grid(axis="x", visible=False)
ax_top.set_ylim(330, 1140)
ax_bottom.set_ylim(0, 45)
for ax, lo, hi in ((ax_top, 330, 1140), (ax_bottom, 0, 45)):
    for i, d in enumerate(dims):
        if lo <= d <= hi:
            ax.text(i, min(d + hi * 0.03, hi * 0.98), f"{d}", ha="center",
                    fontsize=10, clip_on=True)
ax_top.set_ylabel("Dimension")
ax_bottom.set_ylabel("Dimension")
ax_bottom.tick_params(axis="x", labelsize=9)

finalize(
    fig,
    caption="Dimensionnalité des jeux de données",
    output_path=run_dir / "figures" / "FIG-D2_dimensionnalite.png",
)
plt.close(fig)
print("FIG-D2 enregistrée")
"""),

    ("code", """\
# FIG-D3 — Composition des datasets (tableau) + profil train/test
rows = []
for key in ds_cfg.ORDER:
    X_train, y_train, X_test, y_test, _, _ = adapters.get_split(key)
    rows.append({
        "dataset": key,
        "label": ds_cfg.DATASETS[key]["label"],
        "n": int(len(y_train) + len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "d": int(X_train.shape[1]),
        "type": ds_cfg.DATASETS[key]["description"],
        "scaling": ds_cfg.DATASETS[key]["scaling"],
        "pos_ratio_train": round(float(np.mean(y_train)), 4),
    })
    print(rows[-1])

profile = pd.DataFrame(rows)
profile.to_csv(run_dir / "data_profile.csv", index=False)

# 2) Tableau : colWidths proportionnels à la longueur max du contenu,
#    type raccourci (détail complet dans data_profile.csv)
SHORT_TYPES = {
    "Tabulaire, 13 attributs cliniques, ~46 % positifs": "Tabulaire (13 attrs)",
    "Mixte, 8 numériques + one-hot (dim finale 390), 11,2 % positifs": "Mixte, 8 num + one-hot (390)",
    "Multimodal, embeddings 1024-D, 46,2 % positifs": "Multimodal, embeddings (1024)",
}
cols = ["dataset", "n", "n_train", "n_test", "d", "type", "scaling", "pos_ratio_train"]
headers = ["dataset", "n", "n_train", "n_test", "d", "type", "scaling", "pos train"]
cell_rows = []
for r in rows:
    rr = {c: r[c] for c in cols}
    rr["type"] = SHORT_TYPES.get(str(r["type"]), str(r["type"]))
    cell_rows.append(rr)

maxlens = []
for ci, col in enumerate(cols):
    maxlens.append(max(len(str(headers[ci])), *(len(str(rr[col])) for rr in cell_rows)))
col_widths = [ml / sum(maxlens) for ml in maxlens]

fig, ax = plt.subplots(figsize=(17, 2.6))
ax.axis("off")
table = ax.table(
    cellText=[[str(rr[c]) for c in cols] for rr in cell_rows],
    colLabels=headers,
    colWidths=col_widths,
    loc="center",
    cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
finalize(
    fig,
    caption="Composition des jeux de données",
    output_path=run_dir / "figures" / "FIG-D3_composition_datasets.png",
)
plt.close(fig)
print("FIG-D3 enregistrée ; data_profile.csv écrit")
"""),

    ("code", """\
# FIG-D4 — Projection PCA 2D colorée par classe (échantillonné pour D130/Chest)
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

rng = np.random.default_rng(seeds.SAMPLING_SEED)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

for ax, key in zip(axes, ds_cfg.ORDER):
    X_train, y_train, _, _, _, _ = adapters.get_split(key)
    if len(X_train) > 15000:
        idx = rng.choice(len(X_train), 15000, replace=False)
        Xs, ys = X_train[idx], y_train[idx]
    else:
        Xs, ys = X_train, y_train
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seeds.SAMPLING_SEED).fit(Xs)
    P = pca.transform(Xs)
    for c, color in zip(sorted(set(ys)), CLASS_COLORS):
        m = ys == c
        ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.45, color=color, label=f"classe {c}")
    title(ax, ds_cfg.DATASETS[key]["label"])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(markerscale=3, fontsize=9)

finalize(
    fig,
    caption="Projection PCA 2D colorée par classe",
    output_path=run_dir / "figures" / "FIG-D4_PCA_2D.png",
)
plt.close(fig)
print("FIG-D4 enregistrée")
"""),

    ("code", """\
# FIG-D5 — Distance au plus proche voisin (justifie le seuil de redondance T3.1)
from matplotlib.ticker import MaxNLocator

N_REF, N_QRY = 2000, 400
rng = np.random.default_rng(seeds.SAMPLING_SEED)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

for ax, key in zip(axes, ds_cfg.ORDER):
    X_train, _, _, _, _, _ = adapters.get_split(key)
    X = X_train.astype(np.float32)
    n = len(X)
    ref = X[rng.choice(n, min(N_REF, n), replace=False)]
    qry = X[rng.choice(n, min(N_QRY, n), replace=False)]
    qn = np.einsum("ij,ij->i", qry, qry)[:, None]
    rn = np.einsum("ij,ij->i", ref, ref)[None, :]
    d2 = qn + rn - 2.0 * (qry @ ref.T)
    np.maximum(d2, 0, out=d2)
    nn_dist = np.sqrt(d2.min(axis=1))
    ax.hist(nn_dist, bins=40, color=DIM_COLOR, alpha=0.85)
    title(ax, f"{ds_cfg.DATASETS[key]['label']}\\nmédiane = {np.median(nn_dist):.3f}")
    ax.set_xlabel("Distance euclidienne au plus proche voisin")
    ax.set_ylabel("Requêtes")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

finalize(
    fig,
    caption="Distance au plus proche voisin",
    output_path=run_dir / "figures" / "FIG-D5_distance_nn.png",
)
plt.close(fig)
print("FIG-D5 enregistrée")
"""),

    ("code", """\
# FIG-D6 — Diabetes-130 : numériques vs one-hot (échelle cassée) + cardinalité
from sklearn.preprocessing import OneHotEncoder

df = adapters.load_dataframe("diabetes130")
cat_cols = [c for c in diabetes130.CATEGORICAL_COLS if c in df.columns]
num_cols = [c for c in diabetes130.NUMERIC_COLS if c in df.columns]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_arr = ohe.fit_transform(df[cat_cols].astype(str))
n_cat, n_num = cat_arr.shape[1], len(num_cols)

fig = plt.figure(figsize=(13, 4.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=(3, 1),
                      hspace=0.28, wspace=0.42)
ax_t = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])
ax_card = fig.add_subplot(gs[:, 1])

for ax in (ax_t, ax_b):
    ax.bar(["Num.", "One-hot"], [n_num, n_cat], color=[DIM_COLOR, CLASS_COLORS[1]], width=0.5)
    ax.grid(axis="x", visible=False)
ax_t.set_ylim(330, 400)
ax_b.set_ylim(0, 15)
if ax_t.yaxis.get_major_ticks():
    ax_t.yaxis.get_major_ticks()[0].label1.set_visible(False)
if ax_b.yaxis.get_major_ticks():
    ax_b.yaxis.get_major_ticks()[-1].label1.set_visible(False)
for ax, lo, hi in ((ax_t, 330, 400), (ax_b, 0, 15)):
    for i, v in enumerate([n_num, n_cat]):
        if lo <= v <= hi:
            ax.text(i, min(v + hi * 0.03, hi * 0.98), f"{v}", ha="center",
                    fontsize=10, clip_on=True)

d = 0.015
kw = dict(transform=ax_t.transAxes, color="#0F172A", lw=1.0, clip_on=False)
ax_t.plot([-d, d], [-d, d], **kw)
ax_t.plot([1 - d, 1 + d], [-d, d], **kw)
kw.update(transform=ax_b.transAxes)
ax_b.plot([-d, d], [1 - d, 1 + d], **kw)
ax_b.plot([1 - d, 1 + d], [1 - d, 1 + d], **kw)
ax_t.spines.bottom.set_visible(False)
ax_b.spines.top.set_visible(False)
ax_t.tick_params(labelbottom=False)

ax_b.set_ylabel("Nb features")
ax_b.set_xticklabels(["Numériques", "One-hot"], fontsize=9)
title(ax_b, f"Composition : {n_num} + {n_cat} = {n_num + n_cat} dims", fontsize=9)

card = pd.Series({c: int(df[c].nunique(dropna=False)) for c in cat_cols}).sort_values(ascending=False)
top = card.head(12)
ax_card.barh(np.arange(len(top))[::-1], top.values, color=DIM_COLOR)
ax_card.set_yticks(np.arange(len(top))[::-1], top.index, fontsize=8.5)
ax_card.set_xlabel("Modalités")
title(ax_card, "Cardinalité des 12 catégories les plus riches", fontsize=10)

finalize(
    fig,
    caption="Diabetes-130 : numériques vs one-hot",
    output_path=run_dir / "figures" / "FIG-D6_diabetes_composition.png",
)
plt.close(fig)
print("FIG-D6 enregistrée")
"""),

    ("md", """\
## Bilan NB01

- [x] Profil brut + profil train/test (`data_profile_raw.csv`, `data_profile.csv`)
- [x] FIG-D1 répartition des classes (échelles par panneau — barres petites visibles)
- [x] FIG-D2 dimensionnalité (échelle cassée : 13, 390, 1024 distincts)
- [x] FIG-D3 composition (tableau)
- [x] FIG-D4 PCA 2D par dataset
- [x] FIG-D5 distance au plus proche voisin (seuil de redondance)
- [x] FIG-D6 Diabetes-130 : numériques vs one-hot (échelle cassée)

Charte respectée : Space Grotesk (titres) + Manrope (corps), légende en bas,
couleurs conservées.

**Étape suivante** : NB02 — métriques & protocole de timing (T0.4-T0.6)."""),
]
