"""Assemble les paquets à déposer sur Kaggle (datasets + notebooks).

Produit un dossier `kaggle/` (non versionné, cf. .gitignore) avec un sous-dossier
par slug Kaggle, conforme aux chemins déclarés dans config/datasets.py et
config/benchmark.py :

  kaggle/hecbr-exp2-code/        -> src/, config/, requirements.txt   (code exp2)
  kaggle/heart-disease/          -> processed.cleveland.data
  kaggle/diabetes130/            -> diabetic_data.csv
  kaggle/chestxray14-embeddings/ -> chestxray14_embeddings.parquet + labels.parquet
  kaggle/hecbr-diabetes130-codes/-> hecbr_diabetes130_codes_final.parquet
  kaggle/notebooks/<nb-slug>/    -> notebooks Kaggle prêts à pousser

Chaque dossier reçoit ses métadonnées Kaggle (`dataset-metadata.json` /
`kernel-metadata.json` avec les datasets attachés), ce qui rend l'upload déterministe :

  kaggle datasets create -p kaggle/<slug>          (ou -u pour mettre à jour)
  kaggle kernels push -p kaggle/notebooks/<nb-slug>

Les notebooks Kaggle (03, 05, 07) découvrent le code via /kaggle/input/* contenant
src/ + config/ (cellule EXP2) ; ils attachent donc tous `hecbr-exp2-code`.

Usage : python tools/stage_kaggle.py [--with-notebooks] [--username <kaggle user>]
"""

import argparse
import json
import shutil
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
STAGE = CODE_ROOT / "kaggle"

# slug -> liste de (destination, source relative) — sources lues depuis le repo
DATA_PAYLOADS = {
    "heart-disease": [
        ("processed.cleveland.data", "data/raw/heart_disease/processed.cleveland.data"),
    ],
    "diabetes130": [
        ("diabetic_data.csv", "data/raw/diabetes_130/diabetic_data.csv"),
    ],
    "chestxray14-embeddings": [
        ("chestxray14_embeddings.parquet", "data/processed/chestxray14_embeddings.parquet"),
        ("chestxray14_labels.parquet", "data/processed/chestxray14_labels.parquet"),
    ],
    "hecbr-diabetes130-codes": [
        ("hecbr_diabetes130_codes_final.parquet",
         "data/processed/hecbr_diabetes130_codes_final.parquet"),
    ],
}

CODE_PAYLOAD = "hecbr-exp2-code"

DATASET_TITLES = {
    "hecbr-exp2-code": "HeCBR Exp2 - code",
    "heart-disease": "Heart Disease (Cleveland)",
    "diabetes130": "Diabetes 130-US",
    "chestxray14-embeddings": "ChestX-ray14 embeddings DenseNet121",
    "hecbr-diabetes130-codes": "HeCBR Diabetes130 binary codes",
}

# slug notebook -> (fichier, datasets à attacher)
KAGGLE_NOTEBOOKS = {
    "nb03-benchmark-index-seuls": (
        "NB03_benchmark_index_seuls.ipynb",
        ["hecbr-exp2-code", "heart-disease", "diabetes130",
         "chestxray14-embeddings", "hecbr-diabetes130-codes"],
    ),
    "nb05-scaling": (
        "NB05_scaling.ipynb",
        ["hecbr-exp2-code", "heart-disease", "diabetes130", "chestxray14-embeddings"],
    ),
    "nb07-maintenance": (
        "NB07_maintenance.ipynb",
        ["hecbr-exp2-code", "heart-disease", "diabetes130",
         "chestxray14-embeddings", "hecbr-diabetes130-codes"],
    ),
}


def _copy_tree(src, dst, ignore_names=("__pycache__",)):
    def ignore(_dir, names):
        return {n for n in names if n in ignore_names or n.endswith(".pyc")}
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def stage_code_payload():
    dest = STAGE / CODE_PAYLOAD
    _copy_tree(CODE_ROOT / "exp2/src", dest / "src")
    _copy_tree(CODE_ROOT / "exp2/config", dest / "config")
    shutil.copy2(CODE_ROOT / "exp2/requirements.txt", dest / "requirements.txt")
    print(f"[code] {CODE_PAYLOAD}/ -> src/, config/, requirements.txt")


def stage_data_payloads():
    for slug, files in DATA_PAYLOADS.items():
        dest = STAGE / slug
        for name, rel in files:
            src = CODE_ROOT / rel
            if not src.exists():
                raise FileNotFoundError(f"source manquante: {src}")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / name)
        print(f"[data] {slug}/ <- " + ", ".join(f[0] for f in files))


def stage_notebooks():
    dest = STAGE / "notebooks"
    dest.mkdir(parents=True, exist_ok=True)
    for slug, (name, _) in KAGGLE_NOTEBOOKS.items():
        src = CODE_ROOT / "exp2/notebooks" / name
        if not src.exists():
            raise FileNotFoundError(f"notebook manquant: {src}")
        shutil.copy2(src, dest / name)
        print(f"[nb] notebooks/{name}")


def write_metadata(username):
    """Écrit dataset-metadata.json / kernel-metadata.json (upload déterministe)."""
    for slug, title in DATASET_TITLES.items():
        meta = {
            "id": f"{username}/{slug}",
            "title": title,
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        }
        (STAGE / slug / "dataset-metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[meta] {slug}/dataset-metadata.json")

    for nb_slug, (name, datasets) in KAGGLE_NOTEBOOKS.items():
        folder = STAGE / "notebooks" / nb_slug
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STAGE / "notebooks" / name, folder / name)
        meta = {
            "id": f"{username}/{nb_slug}",
            "title": f"{name.rsplit('.', 1)[0].replace('_', ' ')}",
            "code_file": name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "competition_sources": [],
            "dataset_sources": [f"{username}/{d}" for d in datasets],
            "kernel_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[meta] notebooks/{nb_slug}/kernel-metadata.json")


def write_manifest():
    lines = [
        "# Manifeste Kaggle — expérience 2 (hachage + maintenance)",
        "",
        "Chaque dossier `kaggle/<slug>/` est un dataset Kaggle à créer (même slug).",
        "Commandes (CLI kaggle, depuis la racine du dépôt) :",
        "  kaggle datasets create -p kaggle/<slug>                     # 1re création",
        "  kaggle datasets create -p kaggle/hecbr-exp2-code --dir-mode tar  # code (src/, config/)",
        "  kaggle datasets version -p kaggle/<slug> -m '<message>' -r tar  # mise à jour (CLI 2.2.x : create -u refusé si existe)",
        "  kaggle kernels push -p kaggle/notebooks/<nb-slug>          # notebook",
        "  kaggle kernels push -r ... / kernels run ...               # (non dispo en 2.2.x : lancer depuis l'UI)",
        "",
        "| Slug | Contenu | Utilisé par |",
        "|---|---|---|",
        f"| {CODE_PAYLOAD} | src/, config/, requirements.txt | Tous les notebooks (découverte EXP2) |",
        "| heart-disease | processed.cleveland.data | NB03, NB05, NB07, NB09 |",
        "| diabetes130 | diabetic_data.csv | NB03, NB05, NB07, NB09 |",
        "| chestxray14-embeddings | embeddings + labels parquet | NB03, NB05, NB07, NB09 |",
        "| hecbr-diabetes130-codes | codes binaires parquet | NB03 (HeCBR), NB07 |",
        "",
        "Notebooks Kaggle : kaggle/notebooks/<nb-slug>/ — poussés avec leurs datasets"
        "attachés (dataset_sources dans kernel-metadata.json). Sorties écrites dans "
        "`/kaggle/working` (checkpoint CSV incrémental, reprise sans doublon).",
        "",
        "Montage des datasets (Kaggle) : layouts `.../input/<slug>` (ancien) et "
        "`.../input/datasets/<owner>/<slug>` (nouveau) — résolus automatiquement par "
        "`config.paths.kaggle_input` et la cellule de découverte EXP2 des notebooks.",
        "",
        "Variables de réduction (optionnelles, sanity locale uniquement) : NB03_SEEDS, "
        "NB03_QUERY_CAP (NB05_SEEDS/QUERY_CAP, NB07_SEEDS/QUERY_CAP/TRAIN_CAP/SUBSAMPLE). "
        "Sur Kaggle, exécution complète par défaut (6 seeds, n_queries du protocole).",
        "",
        "Chemin critique (plan §6) : NB03 + NB05 + NB07 en parallèle sur Kaggle, "
        "téléchargement des CSV, puis NB04/06/08 (figures locales).",
    ]
    (STAGE / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print("[manifest] kaggle/MANIFEST.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-notebooks", action="store_true",
                        help="prépare aussi les notebooks dans kaggle/notebooks/")
    parser.add_argument("--username", default=None,
                        help="nom d'utilisateur Kaggle (métadonnées id=user/slug)")
    args = parser.parse_args()

    STAGE.mkdir(parents=True, exist_ok=True)
    stage_code_payload()
    stage_data_payloads()
    if args.with_notebooks:
        stage_notebooks()
    if args.username:
        write_metadata(args.username)
    write_manifest()
    print(f"\nTerminé. Paquets dans : {STAGE}")


if __name__ == "__main__":
    main()
