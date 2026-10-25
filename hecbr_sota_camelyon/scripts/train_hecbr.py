"""Point d'entrée CLI : Entraînement supervisé du modèle Deep Hashing HeCBR."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import HECBR_MODEL_PATH
from config.protocol import HECBR_BIT_LENGTH, HECBR_EPOCHS, HECBR_BATCH_SIZE, HECBR_LR
from config.seeds import SPLIT_SEED
from data.loader import load_slides, get_slide_ids
from data.splitter import split_slides, build_split_matrices
from models.train_hecbr import train_hecbr_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_hecbr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement du modèle HeCBR (Deep Hashing).")
    p.add_argument("--bits", type=int, default=HECBR_BIT_LENGTH, help="Nombre de bits (défaut: 64).")
    p.add_argument("--epochs", type=int, default=HECBR_EPOCHS, help="Nombre d'époques (défaut: 15).")
    p.add_argument("--batch-size", type=int, default=HECBR_BATCH_SIZE, help="Taille des batchs (défaut: 1024).")
    p.add_argument("--lr", type=float, default=HECBR_LR, help="Learning rate (défaut: 0.001).")
    p.add_argument("--max-train-samples", type=int, default=300_000, help="Échantillon max pour l'entraînement rapide.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== 1. Chargement des données pour l'entraînement HeCBR ===")
    slide_data, slide_labels = load_slides()
    slide_ids, _ = get_slide_ids(slide_labels)

    base_ids, val_ids, test_ids = split_slides(slide_ids, slide_labels, seed=SPLIT_SEED)
    _, _, X_val, y_val, _, _ = build_split_matrices(
        base_ids=[], val_ids=val_ids, test_ids=None,
        slide_data=slide_data, slide_labels=slide_labels, normalize_l2=True
    )
    del slide_data

    # Sous-échantillonnage pour un entraînement rapide (~10-15 min)
    if args.max_train_samples and X_val.shape[0] > args.max_train_samples:
        rng = np.random.default_rng(SPLIT_SEED)
        idx = rng.choice(X_val.shape[0], args.max_train_samples, replace=False)
        X_train = X_val[idx]
        y_train = y_val[idx]
    else:
        X_train = X_val
        y_train = y_val

    logger.info("Données d'entraînement : %d patches (512-D), %d tumoraux, %d normaux.",
                X_train.shape[0], int(np.sum(y_train == 1)), int(np.sum(y_train == 0)))

    logger.info("=== 2. Lancement de l'entraînement HeCBR (%d bits) ===", args.bits)
    model = train_hecbr_model(
        X_train=X_train,
        y_train=y_train,
        bit_length=args.bits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=HECBR_MODEL_PATH,
    )
    logger.info("=== Modèle HeCBR entraîné et prêt à l'emploi dans %s ===", HECBR_MODEL_PATH)


if __name__ == "__main__":
    main()
