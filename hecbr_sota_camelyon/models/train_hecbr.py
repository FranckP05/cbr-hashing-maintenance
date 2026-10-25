"""Entraînement supervisé du réseau HeCBR sur les embeddings de patches Camelyon16."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.protocol import (
    HECBR_BIT_LENGTH, HECBR_HIDDEN_DIM, HECBR_EPOCHS,
    HECBR_BATCH_SIZE, HECBR_LR, HECBR_QUANT_WEIGHT,
)
from config.paths import HECBR_MODEL_PATH
from models.hecbr_net import HeCBRNet, ContrastiveHashingLoss

logger = logging.getLogger(__name__)


def train_hecbr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bit_length: int = HECBR_BIT_LENGTH,
    hidden_dim: int = HECBR_HIDDEN_DIM,
    epochs: int = HECBR_EPOCHS,
    batch_size: int = HECBR_BATCH_SIZE,
    lr: float = HECBR_LR,
    quant_weight: float = HECBR_QUANT_WEIGHT,
    save_path: Path = HECBR_MODEL_PATH,
    device: str | None = None,
) -> HeCBRNet:
    """Entraîne et sauvegarde le modèle de hachage profond HeCBR.

    Args:
        X_train    : (N, 512) float32 normalisé L2.
        y_train    : (N,) labels entiers.
        bit_length : nombre de bits (ex: 64).
        save_path  : destination du fichier .pt.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Entraînement HeCBR sur %s (%d exemples, bits=%d) ...", device, X_train.shape[0], bit_length)

    model = HeCBRNet(in_dim=X_train.shape[1], hidden_dim=hidden_dim, bit_length=bit_length).to(device)
    criterion = ContrastiveHashingLoss(margin=0.5, quant_weight=quant_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Création du DataLoader
    tensor_x = torch.from_numpy(X_train).float()
    tensor_y = torch.from_numpy(y_train).long()
    dataset   = TensorDataset(tensor_x, tensor_y)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            u = model(bx)
            loss = criterion(u, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        if epoch % 3 == 0 or epoch == epochs:
            logger.info("  Époque [%2d/%2d] — Loss: %.4f (lr=%.5f)", epoch, epochs, avg_loss, scheduler.get_last_lr()[0])

    elapsed = time.perf_counter() - t0
    logger.info("Entraînement HeCBR terminé en %.2f s.", elapsed)

    # Sauvegarde du modèle
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X_train.shape[1],
        "hidden_dim": hidden_dim,
        "bit_length": bit_length,
    }, save_path)
    logger.info("Modèle sauvegardé dans %s", save_path)

    return model


def load_hecbr_model(model_path: Path = HECBR_MODEL_PATH, device: str = "cpu") -> HeCBRNet:
    """Charge un modèle HeCBR pré-entraîné."""
    checkpoint = torch.load(model_path, map_location=device)
    model = HeCBRNet(
        in_dim=checkpoint.get("in_dim", 512),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        bit_length=checkpoint.get("bit_length", 64),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
