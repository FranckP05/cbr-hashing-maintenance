"""Réseau de hachage profond HeCBR (Deep Hashing Network en PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HeCBRNet(nn.Module):
    """Réseau de projection non linéaire 512-D vers code binaire compact B-bits."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, bit_length: int = 64) -> None:
        super().__init__()
        self.in_dim     = in_dim
        self.hidden_dim = hidden_dim
        self.bit_length = bit_length

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, bit_length),
            nn.Tanh(),  # Sortie continue dans [-1, 1]^B
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la représentation continue u in [-1, 1]^B."""
        return self.encoder(x)

    def encode_binary(self, x: torch.Tensor | np.ndarray, device: str = "cpu") -> np.ndarray:
        """Encode un lot de vecteurs en codes binaires booléens (N, B)."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x).float().to(device)
            else:
                t = x.to(device)
            u = self.forward(t)
            bits = (u >= 0).cpu().numpy()
        return bits


class ContrastiveHashingLoss(nn.Module):
    """Perte conjointe de similarité sémantique (Cosine/Contrastive) et de quantification."""

    def __init__(self, margin: float = 0.5, quant_weight: float = 0.01) -> None:
        super().__init__()
        self.margin = margin
        self.quant_weight = quant_weight

    def forward(self, u: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u      : (batch_size, B) sorties continues dans [-1, 1].
            labels : (batch_size,) labels de classe (0 ou 1).
        """
        # Matrice de similarité des labels : 1 si même classe, 0 sinon
        labels = labels.view(-1, 1)
        S = (labels == labels.T).float()  # (batch_size, batch_size)

        # Distance cosinus normalisée entre les représentations continues
        u_norm = F.normalize(u, p=2, dim=1)
        cos_sim = torch.mm(u_norm, u_norm.T)

        # Perte contrastive : maximiser cos_sim si même classe, minimiser au-delà de la marge sinon
        pos_loss = S * (1.0 - cos_sim)
        neg_loss = (1.0 - S) * F.relu(cos_sim - self.margin)
        sim_loss = (pos_loss + neg_loss).mean()

        # Perte de quantification : force |u_j| vers 1 pour éliminer l'erreur de binarisation
        quant_loss = torch.mean((torch.abs(u) - 1.0) ** 2)

        return sim_loss + self.quant_weight * quant_loss
