"""Refonte v3 : entraînement du hachage profond HeCBR — objectif anti-effondrement robuste.

Cible réaliste (plafond F1~0.6 par patch) : produire des codes NON-EFFONDRÉS,
bit-équilibrés, dont les voisins Hamming recoupent les voisins L2, pour rendre
le multi-probe efficace et la structure de voisinage cohérente.

Mécanismes anti-effondrement :
  1. Standardisation z-score de l'entrée (dynamique stable).
  2. Batchs garantis 50/50 par classe (WeightedRandomSampler).
  3. Supervised Contrastive (InfoNCE) sur les embeddings continus.
  4. Régularisation de DÉCENTRATION des bits (force la moyenne de chaque bit -> 0).
  5. Quantification progressive (poids montant au fil des époques).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH
from models.hecbr_net import HeCBRNet


class HashNetV3(nn.Module):
    def __init__(self, in_dim=512, hidden=512, bit_length=64):
        super().__init__()
        self.in_dim = in_dim
        self.bit_length = bit_length
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, bit_length),
        )

    def forward(self, x):
        return torch.tanh(self.encoder(x))


def supervised_contrastive_loss(u_norm, labels, temperature=0.15):
    """InfoNCE supervisé : tout point de même classe est un positif (moyenné)."""
    labels = labels.view(-1, 1)
    mask = (labels == labels.T).float()              # (B,B)
    mask.fill_diagonal_(0.0)
    sim = torch.mm(u_norm, u_norm.T) / temperature
    exp = torch.exp(sim)
    exp = exp * (1.0 - torch.eye(len(u_norm), device=u_norm.device))
    denom = exp.sum(dim=1, keepdim=True) + 1e-9
    pos = (mask * sim).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
    loss = -pos + torch.log(denom.squeeze(1))
    return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temp", type=float, default=0.15)
    ap.add_argument("--decenter", type=float, default=1.0)
    ap.add_argument("--quant0", type=float, default=0.0001)
    ap.add_argument("--quant1", type=float, default=0.2)
    ap.add_argument("--save", type=str, default=str(HECBR_MODEL_PATH))
    ap.add_argument("--n_train", type=int, default=120000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    sd, sl = load_slides()
    sids = sorted(sl.keys())
    b0, bv, bt = split_slides(sids, sl, seed=SPLIT_SEED)
    Xb, yb, _, _, Xt, yt = build_split_matrices(
        base_ids=b0 + bv, val_ids=[], test_ids=bt,
        slide_data=sd, slide_labels=sl, normalize_l2=True)

    rng = np.random.default_rng(0)
    tr = np.concatenate([
        rng.choice(np.flatnonzero(yb == 1), args.n_train // 2, replace=False),
        rng.choice(np.flatnonzero(yb == 0), args.n_train // 2, replace=False),
    ])
    rng.shuffle(tr)
    te = np.concatenate([
        rng.choice(np.flatnonzero(yt == 1), 3000, replace=False),
        rng.choice(np.flatnonzero(yt == 0), 3000, replace=False),
    ])
    rng.shuffle(te)

    Xtr = torch.tensor(Xb[tr], dtype=torch.float32)
    ytr = torch.tensor(yb[tr], dtype=torch.long)
    Xte = torch.tensor(Xt[te], dtype=torch.float32)
    yte = torch.tensor(yt[te], dtype=torch.long)

    mu = Xtr.mean(0); sd_ = Xtr.std(0) + 1e-6
    def norm(X): return (X - mu) / sd_
    Xtr = norm(Xtr); Xte = norm(Xte)

    model = HashNetV3(in_dim=512, hidden=512, bit_length=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_batches = len(Xtr) // args.batch

    # repos d'échantillonnage équilibré
    n_half = args.batch // 2
    idx1 = torch.nonzero(ytr == 1).squeeze(1)
    idx0 = torch.nonzero(ytr == 0).squeeze(1)

    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        qw = args.quant0 + (args.quant1 - args.quant0) * (ep / max(1, args.epochs - 1))
        total = 0.0
        for _ in range(n_batches):
            p1 = idx1[torch.randint(0, len(idx1), (n_half,))]
            p0 = idx0[torch.randint(0, len(idx0), (n_half,))]
            perm = torch.randperm(args.batch)
            b_idx = torch.cat([p1, p0])[perm]
            bx, by = Xtr[b_idx].to(device), ytr[b_idx].to(device)
            opt.zero_grad()
            u = model(bx)
            u_norm = F.normalize(u, p=2, dim=1)
            sc = supervised_contrastive_loss(u_norm, by, temperature=args.temp)
            dec = u.mean(dim=0).abs().mean()
            quant = ((u.abs() - 1.0) ** 2).mean()
            loss = sc + args.decenter * dec + qw * quant
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            ue = model(Xte.to(device))
            bits = (ue >= 0).cpu().numpy()
            pr = torch.tensor([np.bincount(r, minlength=2).argmax() if np.bincount(r, minlength=2).sum()>0 else 0 for r in bits])
            f1 = f1_score(yte.numpy(), pr.numpy(), average="macro")
        mean_abs = ue.detach().cpu().mean(0).abs().mean().item()
        print("ep%02d loss=%.4f sc=%.3f dec=%.3f quant=%.3f | bit|mean|=%.3f F1=%.4f" % (ep, total / n_batches, sc.item(), dec.item(), quant.item(), mean_abs, f1))
    print("train time", round(time.perf_counter() - t0, 1), "s")

    # éval code sur validation : uniques + équilibre
    model.eval()
    with torch.no_grad():
        U = model(Xte.to(device))
    bits = (U >= 0).cpu().numpy()
    uniq = len(np.unique(bits, axis=0))
    bit_frac = bits.mean(0)
    print("codes uniques test: %d/%d | fraction bits=1 min=%.3f max=%.3f" % (uniq, len(bits), bit_frac.min(), bit_frac.max()))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "in_dim": 512, "hidden_dim": 512, "bit_length": 64}, args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
