"""Refonte v4 : hachage à préservation de VOISINAGE L2 (et non de classes).

Retour sur méthodo : les patches Camelyon sont faiblement séparables par CLASSE
(F1<=0.6 même MLP), donc le contraste supervisé par classe ne trouve pas de signal
et produit soit des codes effondrés, soit des codes aléatoires (Recall Hamming 0).

Cible juste = préserver la STRUCTURE DE VOISINAGE L2 réelle (CBR / Case-Based):
   - paires POSITIVES  = voisins L2 (K-plus proches)
   - paires NÉGATIVES  = non-voisins L2
   Perte de tri/contraste sur la distance de Hamming des codes, + décentration
   (anti-effondrement) + quantification progressive.

Étapes :
   1. Échantillon de M points ; calcul de la matrice de similarité cosinus (BLAS).
   2. Construction de paires positives (voisins) et négatives (lointains).
   3. Entraînement du réseau à préserver ces relations en Hamming.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/teamspace/studios/this_studio/hecbr_sota_camelyon")
from data.loader import load_slides
from data.splitter import split_slides, build_split_matrices
from config.seeds import SPLIT_SEED
from config.paths import HECBR_MODEL_PATH


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


def pair_ranking_loss(u, a, p, n, margin=1.0):
    """Perte de tri : dist_H(a,p) < dist_H(a,n) - margin, sur continus (u).

    Approximation différentiable de la distance Hamming via les sorties continues :
    d(a,b) ~ (B - u_a . u_b) / 2, avec u normalisé. Maximiser le produit des paires
    positives, minimiser celui des paires négatives.
    """
    ua = F.normalize(u[a], p=2, dim=1)
    up = F.normalize(u[p], p=2, dim=1)
    un = F.normalize(u[n], p=2, dim=1)
    pos = (ua * up).sum(1)          # plus haut = plus proche (voulu)
    neg = (ua * un).sum(1)          # plus bas = plus loin (voulu)
    loss = F.relu(neg - pos + margin).mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--Nneg", type=int, default=8)
    ap.add_argument("--decenter", type=float, default=1.0)
    ap.add_argument("--quant0", type=float, default=0.0001)
    ap.add_argument("--quant1", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=str(HECBR_MODEL_PATH))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sd, sl = load_slides()
    sids = sorted(sl.keys())
    b0, bv, bt = split_slides(sids, sl, seed=SPLIT_SEED)
    Xb, yb, _, _, Xt, yt = build_split_matrices(
        base_ids=b0 + bv, val_ids=[], test_ids=bt,
        slide_data=sd, slide_labels=sl, normalize_l2=True)

    # échantillon training M
    idx = rng.choice(np.arange(len(Xb)), args.M, replace=False)
    Xm = Xb[idx]
    mu = Xm.mean(0); sd_ = Xm.std(0) + 1e-6
    def norm(X): return (X - mu) / sd_

    print("calcul similarité cosinus %dx%d (BLAS)..." % (args.M, args.M))
    t0 = time.perf_counter()
    C = Xm @ Xm.T  # cosinus car L2-normalisé
    C = C - np.eye(args.M)
    # paires positives = top-K voisins (colonnes par ligne)
    t0 = time.perf_counter()
    pos_idx = np.argpartition(-C, args.K, axis=1)[:, :args.K]
    # scores
    pos_scores = C[np.arange(args.M)[:, None], pos_idx]
    # négatives : échantillonner loin (basse similarité) : prendre combinaisons de bas de C
    # on échantillonne Nneg non-voisins par point parmi les ~M- (K*3) plus éloignés
    low_range = args.M - args.K * 3
    neg_idx = np.zeros((args.M, args.Nneg), dtype=np.int64)
    for j in range(args.Nneg):
        # tire partout sauf très proche
        cand = np.sort(rng.choice(np.arange(low_range), args.M))  # placeholder
    # Méthode plus simple : négatifs = points de basse similarité échantillonnés
    order_lo = np.argpartition(C, low_range - 1, axis=1)[:, :low_range]
    for j in range(args.Nneg):
        jj = rng.integers(0, low_range, args.M)
        neg_idx[:, j] = order_lo[np.arange(args.M), jj]
    print("paires construites en %.1fs" % (time.perf_counter() - t0))

    # préparation tenseurs
    Xn = torch.tensor(norm(Xm), dtype=torch.float32)
    A = torch.arange(args.M)
    neg_idx_t = torch.tensor(neg_idx, dtype=torch.long)
    pos_idx_t = torch.tensor(pos_idx, dtype=torch.long)

    model = HashNetV3(512, 512, 64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_steps = args.M // args.batch

    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        qw = args.quant0 + (args.quant1 - args.quant0) * (ep / max(1, args.epochs - 1))
        perm = torch.randperm(args.M)
        total = 0.0
        for s in range(n_steps):
            a = perm[s * args.batch:(s + 1) * args.batch]
            kk = torch.randint(0, args.K, (len(a),))
            jj = torch.randint(0, args.Nneg, (len(a),))
            p = pos_idx_t[a, kk]
            n = neg_idx_t[a, jj]
            b_idx = torch.cat([a, p, n]).unique()
            bx = Xn[b_idx].to(device)
            u = model(bx)
            loc = {int(v): i for i, v in enumerate(b_idx.tolist())}
            al = torch.tensor([loc[int(v)] for v in a.tolist()], device=device)
            pl = torch.tensor([loc[int(v)] for v in p.tolist()], device=device)
            nl = torch.tensor([loc[int(v)] for v in n.tolist()], device=device)
            opt.zero_grad()
            rl = pair_ranking_loss(u, al, pl, nl, margin=args.margin)
            dec = u.mean(dim=0).abs().mean()
            quant = ((u.abs() - 1.0) ** 2).mean()
            loss = rl + args.decenter * dec + qw * quant
            loss.backward(); opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            ue = model(Xn.to(device))
        mean_abs = ue.mean(0).abs().mean().item()
        print("ep%02d loss=%.4f rank=%.4f dec=%.3f quant=%.3f |bit|mean|=%.3f" % (ep, total / n_steps, rl.item(), dec.item(), quant.item(), mean_abs))
    print("train time %.1fs" % (time.perf_counter() - t0))

    model.eval()
    with torch.no_grad():
        U = model(Xn.to(device)); bits = (U >= 0).cpu().numpy()
    uniq = len(np.unique(bits, axis=0)); bf = bits.mean(0)
    print("codes uniques: %d/%d | bit_frac min=%.3f max=%.3f" % (uniq, args.M, bf.min(), bf.max()))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "in_dim": 512, "hidden_dim": 512, "bit_length": 64}, args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
