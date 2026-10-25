"""Tests pour HeCBR (Réseau et Index Hamming)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.hecbr_net import HeCBRNet, ContrastiveHashingLoss
from indexes.hecbr_index import HeCBRIndex


def test_hecbr_net_forward_and_binarize():
    net = HeCBRNet(in_dim=32, hidden_dim=64, bit_length=64)
    x = torch.randn(10, 32)
    u = net(x)
    assert u.shape == (10, 64)
    assert torch.all(u >= -1.0) and torch.all(u <= 1.0)

    bits = net.encode_binary(x.numpy())
    assert bits.shape == (10, 64)
    assert bits.dtype == bool


def test_hecbr_loss():
    criterion = ContrastiveHashingLoss(margin=0.5, quant_weight=0.01)
    u = torch.randn(8, 64)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    loss = criterion(u, labels)
    assert loss.item() > 0.0


def test_hecbr_index_query():
    rng = np.random.default_rng(42)
    codes = rng.choice([True, False], size=(100, 64))
    query = codes[0].copy()

    index = HeCBRIndex(bit_length=64)
    index.build(codes)

    idx, dists = index.query(query, k=5)
    assert idx[0] == 0
    assert dists[0] == 0.0
