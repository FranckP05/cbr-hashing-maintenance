"""Tests de validation du découpage des données — absence de fuite par lame."""

from __future__ import annotations

import numpy as np
import pytest


# Données synthétiques de lames pour les tests
def make_fake_slides(n_slides: int = 60, n_patches_per_slide: int = 100, dim: int = 8):
    """Génère des lames synthétiques avec labels alternés 0/1."""
    slide_data   = {}
    slide_labels = {}
    for i in range(n_slides):
        sid = f"slide_{i:04d}"
        slide_data[sid]   = np.random.rand(n_patches_per_slide, dim).astype(np.float32)
        slide_labels[sid] = i % 2   # alternance normal/tumeur
    return slide_data, slide_labels


class TestSplitter:
    """Tests sur la fonction split_slides."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from data.loader import get_slide_ids
        from data.splitter import split_slides

        self.split_slides  = split_slides
        self.get_slide_ids = get_slide_ids

        self.slide_data, self.slide_labels = make_fake_slides(n_slides=60)
        self.slide_ids, _ = self.get_slide_ids(self.slide_labels)

    def test_no_overlap_base_test(self):
        """Aucune lame ne doit appartenir simultanément à la base et au test."""
        base_ids, val_ids, test_ids = self.split_slides(self.slide_ids, self.slide_labels)
        assert set(base_ids) & set(test_ids) == set(), "FUITE : base ∩ test ≠ ∅"

    def test_no_overlap_val_test(self):
        """Aucune lame ne doit appartenir simultanément à la validation et au test."""
        base_ids, val_ids, test_ids = self.split_slides(self.slide_ids, self.slide_labels)
        assert set(val_ids) & set(test_ids) == set(), "FUITE : val ∩ test ≠ ∅"

    def test_no_overlap_base_val(self):
        """Aucune lame ne doit appartenir simultanément à la base effective et à la validation."""
        base_ids, val_ids, test_ids = self.split_slides(self.slide_ids, self.slide_labels)
        assert set(base_ids) & set(val_ids) == set(), "FUITE : base ∩ val ≠ ∅"

    def test_all_slides_covered(self):
        """L'union des trois sous-ensembles doit couvrir l'intégralité des lames."""
        base_ids, val_ids, test_ids = self.split_slides(self.slide_ids, self.slide_labels)
        all_split = set(base_ids) | set(val_ids) | set(test_ids)
        assert all_split == set(self.slide_ids), "Des lames sont perdues dans le split."

    def test_test_ratio_approx(self):
        """L'ensemble de test doit représenter environ 10% des lames."""
        base_ids, val_ids, test_ids = self.split_slides(self.slide_ids, self.slide_labels)
        ratio = len(test_ids) / len(self.slide_ids)
        assert 0.05 <= ratio <= 0.20, f"Ratio test inattendu : {ratio:.2f}"

    def test_class_stratification(self):
        """Les deux classes doivent être représentées dans chaque sous-ensemble."""
        base_ids, val_ids, test_ids = self.split_slides(self.slide_ids, self.slide_labels)
        for name, ids in [("base", base_ids), ("val", val_ids), ("test", test_ids)]:
            labels = set(self.slide_labels[sid] for sid in ids)
            assert 0 in labels and 1 in labels, \
                f"Classe manquante dans '{name}' : classes présentes = {labels}"

    def test_determinism(self):
        """Le split doit être déterministe avec la même graine."""
        b1, v1, t1 = self.split_slides(self.slide_ids, self.slide_labels, seed=42)
        b2, v2, t2 = self.split_slides(self.slide_ids, self.slide_labels, seed=42)
        assert sorted(b1) == sorted(b2)
        assert sorted(t1) == sorted(t2)
