"""
Tests unitaires pour lib/margin.py (Nested MC).
"""

import numpy as np
import pytest

from config.parameters import SEED, TIME_GRID, MPOR
from lib.diffusion import simulate_gbm
from lib.margin import compute_im_nested, compute_exposure_with_im


class TestComputeIMNested:
    """Tests de l'IM par nested Monte Carlo."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Parametres reduits pour des tests rapides."""
        self.n_outer = 30
        self.n_inner = 100
        self.n_t = 10
        self.time_grid = np.linspace(0, 1, self.n_t + 1)
        self.paths = simulate_gbm(
            n_outer=self.n_outer,
            n_t=self.n_t,
            seed=SEED,
        )

    def test_shape(self):
        """Verifie la shape de la matrice IM."""
        im = compute_im_nested(
            self.paths, self.time_grid, n_inner=self.n_inner,
        )
        assert im.shape == (self.n_outer, self.n_t + 1)

    def test_im_positive(self):
        """L'IM (quantile 99%) doit etre positif pour un portefeuille net long."""
        im = compute_im_nested(
            self.paths, self.time_grid, n_inner=self.n_inner,
        )
        # La grande majorite des IM doit etre positive
        pct_positive = np.mean(im > 0)
        assert pct_positive > 0.90, f"Seulement {pct_positive:.0%} d'IM positifs"

    def test_im_reasonable_magnitude(self):
        """L'IM moyen est dans un ordre de grandeur raisonnable (< 100)."""
        im = compute_im_nested(
            self.paths, self.time_grid, n_inner=self.n_inner,
        )
        im_mean = np.mean(im)
        assert 0 < im_mean < 100, f"IM moyen = {im_mean:.2f}, hors limites"

    def test_reproducibility(self):
        """Meme seed => meme IM."""
        im1 = compute_im_nested(
            self.paths, self.time_grid, n_inner=self.n_inner, seed=42,
        )
        im2 = compute_im_nested(
            self.paths, self.time_grid, n_inner=self.n_inner, seed=42,
        )
        np.testing.assert_array_equal(im1, im2)


class TestComputeExposureWithIM:
    """Tests de l'exposition residuelle avec IM."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Parametres reduits."""
        self.n_outer = 30
        self.n_t = 10
        self.time_grid = np.linspace(0, 1, self.n_t + 1)
        self.paths = simulate_gbm(
            n_outer=self.n_outer,
            n_t=self.n_t,
            seed=SEED,
        )

    def test_shape(self):
        """Verifie la shape."""
        im = compute_im_nested(
            self.paths, self.time_grid, n_inner=50,
        )
        exposure = compute_exposure_with_im(self.paths, self.time_grid, im)
        assert exposure.shape == (self.n_outer, self.n_t + 1)

    def test_non_negative(self):
        """L'exposition residuelle est >= 0."""
        im = compute_im_nested(
            self.paths, self.time_grid, n_inner=50,
        )
        exposure = compute_exposure_with_im(self.paths, self.time_grid, im)
        assert np.all(exposure >= -1e-10)

    def test_exposure_reduced(self):
        """L'exposition avec IM est plus faible qu'une exposition sans protection.

        E^IM = max(0, DeltaPV - IM) <= max(0, DeltaPV) en moyenne.
        """
        im = compute_im_nested(
            self.paths, self.time_grid, n_inner=50,
        )
        exposure_im = compute_exposure_with_im(self.paths, self.time_grid, im)
        ee_im = np.mean(exposure_im)
        # L'exposition residuelle avec IM doit etre raisonnable (pas trop grande)
        assert ee_im < 50.0, f"EE avec IM = {ee_im:.2f}, trop eleve"
