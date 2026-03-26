"""
Tests unitaires pour lib/portfolio.py.
"""

import numpy as np
import pytest

from config.parameters import SPOTS, VOLS, RISK_FREE_RATE, PORTFOLIO, TIME_GRID
from lib.portfolio import compute_mtm, compute_mtm_full
from lib.black_scholes import bs_call, bs_put
from lib.diffusion import simulate_gbm


class TestComputeMtM:
    """Tests du calcul MtM."""

    def test_mtm_t0_constant(self):
        """A t=0, tous les scenarios ont les memes spots => MtM constant."""
        n_outer = 50
        spots_t0 = np.tile(SPOTS, (n_outer, 1))
        mtm = compute_mtm(spots_t0, t_j=0.0)
        # Tous les scenarios doivent avoir le meme MtM
        assert np.std(mtm) < 1e-10

    def test_mtm_t0_value(self):
        """Verifie le MtM a t=0 par calcul manuel."""
        spots_t0 = SPOTS[np.newaxis, :]  # (1, 3)
        mtm = compute_mtm(spots_t0, t_j=0.0)

        # Calcul manuel
        r = RISK_FREE_RATE
        expected = 0.0
        for opt in PORTFOLIO:
            S = SPOTS[opt["asset_idx"]]
            K = opt["strike"]
            tau = opt["maturity"]  # T_opt - 0 = T_opt
            sigma = VOLS[opt["asset_idx"]]
            if opt["type"] == "call":
                price = bs_call(np.array([S]), K, np.array([tau]), r, sigma)[0]
            else:
                price = bs_put(np.array([S]), K, np.array([tau]), r, sigma)[0]
            expected += opt["position"] * price

        assert abs(mtm[0] - expected) < 1e-10

    def test_mtm_positive_at_t0(self):
        """Le MtM initial du portefeuille est positif (net long)."""
        spots_t0 = SPOTS[np.newaxis, :]
        mtm = compute_mtm(spots_t0, t_j=0.0)
        assert mtm[0] > 0

    def test_mtm_shape(self):
        """Verifie la shape du MtM."""
        n_outer = 100
        spots = np.tile(SPOTS, (n_outer, 1))
        mtm = compute_mtm(spots, t_j=0.5)
        assert mtm.shape == (n_outer,)

    def test_mtm_expired_option(self):
        """Apres expiration, la valeur est intrinseque."""
        # Avec t_j = 2.5 > T_opt = 2.0, toutes les options sont expirees
        n_outer = 10
        spots = np.tile(SPOTS, (n_outer, 1)) * 1.5  # Spots eleves
        mtm = compute_mtm(spots, t_j=2.5)

        # Calcul manuel : valeur intrinseque
        expected = 0.0
        for opt in PORTFOLIO:
            S = SPOTS[opt["asset_idx"]] * 1.5
            K = opt["strike"]
            if opt["type"] == "call":
                intrinsic = max(S - K, 0.0)
            else:
                intrinsic = max(K - S, 0.0)
            expected += opt["position"] * intrinsic

        np.testing.assert_allclose(mtm[0], expected, atol=1e-10)


class TestComputeMtMFull:
    """Tests du calcul MtM sur toute la grille temporelle."""

    def test_shape(self):
        """Verifie la shape de la matrice MtM."""
        paths = simulate_gbm(n_outer=50, n_t=52, seed=42)
        mtm = compute_mtm_full(paths, TIME_GRID)
        assert mtm.shape == (50, 53)

    def test_t0_column_constant(self):
        """La colonne t=0 est constante."""
        paths = simulate_gbm(n_outer=50, n_t=52, seed=42)
        mtm = compute_mtm_full(paths, TIME_GRID)
        assert np.std(mtm[:, 0]) < 1e-10

    def test_consistency_with_single(self):
        """compute_mtm_full est coherent avec compute_mtm appele date par date."""
        paths = simulate_gbm(n_outer=20, n_t=10, seed=42)
        time_grid = np.linspace(0, 1, 11)
        mtm_full = compute_mtm_full(paths, time_grid)

        for j, t_j in enumerate(time_grid):
            mtm_j = compute_mtm(paths[:, j, :], t_j)
            np.testing.assert_allclose(mtm_full[:, j], mtm_j, atol=1e-12)
