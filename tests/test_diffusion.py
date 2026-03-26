"""
Tests unitaires pour lib/diffusion.py.
"""

import numpy as np
import pytest

from config.parameters import (
    SEED, RISK_FREE_RATE, SPOTS, VOLS, CORRELATION_MATRIX,
    N_T, DELTA_T, MPOR,
)
from lib.diffusion import simulate_gbm, simulate_gbm_from_spot, cholesky_decomposition


class TestCholesky:
    """Tests de la decomposition de Cholesky."""

    def test_decomposition_valid(self):
        """Cholesky de la matrice de correlation du projet."""
        L = cholesky_decomposition(CORRELATION_MATRIX)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, CORRELATION_MATRIX, atol=1e-12)

    def test_lower_triangular(self):
        """La matrice L est bien triangulaire inferieure."""
        L = cholesky_decomposition(CORRELATION_MATRIX)
        assert np.allclose(L, np.tril(L))

    def test_not_positive_definite(self):
        """Matrice non definie positive => erreur."""
        bad_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(np.linalg.LinAlgError):
            cholesky_decomposition(bad_matrix)


class TestSimulateGBM:
    """Tests de la simulation GBM multidimensionnelle."""

    def test_shape(self):
        """Verifie la shape des chemins simules."""
        n_outer, n_t = 100, 52
        paths = simulate_gbm(n_outer=n_outer, n_t=n_t)
        assert paths.shape == (n_outer, n_t + 1, len(SPOTS))

    def test_initial_conditions(self):
        """Tous les chemins demarrent au spot initial."""
        paths = simulate_gbm(n_outer=50, n_t=10)
        for i in range(50):
            np.testing.assert_allclose(paths[i, 0, :], SPOTS)

    def test_positive_prices(self):
        """Les prix simules sont strictement positifs (GBM)."""
        paths = simulate_gbm(n_outer=200, n_t=52)
        assert np.all(paths > 0)

    def test_risk_neutral_drift(self):
        """Sous Q, E[S(T)] = S0 * exp(rT) a 1% pres (N_outer=5000)."""
        n_outer = 5000
        n_t = 52
        paths = simulate_gbm(n_outer=n_outer, n_t=n_t)
        T = n_t * DELTA_T
        expected = SPOTS * np.exp(RISK_FREE_RATE * T)
        realized = np.mean(paths[:, -1, :], axis=0)
        rel_error = np.abs(realized - expected) / expected
        assert np.all(rel_error < 0.01), f"Erreur drift: {rel_error}"

    def test_volatility(self):
        """La volatilite empirique est coherente avec sigma (a 5% pres)."""
        n_outer = 5000
        n_t = 52
        paths = simulate_gbm(n_outer=n_outer, n_t=n_t)
        T = n_t * DELTA_T
        log_returns = np.log(paths[:, -1, :] / SPOTS)
        realized_vol = np.std(log_returns, axis=0) / np.sqrt(T)
        rel_error = np.abs(realized_vol - VOLS) / VOLS
        assert np.all(rel_error < 0.05), f"Erreur vol: {rel_error}"

    def test_correlations(self):
        """Les correlations empiriques sont proches des theoriques (a 5% pres)."""
        n_outer = 5000
        paths = simulate_gbm(n_outer=n_outer, n_t=52)
        log_returns = np.log(paths[:, -1, :] / SPOTS)
        corr_emp = np.corrcoef(log_returns.T)
        # Verifier les elements hors-diagonale
        for i in range(3):
            for j in range(i + 1, 3):
                err = abs(corr_emp[i, j] - CORRELATION_MATRIX[i, j])
                assert err < 0.05, (
                    f"Correlation ({i},{j}): empirique={corr_emp[i,j]:.3f}, "
                    f"theorique={CORRELATION_MATRIX[i,j]:.3f}"
                )

    def test_reproducibility(self):
        """Meme seed => memes chemins."""
        p1 = simulate_gbm(n_outer=10, n_t=5, seed=123)
        p2 = simulate_gbm(n_outer=10, n_t=5, seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds(self):
        """Seeds differentes => chemins differents."""
        p1 = simulate_gbm(n_outer=10, n_t=5, seed=123)
        p2 = simulate_gbm(n_outer=10, n_t=5, seed=456)
        assert not np.allclose(p1, p2)


class TestSimulateGBMFromSpot:
    """Tests de la simulation GBM a partir d'un spot donne."""

    def test_shape_single(self):
        """Shape correcte pour un seul spot."""
        spots = SPOTS.copy()
        result = simulate_gbm_from_spot(spots, dt=MPOR)
        assert result.shape == (1, len(SPOTS))

    def test_shape_batch(self):
        """Shape correcte pour un batch de spots."""
        n = 100
        spots = np.tile(SPOTS, (n, 1))
        result = simulate_gbm_from_spot(spots, dt=MPOR)
        assert result.shape == (n, len(SPOTS))

    def test_positive(self):
        """Les prix simules sont positifs."""
        spots = np.tile(SPOTS, (500, 1))
        result = simulate_gbm_from_spot(spots, dt=MPOR)
        assert np.all(result > 0)

    def test_mean_drift(self):
        """E[S(t+dt)] ~ S(t) * exp(r*dt) a 2% pres (N=10000)."""
        n = 10000
        spots = np.tile(SPOTS, (n, 1))
        rng = np.random.default_rng(42)
        result = simulate_gbm_from_spot(spots, dt=MPOR, rng=rng)
        expected = SPOTS * np.exp(RISK_FREE_RATE * MPOR)
        realized = np.mean(result, axis=0)
        rel_error = np.abs(realized - expected) / expected
        assert np.all(rel_error < 0.02), f"Erreur drift: {rel_error}"
