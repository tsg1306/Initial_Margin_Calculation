"""
Tests unitaires pour lib/johnson.py.
"""

import numpy as np
import pytest
from scipy.stats import norm

from lib.johnson import (
    _johnson_type,
    _su_skew_kurt,
    _fit_johnson_su,
    _fit_johnson_sn,
    johnson_quantile,
    _polynomial_regression,
    estimate_conditional_moments,
    compute_im_johnson,
)
from lib.diffusion import simulate_gbm
from config.parameters import SEED, TIME_GRID


class TestJohnsonType:
    """Tests de la selection du type de distribution Johnson."""

    def test_normal(self):
        """Skewness = 0, kurtosis = 3 => SN."""
        assert _johnson_type(0.0, 3.0) == "SN"

    def test_symmetric_heavy_tails(self):
        """Skewness = 0, kurtosis > 3 => SU."""
        assert _johnson_type(0.0, 5.0) == "SU"

    def test_symmetric_light_tails(self):
        """Skewness = 0, kurtosis < 3 => SB."""
        assert _johnson_type(0.0, 2.0) == "SB"

    def test_asymmetric_heavy_tails(self):
        """Skewness^2 = 1, kurtosis = 8 => SU (au-dessus de la frontiere lognormale)."""
        result = _johnson_type(1.0, 8.0)
        assert result == "SU", f"Attendu SU, obtenu {result}"

    def test_asymmetric_bounded(self):
        """Skewness^2 = 0.5, kurtosis = 2.5 => SB (en dessous de la frontiere)."""
        result = _johnson_type(0.5, 2.5)
        assert result == "SB", f"Attendu SB, obtenu {result}"

    def test_lognormal_boundary(self):
        """Sur la courbe lognormale => SL (avec tolerance)."""
        # Pour omega=2 : beta1 = (2-1)(2+2)^2 = 16, beta2 = 16+16+12-3 = 41
        result = _johnson_type(16.0, 41.0)
        assert result == "SL", f"Attendu SL, obtenu {result}"


class TestSUSkewKurt:
    """Tests des formules exactes de skewness/kurtosis pour Johnson SU."""

    def test_symmetric_case(self):
        """Gamma = 0 => skewness = 0."""
        skew, kurt = _su_skew_kurt(omega=2.0, Gamma=0.0)
        assert abs(skew) < 1e-10

    def test_kurtosis_gt_3(self):
        """La kurtosis d'une Johnson SU est > 3 (queues lourdes)."""
        _, kurt = _su_skew_kurt(omega=1.5, Gamma=0.0)
        assert kurt > 3.0

    def test_degenerate(self):
        """omega proche de 1 => quasi-normal (skew=0, kurt~3)."""
        skew, kurt = _su_skew_kurt(omega=1.001, Gamma=0.0)
        assert abs(skew) < 0.1
        assert abs(kurt - 3.0) < 0.5


class TestFitJohnsonSU:
    """Tests du fit Johnson SU par les moments."""

    def test_symmetric(self):
        """Fit d'une distribution symetrique (skew=0, kurt=5)."""
        xi, lam, gamma, delta_j = _fit_johnson_su(
            mean=10.0, var=4.0, skew=0.0, kurt=5.0
        )
        # Verifier que le quantile 99% est raisonnable
        q99 = johnson_quantile(0.99, "SU", xi, lam, gamma, delta_j)
        assert q99 > 10.0  # Doit etre au-dessus de la moyenne
        assert q99 < 20.0  # Pas excessif

    def test_skewed(self):
        """Fit d'une distribution asymetrique (skew=1.0, kurt=6)."""
        xi, lam, gamma, delta_j = _fit_johnson_su(
            mean=5.0, var=9.0, skew=1.0, kurt=6.0
        )
        q99 = johnson_quantile(0.99, "SU", xi, lam, gamma, delta_j)
        # Avec skew positif, le quantile 99% doit etre > mean + 2*sigma
        assert q99 > 5.0 + 2 * 3.0, f"Q99 = {q99:.4f}, trop bas pour skew=1"

    def test_known_moments_su(self):
        """Genere des donnees Johnson SU, fit les moments, verifie le quantile.

        X = xi + lam * sinh((Z - gamma) / delta), Z ~ N(0,1)
        Parametres : xi=10, lam=3, gamma=-0.5, delta=1.5
        """
        np.random.seed(42)
        Z = np.random.randn(100000)
        xi_true, lam_true, gam_true, delta_true = 10.0, 3.0, -0.5, 1.5
        X = xi_true + lam_true * np.sinh((Z - gam_true) / delta_true)

        mean, var = np.mean(X), np.var(X)
        from scipy.stats import skew as _sk, kurtosis as _kt
        skew = _sk(X)
        kurt = _kt(X, fisher=False)

        xi, lam, gamma, delta_j = _fit_johnson_su(mean, var, skew, kurt)
        q99_fit = johnson_quantile(0.99, "SU", xi, lam, gamma, delta_j)
        q99_exact = johnson_quantile(0.99, "SU", xi_true, lam_true, gam_true, delta_true)

        rel_err = abs(q99_fit - q99_exact) / abs(q99_exact)
        assert rel_err < 0.05, f"Erreur relative Q99 = {rel_err:.2%}"


class TestFitJohnsonSN:
    """Tests du fit Johnson SN (distribution normale)."""

    def test_sn_params(self):
        """Fit SN : xi = mean, lam = sigma."""
        xi, lam, gamma, delta_j = _fit_johnson_sn(mean=5.0, var=9.0)
        assert abs(xi - 5.0) < 1e-10
        assert abs(lam - 3.0) < 1e-10
        assert abs(gamma) < 1e-10
        assert abs(delta_j - 1.0) < 1e-10


class TestJohnsonQuantile:
    """Tests du quantile Johnson."""

    def test_sn_quantile(self):
        """Pour SN, le quantile doit correspondre a la normale."""
        xi, lam = 10.0, 3.0
        q = johnson_quantile(0.99, "SN", xi, lam, 0.0, 1.0)
        expected = xi + lam * norm.ppf(0.99)
        assert abs(q - expected) < 1e-10

    def test_su_quantile_median(self):
        """Pour alpha=0.5 (mediane), Q = xi + lam * sinh(-gamma/delta)."""
        xi, lam, gamma, delta = 10.0, 3.0, 0.5, 1.5
        q50 = johnson_quantile(0.50, "SU", xi, lam, gamma, delta)
        expected = xi + lam * np.sinh(-gamma / delta)
        assert abs(q50 - expected) < 1e-10

    def test_invalid_type(self):
        """Type invalide => ValueError."""
        with pytest.raises(ValueError):
            johnson_quantile(0.99, "XX", 0, 1, 0, 1)

    def test_su_quantile_gt_mean(self):
        """Q99 > mean pour une Johnson SU centree."""
        q99 = johnson_quantile(0.99, "SU", xi=0.0, lam=1.0, gamma=0.0, delta_j=1.0)
        assert q99 > 0.0  # Quantile 99% au-dessus de la moyenne


class TestPolynomialRegression:
    """Tests de la regression polynomiale."""

    def test_perfect_fit(self):
        """Regression parfaite sur un polynome de degre correct."""
        x = np.linspace(-2, 2, 100)
        y = 3 * x**2 - 2 * x + 1
        y_hat = _polynomial_regression(x, y, degree=2)
        np.testing.assert_allclose(y_hat, y, atol=1e-8)

    def test_constant_x(self):
        """Si x est constant, retourne la moyenne de y."""
        x = np.full(50, 5.0)
        y = np.random.randn(50)
        y_hat = _polynomial_regression(x, y, degree=3)
        np.testing.assert_allclose(y_hat, np.mean(y) * np.ones(50), atol=1e-10)


class TestEstimateConditionalMoments:
    """Tests de l'estimation des moments conditionnels."""

    def test_output_shapes(self):
        """Verifie les shapes de sortie."""
        n = 100
        pv = np.random.randn(n)
        dpv = np.random.randn(n)
        m, v, s, k = estimate_conditional_moments(pv, dpv, degree=3)
        assert m.shape == (n,)
        assert v.shape == (n,)
        assert s.shape == (n,)
        assert k.shape == (n,)

    def test_variance_positive(self):
        """La variance conditionnelle est toujours > 0."""
        n = 200
        pv = np.random.randn(n)
        dpv = np.random.randn(n) * 5
        _, v, _, _ = estimate_conditional_moments(pv, dpv)
        assert np.all(v > 0)

    def test_kurtosis_constraint(self):
        """kurtosis >= skewness^2 + 1 (contrainte theorique)."""
        n = 200
        pv = np.random.randn(n)
        dpv = np.random.randn(n) * 3
        _, _, s, k = estimate_conditional_moments(pv, dpv)
        assert np.all(k >= s**2 + 1.0 - 1e-6)


class TestComputeIMJohnson:
    """Tests du pipeline Johnson complet."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Parametres reduits."""
        self.n_outer = 30
        self.n_t = 5
        self.time_grid = np.linspace(0, 1, self.n_t + 1)
        self.paths = simulate_gbm(
            n_outer=self.n_outer,
            n_t=self.n_t,
            seed=SEED,
        )

    def test_shape(self):
        """Verifie la shape."""
        im = compute_im_johnson(self.paths, self.time_grid)
        assert im.shape == (self.n_outer, self.n_t + 1)

    def test_im_non_negative(self):
        """L'IM Johnson est >= 0 (force dans le code)."""
        im = compute_im_johnson(self.paths, self.time_grid)
        assert np.all(im >= 0)

    def test_im_reasonable_magnitude(self):
        """L'IM moyen est dans un ordre de grandeur raisonnable."""
        im = compute_im_johnson(self.paths, self.time_grid)
        im_mean = np.mean(im)
        assert 0 < im_mean < 100, f"IM moyen Johnson = {im_mean:.2f}"

    def test_reproducibility(self):
        """Meme seed => meme IM."""
        im1 = compute_im_johnson(self.paths, self.time_grid, seed=42)
        im2 = compute_im_johnson(self.paths, self.time_grid, seed=42)
        np.testing.assert_array_equal(im1, im2)
