"""
Tests unitaires pour lib/black_scholes.py.
"""

import numpy as np
import pytest
from scipy.stats import norm

from lib.black_scholes import bs_call, bs_put, bs_price, bs_d1, bs_d2


class TestBSPricing:
    """Tests de pricing Black-Scholes."""

    def test_call_reference_value(self):
        """Verifie le prix d'un call ATM contre une valeur de reference connue.

        S=100, K=100, T=1, r=5%, sigma=20% => C ~ 10.4506
        """
        S = np.array([100.0])
        C = bs_call(S, K=100.0, tau=np.array([1.0]), r=0.05, sigma=0.20)
        assert abs(C[0] - 10.4506) < 0.001

    def test_put_reference_value(self):
        """Verifie le prix d'un put ATM par parite call-put.

        P = C - S + K*exp(-rT) ~ 10.4506 - 100 + 100*exp(-0.05) ~ 5.5735
        """
        S = np.array([100.0])
        P = bs_put(S, K=100.0, tau=np.array([1.0]), r=0.05, sigma=0.20)
        expected = 10.4506 - 100.0 + 100.0 * np.exp(-0.05)
        assert abs(P[0] - expected) < 0.01

    def test_call_put_parity(self):
        """Parite call-put : C - P = S - K*exp(-rT)."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        tau = np.array([1.0, 1.0, 1.0])
        r, sigma = 0.03, 0.25

        C = bs_call(S, K, tau, r, sigma)
        P = bs_put(S, K, tau, r, sigma)
        parity = S - K * np.exp(-r * tau)

        np.testing.assert_allclose(C - P, parity, atol=1e-10)

    def test_call_positive(self):
        """Le prix d'un call est toujours positif."""
        S = np.array([50.0, 100.0, 150.0])
        C = bs_call(S, K=100.0, tau=np.array([1.0, 1.0, 1.0]), r=0.03, sigma=0.20)
        assert np.all(C > 0)

    def test_put_positive(self):
        """Le prix d'un put est toujours positif."""
        S = np.array([50.0, 100.0, 150.0])
        P = bs_put(S, K=100.0, tau=np.array([1.0, 1.0, 1.0]), r=0.03, sigma=0.20)
        assert np.all(P > 0)

    def test_call_monotone_in_spot(self):
        """Un call est croissant en S."""
        S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        C = bs_call(S, K=100.0, tau=np.full(5, 1.0), r=0.03, sigma=0.20)
        assert np.all(np.diff(C) > 0)

    def test_put_monotone_in_spot(self):
        """Un put est decroissant en S."""
        S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        P = bs_put(S, K=100.0, tau=np.full(5, 1.0), r=0.03, sigma=0.20)
        assert np.all(np.diff(P) < 0)

    def test_call_intrinsic_value_bound(self):
        """C >= max(0, S - K*exp(-rT))."""
        S = np.array([80.0, 100.0, 120.0])
        K, r, tau = 100.0, 0.03, np.full(3, 1.0)
        C = bs_call(S, K, tau, r, 0.20)
        lower = np.maximum(0, S - K * np.exp(-r * tau))
        assert np.all(C >= lower - 1e-10)

    def test_bs_price_call(self):
        """bs_price("call", ...) == bs_call(...)."""
        S = np.array([100.0])
        tau = np.array([1.0])
        assert bs_price("call", S, 100.0, tau, 0.03, 0.20) == bs_call(S, 100.0, tau, 0.03, 0.20)

    def test_bs_price_put(self):
        """bs_price("put", ...) == bs_put(...)."""
        S = np.array([100.0])
        tau = np.array([1.0])
        assert bs_price("put", S, 100.0, tau, 0.03, 0.20) == bs_put(S, 100.0, tau, 0.03, 0.20)

    def test_bs_price_invalid_type(self):
        """bs_price leve ValueError pour un type inconnu."""
        with pytest.raises(ValueError, match="Type d'option inconnu"):
            bs_price("barrier", np.array([100.0]), 100.0, np.array([1.0]), 0.03, 0.20)

    def test_vectorized(self):
        """Le pricing est bien vectorise (meme resultat element par element)."""
        S = np.array([80.0, 100.0, 120.0])
        tau = np.full(3, 1.5)
        K, r, sigma = 100.0, 0.03, 0.25

        C_vec = bs_call(S, K, tau, r, sigma)
        C_loop = np.array([bs_call(np.array([s]), K, np.array([1.5]), r, sigma)[0] for s in S])

        np.testing.assert_allclose(C_vec, C_loop, atol=1e-12)

    def test_deep_itm_call_approx_intrinsic(self):
        """Un call deep ITM a un prix proche de S - K*exp(-rT)."""
        S = np.array([200.0])
        K, tau, r, sigma = 100.0, np.array([0.1]), 0.03, 0.20
        C = bs_call(S, K, tau, r, sigma)
        intrinsic = S - K * np.exp(-r * tau)
        assert abs(C[0] - intrinsic[0]) / intrinsic[0] < 0.01  # < 1% d'ecart
