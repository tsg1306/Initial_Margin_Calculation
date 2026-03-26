"""
Tests unitaires pour lib/exposure.py.
"""

import numpy as np
import pytest

from lib.exposure import (
    compute_exposure, compute_ee, compute_eee, compute_eepe,
    compute_all_exposure_metrics,
)


class TestComputeExposure:
    """Tests de l'exposition E(t) = max(0, MtM)."""

    def test_positive_mtm(self):
        """Si MtM > 0, E = MtM."""
        mtm = np.array([10.0, 20.0, 5.0])
        np.testing.assert_array_equal(compute_exposure(mtm), mtm)

    def test_negative_mtm(self):
        """Si MtM < 0, E = 0."""
        mtm = np.array([-10.0, -20.0, -5.0])
        np.testing.assert_array_equal(compute_exposure(mtm), np.zeros(3))

    def test_mixed(self):
        """Cas mixte : max(0, MtM)."""
        mtm = np.array([10.0, -5.0, 0.0, 3.0])
        expected = np.array([10.0, 0.0, 0.0, 3.0])
        np.testing.assert_array_equal(compute_exposure(mtm), expected)

    def test_2d(self):
        """Fonctionne aussi en 2D."""
        mtm = np.array([[1.0, -2.0], [-3.0, 4.0]])
        expected = np.array([[1.0, 0.0], [0.0, 4.0]])
        np.testing.assert_array_equal(compute_exposure(mtm), expected)


class TestComputeEE:
    """Tests de l'Expected Exposure."""

    def test_simple(self):
        """EE = moyenne des max(0, MtM) par colonne."""
        mtm = np.array([
            [10.0, -5.0, 3.0],
            [20.0, 5.0, -1.0],
        ])
        expected = np.array([15.0, 2.5, 1.5])
        np.testing.assert_allclose(compute_ee(mtm), expected)

    def test_all_positive(self):
        """Si tout MtM > 0, EE = mean(MtM)."""
        mtm = np.array([[10.0, 20.0], [30.0, 40.0]])
        expected = np.array([20.0, 30.0])
        np.testing.assert_allclose(compute_ee(mtm), expected)

    def test_ee_shape(self):
        """EE a la bonne shape."""
        mtm = np.random.randn(100, 53)
        ee = compute_ee(mtm)
        assert ee.shape == (53,)


class TestComputeEEE:
    """Tests de l'Effective Expected Exposure (running maximum)."""

    def test_non_decreasing(self):
        """EEE est non-decroissante."""
        ee = np.array([5.0, 3.0, 7.0, 2.0, 8.0])
        eee = compute_eee(ee)
        assert np.all(np.diff(eee) >= 0)

    def test_running_max(self):
        """EEE = running maximum de EE."""
        ee = np.array([5.0, 3.0, 7.0, 2.0, 8.0])
        expected = np.array([5.0, 5.0, 7.0, 7.0, 8.0])
        np.testing.assert_array_equal(compute_eee(ee), expected)

    def test_already_increasing(self):
        """Si EE est deja croissante, EEE = EE."""
        ee = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(compute_eee(ee), ee)


class TestComputeEEPE:
    """Tests de l'Effective Expected Positive Exposure."""

    def test_constant_eee(self):
        """Si EEE est constante, EEPE = cette constante."""
        eee = np.full(53, 10.0)
        time_grid = np.linspace(0, 1, 53)
        eepe = compute_eepe(eee, time_grid)
        assert abs(eepe - 10.0) < 1e-10

    def test_positive(self):
        """EEPE est positif si EEE > 0."""
        eee = np.abs(np.random.randn(53)) + 1.0
        time_grid = np.linspace(0, 1, 53)
        eepe = compute_eepe(eee, time_grid)
        assert eepe > 0

    def test_linear_eee(self):
        """Si EEE croit lineairement de a a b, EEPE = (a+b)/2."""
        time_grid = np.linspace(0, 1, 101)
        a, b = 5.0, 15.0
        eee = np.linspace(a, b, 101)
        eepe = compute_eepe(eee, time_grid)
        expected = (a + b) / 2.0
        assert abs(eepe - expected) < 1e-6

    def test_zero_horizon(self):
        """Si T = 0, EEPE = 0."""
        eee = np.array([10.0])
        time_grid = np.array([0.0])
        eepe = compute_eepe(eee, time_grid)
        assert eepe == 0.0


class TestComputeAllMetrics:
    """Tests de la fonction globale."""

    def test_keys(self):
        """Le dictionnaire contient les bonnes cles."""
        mtm = np.random.randn(100, 53)
        time_grid = np.linspace(0, 1, 53)
        metrics = compute_all_exposure_metrics(mtm, time_grid)
        assert "ee" in metrics
        assert "eee" in metrics
        assert "eepe" in metrics

    def test_eee_ge_ee(self):
        """EEE >= EE a chaque date."""
        mtm = np.random.randn(200, 53)
        time_grid = np.linspace(0, 1, 53)
        metrics = compute_all_exposure_metrics(mtm, time_grid)
        assert np.all(metrics["eee"] >= metrics["ee"] - 1e-12)
