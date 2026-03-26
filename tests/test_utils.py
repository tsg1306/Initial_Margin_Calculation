"""
Tests unitaires pour lib/utils.py.
"""

import io
import sys
import numpy as np
import pytest

from lib.utils import validate_correlation_matrix, compute_ead, Timer, print_header, print_table
from config.parameters import CORRELATION_MATRIX, ALPHA_EAD


class TestValidateCorrelationMatrix:
    """Tests de la validation de la matrice de correlation."""

    def test_valid_matrix(self):
        """La matrice du projet est valide."""
        validate_correlation_matrix(CORRELATION_MATRIX)  # Ne doit pas lever

    def test_identity(self):
        """La matrice identite est valide."""
        validate_correlation_matrix(np.eye(3))

    def test_not_symmetric(self):
        """Matrice non symetrique => ValueError."""
        bad = np.array([[1.0, 0.5, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="symetrique"):
            validate_correlation_matrix(bad)

    def test_diagonal_not_one(self):
        """Diagonale != 1 => ValueError."""
        bad = np.array([[2.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="diagonale"):
            validate_correlation_matrix(bad)

    def test_not_positive_definite(self):
        """Matrice non definie positive => ValueError."""
        bad = np.array([[1.0, 0.99], [0.99, 1.0]])
        bad[0, 1] = 2.0  # Casse la positivite
        bad[1, 0] = 2.0
        with pytest.raises(ValueError, match="definie positive"):
            validate_correlation_matrix(bad)


class TestComputeEAD:
    """Tests du calcul EAD."""

    def test_basic(self):
        """EAD = alpha * EEPE."""
        assert compute_ead(10.0) == ALPHA_EAD * 10.0

    def test_custom_alpha(self):
        """EAD avec alpha personnalise."""
        assert compute_ead(10.0, alpha=2.0) == 20.0

    def test_zero(self):
        """EEPE = 0 => EAD = 0."""
        assert compute_ead(0.0) == 0.0


class TestTimer:
    """Tests du context manager Timer."""

    def test_elapsed(self):
        """Le timer mesure un temps > 0."""
        import time
        with Timer("test") as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04  # Au moins ~40ms

    def test_name(self):
        """Le nom est stocke."""
        with Timer("mon_timer") as t:
            pass
        assert t.name == "mon_timer"


class TestPrintFunctions:
    """Tests des fonctions d'affichage."""

    def test_print_header(self, capsys):
        """print_header affiche le titre."""
        print_header("Test")
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "=" in captured.out

    def test_print_table(self, capsys):
        """print_table affiche les donnees."""
        headers = ["Col1", "Col2"]
        rows = [["a", "b"], ["c", "d"]]
        print_table(headers, rows)
        captured = capsys.readouterr()
        assert "Col1" in captured.out
        assert "a" in captured.out
        assert "d" in captured.out
