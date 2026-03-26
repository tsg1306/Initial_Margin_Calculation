"""
Fixtures partagees pour les tests unitaires.
"""

import numpy as np
import pytest

from config.parameters import (
    SEED, RISK_FREE_RATE, SPOTS, VOLS, CORRELATION_MATRIX,
    PORTFOLIO, TIME_GRID, N_T, DELTA_T, MPOR,
)
from lib.diffusion import simulate_gbm


@pytest.fixture
def rng():
    """Generateur aleatoire reproductible."""
    return np.random.default_rng(SEED)


@pytest.fixture
def small_paths():
    """Chemins GBM avec parametres reduits (N_outer=50, seed fixe)."""
    return simulate_gbm(n_outer=50, n_t=N_T, seed=SEED)


@pytest.fixture
def medium_paths():
    """Chemins GBM avec parametres moyens (N_outer=200, seed fixe)."""
    return simulate_gbm(n_outer=200, n_t=N_T, seed=SEED)
