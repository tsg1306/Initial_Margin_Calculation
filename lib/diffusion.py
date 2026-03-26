"""
Diffusion GBM multidimensionnelle corrélée.

Simulation des sous-jacents sous la mesure risque-neutre Q
par solution exacte du GBM (pas d'erreur de discrétisation).
"""

import numpy as np
from config.parameters import (
    SEED, RISK_FREE_RATE, N_ASSETS, SPOTS, VOLS,
    CORRELATION_MATRIX, N_OUTER, N_T, DELTA_T, TIME_GRID,
)


def cholesky_decomposition(corr_matrix: np.ndarray) -> np.ndarray:
    """Décomposition de Cholesky de la matrice de corrélation.

    Args:
        corr_matrix: Matrice de corrélation (d x d), doit être définie positive.

    Returns:
        L: Matrice triangulaire inférieure telle que corr_matrix = L @ L.T

    Raises:
        np.linalg.LinAlgError: Si la matrice n'est pas définie positive.
    """
    return np.linalg.cholesky(corr_matrix)


def simulate_gbm(
    n_outer: int = N_OUTER,
    n_t: int = N_T,
    dt: float = DELTA_T,
    spots: np.ndarray = SPOTS,
    vols: np.ndarray = VOLS,
    r: float = RISK_FREE_RATE,
    corr_matrix: np.ndarray = CORRELATION_MATRIX,
    seed: int = SEED,
) -> np.ndarray:
    """Simule les chemins GBM multidimensionnels corrélés.

    S^i(t_{j+1}) = S^i(t_j) * exp[(r - sigma_i^2/2)*dt + sigma_i*sqrt(dt)*Z^i]
    avec Z = L @ epsilon, epsilon ~ N(0, I)

    Args:
        n_outer: Nombre de scénarios.
        n_t: Nombre de pas de temps.
        dt: Pas de temps.
        spots: Prix spots initiaux (d,).
        vols: Volatilités (d,).
        r: Taux sans risque.
        corr_matrix: Matrice de corrélation (d, d).
        seed: Seed aléatoire.

    Returns:
        paths: Array (n_outer, n_t + 1, d) des prix simulés.
               paths[:, 0, :] = spots (conditions initiales).
    """
    rng = np.random.default_rng(seed)
    d = len(spots)
    L = cholesky_decomposition(corr_matrix)

    # Drift et diffusion (constants)
    drift = (r - 0.5 * vols**2) * dt         # (d,)
    diffusion = vols * np.sqrt(dt)             # (d,)

    # Initialisation des chemins
    paths = np.empty((n_outer, n_t + 1, d))
    paths[:, 0, :] = spots

    # Simulation pas à pas
    # On génère tous les aléas d'un coup : (n_outer, n_t, d)
    epsilon = rng.standard_normal((n_outer, n_t, d))
    # Corrélation : Z = epsilon @ L.T (chaque ligne epsilon[i,j,:] est multipliée par L.T)
    Z = epsilon @ L.T  # (n_outer, n_t, d)

    # Log-rendements
    log_returns = drift[np.newaxis, np.newaxis, :] + diffusion[np.newaxis, np.newaxis, :] * Z

    # Cumul des log-rendements et exponentiation
    cum_log_returns = np.cumsum(log_returns, axis=1)
    paths[:, 1:, :] = spots[np.newaxis, np.newaxis, :] * np.exp(cum_log_returns)

    return paths


def simulate_gbm_from_spot(
    spots_t: np.ndarray,
    dt: float,
    vols: np.ndarray = VOLS,
    r: float = RISK_FREE_RATE,
    corr_matrix: np.ndarray = CORRELATION_MATRIX,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simule un pas GBM à partir de spots donnés (pour le nested MC).

    Utilisé pour simuler S(t + delta) à partir de S(t) sur le MPOR.

    Args:
        spots_t: Prix spots au temps t, shape (n_scenarios, d) ou (d,).
        dt: Pas de temps (typiquement MPOR).
        vols: Volatilités (d,).
        r: Taux sans risque.
        corr_matrix: Matrice de corrélation (d, d).
        rng: Générateur aléatoire NumPy.

    Returns:
        spots_tpdt: Prix spots au temps t + dt, même shape que spots_t.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(vols)
    L = cholesky_decomposition(corr_matrix)

    if spots_t.ndim == 1:
        spots_t = spots_t[np.newaxis, :]

    n = spots_t.shape[0]

    drift = (r - 0.5 * vols**2) * dt
    diffusion = vols * np.sqrt(dt)

    epsilon = rng.standard_normal((n, d))
    Z = epsilon @ L.T

    log_return = drift[np.newaxis, :] + diffusion[np.newaxis, :] * Z
    spots_tpdt = spots_t * np.exp(log_return)

    return spots_tpdt
