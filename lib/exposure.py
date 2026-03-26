"""
Métriques d'exposition : E, EE, EEE, EEPE.
"""

import numpy as np
from config.parameters import TIME_GRID, T_EEPE, DELTA_T


def compute_exposure(mtm: np.ndarray) -> np.ndarray:
    """Exposition : E(t) = max(0, MtM(t)).

    Args:
        mtm: MtM, shape (n_outer,) ou (n_outer, n_steps).

    Returns:
        Exposition, même shape.
    """
    return np.maximum(0.0, mtm)


def compute_ee(mtm_matrix: np.ndarray) -> np.ndarray:
    """Expected Exposure : EE(t_j) = mean_i(max(0, MtM_i(t_j))).

    Args:
        mtm_matrix: MtM, shape (n_outer, n_steps).

    Returns:
        ee: Expected Exposure, shape (n_steps,).
    """
    exposure = compute_exposure(mtm_matrix)
    return np.mean(exposure, axis=0)


def compute_eee(ee: np.ndarray) -> np.ndarray:
    """Effective Expected Exposure : EEE(t_j) = max_{l<=j} EE(t_l).

    Contrainte de non-décroissance (running maximum).

    Args:
        ee: Expected Exposure, shape (n_steps,).

    Returns:
        eee: Effective EE (non-décroissant), shape (n_steps,).
    """
    return np.maximum.accumulate(ee)


def compute_eepe(eee: np.ndarray, time_grid: np.ndarray = TIME_GRID) -> float:
    """Effective Expected Positive Exposure : moyenne temporelle de l'EEE.

    EEPE = (1/T) * integral_0^T EEE(t) dt, discrétisé par les trapèzes.

    Args:
        eee: Effective EE, shape (n_steps,).
        time_grid: Grille temporelle, shape (n_steps,).

    Returns:
        eepe: Scalaire.
    """
    T = time_grid[-1] - time_grid[0]
    if T <= 0:
        return 0.0
    return float(np.trapezoid(eee, time_grid) / T)


def compute_all_exposure_metrics(mtm_matrix: np.ndarray, time_grid: np.ndarray = TIME_GRID) -> dict:
    """Calcule toutes les métriques d'exposition.

    Args:
        mtm_matrix: MtM, shape (n_outer, n_steps).
        time_grid: Grille temporelle.

    Returns:
        Dictionnaire avec les clés : "ee", "eee", "eepe".
    """
    ee = compute_ee(mtm_matrix)
    eee = compute_eee(ee)
    eepe = compute_eepe(eee, time_grid)
    return {"ee": ee, "eee": eee, "eepe": eepe}
