"""
Portefeuille d'options et calcul du Mark-to-Market (MtM).
"""

import numpy as np
from lib.black_scholes import bs_price
from config.parameters import PORTFOLIO, RISK_FREE_RATE, VOLS


def compute_mtm(
    paths: np.ndarray,
    t_j: float,
    portfolio: list[dict] = PORTFOLIO,
    r: float = RISK_FREE_RATE,
    vols: np.ndarray = VOLS,
) -> np.ndarray:
    """Calcule le MtM du portefeuille pour tous les scénarios à la date t_j.

    MtM_i(t_j) = sum_k eta_k * V_k(t_j, S^{i(k)}_i(t_j))

    Args:
        paths: Prix spots simulés, shape (n_outer, d) — les spots à t_j.
        t_j: Date courante.
        portfolio: Liste des options du portefeuille.
        r: Taux sans risque.
        vols: Volatilités des sous-jacents.

    Returns:
        mtm: MtM du portefeuille, shape (n_outer,).
    """
    n_outer = paths.shape[0]
    mtm = np.zeros(n_outer)

    for option in portfolio:
        asset_idx = option["asset_idx"]
        K = option["strike"]
        T_opt = option["maturity"]
        option_type = option["type"]
        position = option["position"]
        sigma = vols[asset_idx]

        tau = T_opt - t_j  # Maturité résiduelle

        if tau <= 0:
            # Option expirée : valeur intrinsèque
            S = paths[:, asset_idx]
            if option_type == "call":
                value = np.maximum(S - K, 0.0)
            else:
                value = np.maximum(K - S, 0.0)
        else:
            S = paths[:, asset_idx]
            tau_arr = np.full(n_outer, tau)
            value = bs_price(option_type, S, K, tau_arr, r, sigma)

        mtm += position * value

    return mtm


def compute_mtm_full(
    all_paths: np.ndarray,
    time_grid: np.ndarray,
    portfolio: list[dict] = PORTFOLIO,
    r: float = RISK_FREE_RATE,
    vols: np.ndarray = VOLS,
) -> np.ndarray:
    """Calcule le MtM pour tous les scénarios et toutes les dates.

    Args:
        all_paths: Chemins simulés, shape (n_outer, n_t + 1, d).
        time_grid: Grille temporelle, shape (n_t + 1,).
        portfolio: Liste des options du portefeuille.
        r: Taux sans risque.
        vols: Volatilités.

    Returns:
        mtm_matrix: MtM, shape (n_outer, n_t + 1).
    """
    n_outer, n_steps, _ = all_paths.shape
    mtm_matrix = np.zeros((n_outer, n_steps))

    for j, t_j in enumerate(time_grid):
        mtm_matrix[:, j] = compute_mtm(all_paths[:, j, :], t_j, portfolio, r, vols)

    return mtm_matrix
