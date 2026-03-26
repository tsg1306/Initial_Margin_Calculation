"""
Calcul de l'Initial Margin (IM) stochastique.

- Nested Monte Carlo (Livrable 1)
- Interface commune pour l'approximation Johnson (Livrable 2, cf. johnson.py)
"""

import numpy as np
from lib.diffusion import simulate_gbm_from_spot
from lib.portfolio import compute_mtm
from config.parameters import (
    PORTFOLIO, RISK_FREE_RATE, VOLS, CORRELATION_MATRIX,
    N_INNER, MPOR, IM_CONFIDENCE, SEED,
)


def compute_im_nested(
    paths: np.ndarray,
    time_grid: np.ndarray,
    portfolio: list[dict] = PORTFOLIO,
    r: float = RISK_FREE_RATE,
    vols: np.ndarray = VOLS,
    corr_matrix: np.ndarray = CORRELATION_MATRIX,
    n_inner: int = N_INNER,
    mpor: float = MPOR,
    confidence: float = IM_CONFIDENCE,
    seed: int = SEED,
) -> np.ndarray:
    """Calcule l'IM par nested Monte Carlo pour tous les nœuds.

    Pour chaque nœud (i, t_j) :
    1. Simuler N_inner sous-scénarios sur le MPOR
    2. Pricer le portefeuille à t_j + delta
    3. Calculer les P&L : DeltaPV = PV(t_j) - PV(t_j + delta)
    4. IM = quantile 99% des DeltaPV

    Args:
        paths: Chemins simulés, shape (n_outer, n_t + 1, d).
        time_grid: Grille temporelle, shape (n_t + 1,).
        portfolio: Portefeuille.
        r: Taux sans risque.
        vols: Volatilités.
        corr_matrix: Matrice de corrélation.
        n_inner: Nombre de scénarios intérieurs.
        mpor: Margin Period of Risk.
        confidence: Niveau de confiance (0.99).
        seed: Seed aléatoire.

    Returns:
        im_matrix: IM, shape (n_outer, n_t + 1).
    """
    n_outer, n_steps, d = paths.shape
    im_matrix = np.zeros((n_outer, n_steps))
    rng = np.random.default_rng(seed + 1_000_000)  # Seed distincte du MC extérieur

    for j in range(n_steps):
        t_j = time_grid[j]

        # PV(t_j) pour tous les scénarios extérieurs
        pv_tj = compute_mtm(paths[:, j, :], t_j, portfolio, r, vols)

        # Pour chaque scénario extérieur, simuler N_inner sous-scénarios
        for i in range(n_outer):
            spots_i = paths[i, j, :]  # (d,)

            # Simuler N_inner chemins sur le MPOR
            spots_inner = simulate_gbm_from_spot(
                np.tile(spots_i, (n_inner, 1)),  # (n_inner, d)
                dt=mpor,
                vols=vols,
                r=r,
                corr_matrix=corr_matrix,
                rng=rng,
            )  # (n_inner, d)

            # PV(t_j + delta) pour chaque sous-scénario
            pv_tjpd = compute_mtm(spots_inner, t_j + mpor, portfolio, r, vols)

            # P&L : DeltaPV = PV(t_j) - PV(t_j + delta)
            delta_pv = pv_tj[i] - pv_tjpd  # (n_inner,)

            # IM = quantile 99% empirique
            im_matrix[i, j] = np.quantile(delta_pv, confidence)

    return im_matrix


def compute_exposure_with_im(
    paths: np.ndarray,
    time_grid: np.ndarray,
    im_matrix: np.ndarray,
    portfolio: list[dict] = PORTFOLIO,
    r: float = RISK_FREE_RATE,
    vols: np.ndarray = VOLS,
    corr_matrix: np.ndarray = CORRELATION_MATRIX,
    mpor: float = MPOR,
    seed: int = SEED,
) -> np.ndarray:
    """Calcule l'exposition résiduelle avec IM pour le nested MC.

    E^IM_i(t_j) = max(0, DeltaPV_i(t_j) - IM_i(t_j))

    On utilise un DeltaPV « réalisé » indépendant (un seul scénario MPOR par nœud)
    pour calculer l'exposition résiduelle.

    Args:
        paths: Chemins simulés, shape (n_outer, n_t + 1, d).
        time_grid: Grille temporelle.
        im_matrix: IM calculé par nested MC, shape (n_outer, n_t + 1).
        portfolio: Portefeuille.
        r: Taux sans risque.
        vols: Volatilités.
        corr_matrix: Matrice de corrélation.
        mpor: MPOR.
        seed: Seed.

    Returns:
        exposure_im: Exposition résiduelle, shape (n_outer, n_t + 1).
    """
    n_outer, n_steps, d = paths.shape
    exposure_im = np.zeros((n_outer, n_steps))
    rng = np.random.default_rng(seed + 2_000_000)

    for j in range(n_steps):
        t_j = time_grid[j]
        pv_tj = compute_mtm(paths[:, j, :], t_j, portfolio, r, vols)

        # Simuler un scénario MPOR pour chaque scénario extérieur
        spots_mpor = simulate_gbm_from_spot(
            paths[:, j, :],
            dt=mpor,
            vols=vols,
            r=r,
            corr_matrix=corr_matrix,
            rng=rng,
        )
        pv_tjpd = compute_mtm(spots_mpor, t_j + mpor, portfolio, r, vols)

        delta_pv = pv_tj - pv_tjpd
        exposure_im[:, j] = np.maximum(0.0, delta_pv - im_matrix[:, j])

    return exposure_im
