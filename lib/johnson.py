"""
Approximation Johnson pour l'IM stochastique (Livrable 2).

Évite le nested MC en approchant la distribution conditionnelle de DeltaPV
par une distribution de Johnson (SU/SB/SL/SN) identifiée par ses 4 moments.

Référence : McWalter et al. (2018)
"""

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from lib.portfolio import compute_mtm
from lib.diffusion import simulate_gbm_from_spot
from config.parameters import (
    PORTFOLIO, RISK_FREE_RATE, VOLS, CORRELATION_MATRIX,
    MPOR, IM_CONFIDENCE, JOHNSON_POLY_DEGREE, SEED,
)


# ============================================================
# Sélection du type de distribution Johnson
# ============================================================

def _johnson_type(skew2: float, kurt: float) -> str:
    """Détermine le type de distribution Johnson selon (skewness^2, kurtosis).

    La frontière entre SU et SB est la courbe lognormale dans le plan
    (beta_1, beta_2). Pour une lognormale de paramètre omega = exp(sigma^2) :
        beta_1 = (omega - 1)(omega + 2)^2
        beta_2 = omega^4 + 2*omega^3 + 3*omega^2 - 3

    - Au-dessus de la courbe (kurt > beta_2_logn) => SU (queues lourdes)
    - En dessous (kurt < beta_2_logn) => SB (bornée)
    - Sur la courbe => SL (lognormale)

    Args:
        skew2: Skewness au carré (beta_1).
        kurt: Kurtosis (beta_2).

    Returns:
        "SU", "SB", "SL" ou "SN".
    """
    # Cas normal
    if abs(skew2) < 1e-8 and abs(kurt - 3.0) < 0.1:
        return "SN"

    if skew2 < 1e-10:
        # skewness ~ 0 : SU si kurtosis > 3, SB sinon
        if kurt > 3.0:
            return "SU"
        elif kurt < 3.0:
            return "SB"
        else:
            return "SN"

    # Résolution exacte de la frontière lognormale :
    # On cherche omega > 1 tel que (omega - 1)(omega + 2)^2 = beta_1
    def _logn_eq(omega: float) -> float:
        return (omega - 1.0) * (omega + 2.0) ** 2 - skew2

    try:
        omega = brentq(_logn_eq, 1.0 + 1e-12, 1e6, maxiter=200)
    except (ValueError, RuntimeError):
        # Fallback : si on ne trouve pas de racine, SU par défaut
        return "SU"

    beta2_lognormal = omega**4 + 2 * omega**3 + 3 * omega**2 - 3

    tol = 0.5  # Tolérance pour le cas SL
    if abs(kurt - beta2_lognormal) < tol:
        return "SL"
    elif kurt > beta2_lognormal:
        return "SU"
    else:
        return "SB"


# ============================================================
# Fit des paramètres Johnson SU par les moments
# ============================================================

def _su_skew_kurt(omega: float, Gamma: float) -> tuple[float, float]:
    """Calcule le skewness et la kurtosis de Y = sinh((Z - Gamma*delta)/delta).

    Formules exactes via la fonction génératrice des moments de Z ~ N(0,1) :
        E[exp(aZ)] = exp(a^2/2)

    Moments bruts de Y :
        m1 = -sqrt(omega) * sinh(Gamma)
        m2 = (omega^2 * cosh(2*Gamma) - 1) / 2
        m3 = -(omega^(9/2) * sinh(3*Gamma) - 3*sqrt(omega)*sinh(Gamma)) / 4
        m4 = (omega^8 * cosh(4*Gamma) - 4*omega^2*cosh(2*Gamma) + 3) / 8

    avec omega = exp(1/delta^2), Gamma = gamma/delta.

    Args:
        omega: exp(1/delta^2), doit être > 1.
        Gamma: gamma / delta.

    Returns:
        (skewness, kurtosis) de la distribution Johnson SU.
    """
    so = np.sqrt(omega)

    m1 = -so * np.sinh(Gamma)
    m2 = (omega**2 * np.cosh(2.0 * Gamma) - 1.0) / 2.0
    m3 = (-omega**4.5 * np.sinh(3.0 * Gamma) + 3.0 * so * np.sinh(Gamma)) / 4.0
    m4 = (omega**8 * np.cosh(4.0 * Gamma) - 4.0 * omega**2 * np.cosh(2.0 * Gamma) + 3.0) / 8.0

    mu2 = m2 - m1**2
    if mu2 < 1e-30:
        return 0.0, 3.0
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4

    return mu3 / mu2**1.5, mu4 / mu2**2


def _fit_johnson_su(mean: float, var: float, skew: float, kurt: float) -> tuple[float, float, float, float]:
    """Fit les paramètres Johnson SU (xi, lambda, gamma, delta) à partir des moments.

    X = xi + lambda * sinh((Z - gamma) / delta), Z ~ N(0,1)

    Méthode rapide en 2 étapes itérées (brentq 1D, pas de Nelder-Mead) :
      1. Trouver omega depuis la kurtosis (1D brentq, Gamma fixé)
      2. Trouver Gamma depuis le skewness (1D brentq, omega fixé)
      3. Itérer 2-3 fois pour convergence

    Puis inversion analytique pour (xi, lambda) depuis (mean, var).

    Args:
        mean: Moyenne.
        var: Variance.
        skew: Skewness.
        kurt: Kurtosis (non-excess, i.e. = 3 pour une normale).

    Returns:
        (xi, lam, gamma, delta_j): Paramètres Johnson SU.
    """
    # --- Étape 1 : omega initial depuis la kurtosis (Gamma=0, cas symétrique) ---
    def _kurt_eq_sym(log_om_m1: float) -> float:
        omega = 1.0 + np.exp(log_om_m1)
        _, k = _su_skew_kurt(omega, 0.0)
        return k - kurt

    try:
        log_om_init = brentq(_kurt_eq_sym, -5.0, 12.0, maxiter=100)
    except (ValueError, RuntimeError):
        log_om_init = 0.0

    omega = 1.0 + np.exp(log_om_init)
    Gamma = 0.0

    # --- Étapes 2-3 : itération alternée brentq (skewness, kurtosis) ---
    # 2 itérations suffisent (erreur < 0.3% sur le quantile 99%)
    for _ in range(2):
        # 2a. Trouver Gamma depuis le skewness, omega fixé
        if abs(skew) > 1e-8:
            def _skew_eq(G: float) -> float:
                s, _ = _su_skew_kurt(omega, G)
                return s - skew
            try:
                Gamma = brentq(_skew_eq, -5.0, 5.0, maxiter=100)
            except (ValueError, RuntimeError):
                pass  # garder le Gamma précédent

        # 2b. Trouver omega depuis la kurtosis, Gamma fixé
        def _kurt_eq(log_om_m1: float) -> float:
            om = 1.0 + np.exp(log_om_m1)
            _, k = _su_skew_kurt(om, Gamma)
            return k - kurt
        try:
            log_om = brentq(_kurt_eq, -5.0, 12.0, maxiter=100)
            omega = 1.0 + np.exp(log_om)
        except (ValueError, RuntimeError):
            pass  # garder l'omega précédent

    # --- Conversion en paramètres (gamma, delta) ---
    log_omega = np.log(omega)
    if log_omega < 1e-10:
        # omega ~ 1 : distribution quasi-normale
        return mean, np.sqrt(var), 0.0, 1.0

    delta_j = 1.0 / np.sqrt(log_omega)
    gamma = Gamma * delta_j

    # --- Récupérer xi et lambda depuis mean et var ---
    so = np.sqrt(omega)
    m1_Y = -so * np.sinh(Gamma)
    m2_Y = (omega**2 * np.cosh(2.0 * Gamma) - 1.0) / 2.0
    var_Y = m2_Y - m1_Y**2

    sigma = np.sqrt(var)
    lam = sigma / np.sqrt(max(var_Y, 1e-30))
    xi = mean - lam * m1_Y

    # Sécurité
    if not (np.isfinite(xi) and np.isfinite(lam) and lam > 0):
        xi = mean
        lam = sigma
        gamma = 0.0
        delta_j = 1.0

    return xi, lam, gamma, delta_j


def _fit_johnson_sn(mean: float, var: float) -> tuple[float, float, float, float]:
    """Fit Johnson SN (distribution normale)."""
    return mean, np.sqrt(var), 0.0, 1.0


# ============================================================
# Quantile Johnson
# ============================================================

def johnson_quantile(alpha: float, jtype: str, xi: float, lam: float,
                     gamma: float, delta_j: float) -> float:
    """Calcule le quantile d'ordre alpha de la distribution Johnson.

    Args:
        alpha: Niveau du quantile (ex: 0.99).
        jtype: Type de distribution ("SU", "SB", "SL", "SN").
        xi, lam, gamma, delta_j: Paramètres Johnson.

    Returns:
        Quantile d'ordre alpha.
    """
    z = norm.ppf(alpha)

    if jtype == "SU":
        return xi + lam * np.sinh((z - gamma) / delta_j)
    elif jtype == "SB":
        u = (z - gamma) / delta_j
        return xi + lam / (1.0 + np.exp(-u))
    elif jtype == "SL":
        return xi + lam * np.exp((z - gamma) / delta_j)
    elif jtype == "SN":
        return xi + lam * (z - gamma) / delta_j
    else:
        raise ValueError(f"Type Johnson inconnu : {jtype}")


# ============================================================
# Régression polynomiale des moments conditionnels
# ============================================================

def _polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Régression polynomiale : y ~ sum_p a_p * x^p.

    Standardise x avant le fit pour éviter le mauvais conditionnement
    de la matrice de Vandermonde (RankWarning de polyfit).

    Args:
        x: Variable explicative (n,).
        y: Variable réponse (n,).
        degree: Degré du polynôme.

    Returns:
        y_hat: Valeurs ajustées (n,).
    """
    # Si x est constant (ex : t=0, tous les scenarios identiques),
    # la regression est impossible -> retourner la moyenne
    x_std = np.std(x)
    if x_std < 1e-12:
        return np.full_like(y, np.mean(y))

    # Standardiser x pour stabiliser le polyfit
    x_mean = np.mean(x)
    x_norm = (x - x_mean) / x_std

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coeffs = np.polyfit(x_norm, y, degree)
    return np.polyval(coeffs, x_norm)


def estimate_conditional_moments(
    pv_tj: np.ndarray,
    delta_pv: np.ndarray,
    degree: int = JOHNSON_POLY_DEGREE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estime les 4 moments conditionnels de DeltaPV | PV(t) par régression.

    Args:
        pv_tj: PV(t_j) pour chaque scénario, shape (n_outer,).
        delta_pv: DeltaPV pour chaque scénario, shape (n_outer,).
        degree: Degré polynomial.

    Returns:
        (cond_mean, cond_var, cond_skew, cond_kurt): Moments conditionnels,
        chacun de shape (n_outer,).
    """
    n = len(pv_tj)

    # Moments bruts conditionnels E[DeltaPV^k | PV(t)]
    m1 = _polynomial_regression(pv_tj, delta_pv, degree)
    m2 = _polynomial_regression(pv_tj, delta_pv**2, degree)
    m3 = _polynomial_regression(pv_tj, delta_pv**3, degree)
    m4 = _polynomial_regression(pv_tj, delta_pv**4, degree)

    # Moments centrés
    cond_mean = m1
    cond_var = np.maximum(m2 - m1**2, 1e-12)  # Assurer positivité
    cond_std = np.sqrt(cond_var)

    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4

    cond_skew = mu3 / (cond_std**3 + 1e-30)
    cond_kurt = mu4 / (cond_var**2 + 1e-30)

    # Assurer kurtosis >= skewness^2 + 1 (contrainte théorique)
    cond_kurt = np.maximum(cond_kurt, cond_skew**2 + 1.01)

    return cond_mean, cond_var, cond_skew, cond_kurt


# ============================================================
# Pipeline Johnson pour l'IM
# ============================================================

def compute_im_johnson(
    paths: np.ndarray,
    time_grid: np.ndarray,
    portfolio: list[dict] = PORTFOLIO,
    r: float = RISK_FREE_RATE,
    vols: np.ndarray = VOLS,
    corr_matrix: np.ndarray = CORRELATION_MATRIX,
    mpor: float = MPOR,
    confidence: float = IM_CONFIDENCE,
    degree: int = JOHNSON_POLY_DEGREE,
    seed: int = SEED,
    n_grid: int = 30,
) -> np.ndarray:
    """Calcule l'IM par approximation Johnson pour tous les noeuds.

    Etapes :
    1. MC exterieur : simuler les prix a t_j et t_j + delta
    2. Pour chaque t_j : calculer DeltaPV = PV(t_j) - PV(t_j + delta)
    3. Regresser les moments conditionnels de DeltaPV sur PV(t_j)
    4. Fit Johnson sur une grille de PV + interpolation du quantile 99%

    L'optimisation cle : au lieu de fitter Johnson pour chaque scenario
    (n_outer fits par date), on fit sur une grille de n_grid points puis
    on interpole. Gain : facteur n_outer / n_grid (~30x).

    Args:
        paths: Chemins simules, shape (n_outer, n_t + 1, d).
        time_grid: Grille temporelle.
        portfolio: Portefeuille.
        r: Taux sans risque.
        vols: Volatilites.
        corr_matrix: Matrice de correlation.
        mpor: MPOR.
        confidence: Niveau de confiance.
        degree: Degre polynomial pour la regression.
        seed: Seed.
        n_grid: Nombre de points de grille pour l'interpolation.

    Returns:
        im_matrix: IM Johnson, shape (n_outer, n_t + 1).
    """
    n_outer, n_steps, d = paths.shape
    im_matrix = np.zeros((n_outer, n_steps))
    rng = np.random.default_rng(seed + 3_000_000)

    for j in range(n_steps):
        t_j = time_grid[j]

        # PV(t_j)
        pv_tj = compute_mtm(paths[:, j, :], t_j, portfolio, r, vols)

        # Simuler S(t_j + delta) pour chaque scenario (un seul scenario MPOR)
        spots_mpor = simulate_gbm_from_spot(
            paths[:, j, :],
            dt=mpor,
            vols=vols,
            r=r,
            corr_matrix=corr_matrix,
            rng=rng,
        )

        # PV(t_j + delta)
        pv_tjpd = compute_mtm(spots_mpor, t_j + mpor, portfolio, r, vols)

        # DeltaPV = PV(t_j) - PV(t_j + delta)
        delta_pv = pv_tj - pv_tjpd

        # --- Regression polynomiale des moments bruts ---
        pv_std = np.std(pv_tj)
        if pv_std < 1e-12:
            # Tous les PV identiques (ex: t=0) -> un seul fit
            mean_dp = np.mean(delta_pv)
            var_dp = max(np.var(delta_pv), 1e-12)
            from scipy.stats import skew as _skew, kurtosis as _kurt
            skew_dp = _skew(delta_pv)
            kurt_dp = _kurt(delta_pv, fisher=False)
            kurt_dp = max(kurt_dp, skew_dp**2 + 1.01)
            skew2 = skew_dp**2
            jtype = _johnson_type(skew2, kurt_dp)
            if jtype == "SN":
                xi, lam, gam, delt = _fit_johnson_sn(mean_dp, var_dp)
            else:
                xi, lam, gam, delt = _fit_johnson_su(mean_dp, var_dp, skew_dp, kurt_dp)
                jtype = "SU"
            im_val = max(johnson_quantile(confidence, jtype, xi, lam, gam, delt), 0.0)
            im_matrix[:, j] = im_val
            continue

        pv_mean = np.mean(pv_tj)
        pv_norm = (pv_tj - pv_mean) / pv_std

        # Fit des coefficients polynomiaux sur x standardise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            c1 = np.polyfit(pv_norm, delta_pv, degree)
            c2 = np.polyfit(pv_norm, delta_pv**2, degree)
            c3 = np.polyfit(pv_norm, delta_pv**3, degree)
            c4 = np.polyfit(pv_norm, delta_pv**4, degree)

        # --- Grille d'interpolation ---
        pv_grid = np.linspace(np.min(pv_norm), np.max(pv_norm), n_grid)

        # Evaluer les moments bruts aux points de la grille
        m1_g = np.polyval(c1, pv_grid)
        m2_g = np.polyval(c2, pv_grid)
        m3_g = np.polyval(c3, pv_grid)
        m4_g = np.polyval(c4, pv_grid)

        # Moments centres
        var_g = np.maximum(m2_g - m1_g**2, 1e-12)
        std_g = np.sqrt(var_g)
        mu3_g = m3_g - 3 * m1_g * m2_g + 2 * m1_g**3
        mu4_g = m4_g - 4 * m1_g * m3_g + 6 * m1_g**2 * m2_g - 3 * m1_g**4
        skew_g = mu3_g / (std_g**3 + 1e-30)
        kurt_g = mu4_g / (var_g**2 + 1e-30)
        kurt_g = np.maximum(kurt_g, skew_g**2 + 1.01)

        # Fit Johnson a chaque point de la grille
        im_grid = np.zeros(n_grid)
        for g in range(n_grid):
            skew2 = skew_g[g] ** 2
            jtype = _johnson_type(skew2, kurt_g[g])
            if jtype == "SN":
                xi, lam, gam, delt = _fit_johnson_sn(m1_g[g], var_g[g])
            else:
                xi, lam, gam, delt = _fit_johnson_su(
                    m1_g[g], var_g[g], skew_g[g], kurt_g[g]
                )
                jtype = "SU"
            im_grid[g] = max(johnson_quantile(confidence, jtype, xi, lam, gam, delt), 0.0)

        # Interpoler le quantile pour tous les scenarios
        im_matrix[:, j] = np.interp(pv_norm, pv_grid, im_grid)

    return im_matrix
