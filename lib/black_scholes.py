"""
Pricing Black-Scholes pour options européennes vanilles.

Formules vectorisées (NumPy) pour call et put européens.
"""

import numpy as np
from scipy.stats import norm


def bs_d1(S: np.ndarray, K: float, tau: np.ndarray, r: float, sigma: float) -> np.ndarray:
    """Calcule d1 de la formule Black-Scholes.

    Args:
        S: Prix spot (array).
        K: Prix d'exercice.
        tau: Maturité résiduelle T - t (array, même shape que S).
        r: Taux sans risque.
        sigma: Volatilité.

    Returns:
        d1 (array, même shape que S).
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_d2(d1: np.ndarray, sigma: float, tau: np.ndarray) -> np.ndarray:
    """Calcule d2 = d1 - sigma * sqrt(tau)."""
    return d1 - sigma * np.sqrt(tau)


def bs_call(S: np.ndarray, K: float, tau: np.ndarray, r: float, sigma: float) -> np.ndarray:
    """Prix d'un call européen par Black-Scholes.

    C = S * Phi(d1) - K * exp(-r*tau) * Phi(d2)

    Args:
        S: Prix spot (array).
        K: Prix d'exercice.
        tau: Maturité résiduelle (array, > 0).
        r: Taux sans risque.
        sigma: Volatilité.

    Returns:
        Prix du call (array, même shape que S).
    """
    d1 = bs_d1(S, K, tau, r, sigma)
    d2 = bs_d2(d1, sigma, tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_put(S: np.ndarray, K: float, tau: np.ndarray, r: float, sigma: float) -> np.ndarray:
    """Prix d'un put européen par Black-Scholes.

    P = -S * Phi(-d1) + K * exp(-r*tau) * Phi(-d2)

    Args:
        S: Prix spot (array).
        K: Prix d'exercice.
        tau: Maturité résiduelle (array, > 0).
        r: Taux sans risque.
        sigma: Volatilité.

    Returns:
        Prix du put (array, même shape que S).
    """
    d1 = bs_d1(S, K, tau, r, sigma)
    d2 = bs_d2(d1, sigma, tau)
    return -S * norm.cdf(-d1) + K * np.exp(-r * tau) * norm.cdf(-d2)


def bs_price(option_type: str, S: np.ndarray, K: float, tau: np.ndarray,
             r: float, sigma: float) -> np.ndarray:
    """Prix Black-Scholes selon le type d'option.

    Args:
        option_type: "call" ou "put".
        S: Prix spot (array).
        K: Prix d'exercice.
        tau: Maturité résiduelle (array, > 0).
        r: Taux sans risque.
        sigma: Volatilité.

    Returns:
        Prix de l'option (array).

    Raises:
        ValueError: Si option_type n'est pas "call" ou "put".
    """
    if option_type == "call":
        return bs_call(S, K, tau, r, sigma)
    elif option_type == "put":
        return bs_put(S, K, tau, r, sigma)
    else:
        raise ValueError(f"Type d'option inconnu : {option_type!r}. Attendu 'call' ou 'put'.")
