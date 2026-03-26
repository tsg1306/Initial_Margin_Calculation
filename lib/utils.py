"""
Helpers, validation et utilitaires pour le projet IM Stochastique CCR.
"""

import time
import numpy as np
from config.parameters import CORRELATION_MATRIX, ALPHA_EAD


def validate_correlation_matrix(corr: np.ndarray) -> None:
    """Verifie que la matrice de correlation est valide.

    Raises:
        ValueError: Si la matrice n'est pas symetrique, definie positive,
                    ou si la diagonale n'est pas 1.
    """
    if not np.allclose(corr, corr.T):
        raise ValueError("La matrice de correlation n'est pas symetrique.")
    if not np.allclose(np.diag(corr), 1.0):
        raise ValueError("La diagonale de la matrice doit etre 1.")
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.any(eigenvalues <= 0):
        raise ValueError(
            f"La matrice n'est pas definie positive. "
            f"Valeurs propres : {eigenvalues}"
        )


def compute_ead(eepe: float, alpha: float = ALPHA_EAD) -> float:
    """Calcule l'Exposure at Default : EAD = alpha * EEPE."""
    return alpha * eepe


class Timer:
    """Context manager pour mesurer le temps d'execution."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            print(f"  [{self.name}] {self.elapsed:.2f}s")


def print_header(title: str, width: int = 70) -> None:
    """Affiche un titre encadre."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None) -> None:
    """Affiche un tableau formate."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
        col_widths = [w + 2 for w in col_widths]

    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * sum(col_widths))
    for row in rows:
        row_str = "".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
        print(row_str)
