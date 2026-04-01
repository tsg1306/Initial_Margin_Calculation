"""
Paramètres centralisés du projet IM Stochastique CCR.

Tous les paramètres modifiables sont définis ici.
Le reste du code importe depuis ce module.
"""

import numpy as np

# ============================================================
# Seed pour la reproductibilité
# ============================================================
SEED: int = 42

# ============================================================
# Paramètres de marché
# ============================================================
RISK_FREE_RATE: float = 0.03  # Taux sans risque continu (3%)

# ============================================================
# Paramètres temporels
# ============================================================
T_EEPE: float = 1.0       # Horizon EEPE (1 an)
N_T: int = 52              # Nombre de pas de temps (hebdomadaires)
DELTA_T: float = T_EEPE / N_T  # Pas de temps

# MPOR : Margin Period of Risk (10 jours business ≈ 2 semaines)
MPOR: float = 2.0 / 52.0  # ≈ 0.03846 années

# Grille temporelle : t_j = j * DELTA_T, j = 0, ..., N_T
TIME_GRID: np.ndarray = np.linspace(0.0, T_EEPE, N_T + 1)

# ============================================================
# Paramètres Monte Carlo
# ============================================================
N_OUTER: int = 1000   # Scénarios extérieurs
N_INNER: int = 1000    # Scénarios intérieurs (nested MC)

# ============================================================
# Niveau de confiance pour l'IM
# ============================================================
IM_CONFIDENCE: float = 0.99  # Quantile 99%

# ============================================================
# Sous-jacents
# ============================================================
N_ASSETS: int = 3  # Nombre de sous-jacents (A, B, C)

SPOTS: np.ndarray = np.array([100.0, 150.0, 80.0])          # S0 pour A, B, C
VOLS: np.ndarray = np.array([0.20, 0.25, 0.30])             # σ pour A, B, C
ASSET_NAMES: list[str] = ["A", "B", "C"]

# Matrice de corrélation
CORRELATION_MATRIX: np.ndarray = np.array([
    [1.0, 0.6, 0.4],
    [0.6, 1.0, 0.5],
    [0.4, 0.5, 1.0],
])

# ============================================================
# Portefeuille : liste de dictionnaires décrivant chaque option
# ============================================================
# Chaque option est définie par :
#   - type : "call" ou "put"
#   - asset_idx : indice du sous-jacent (0=A, 1=B, 2=C)
#   - strike : prix d'exercice K
#   - maturity : maturité T_opt (en années)
#   - position : +1 (long) ou -1 (short)

PORTFOLIO: list[dict] = [
    {"type": "call", "asset_idx": 0, "strike": 105.0, "maturity": 2.0, "position": +1},
    {"type": "put",  "asset_idx": 0, "strike": 95.0,  "maturity": 2.0, "position": +1},
    {"type": "call", "asset_idx": 1, "strike": 160.0, "maturity": 2.0, "position": +1},
    {"type": "put",  "asset_idx": 2, "strike": 75.0,  "maturity": 2.0, "position": -1},
    {"type": "call", "asset_idx": 2, "strike": 85.0,  "maturity": 2.0, "position": +1},
]

N_OPTIONS: int = len(PORTFOLIO)

# ============================================================
# Paramètres pour l'approximation Johnson
# ============================================================
JOHNSON_POLY_DEGREE: int = 5  # Degré de la régression polynomiale

# ============================================================
# Paramètres alpha pour RWA
# ============================================================
ALPHA_EAD: float = 1.4  # Facteur alpha réglementaire
