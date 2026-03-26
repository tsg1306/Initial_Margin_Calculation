# Stochastic Initial Margin for Counterparty Credit Risk (CCR)

A Python framework for computing **stochastic Initial Margin (IM)** in the context of Counterparty Credit Risk, comparing brute-force Nested Monte Carlo with the Johnson distribution approximation (McWalter et al., 2018).

## Overview

This project computes exposure metrics (EE, EEE, EEPE) for a portfolio of European options under bilateral CSA with stochastic IM. It implements two approaches to estimate the conditional 99% VaR of P&L changes over the Margin Period of Risk (MPOR):

1. **Nested Monte Carlo** (brute-force) — inner simulations at each outer node to build the empirical P&L distribution
2. **Johnson approximation** — fits a Johnson distribution (SU/SB/SL/SN) using the first four conditional moments, avoiding the costly inner MC loop

The comparison demonstrates that the Johnson method achieves comparable accuracy with a speedup factor proportional to the number of inner scenarios.

## Key Features

- **Multi-asset correlated GBM** diffusion under the risk-neutral measure (exact discretization via Cholesky)
- **Black-Scholes closed-form pricing** for European calls and puts
- **Nested Monte Carlo IM** with configurable inner/outer scenario counts
- **Johnson distribution fit** using moment matching (skewness/kurtosis plane classification)
- **Exposure metrics**: Expected Exposure (EE), Effective EE, EEPE, and EAD computation
- **Centralized configuration** — all parameters in a single file (`config/parameters.py`)
- Fully vectorized NumPy implementation for performance

## Portfolio

| # | Type | Underlying | S₀  | Strike | Maturity | Vol  | Position |
|---|------|------------|-----|--------|----------|------|----------|
| 1 | Call | A          | 100 | 105    | 2Y       | 20%  | Long     |
| 2 | Put  | A          | 100 | 95     | 2Y       | 20%  | Long     |
| 3 | Call | B          | 150 | 160    | 2Y       | 25%  | Long     |
| 4 | Put  | C          | 80  | 75     | 2Y       | 30%  | Short    |
| 5 | Call | C          | 80  | 85     | 2Y       | 30%  | Long     |

Correlations: ρ(A,B) = 0.6, ρ(A,C) = 0.4, ρ(B,C) = 0.5

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Risk-free rate | 3% | Continuous compounding |
| MPOR | 10 business days (≈ 2/52 yr) | Standard bilateral CSA |
| EEPE horizon | 1 year | Regulatory standard |
| IM confidence | 99% | VaR quantile level |
| N_outer | 500 | Outer Monte Carlo scenarios |
| N_inner | 500 | Inner MC scenarios (nested only) |
| Time steps | 52 | Weekly grid over 1 year |

## Project Structure

```
├── config/
│   └── parameters.py          # All configurable parameters
├── lib/
│   ├── black_scholes.py       # BS pricing (call/put)
│   ├── diffusion.py           # Correlated multi-asset GBM simulation
│   ├── portfolio.py           # Portfolio class and MtM computation
│   ├── margin.py              # IM calculation (nested MC + exposure with IM)
│   ├── exposure.py            # EE, EEE, EEPE metrics
│   ├── johnson.py             # Johnson distribution fit and IM approximation
│   └── utils.py               # Helpers, validation, timing
├── notebooks/
│   ├── 01_test_phase1.ipynb   # Validation: BS pricing, GBM, exposure
│   ├── 02_test_phase2.ipynb   # Validation: nested MC IM
│   └── 03_test_phase3.ipynb   # Validation: Johnson approximation
├── tests/                     # Unit tests (pytest)
├── report/                    # Generated figures and theoretical writeup
├── main.py                    # Full pipeline: both methods + comparison
└── CLAUDE.md                  # Project specification
```

## Quick Start

### Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib

### Installation

```bash
pip install numpy scipy matplotlib
```

### Run

```bash
# Full run (N_outer=500, N_inner=500)
python main.py

# Fast mode for quick testing (N_outer=200, N_inner=200)
python main.py --fast
```

### Tests

```bash
pytest tests/
```

## Output

The pipeline produces:

- **Console output**: EEPE values, IM statistics, comparison table, timing
- **Figures** saved in `report/`:
  - `fig1` — Sample GBM paths for each underlying
  - `fig2` — EE/EEE profiles without IM
  - `fig3` — EE/EEE comparison: no IM vs nested MC vs Johnson
  - `fig4` — Scatter plots: nested MC IM vs Johnson IM
  - `fig5` — Mean IM over time + MAE between methods
  - `fig6` — IM distribution at t = 0.5Y

## Methodology

### IM Computation

The Initial Margin at time t is defined as:

```
IM(t) = Q_99%(ΔPV | F_t)    where ΔPV = PV(t) - PV(t + δ)
```

A positive ΔPV represents a portfolio loss over the MPOR period δ.

### Nested Monte Carlo (Deliverable 1)

For each outer scenario at each time step:
1. Simulate N_inner sub-scenarios over the MPOR
2. Price the portfolio at t + δ using Black-Scholes
3. Compute P&L changes: ΔPV = PV(t) - PV(t + δ)
4. IM = empirical 99th percentile of ΔPV

### Johnson Approximation (Deliverable 2)

Replaces the inner MC loop by:
1. Estimating the first four conditional moments of ΔPV via polynomial regression
2. Fitting a Johnson distribution (SU/SB/SL/SN) based on the (skewness², kurtosis) pair
3. Computing the 99% quantile analytically

### Exposure with IM (Deliverable 3)

Residual exposure after IM collateralization:

```
E_IM(t) = max(0, ΔPV(t, t+δ) - IM(t))
```

Then EE → EEE → EEPE are computed as usual.

## References

- McWalter, T., Kienitz, J., Nowaczyk, N., Rudd, R., & Acar, S. (2018). *Dynamic Initial Margin Estimation Based on Quantiles of Johnson Distributions*
- CRR Art. 284 — EEPE computation under the risk-neutral measure
- BCBS/IOSCO — Margin requirements for non-centrally cleared derivatives

## License

Academic project — EXIOM Partners × CentraleSupélec (M2 Quantitative Finance).
