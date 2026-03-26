# CLAUDE.md — Projet IM Stochastique CCR

## Contexte projet

Projet académique EXIOM Partners (3-4 semaines) : calcul de l'Initial Margin stochastique pour le risque de contrepartie (CCR). Étudiant M2 CentraleSupélec, spécialisation finance quantitative / ML.

## Documents de référence

- `NestedMcStochasticIm.pdf` : description du projet + bibliographie (McWalter et al. 2018)
- `Initial Margin.pdf` (EXIOM, 15 pages) : définitions CCR, réglementation CRR/TRIM, calcul RWA_CCR et RWA_CVA, métriques E/EE/EEE/EEPE, netting, collatéralisation VM/IM
- `ST7-1-MeasureProbability.pdf`, `ST7-2-Stochastic_processes.pdf`, `ST7-5-Multi-period_model.pdf` : cours CentraleSupélec (Carassus & Guo) — notation et cadre théorique à respecter
- Capture d'écran des formules Black-Scholes (call/put européen)

## Livrables

1. **Livrable 1** : Calcul IM stochastique par nested Monte Carlo (brute force)
2. **Livrable 2** : Approximation Johnson pour éviter le nested MC (McWalter et al. 2018)
3. **Livrable 3** : Comparaison EEPE, analyse erreur et performance
4. **Rapport** : document théorique + résultats (Word/PDF)

---

## Hypothèses fixées

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Mesure de diffusion | Risque-neutre (drift = r) | Standard EEPE réglementaire (CRR art. 284), cohérent avec pricing BS |
| Taux sans risque r | 3% | — |
| MPOR δ | 10 jours business ≈ 2/52 an | Standard bilatéral CSA |
| CSA | Simplifié : VM = MtM, threshold = 0, MTA = 0 | Isole l'effet IM sur la période MPOR |
| Maturité options T_opt | 2 ans | Options vivantes sur tout l'horizon EEPE (1 an) |
| Horizon EEPE | 1 an | Standard réglementaire |
| Pricing | Black-Scholes (formules fermées) | Options vanilles européennes, formules fournies par le sujet |
| Diffusion | GBM multidimensionnel corrélé (Cholesky) | Solution exacte (pas d'erreur de discrétisation) |
| Nombre de sous-jacents d | 3 (A, B, C) | — |

## Portefeuille

5 options sur 3 sous-jacents :

| # | Type | Sous-jacent | S₀ | K | T | σ | Position |
|---|------|-------------|-----|-----|-------|------|----------|
| 1 | Call | A | 100 | 105 | 2Y | 20% | Long (+1) |
| 2 | Put  | A | 100 | 95  | 2Y | 20% | Long (+1) |
| 3 | Call | B | 150 | 160 | 2Y | 25% | Long (+1) |
| 4 | Put  | C | 80  | 75  | 2Y | 30% | Short (-1) |
| 5 | Call | C | 80  | 85  | 2Y | 30% | Long (+1) |

Corrélations : ρ(A,B)=0.6, ρ(A,C)=0.4, ρ(B,C)=0.5

Matrice de corrélation Σ (à vérifier définie positive via Cholesky) :
```
Σ = [[1.0, 0.6, 0.4],
     [0.6, 1.0, 0.5],
     [0.4, 0.5, 1.0]]
```

## Paramètres MC suggérés

- N_outer = 500-1000 (scénarios extérieurs)
- N_inner = 500-1000 (scénarios intérieurs pour l'IM nested)
- N_t = 52 (pas de temps hebdomadaires sur 1 an)
- Grille temporelle : t_j = j/52, j = 0, ..., 52

---

## Architecture code

```
project/
├── config/
│   └── parameters.py          # TOUS les paramètres modifiables ici
├── lib/
│   ├── __init__.py
│   ├── black_scholes.py       # Pricing BS : call, put (+ greeks si besoin)
│   ├── diffusion.py           # GBM multidim corrélé, Cholesky
│   ├── portfolio.py           # Classe Portfolio, MtM, netting
│   ├── margin.py              # IM stochastique (nested MC + Johnson), VM, CSA
│   ├── exposure.py            # E, EE, EEE, EEPE
│   ├── johnson.py             # Distribution Johnson (SU/SB/SL/SN), fit, quantile
│   └── utils.py               # Helpers, validation, stats
├── notebooks/                 # Exploration / validation
│   ├── 01_validation_bs.ipynb
│   ├── 02_nested_mc.ipynb
│   ├── 03_johnson_approx.ipynb
│   └── 04_comparison.ipynb
├── main.py                    # Pipeline complet (exécutable)
├── tests/                     # Tests unitaires
├── CLAUDE.md                  # Ce fichier
└── report/                    # Rapport final
```

**Principe clé :** `parameters.py` centralise tout ce qu'on veut faire varier (drift, portefeuille, N_outer, N_inner, grille, etc.). Le reste du code importe depuis `parameters.py`. On peut changer un paramètre et relancer sans toucher au code.

---

## Spécifications théoriques détaillées

### Notation (conforme aux cours ST7 de CentraleSupélec)

- Espace probabilisé filtré : (Ω, F, 𝔽, ℚ) avec 𝔽 = (ℱ_t)
- Mesure risque-neutre : ℚ
- Processus actualisé : S̃ = S/S⁰
- FTAP : absence d'arbitrage ⟺ ∃ mesure martingale équivalente ℚ ~ ℙ
- Prix sans arbitrage d'un payoff H : π_t = 𝔼^ℚ[H/S⁰_T | ℱ_t]
- Densité de Radon-Nikodym : dℚ/dℙ = Z

### Dynamique des sous-jacents sous ℚ

```
dSⁱ_t = r Sⁱ_t dt + σᵢ Sⁱ_t dWⁱ_t,  i = 1,...,d
```

avec dWⁱ·dWʲ = ρᵢⱼ dt.

Solution exacte (discrétisation sans erreur) :
```
Sⁱ(t_{j+1}) = Sⁱ(t_j) × exp[(r - σᵢ²/2)Δt + σᵢ √Δt Zⁱ_{j+1}]
```

avec Z = (Z¹,...,Zᵈ)ᵀ ~ 𝒩(0, Σ), simulé via Cholesky : Z = L·ε, ε ~ 𝒩(0,I).

### Pricing Black-Scholes

Call :
```
C(t, S, K, T, r, σ) = S·Φ(d₁) - K·e^{-r(T-t)}·Φ(d₂)
```

Put :
```
P(t, S, K, T, r, σ) = -S·Φ(-d₁) + K·e^{-r(T-t)}·Φ(-d₂)
```

avec :
```
d₁ = [ln(S/K) + (r + σ²/2)(T-t)] / [σ√(T-t)]
d₂ = d₁ - σ√(T-t)
```

### MtM du portefeuille

```
MtM_i(t_j) = Σ_{k=1}^{N_p} η_k · V_k(t_j, S^{i(k)}_i(t_j))
```

η_k = +1 (long) ou -1 (short), V_k = prix BS de l'option k au spot simulé.

### Métriques d'exposition

```
E(t) = max(0, MtM(t))                          # Exposition
EE(t_j) = (1/N_outer) Σ max(0, MtM_i(t_j))     # Expected Exposure
EEE(t_j) = max_{l≤j} EE(t_l)                    # Effective EE (non-décroissant)
EEPE(T) = (1/T) ∫₀ᵀ EEE(t) dt                  # Effective EPE (moyenne temporelle)
```

EEPE discrétisé par méthode des trapèzes.

### Convention de signe pour l'IM

**CRITIQUE :** On définit la perte comme ΔPV = PV(t) - PV(t+δ), de sorte qu'une dépréciation du portefeuille donne ΔPV > 0.

```
IM(t) = Q_{99%}(ΔPV(t, t+δ) | ℱ_t)    avec ΔPV = PV(t) - PV(t+δ)
```

### Nested Monte Carlo (Livrable 1)

Pour chaque nœud (i, t_j) du MC extérieur :

1. Simuler N_inner sous-scénarios sur le MPOR : S^{(i,k)}(t_j+δ) à partir de S^{(i)}(t_j)
2. Pricer le portefeuille : PV_{i,k}(t_j+δ) par BS
3. Calculer les P&L : ΔPV_{i,k} = PV_i(t_j) - PV_{i,k}(t_j+δ)
4. Estimer l'IM : IM̂_i(t_j) = quantile empirique 99% des {ΔPV_{i,k}}

Coût total : N_outer × N_t × (1 + N_inner) × N_p pricings BS.

### Approximation Johnson (Livrable 2)

Idée : approcher la distribution conditionnelle de ΔPV | ℱ_t par une distribution de Johnson (4 moments).

**Étapes :**
1. MC extérieur classique (pas de MC intérieur)
2. Pour chaque t_j, estimer les 4 premiers moments conditionnels de ΔPV par régression sur PV(t_j)
3. Fit d'une distribution Johnson (SU/SB/SL/SN) selon le couple (skewness², kurtosis)
4. Calcul analytique du quantile 99%

Distribution Johnson SU :
```
X = ξ + λ·sinh((Z - γ)/δ_J),  Z ~ 𝒩(0,1)
Q_α(X) = ξ + λ·sinh((Φ⁻¹(α) - γ)/δ_J)
```

Gain computationnel : facteur ≈ N_inner.

### Exposition résiduelle avec IM

```
E^IM_i(t_j) = max(0, ΔPV_i(t_j, t_j+δ) - IM_i(t_j))
```

Puis EE^IM → EEE^IM → EEPE^IM comme ci-dessus.

### Comparaison (Livrable 3)

Métriques :
- Erreur absolue/relative sur EEPE
- Erreur nœud par nœud sur l'IM : (1/N_outer) Σ |IM^nested - IM^Johnson|
- Ratio de temps de calcul
- Graphiques : profils EE/EEE comparés, scatter IM nested vs Johnson

---

## Tâche 1 : Formalisation théorique (document Markdown/LaTeX)

Créer un fichier `report/formalisation_theorique.md` contenant la formalisation mathématique complète du projet. Structure :

1. Cadre probabiliste et notations (notation ST7 CentraleSupélec)
2. Dynamique des sous-jacents (GBM multidim, Cholesky)
3. Pricing BS (formules, fondement théorique via FTAP)
4. Portefeuille et MtM
5. Métriques d'exposition (E, EE, EEE, EEPE)
6. Collatéralisation et IM (VM, CSA, convention de signe)
7. Nested Monte Carlo pour l'IM
8. Approximation Johnson (famille SU/SB/SL/SN, estimation moments, fit, quantile)
9. Calcul EEPE avec IM stochastique
10. Analyse d'erreur et convergence
11. Hypothèses du projet (tableau récapitulatif)

**Exigences :**
- Notation conforme aux cours ST7 (ℚ pour risque-neutre, ℱ_t pour filtration, etc.)
- Formules LaTeX complètes
- Références explicites aux slides ST7 quand pertinent
- Justifications des choix (mesure ℚ, MPOR 10j, CSA simplifié)
- Convention de signe clairement expliquée

## Tâche 2 : Implémentation Python

### Phase 1 — Briques de base

- `config/parameters.py` : tous les paramètres
- `lib/black_scholes.py` : pricing call/put BS vectorisé (numpy)
- `lib/diffusion.py` : GBM multidim corrélé, vérification Cholesky
- `lib/portfolio.py` : classe Portfolio, calcul MtM vectorisé
- `lib/exposure.py` : E, EE, EEE, EEPE

### Phase 2 — Nested MC (Livrable 1)

- `lib/margin.py` : calcul IM par nested MC
- Pipeline complet : diffusion → pricing → nested MC IM → exposition → EEPE

### Phase 3 — Johnson (Livrable 2)

- `lib/johnson.py` : fit Johnson, estimation moments conditionnels, calcul quantile
- Pipeline : diffusion → pricing → Johnson IM → exposition → EEPE

### Phase 4 — Comparaison (Livrable 3)

- `main.py` : exécute les deux approches, compare, génère graphiques
- Graphiques : profils EE/EEE, scatter IM, distributions, temps de calcul

### Exigences techniques

- Python 3.10+, NumPy, SciPy, Matplotlib
- Code vectorisé (pas de boucles Python sur les scénarios si possible)
- Reproductibilité : seed fixé dans parameters.py
- Docstrings complètes
- Type hints

---

## Plan d'attaque (4 semaines)

- **Semaine 1 :** Formalisation théorique + briques de base (pricing, diffusion, exposition)
- **Semaine 2 :** Nested MC complet (Livrable 1) + validation
- **Semaine 3 :** Approximation Johnson (Livrable 2) + benchmarks
- **Semaine 4 :** Comparaison (Livrable 3) + rapport final

---

## Points d'attention

1. **Convention de signe IM :** ΔPV = PV(t) - PV(t+δ), perte = positif
2. **Maturité résiduelle :** à chaque nœud, τ = T_opt - t_j. Les options restent vivantes (T_opt=2Y > horizon EEPE=1Y)
3. **Cholesky :** vérifier que Σ est définie positive avant de décomposer
4. **Quantile empirique 99% :** avec N_inner=500, le quantile 99% = 5ème plus grande valeur → variance significative, en discuter dans le rapport
5. **Johnson SU vs SB :** le choix dépend du couple (skewness², kurtosis) de la distribution empirique des ΔPV
6. **Régression des moments :** tester polynomiale vs kernel, documenter la sensibilité
7. **Seed :** fixer np.random.seed pour reproductibilité
