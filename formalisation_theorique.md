# Formalisation Théorique — IM Stochastique pour le Risque de Contrepartie (CCR)

---

## Table des matières

1. [Cadre probabiliste et notations](#1-cadre-probabiliste-et-notations)
2. [Dynamique des sous-jacents](#2-dynamique-des-sous-jacents)
3. [Pricing Black-Scholes](#3-pricing-black-scholes)
4. [Portefeuille et Mark-to-Market](#4-portefeuille-et-mark-to-market)
5. [Métriques d'exposition](#5-métriques-dexposition)
6. [Collatéralisation et Initial Margin](#6-collatéralisation-et-initial-margin)
7. [Nested Monte Carlo pour l'IM](#7-nested-monte-carlo-pour-lim)
8. [Approximation Johnson](#8-approximation-johnson)
9. [Calcul de l'EEPE avec IM stochastique](#9-calcul-de-leepe-avec-im-stochastique)
10. [Analyse d'erreur et convergence](#10-analyse-derreur-et-convergence)
11. [Hypothèses du projet](#11-hypothèses-du-projet)

---

## 1. Cadre probabiliste et notations

### 1.1 Espace probabilisé filtré

On se place sur un espace probabilisé filtré $(\Omega, \mathcal{F}, \mathbb{F}, \mathbb{Q})$ où :

- $\Omega$ est l'ensemble des états du monde.
- $\mathcal{F}$ est la tribu complète.
- $\mathbb{F} = (\mathcal{F}_t)_{t \geq 0}$ est la filtration naturelle (satisfaisant les conditions habituelles de complétude et de continuité à droite).
- $\mathbb{Q}$ est la mesure martingale équivalente (mesure risque-neutre).

### 1.2 Mesure risque-neutre

Par le **Premier Théorème Fondamental de l'Asset Pricing** (FTAP, cf. cours ST7-1), l'absence d'opportunité d'arbitrage est équivalente à l'existence d'une mesure de probabilité $\mathbb{Q}$ équivalente à $\mathbb{P}$ ($\mathbb{Q} \sim \mathbb{P}$) telle que tout processus de prix actualisé $\tilde{S} = S / S^0$ soit une $\mathbb{Q}$-martingale.

Le lien entre les deux mesures est donné par la densité de Radon-Nikodym :

$$\frac{d\mathbb{Q}}{d\mathbb{P}} = Z$$

### 1.3 Prix sans arbitrage

Par le **Second Théorème Fondamental** (ST7-5), dans un marché complet, le prix sans arbitrage d'un actif contingent de payoff $H$ à maturité $T$ est :

$$\pi_t = \mathbb{E}^{\mathbb{Q}}\left[\frac{H}{S^0_T} \bigg| \mathcal{F}_t\right]$$

soit, dans le cas d'un taux sans risque constant $r$ :

$$\pi_t = e^{-r(T-t)} \, \mathbb{E}^{\mathbb{Q}}\left[H \,|\, \mathcal{F}_t\right]$$

### 1.4 Choix de la mesure $\mathbb{Q}$

**Justification réglementaire :** Le calcul de l'EEPE au sens du CRR (article 284) requiert la simulation des facteurs de risque sous une mesure cohérente avec le pricing. Nous utilisons donc la mesure risque-neutre $\mathbb{Q}$, ce qui assure la cohérence entre la diffusion des sous-jacents et le pricing Black-Scholes.

### 1.5 Notations récapitulatives

| Symbole | Signification |
|---------|---------------|
| $d$ | Nombre de sous-jacents ($d = 3$) |
| $r$ | Taux sans risque continu |
| $\sigma_i$ | Volatilité du sous-jacent $i$ |
| $\rho_{ij}$ | Corrélation entre les sous-jacents $i$ et $j$ |
| $\Sigma$ | Matrice de corrélation ($d \times d$) |
| $T_{\text{opt}}$ | Maturité des options |
| $T_{\text{EEPE}}$ | Horizon EEPE (1 an) |
| $\delta$ | MPOR (Margin Period of Risk) |
| $N_{\text{outer}}$ | Nombre de scénarios extérieurs MC |
| $N_{\text{inner}}$ | Nombre de scénarios intérieurs (nested MC) |
| $N_t$ | Nombre de pas de temps |
| $\Delta t$ | Pas de temps ($= T_{\text{EEPE}} / N_t$) |

---

## 2. Dynamique des sous-jacents

### 2.1 GBM multidimensionnel sous $\mathbb{Q}$

Sous la mesure risque-neutre $\mathbb{Q}$, chaque sous-jacent $S^i$ suit un mouvement brownien géométrique :

$$dS^i_t = r \, S^i_t \, dt + \sigma_i \, S^i_t \, dW^i_t, \quad i = 1, \ldots, d$$

où $(W^1, \ldots, W^d)$ est un $\mathbb{Q}$-mouvement brownien $d$-dimensionnel corrélé :

$$dW^i_t \cdot dW^j_t = \rho_{ij} \, dt$$

### 2.2 Solution exacte

Le GBM admet une solution fermée. Pour la simulation, on utilise la solution exacte (pas d'erreur de discrétisation) :

$$S^i(t_{j+1}) = S^i(t_j) \times \exp\left[\left(r - \frac{\sigma_i^2}{2}\right)\Delta t + \sigma_i \sqrt{\Delta t} \, Z^i_{j+1}\right]$$

### 2.3 Corrélation par décomposition de Cholesky

Le vecteur gaussien corrélé $Z = (Z^1, \ldots, Z^d)^\top \sim \mathcal{N}(0, \Sigma)$ est obtenu via la décomposition de Cholesky de la matrice de corrélation :

$$\Sigma = L \, L^\top$$

où $L$ est triangulaire inférieure. On simule alors :

$$Z = L \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_d)$$

**Condition nécessaire :** La matrice $\Sigma$ doit être **symétrique définie positive** pour que la décomposition de Cholesky existe. Ceci est vérifié programmatiquement avant toute simulation.

### 2.4 Matrice de corrélation du projet

$$\Sigma = \begin{pmatrix} 1.0 & 0.6 & 0.4 \\ 0.6 & 1.0 & 0.5 \\ 0.4 & 0.5 & 1.0 \end{pmatrix}$$

Valeurs propres : toutes strictement positives $\Rightarrow$ $\Sigma$ est définie positive.

---

## 3. Pricing Black-Scholes

### 3.1 Fondement théorique

Dans le cadre du modèle de Black-Scholes-Merton, le marché est complet (un seul facteur de risque par sous-jacent). Par le second FTAP, le prix d'une option européenne est uniquement déterminé par l'espérance sous $\mathbb{Q}$ du payoff actualisé (cf. ST7-5, Théorème 5.2.9).

### 3.2 Formules de pricing

**Call européen :**

$$C(t, S, K, T, r, \sigma) = S \, \Phi(d_1) - K \, e^{-r(T-t)} \, \Phi(d_2)$$

**Put européen :**

$$P(t, S, K, T, r, \sigma) = -S \, \Phi(-d_1) + K \, e^{-r(T-t)} \, \Phi(-d_2)$$

avec :

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T - t)}{\sigma\sqrt{T - t}}, \qquad d_2 = d_1 - \sigma\sqrt{T - t}$$

où $\Phi$ désigne la fonction de répartition de la loi normale centrée réduite.

### 3.3 Vérifications

- **Parité call-put :** $C - P = S - K e^{-r(T-t)}$ (vérifiée dans les tests unitaires).
- **Valeurs limites :** call deep ITM $\approx S - K e^{-r\tau}$, put deep OTM $\approx 0$.
- **Vectorisation :** les formules sont implémentées en NumPy vectorisé (pas de boucle Python sur les scénarios).

---

## 4. Portefeuille et Mark-to-Market

### 4.1 Composition du portefeuille

Le portefeuille contient $N_p = 5$ options européennes sur $d = 3$ sous-jacents :

| # | Type | Sous-jacent | $S_0$ | $K$ | $T_{\text{opt}}$ | $\sigma$ | Position $\eta_k$ |
|---|------|-------------|-------|-----|-------------------|----------|-------------------|
| 1 | Call | A | 100 | 105 | 2Y | 20% | +1 (Long) |
| 2 | Put  | A | 100 | 95  | 2Y | 20% | +1 (Long) |
| 3 | Call | B | 150 | 160 | 2Y | 25% | +1 (Long) |
| 4 | Put  | C | 80  | 75  | 2Y | 30% | -1 (Short) |
| 5 | Call | C | 80  | 85  | 2Y | 30% | +1 (Long) |

**Note :** $T_{\text{opt}} = 2\text{Y} > T_{\text{EEPE}} = 1\text{Y}$, donc toutes les options restent vivantes sur l'intégralité de l'horizon EEPE. La maturité résiduelle à chaque nœud est $\tau_k(t_j) = T_{\text{opt}} - t_j > 0$.

### 4.2 Mark-to-Market

Le MtM du portefeuille au nœud $(i, t_j)$ (scénario $i$, date $t_j$) est :

$$\text{MtM}_i(t_j) = \sum_{k=1}^{N_p} \eta_k \cdot V_k\left(t_j, S^{i(k)}_i(t_j)\right)$$

où :
- $\eta_k = +1$ (long) ou $-1$ (short) est la position.
- $V_k$ est le prix Black-Scholes de l'option $k$.
- $S^{i(k)}_i(t_j)$ est le prix simulé du sous-jacent de l'option $k$ au scénario $i$, date $t_j$.

---

## 5. Métriques d'exposition

### 5.1 Exposition

L'exposition (positive) du portefeuille au scénario $i$, date $t_j$ :

$$E_i(t_j) = \max\left(0, \, \text{MtM}_i(t_j)\right)$$

L'exposition représente la perte potentielle en cas de défaut de la contrepartie : elle est positive quand le portefeuille a une valeur de marché positive (la contrepartie nous doit de l'argent).

### 5.2 Expected Exposure (EE)

Moyenne de l'exposition sur les scénarios à chaque date :

$$\text{EE}(t_j) = \frac{1}{N_{\text{outer}}} \sum_{i=1}^{N_{\text{outer}}} E_i(t_j) = \frac{1}{N_{\text{outer}}} \sum_{i=1}^{N_{\text{outer}}} \max\left(0, \, \text{MtM}_i(t_j)\right)$$

### 5.3 Effective Expected Exposure (EEE)

L'EEE impose la contrainte de non-décroissance (running maximum) :

$$\text{EEE}(t_j) = \max_{l \leq j} \text{EE}(t_l)$$

**Justification réglementaire :** La contrainte de non-décroissance reflète le fait qu'un netting set ne peut pas « oublier » son exposition passée maximale. Elle évite de sous-estimer l'EEPE pour des portefeuilles dont l'EE décroît (par exemple des options proches de l'expiration).

### 5.4 Effective Expected Positive Exposure (EEPE)

Moyenne temporelle de l'EEE sur l'horizon $[0, T_{\text{EEPE}}]$ :

$$\text{EEPE} = \frac{1}{T_{\text{EEPE}}} \int_0^{T_{\text{EEPE}}} \text{EEE}(t) \, dt$$

**Discrétisation :** L'intégrale est approchée par la méthode des trapèzes sur la grille temporelle $(t_0, t_1, \ldots, t_{N_t})$ :

$$\text{EEPE} \approx \frac{1}{T_{\text{EEPE}}} \sum_{j=0}^{N_t - 1} \frac{\text{EEE}(t_j) + \text{EEE}(t_{j+1})}{2} \cdot \Delta t$$

### 5.5 Exposure at Default (EAD)

$$\text{EAD} = \alpha \cdot \text{EEPE}$$

avec $\alpha = 1.4$ (facteur réglementaire, CRR art. 284).

---

## 6. Collatéralisation et Initial Margin

### 6.1 Variation Margin (VM)

Sous le CSA simplifié adopté dans ce projet :

$$\text{VM}(t) = \text{MtM}(t)$$

avec threshold $= 0$ et MTA (Minimum Transfer Amount) $= 0$. Le VM compense intégralement la valeur de marché.

### 6.2 Margin Period of Risk (MPOR)

Le MPOR $\delta$ représente la période nécessaire pour fermer ou remplacer les positions en cas de défaut. Standard bilatéral CSA :

$$\delta = 10 \text{ jours business} \approx \frac{2}{52} \text{ an} \approx 0.03846 \text{ an}$$

### 6.3 Initial Margin (IM) stochastique

L'IM couvre les pertes potentielles sur la période du MPOR au-delà de ce que le VM compense déjà. On définit la **perte** comme :

$$\Delta PV(t, t+\delta) = PV(t) - PV(t + \delta)$$

**Convention de signe (CRITIQUE) :** $\Delta PV > 0$ correspond à une dépréciation du portefeuille (perte pour le détenteur).

L'IM est alors défini comme le quantile à 99% de cette perte conditionnelle :

$$\text{IM}(t) = Q_{99\%}\left(\Delta PV(t, t+\delta) \,\big|\, \mathcal{F}_t\right)$$

### 6.4 Exposition résiduelle avec IM

L'IM réduit l'exposition résiduelle. Après prise en compte de l'IM :

$$E^{\text{IM}}_i(t_j) = \max\left(0, \, \Delta PV_i(t_j, t_j + \delta) - \text{IM}_i(t_j)\right)$$

Intuition : l'IM absorbe les pertes jusqu'au 99ème percentile. L'exposition résiduelle ne subsiste que pour les pertes dépassant l'IM (queue au-delà du quantile 99%).

---

## 7. Nested Monte Carlo pour l'IM (Livrable 1)

### 7.1 Principe

Le nested Monte Carlo (brute force) estime l'IM en simulant, pour chaque nœud $(i, t_j)$ du MC extérieur, un ensemble de $N_{\text{inner}}$ sous-scénarios sur le MPOR.

### 7.2 Algorithme

Pour chaque nœud $(i, t_j)$ :

1. **Simulation intérieure :** Générer $N_{\text{inner}}$ réalisations de $S^{(i,k)}(t_j + \delta)$ à partir de $S^{(i)}(t_j)$ :

$$S^{(i,k)}_l(t_j + \delta) = S^{(i)}_l(t_j) \times \exp\left[\left(r - \frac{\sigma_l^2}{2}\right)\delta + \sigma_l \sqrt{\delta} \, Z^{(k)}_l\right], \quad k = 1, \ldots, N_{\text{inner}}$$

2. **Pricing :** Calculer $PV_{i,k}(t_j + \delta)$ par Black-Scholes pour chaque sous-scénario.

3. **P&L :** $\Delta PV_{i,k} = PV_i(t_j) - PV_{i,k}(t_j + \delta)$.

4. **Quantile empirique :** $\widehat{\text{IM}}_i(t_j) = \hat{Q}_{99\%}\left(\{\Delta PV_{i,k}\}_{k=1}^{N_{\text{inner}}}\right)$

### 7.3 Coût computationnel

Le nombre total de pricings Black-Scholes est :

$$\text{Coût} = N_{\text{outer}} \times (N_t + 1) \times (1 + N_{\text{inner}}) \times N_p$$

Avec les paramètres du projet ($N_{\text{outer}} = 500$, $N_t = 52$, $N_{\text{inner}} = 500$, $N_p = 5$) :

$$\text{Coût} \approx 500 \times 53 \times 501 \times 5 \approx 66.4 \text{ millions de pricings}$$

### 7.4 Variance du quantile empirique

Avec $N_{\text{inner}} = 500$, le quantile empirique 99% correspond au 5ème plus grand $\Delta PV$ parmi 500. La variance de cet estimateur est significative :

$$\text{Var}(\hat{Q}_\alpha) \approx \frac{\alpha(1-\alpha)}{n \cdot f(Q_\alpha)^2}$$

où $f$ est la densité de $\Delta PV$ au voisinage du quantile. Cette variance de l'estimateur nested constitue le bruit de référence contre lequel l'approximation Johnson sera comparée.

---

## 8. Approximation Johnson (Livrable 2)

### 8.1 Motivation

L'approximation Johnson (McWalter et al. 2018) évite le nested MC en approchant la distribution conditionnelle de $\Delta PV \,|\, \mathcal{F}_t$ par une distribution de Johnson, identifiée par ses 4 premiers moments. Le gain computationnel est d'un facteur $\approx N_{\text{inner}}$.

### 8.2 Famille de distributions Johnson

La famille Johnson (Johnson, 1949) fournit des transformations de la loi normale qui couvrent un large espace de couples (skewness, kurtosis). Elle comprend quatre types :

#### Johnson SU (Unbounded)

$$X = \xi + \lambda \cdot \sinh\left(\frac{Z - \gamma}{\delta_J}\right), \quad Z \sim \mathcal{N}(0,1)$$

Support : $(-\infty, +\infty)$. Queues plus lourdes que la normale.

**Quantile :**

$$Q_\alpha(X) = \xi + \lambda \cdot \sinh\left(\frac{\Phi^{-1}(\alpha) - \gamma}{\delta_J}\right)$$

#### Johnson SB (Bounded)

$$X = \xi + \frac{\lambda}{1 + \exp\left(-\frac{Z - \gamma}{\delta_J}\right)}, \quad Z \sim \mathcal{N}(0,1)$$

Support : $(\xi, \xi + \lambda)$. Distribution bornée.

#### Johnson SL (Lognormal)

$$X = \xi + \lambda \cdot \exp\left(\frac{Z - \gamma}{\delta_J}\right), \quad Z \sim \mathcal{N}(0,1)$$

Cas limite entre SU et SB, correspondant à la frontière lognormale dans le plan $(\beta_1, \beta_2)$.

#### Johnson SN (Normal)

$$X = \xi + \lambda \cdot \frac{Z - \gamma}{\delta_J}, \quad Z \sim \mathcal{N}(0,1)$$

Cas dégénéré : distribution normale.

### 8.3 Sélection du type

Le type de distribution est déterminé par le couple $(\beta_1, \beta_2) = (\text{skewness}^2, \text{kurtosis})$ dans le diagramme de Pearson :

- **SN :** $\beta_1 \approx 0$ et $\beta_2 \approx 3$ (quasi-normal).
- **SU :** $\beta_2$ au-dessus de la courbe lognormale (queues lourdes).
- **SB :** $\beta_2$ en dessous de la courbe lognormale (bornée).
- **SL :** $\beta_2$ sur la courbe lognormale.

La courbe lognormale est définie implicitement par $\omega = \exp(\sigma^2)$ :

$$\beta_1 = (\omega - 1)(\omega + 2)^2, \qquad \beta_2 = \omega^4 + 2\omega^3 + 3\omega^2 - 3$$

### 8.4 Fit des paramètres Johnson SU par les moments

**Objectif :** Trouver $(\xi, \lambda, \gamma, \delta_J)$ tels que la distribution Johnson SU ait les mêmes 4 premiers moments que la distribution cible.

**Méthode :** Algorithme itératif en 2 étapes (plus rapide que l'optimisation Nelder-Mead) :

1. **Paramétrage :** Poser $\omega = \exp(1/\delta_J^2)$ et $\Gamma = \gamma / \delta_J$.

2. **Moments de la transformée :** Les moments bruts de $Y = \sinh((Z - \Gamma \delta_J) / \delta_J)$ s'expriment analytiquement via la fonction génératrice des moments de la normale :

$$m_1 = -\sqrt{\omega} \, \sinh(\Gamma)$$
$$m_2 = \frac{\omega^2 \cosh(2\Gamma) - 1}{2}$$

3. **Itération alternée :**
   - Trouver $\omega$ depuis la kurtosis ($\Gamma$ fixé) par Brentq 1D.
   - Trouver $\Gamma$ depuis le skewness ($\omega$ fixé) par Brentq 1D.
   - 2 itérations suffisent (erreur $< 0.3\%$ sur le quantile 99%).

4. **Conversion :** $\delta_J = 1 / \sqrt{\ln \omega}$, $\gamma = \Gamma \cdot \delta_J$. Puis $\xi$ et $\lambda$ sont obtenus analytiquement depuis la moyenne et la variance cibles.

### 8.5 Estimation des moments conditionnels par régression

Pour chaque date $t_j$, on estime les 4 moments conditionnels de $\Delta PV$ sachant $PV(t_j)$ par régression polynomiale :

$$\mathbb{E}\left[(\Delta PV)^k \,\big|\, PV(t_j)\right] \approx \sum_{p=0}^{P} a_{k,p} \cdot PV(t_j)^p, \quad k = 1, 2, 3, 4$$

**Degré polynomial :** $P = 5$ (paramétrable). La variable $PV(t_j)$ est standardisée avant la régression pour améliorer le conditionnement de la matrice de Vandermonde.

**Moments centrés conditionnels :**

$$\mu_1 = m_1, \qquad \sigma^2 = m_2 - m_1^2$$
$$\mu_3 = m_3 - 3 m_1 m_2 + 2 m_1^3, \qquad \mu_4 = m_4 - 4 m_1 m_3 + 6 m_1^2 m_2 - 3 m_1^4$$

$$\text{skewness} = \mu_3 / \sigma^3, \qquad \text{kurtosis} = \mu_4 / \sigma^4$$

**Contrainte théorique :** $\text{kurtosis} \geq \text{skewness}^2 + 1$ (forcée par clipping).

### 8.6 Interpolation sur grille

Pour éviter de fitter Johnson pour chaque scénario ($N_{\text{outer}}$ fits par date), on procède par interpolation :

1. Définir une grille de $n_{\text{grid}} = 30$ points uniformément répartis dans l'espace de $PV(t_j)$ standardisé.
2. Calculer les moments conditionnels aux points de la grille.
3. Fitter Johnson à chaque point de la grille ($n_{\text{grid}}$ fits au lieu de $N_{\text{outer}}$).
4. Interpoler le quantile 99% pour chaque scénario par interpolation linéaire.

Gain : facteur $N_{\text{outer}} / n_{\text{grid}} \approx 17$.

### 8.7 Coût computationnel

Le coût total est dominé par le MC extérieur (un seul scénario MPOR par nœud) :

$$\text{Coût} = N_{\text{outer}} \times (N_t + 1) \times 2 \times N_p + n_{\text{grid}} \times (N_t + 1) \times \text{fit}$$

soit un gain d'un facteur $\approx N_{\text{inner}}$ par rapport au nested MC.

---

## 9. Calcul de l'EEPE avec IM stochastique

### 9.1 Pipeline

L'intégration de l'IM stochastique dans le calcul de l'EEPE suit les étapes :

1. **MC extérieur :** Simuler $N_{\text{outer}}$ chemins $(S^{(i)}(t_j))_{j=0}^{N_t}$.
2. **Calcul de l'IM :** Pour chaque nœud $(i, t_j)$, estimer $\text{IM}_i(t_j)$ par :
   - Nested MC (Livrable 1), ou
   - Approximation Johnson (Livrable 2).
3. **Exposition résiduelle :** Simuler un scénario MPOR indépendant et calculer :

$$E^{\text{IM}}_i(t_j) = \max\left(0, \, \Delta PV_i(t_j, t_j + \delta) - \text{IM}_i(t_j)\right)$$

4. **Agrégation :**

$$\text{EE}^{\text{IM}}(t_j) = \frac{1}{N_{\text{outer}}} \sum_{i=1}^{N_{\text{outer}}} E^{\text{IM}}_i(t_j)$$

$$\text{EEE}^{\text{IM}}(t_j) = \max_{l \leq j} \text{EE}^{\text{IM}}(t_l)$$

$$\text{EEPE}^{\text{IM}} = \frac{1}{T_{\text{EEPE}}} \int_0^{T_{\text{EEPE}}} \text{EEE}^{\text{IM}}(t) \, dt$$

### 9.2 Scénarios indépendants pour l'exposition résiduelle

**Point important :** Les scénarios MPOR utilisés pour calculer l'exposition résiduelle doivent être **indépendants** de ceux utilisés pour estimer l'IM. Dans le cas contraire, on introduirait un biais de surestimation de la couverture par l'IM. Ceci est assuré par l'utilisation de seeds aléatoires distinctes.

---

## 10. Analyse d'erreur et convergence

### 10.1 Sources d'erreur

| Source | Impact | Contrôle |
|--------|--------|----------|
| Erreur MC extérieur | $O(N_{\text{outer}}^{-1/2})$ | Augmenter $N_{\text{outer}}$ |
| Erreur nested MC (quantile) | $O(N_{\text{inner}}^{-1/2})$ | Augmenter $N_{\text{inner}}$ |
| Erreur de discrétisation temporelle | Nulle (solution exacte GBM) | — |
| Erreur de discrétisation de l'EEPE | $O(\Delta t^2)$ (trapèzes) | Augmenter $N_t$ |
| Erreur Johnson (biais de modèle) | Dépend de la forme de la distribution | Comparer avec nested MC |
| Erreur de régression polynomiale | Dépend du degré $P$ et de $N_{\text{outer}}$ | Tester $P$ = 3, 5, 7 |

### 10.2 Métriques de comparaison (Livrable 3)

1. **Erreur absolue sur l'EEPE :**

$$\varepsilon_{\text{abs}} = \left|\text{EEPE}^{\text{nested}} - \text{EEPE}^{\text{Johnson}}\right|$$

2. **Erreur relative sur l'EEPE :**

$$\varepsilon_{\text{rel}} = \frac{\varepsilon_{\text{abs}}}{\text{EEPE}^{\text{nested}}}$$

3. **MAE nœud par nœud sur l'IM :**

$$\text{MAE}_{\text{IM}}(t_j) = \frac{1}{N_{\text{outer}}} \sum_{i=1}^{N_{\text{outer}}} \left|\text{IM}^{\text{nested}}_i(t_j) - \text{IM}^{\text{Johnson}}_i(t_j)\right|$$

4. **Ratio de temps de calcul :**

$$\text{Speedup} = \frac{t_{\text{nested}}}{t_{\text{Johnson}}}$$

### 10.3 Convergence du nested MC

L'estimateur du quantile empirique 99% avec $N_{\text{inner}} = 500$ utilise la 5ème plus grande valeur. L'intervalle de confiance asymptotique est :

$$\hat{Q}_{0.99} \pm z_{\alpha/2} \cdot \sqrt{\frac{0.99 \times 0.01}{N_{\text{inner}} \cdot f(Q_{0.99})^2}}$$

Ce bruit intrinsèque du nested MC doit être gardé en tête lors de la comparaison avec l'approximation Johnson : une partie de l'écart observé provient de la variance de l'estimateur de référence lui-même.

---

## 11. Hypothèses du projet

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Mesure de diffusion | Risque-neutre ($\mathbb{Q}$, drift $= r$) | Standard EEPE réglementaire (CRR art. 284), cohérent avec pricing BS |
| Taux sans risque $r$ | 3% | Paramètre fixé |
| MPOR $\delta$ | 10 jours business $\approx 2/52$ an | Standard bilatéral CSA |
| CSA | Simplifié : VM $=$ MtM, threshold $= 0$, MTA $= 0$ | Isole l'effet de l'IM sur la période MPOR |
| Maturité options $T_{\text{opt}}$ | 2 ans | Options vivantes sur tout l'horizon EEPE |
| Horizon EEPE | 1 an | Standard réglementaire |
| Pricing | Black-Scholes (formules fermées) | Options vanilles européennes |
| Diffusion | GBM multidimensionnel corrélé (Cholesky) | Solution exacte, pas d'erreur de discrétisation |
| Nombre de sous-jacents $d$ | 3 (A, B, C) | — |
| $N_{\text{outer}}$ | 500 | Compromis précision/temps de calcul |
| $N_{\text{inner}}$ | 500 | Compromis précision/temps de calcul |
| $N_t$ | 52 | Pas hebdomadaires sur 1 an |
| Confiance IM | 99% | Standard réglementaire |
| Degré polynomial (Johnson) | 5 | Calibré empiriquement |
| Seed aléatoire | 42 | Reproductibilité |
| Facteur $\alpha$ (EAD) | 1.4 | Réglementaire |

---

## Références

- **McWalter, T. A. et al. (2018).** Nested Monte Carlo simulation and stochastic initial margin.
- **CRR — Capital Requirements Regulation**, Article 284 : calcul de l'EPE effective.
- **TRIM — Targeted Review of Internal Models** (ECB).
- **Carassus, L. & Guo, X.** — Cours ST7 : Measure & Probability (ST7-1), Stochastic Processes (ST7-2), Multi-period Model (ST7-5). CentraleSupélec.
- **Johnson, N. L. (1949).** Systems of frequency curves generated by methods of translation. *Biometrika*, 36(1/2), 149–176.
