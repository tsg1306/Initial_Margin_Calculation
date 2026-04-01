"""
Pipeline complet — IM Stochastique CCR (Livrable 3 : Comparaison)

Execute les deux approches (Nested MC et Johnson), compare les resultats,
et genere les graphiques de comparaison.

Usage :
    python main.py
    python main.py --fast     # parametres reduits pour test rapide
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from config.parameters import (
    SEED, RISK_FREE_RATE, N_OUTER, N_INNER, N_T, DELTA_T,
    TIME_GRID, T_EEPE, MPOR, IM_CONFIDENCE, SPOTS, VOLS,
    CORRELATION_MATRIX, PORTFOLIO, N_OPTIONS, ALPHA_EAD,
    ASSET_NAMES,
)
from lib.diffusion import simulate_gbm
from lib.portfolio import compute_mtm, compute_mtm_full
from lib.exposure import (
    compute_ee, compute_eee, compute_eepe, compute_all_exposure_metrics,
)
from lib.margin import compute_im_nested, compute_exposure_with_im
from lib.johnson import compute_im_johnson
from lib.utils import (
    validate_correlation_matrix, compute_ead, Timer,
    print_header, print_table,
)


# ============================================================
# Configuration
# ============================================================

FAST_MODE = "--fast" in sys.argv

if FAST_MODE:
    n_outer = 200
    n_inner = 200
    print("[MODE RAPIDE] N_outer=200, N_inner=200")
else:
    n_outer = N_OUTER
    n_inner = N_INNER


def main():
    """Pipeline principal."""

    # ========================================================
    # 0. Validation des parametres
    # ========================================================
    print_header("0. Validation des parametres")
    validate_correlation_matrix(CORRELATION_MATRIX)
    print(f"  Matrice de correlation : definie positive OK")
    print(f"  Parametres : r={RISK_FREE_RATE}, MPOR={MPOR:.4f}, "
          f"N_outer={n_outer}, N_inner={n_inner}, N_t={N_T}")
    print(f"  Portefeuille : {N_OPTIONS} options sur {len(SPOTS)} sous-jacents")
    print(f"  Seed : {SEED}")

    # ========================================================
    # 1. Simulation MC exterieur
    # ========================================================
    print_header("1. Simulation Monte Carlo exterieur")

    with Timer("Simulation GBM") as t_sim:
        paths = simulate_gbm(n_outer=n_outer, n_t=N_T)

    print(f"  Chemins : {paths.shape} (n_outer, n_t+1, d)")

    with Timer("Calcul MtM") as t_mtm:
        mtm_matrix = compute_mtm_full(paths, TIME_GRID)

    print(f"  MtM(t=0) = {mtm_matrix[0, 0]:.4f}")
    print(f"  MtM moyen a T=1Y : {np.mean(mtm_matrix[:, -1]):.4f}")

    # ========================================================
    # 2. Metriques d'exposition SANS IM
    # ========================================================
    print_header("2. Exposition sans IM")

    metrics_no_im = compute_all_exposure_metrics(mtm_matrix, TIME_GRID)
    eepe_no_im = metrics_no_im["eepe"]
    ead_no_im = compute_ead(eepe_no_im)

    print(f"  EE(0)  = {metrics_no_im['ee'][0]:.4f}")
    print(f"  EEPE   = {eepe_no_im:.4f}")
    print(f"  EAD    = alpha * EEPE = {ALPHA_EAD} * {eepe_no_im:.4f} = {ead_no_im:.4f}")

    # ========================================================
    # 3. Livrable 1 — IM par Nested Monte Carlo
    # ========================================================
    print_header("3. Livrable 1 : IM par Nested Monte Carlo")

    with Timer("Nested MC") as t_nested:
        im_nested = compute_im_nested(
            paths, TIME_GRID, n_inner=n_inner,
        )

    print(f"  IM nested shape : {im_nested.shape}")
    print(f"  IM median : {np.median(im_nested):.4f}")
    print(f"  IM negatif : {100 * np.mean(im_nested < 0):.1f}% des noeuds")

    # Exposition avec IM nested
    with Timer("Exposition avec IM nested"):
        exposure_nested = compute_exposure_with_im(paths, TIME_GRID, im_nested)

    ee_nested_im = np.mean(exposure_nested, axis=0)
    eee_nested_im = np.maximum.accumulate(ee_nested_im)
    eepe_nested_im = np.trapezoid(eee_nested_im, TIME_GRID) / T_EEPE

    print(f"  EEPE avec IM (nested) = {eepe_nested_im:.4f}")
    print(f"  Reduction EEPE : {100 * (1 - eepe_nested_im / eepe_no_im):.1f}%")

    # ========================================================
    # 4. Livrable 2 — IM par approximation Johnson
    # ========================================================
    print_header("4. Livrable 2 : IM par approximation Johnson")

    with Timer("Johnson") as t_johnson:
        im_johnson = compute_im_johnson(paths, TIME_GRID)

    print(f"  IM Johnson shape : {im_johnson.shape}")
    print(f"  IM median : {np.median(im_johnson):.4f}")

    # Exposition avec IM Johnson
    with Timer("Exposition avec IM Johnson"):
        exposure_johnson = compute_exposure_with_im(paths, TIME_GRID, im_johnson)

    ee_johnson_im = np.mean(exposure_johnson, axis=0)
    eee_johnson_im = np.maximum.accumulate(ee_johnson_im)
    eepe_johnson_im = np.trapezoid(eee_johnson_im, TIME_GRID) / T_EEPE

    print(f"  EEPE avec IM (Johnson) = {eepe_johnson_im:.4f}")
    print(f"  Reduction EEPE : {100 * (1 - eepe_johnson_im / eepe_no_im):.1f}%")

    # ========================================================
    # 5. Livrable 3 — Comparaison
    # ========================================================
    print_header("5. Livrable 3 : Comparaison Nested MC vs Johnson")

    # 5a. Tableau recapitulatif
    err_eepe_abs = abs(eepe_nested_im - eepe_johnson_im)
    err_eepe_rel = err_eepe_abs / max(abs(eepe_nested_im), 1e-10)
    mae_im = np.mean(np.abs(im_nested - im_johnson))

    rows = [
        ["EEPE sans IM", f"{eepe_no_im:.4f}", f"{eepe_no_im:.4f}"],
        ["EEPE avec IM", f"{eepe_nested_im:.4f}", f"{eepe_johnson_im:.4f}"],
        ["Reduction EEPE",
         f"{100 * (1 - eepe_nested_im / eepe_no_im):.1f}%",
         f"{100 * (1 - eepe_johnson_im / eepe_no_im):.1f}%"],
        ["IM median", f"{np.median(im_nested):.4f}", f"{np.median(im_johnson):.4f}"],
        ["MAE IM (noeud/noeud)", "ref", f"{mae_im:.4f}"],
        ["Temps de calcul", f"{t_nested.elapsed:.1f}s", f"{t_johnson.elapsed:.1f}s"],
        ["Ratio de vitesse", "1x", f"{t_nested.elapsed / max(t_johnson.elapsed, 0.01):.0f}x"],
    ]
    print_table(["Metrique", "Nested MC", "Johnson"], rows, [25, 18, 18])

    # 5b. Erreur sur l'IM par date
    err_eepe_vs_base = err_eepe_abs / max(abs(eepe_no_im), 1e-10)
    print(f"\n  Erreur absolue EEPE : {err_eepe_abs:.4f}")
    print(f"  Erreur relative EEPE (vs nested) : {err_eepe_rel:.2%}")
    print(f"  Erreur relative EEPE (vs sans IM) : {err_eepe_vs_base:.2%}")

    mae_by_date = np.mean(np.abs(im_nested - im_johnson), axis=0)
    print(f"\n  MAE IM par date (selection) :")
    for idx in [0, 13, 26, 39, 52]:
        print(f"    t={TIME_GRID[idx]:.2f} : MAE={mae_by_date[idx]:.4f}")

    # ========================================================
    # 6. Graphiques
    # ========================================================
    print_header("6. Generation des graphiques")
    _generate_plots(
        paths, mtm_matrix, metrics_no_im,
        im_nested, im_johnson,
        ee_nested_im, eee_nested_im, eepe_nested_im,
        ee_johnson_im, eee_johnson_im, eepe_johnson_im,
        eepe_no_im, mae_by_date,
    )
    print("  Graphiques sauvegardes dans report/")

    # ========================================================
    # Resume final
    # ========================================================
    print_header("RESUME FINAL")
    print(f"  EEPE sans IM         = {eepe_no_im:.4f}")
    print(f"  EEPE nested MC + IM  = {eepe_nested_im:.4f}  "
          f"(reduction {100 * (1 - eepe_nested_im / eepe_no_im):.1f}%)")
    print(f"  EEPE Johnson + IM    = {eepe_johnson_im:.4f}  "
          f"(reduction {100 * (1 - eepe_johnson_im / eepe_no_im):.1f}%)")
    print(f"  Ecart nested/Johnson (vs base) = {err_eepe_vs_base:.2%}")
    print(f"  Speedup Johnson      = {t_nested.elapsed / max(t_johnson.elapsed, 0.01):.0f}x")
    print(f"  EAD sans IM          = {ead_no_im:.4f}")
    print(f"  EAD nested + IM      = {compute_ead(eepe_nested_im):.4f}")
    print(f"  EAD Johnson + IM     = {compute_ead(eepe_johnson_im):.4f}")
    print()


# ============================================================
# Generation des graphiques
# ============================================================

def _generate_plots(
    paths, mtm_matrix, metrics_no_im,
    im_nested, im_johnson,
    ee_nested_im, eee_nested_im, eepe_nested_im,
    ee_johnson_im, eee_johnson_im, eepe_johnson_im,
    eepe_no_im, mae_by_date,
):
    """Genere et sauvegarde tous les graphiques du Livrable 3."""

    import os
    os.makedirs("report", exist_ok=True)

    # ------ Figure 1 : Chemins des sous-jacents ------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, ASSET_NAMES)):
        for k in range(20):
            ax.plot(TIME_GRID, paths[k, :, i], alpha=0.4, linewidth=0.7)
        ax.axhline(SPOTS[i], color='red', linestyle='--', linewidth=1.5,
                   label=f'S0={SPOTS[i]}')
        ax.set_title(f'{name} (S0={SPOTS[i]}, sig={VOLS[i]:.0%})')
        ax.set_xlabel('Temps (annees)')
        ax.set_ylabel('Prix')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Echantillon de chemins GBM sous Q', fontsize=13)
    plt.tight_layout()
    plt.savefig('report/fig1_chemins_gbm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------ Figure 2 : Profils EE / EEE sans IM ------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(TIME_GRID, metrics_no_im['ee'], 'b-', linewidth=2, label='EE')
    ax.plot(TIME_GRID, metrics_no_im['eee'], 'r-', linewidth=2, label='EEE')
    ax.axhline(eepe_no_im, color='green', linestyle='--', linewidth=1.5,
               label=f'EEPE = {eepe_no_im:.2f}')
    ax.fill_between(TIME_GRID, 0, metrics_no_im['eee'], alpha=0.1, color='red')
    ax.set_xlabel('Temps (annees)')
    ax.set_ylabel('Exposition')
    ax.set_title('Profils EE et EEE (sans IM)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/fig2_ee_eee_sans_im.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------ Figure 3 : Comparaison EE avec IM ------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(TIME_GRID, metrics_no_im['ee'], 'gray', linewidth=2,
             label='EE sans IM', alpha=0.7)
    ax1.plot(TIME_GRID, ee_nested_im, 'b-', linewidth=2,
             label='EE nested MC + IM')
    ax1.plot(TIME_GRID, ee_johnson_im, 'r--', linewidth=2,
             label='EE Johnson + IM')
    ax1.set_xlabel('Temps (annees)')
    ax1.set_ylabel('Expected Exposure')
    ax1.set_title('Profils EE')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(TIME_GRID, metrics_no_im['eee'], 'gray', linewidth=2,
             label='EEE sans IM', alpha=0.7)
    ax2.plot(TIME_GRID, eee_nested_im, 'b-', linewidth=2,
             label='EEE nested MC + IM')
    ax2.plot(TIME_GRID, eee_johnson_im, 'r--', linewidth=2,
             label='EEE Johnson + IM')
    ax2.axhline(eepe_no_im, color='gray', linestyle=':', alpha=0.5,
                label=f'EEPE sans IM = {eepe_no_im:.2f}')
    ax2.axhline(eepe_nested_im, color='blue', linestyle=':', alpha=0.5,
                label=f'EEPE nested = {eepe_nested_im:.2f}')
    ax2.axhline(eepe_johnson_im, color='red', linestyle=':', alpha=0.5,
                label=f'EEPE Johnson = {eepe_johnson_im:.2f}')
    ax2.set_xlabel('Temps (annees)')
    ax2.set_ylabel('Effective Expected Exposure')
    ax2.set_title('Profils EEE et EEPE')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Impact de l\'IM sur l\'exposition', fontsize=13)
    plt.tight_layout()
    plt.savefig('report/fig3_comparaison_ee_im.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------ Figure 4 : Scatter IM nested vs Johnson ------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, idx, t_label in zip(axes, [0, 26, 52], ['t=0', 't=0.5Y', 't=1Y']):
        ax.scatter(im_nested[:, idx], im_johnson[:, idx], alpha=0.3, s=10)
        all_vals = np.concatenate([im_nested[:, idx], im_johnson[:, idx]])
        lo, hi = np.min(all_vals) - 1, np.max(all_vals) + 1
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
        ax.set_xlabel('IM nested MC')
        ax.set_ylabel('IM Johnson')
        ax.set_title(f'Scatter IM {t_label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    plt.suptitle('IM Nested MC vs Johnson (chaque point = 1 scenario)', fontsize=13)
    plt.tight_layout()
    plt.savefig('report/fig4_scatter_im.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------ Figure 5 : IM moyen + MAE ------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 3) IM median
    im_nested_q05_t = np.quantile(im_nested, 0.05, axis=0)
    im_nested_q95_t = np.quantile(im_nested, 0.95, axis=0)
    im_johnson_q05_t = np.quantile(im_johnson, 0.05, axis=0)
    im_johnson_q95_t = np.quantile(im_johnson, 0.95, axis=0)
    im_nested_median=np.median(im_nested,axis=0)
    im_johnson_median=np.median(im_johnson,axis=0)
    ax1.plot(TIME_GRID, im_nested_median, 'b-', linewidth=2, label='median IM nested MC')
    ax1.fill_between(TIME_GRID, im_nested_q05_t, im_nested_q95_t, alpha=0.2, color='blue', label='Bande Nested Q5%-Q95%')
    ax1.fill_between(TIME_GRID, im_johnson_q05_t, im_johnson_q95_t, alpha=0.2, color='red', label='Bande Johnson Q5%-Q95%')
    ax1.plot(TIME_GRID, im_johnson_median, 'r--', linewidth=2, label='median IM Johnson')
    ax1.set_xlabel('Temps (annees)')
    ax1.set_ylabel('IM median')
    ax1.set_title('IM median au cours du temps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Profil median avec bandes de confiance

    ax2.plot(TIME_GRID, mae_by_date, 'k-', linewidth=2)
    ax2.set_xlabel('Temps (annees)')
    ax2.set_ylabel('MAE(IM nested - IM Johnson)')
    ax2.set_title('Erreur absolue moyenne sur l\'IM')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Comparaison IM : Nested MC vs Johnson', fontsize=13)
    plt.tight_layout()
    plt.savefig('report/fig5_im_moyen_mae.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------ Figure 6 : Distribution de DeltaPV a t=0.5Y ------
    fig, ax = plt.subplots(figsize=(10, 5))
    t_idx = 26  # t = 0.5Y
    pv_t = compute_mtm(paths[:, t_idx, :], TIME_GRID[t_idx])
    # Un scenario representatif (median)
    median_idx = np.argsort(pv_t)[len(pv_t) // 2]
    ax.hist(
        im_nested[:, t_idx], bins=40, density=True, alpha=0.5,
        edgecolor='black', linewidth=0.3, label='IM nested MC', color='blue'
    )
    ax.hist(
        im_johnson[:, t_idx], bins=40, density=True, alpha=0.5,
        edgecolor='black', linewidth=0.3, label='IM Johnson', color='red'
    )
    ax.set_xlabel('IM')
    ax.set_ylabel('Densite')
    ax.set_title('Distribution de l\'IM a t=0.5Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/fig6_distribution_im.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  6 figures sauvegardees dans report/")


# ============================================================
# Point d'entree
# ============================================================

if __name__ == "__main__":
    main()
