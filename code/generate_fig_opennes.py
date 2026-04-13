"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        generate_fig_opennes.py
Purpose:     Generate Figure 5 — sectoral trade openness, (exports + imports)/GDP,
             by sector (agr, man, trd, bss, fin, nps) for the US (left panel)
             and for the EU4 aggregate built from DEU+FRA+GBR+ITA hours-weighted
             series (right panel). Motivates treating trade as the binding
             mechanism behind EU4 manufacturing reallocation.
Pipeline:    Step 11/19 — Figure 5 (trade openness).
Inputs:      US sectoral openness ratios exp_imp_*_Y from
             model_calibration_USA_open.py and the EU4 country instances
             (DEU, FRA, GBR, ITA) plus EUR4_h_tot from model_test_europe_open.py.
Outputs:     ../output/figures/fig_opennes.pdf (renamed to Figure 5 by the final
             paper-output script).
Dependencies: model_calibration_USA.py (Step 1), model_calibration_USA_open.py
              (Step 4), model_test_europe_open.py (Step 5).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
rc("text", usetex=True)
rc("font", family="serif")

from model_calibration_USA_open import exp_imp_agr_Y, exp_imp_man_Y, exp_imp_trd_Y, exp_imp_bss_Y, exp_imp_fin_Y, exp_imp_nps_Y
from model_test_europe_open import DEU, FRA, GBR, ITA, EUR4_h_tot

fig = plt.figure(1)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(121)
plt.plot(exp_imp_agr_Y, 'D-', markersize=8, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=3, alpha=0.95, label=r'$\texttt{agr}$')
plt.plot(exp_imp_man_Y, 'o-', markersize=8, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=3, alpha=0.95, label=r'$\texttt{man}$')
plt.plot(exp_imp_trd_Y, 's-', markersize=8, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=3, alpha=0.95, label=r'$\texttt{trd}$')
plt.plot(exp_imp_bss_Y, '^-', markersize=8, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=3, alpha=0.95, label=r'$\texttt{bss}$')
plt.plot(exp_imp_fin_Y, 'p-', markersize=8, color='darkcyan', markerfacecolor='cyan', markeredgecolor='darkcyan', markevery=3, alpha=0.95, label=r'$\texttt{fin}$')
plt.plot(exp_imp_nps_Y, 'v-', markersize=8, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markevery=3, alpha=0.95, label=r'$\texttt{nps}$')
 
plt.xlabel('Year', fontsize=14)
plt.ylabel('(Exports + Imports)/GDP', fontsize=14)
plt.title('United States', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.axis([1994, 2020, 0, 0.33])
plt.grid()

ax = plt.subplot(122)
# EU4 openness = sum_c h_c * (exp_imp_sec_Y)_c / sum_c h_c, i.e. an
# hours-weighted aggregate across DEU+FRA+GBR+ITA. Hours-weighting is the same
# convention used elsewhere in the paper for EU4 sectoral aggregates.
plt.plot((DEU.h_tot*DEU.exp_imp_agr_Y + GBR.h_tot*GBR.exp_imp_agr_Y + FRA.h_tot*FRA.exp_imp_agr_Y + ITA.h_tot*ITA.exp_imp_agr_Y)/EUR4_h_tot, 'D-', markersize=8, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=3, alpha=0.95, label=r'$\texttt{agr}$')
plt.plot((DEU.h_tot*DEU.exp_imp_man_Y + GBR.h_tot*GBR.exp_imp_man_Y + FRA.h_tot*FRA.exp_imp_man_Y + ITA.h_tot*ITA.exp_imp_man_Y)/EUR4_h_tot, 'o-', markersize=8, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=3, alpha=0.95, label=r'$\texttt{man}$')
plt.plot((DEU.h_tot*DEU.exp_imp_trd_Y + GBR.h_tot*GBR.exp_imp_trd_Y + FRA.h_tot*FRA.exp_imp_trd_Y + ITA.h_tot*ITA.exp_imp_trd_Y)/EUR4_h_tot, 's-', markersize=8, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=3, alpha=0.95, label=r'$\texttt{trd}$')
plt.plot((DEU.h_tot*DEU.exp_imp_bss_Y + GBR.h_tot*GBR.exp_imp_bss_Y + FRA.h_tot*FRA.exp_imp_bss_Y + ITA.h_tot*ITA.exp_imp_bss_Y)/EUR4_h_tot, '^-', markersize=8, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=3, alpha=0.95, label=r'$\texttt{bss}$')
plt.plot((DEU.h_tot*DEU.exp_imp_fin_Y + GBR.h_tot*GBR.exp_imp_fin_Y + FRA.h_tot*FRA.exp_imp_fin_Y + ITA.h_tot*ITA.exp_imp_fin_Y)/EUR4_h_tot, 'p-', markersize=8, color='darkcyan', markerfacecolor='cyan', markeredgecolor='darkcyan', markevery=3, alpha=0.95, label=r'$\texttt{fin}$')
plt.plot((DEU.h_tot*DEU.exp_imp_nps_Y + GBR.h_tot*GBR.exp_imp_nps_Y + FRA.h_tot*FRA.exp_imp_nps_Y + ITA.h_tot*ITA.exp_imp_nps_Y)/EUR4_h_tot, 'v-', markersize=8, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markevery=3, alpha=0.95, label=r'$\texttt{nps}$')

plt.xlabel('Year', fontsize=14)
plt.ylabel('(Exports + Imports)/GDP', fontsize=14)
plt.title('Europe', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.axis([1994, 2020, 0, 0.33])
plt.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=14)

plt.tight_layout()
plt.savefig('../output/figures/fig_opennes.pdf', bbox_inches="tight")
plt.close()

