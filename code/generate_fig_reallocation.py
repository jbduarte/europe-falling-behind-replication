import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
rc("text", usetex=True)
rc("font", family="serif")
import numpy as np
import pandas as pd

from model_test_europe import (model_country, EUR4_h_tot,
    EUR4_share_agr_nps_m, EUR4_share_man_nps_m, EUR4_share_trd_nps_m,
    EUR4_share_bss_nps_m, EUR4_share_fin_nps_m, EUR4_share_nps_nps_m)
from counterfactuals import counterfactual

DEU = model_country('DEU')
DEU.productivity_series()
DEU.predictions_nps()

FRA = model_country('FRA')
FRA.productivity_series()
FRA.predictions_nps()

GBR = model_country('GBR')
GBR.productivity_series()
GBR.predictions_nps()

ITA = model_country('ITA')
ITA.productivity_series()
ITA.predictions_nps()

'trd'
sec='trd'
DEU_cf=counterfactual('DEU')
DEU_cf.baseline()
DEU_cf.feed_catch_up_growth(0, sec)

GBR_cf=counterfactual('GBR')
GBR_cf.baseline()
GBR_cf.feed_catch_up_growth(0, sec)

FRA_cf=counterfactual('FRA')
FRA_cf.baseline()
FRA_cf.feed_catch_up_growth(0, sec)

ITA_cf=counterfactual('ITA')
ITA_cf.baseline()
ITA_cf.feed_catch_up_growth(0, sec)

EUR4_cf_trd_share_agr = (np.array(DEU.h_tot)*np.array(DEU_cf.share_agr_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_agr_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_agr_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_agr_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_trd_share_man = (np.array(DEU.h_tot)*np.array(DEU_cf.share_man_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_man_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_man_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_man_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_trd_share_trd = (np.array(DEU.h_tot)*np.array(DEU_cf.share_trd_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_trd_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_trd_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_trd_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_trd_share_bss = (np.array(DEU.h_tot)*np.array(DEU_cf.share_bss_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_bss_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_bss_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_bss_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_trd_share_fin = (np.array(DEU.h_tot)*np.array(DEU_cf.share_fin_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_fin_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_fin_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_fin_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_trd_share_nps = (np.array(DEU.h_tot)*np.array(DEU_cf.share_nps_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_nps_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_nps_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_nps_nps_m, dtype=object).flatten())/EUR4_h_tot


'bss'
sec='bss'
DEU_cf=counterfactual('DEU')
DEU_cf.baseline()
DEU_cf.feed_catch_up_growth(0, sec)

GBR_cf=counterfactual('GBR')
GBR_cf.baseline()
GBR_cf.feed_catch_up_growth(0, sec)

FRA_cf=counterfactual('FRA')
FRA_cf.baseline()
FRA_cf.feed_catch_up_growth(0, sec)

ITA_cf=counterfactual('ITA')
ITA_cf.baseline()
ITA_cf.feed_catch_up_growth(0, sec)

EUR4_cf_bss_share_agr = (np.array(DEU.h_tot)*np.array(DEU_cf.share_agr_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_agr_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_agr_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_agr_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_bss_share_man = (np.array(DEU.h_tot)*np.array(DEU_cf.share_man_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_man_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_man_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_man_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_bss_share_bss = (np.array(DEU.h_tot)*np.array(DEU_cf.share_bss_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_bss_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_bss_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_bss_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_bss_share_trd = (np.array(DEU.h_tot)*np.array(DEU_cf.share_trd_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_trd_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_trd_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_trd_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_bss_share_fin = (np.array(DEU.h_tot)*np.array(DEU_cf.share_fin_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_fin_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_fin_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_fin_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_bss_share_nps = (np.array(DEU.h_tot)*np.array(DEU_cf.share_nps_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_nps_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_nps_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_nps_nps_m, dtype=object).flatten())/EUR4_h_tot



'fin'
sec='fin'
DEU_cf=counterfactual('DEU')
DEU_cf.baseline()
DEU_cf.feed_catch_up_growth(0, sec)

GBR_cf=counterfactual('GBR')
GBR_cf.baseline()
GBR_cf.feed_catch_up_growth(0, sec)

FRA_cf=counterfactual('FRA')
FRA_cf.baseline()
FRA_cf.feed_catch_up_growth(0, sec)

ITA_cf=counterfactual('ITA')
ITA_cf.baseline()
ITA_cf.feed_catch_up_growth(0, sec)

EUR4_cf_fin_share_agr = (np.array(DEU.h_tot)*np.array(DEU_cf.share_agr_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_agr_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_agr_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_agr_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_fin_share_man = (np.array(DEU.h_tot)*np.array(DEU_cf.share_man_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_man_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_man_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_man_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_fin_share_trd = (np.array(DEU.h_tot)*np.array(DEU_cf.share_trd_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_trd_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_trd_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_trd_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_fin_share_bss = (np.array(DEU.h_tot)*np.array(DEU_cf.share_bss_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_bss_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_bss_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_bss_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_fin_share_fin = (np.array(DEU.h_tot)*np.array(DEU_cf.share_fin_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_fin_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_fin_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_fin_nps_m, dtype=object).flatten())/EUR4_h_tot
EUR4_cf_fin_share_nps = (np.array(DEU.h_tot)*np.array(DEU_cf.share_nps_nps_m, dtype=object).flatten() + np.array(FRA.h_tot)*np.array(FRA_cf.share_nps_nps_m, dtype=object).flatten() + np.array(GBR.h_tot)*np.array(GBR_cf.share_nps_nps_m, dtype=object).flatten() + np.array(ITA.h_tot)*np.array(ITA_cf.share_nps_nps_m, dtype=object).flatten())/EUR4_h_tot


'''
-----------
	Plot
-----------
'''
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(4)
fig.set_figwidth(8)

ax = plt.subplot(1,3,1)
ax.plot(DEU.year, EUR4_share_agr_nps_m, 'D-', markersize=4, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Baseline')
ax.plot(DEU.year, EUR4_cf_bss_share_agr, 'D--', markersize=4, color='darkgreen', markeredgecolor='darkgreen', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_man_nps_m, 'o-', markersize=4, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Baseline')
ax.plot(DEU.year, EUR4_cf_bss_share_man, 'o--', markersize=4, color='darkblue', markeredgecolor='darkblue', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_trd_nps_m, 's-', markersize=4, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Baseline')
ax.plot(DEU.year, EUR4_cf_bss_share_trd, 's--', markersize=4, color='darkred', markeredgecolor='darkred', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_bss_nps_m, '^-', markersize=4, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Baseline')
ax.plot(DEU.year, EUR4_cf_bss_share_bss, '^--', markersize=4, color='darkmagenta', markeredgecolor='darkmagenta', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_fin_nps_m, 'p-', markersize=4, color='darkcyan', markerfacecolor='cyan', markeredgecolor='darkcyan', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Baseline')
ax.plot(DEU.year, EUR4_cf_bss_share_fin, 'p--', markersize=4, color='darkcyan', markeredgecolor='darkcyan', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_nps_nps_m, 'v-', markersize=4, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Baseline')
ax.plot(DEU.year, EUR4_cf_bss_share_nps, 'v--', markersize=4, color='saddlebrown', markeredgecolor='saddlebrown', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Counterfactual')
plt.title(r'$\texttt{bss}$', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid()



handles, labels = ax.get_legend_handles_labels()

ax = plt.subplot(1,3,2)
ax.plot(DEU.year, EUR4_share_agr_nps_m, 'D-', markersize=4, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Baseline')
ax.plot(DEU.year, EUR4_cf_fin_share_agr, 'D--', markersize=4, color='darkgreen', markeredgecolor='darkgreen', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_man_nps_m, 'o-', markersize=4, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Baseline')
ax.plot(DEU.year, EUR4_cf_fin_share_man, 'o--', markersize=4, color='darkblue', markeredgecolor='darkblue', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_trd_nps_m, 's-', markersize=4, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Baseline')
ax.plot(DEU.year, EUR4_cf_fin_share_trd, 's--', markersize=4, color='darkred', markeredgecolor='darkred', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_bss_nps_m, '^-', markersize=4, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Baseline')
ax.plot(DEU.year, EUR4_cf_fin_share_bss, '^--', markersize=4, color='darkmagenta', markeredgecolor='darkmagenta', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_fin_nps_m, 'p-', markersize=4, color='darkcyan', markerfacecolor='cyan', markeredgecolor='darkcyan', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Baseline')
ax.plot(DEU.year, EUR4_cf_fin_share_fin, 'p--', markersize=4, color='darkcyan', markeredgecolor='darkcyan', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_nps_nps_m, 'v-', markersize=4, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Baseline')
ax.plot(DEU.year, EUR4_cf_fin_share_nps, 'v--', markersize=4, color='saddlebrown', markeredgecolor='saddlebrown', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Counterfactual')
plt.title(r'$\texttt{fin}$', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid()

ax = plt.subplot(1,3,3)
ax.plot(DEU.year, EUR4_share_agr_nps_m, 'D-', markersize=4, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Baseline')
ax.plot(DEU.year, EUR4_cf_trd_share_agr, 'D--', markersize=4, color='darkgreen', markeredgecolor='darkgreen', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_man_nps_m, 'o-', markersize=4, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Baseline')
ax.plot(DEU.year, EUR4_cf_trd_share_man, 'o--', markersize=4, color='darkblue', markeredgecolor='darkblue', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_trd_nps_m, 's-', markersize=4, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Baseline')
ax.plot(DEU.year, EUR4_cf_trd_share_trd, 's--', markersize=4, color='darkred', markeredgecolor='darkred', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_bss_nps_m, '^-', markersize=4, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Baseline')
ax.plot(DEU.year, EUR4_cf_trd_share_bss, '^--', markersize=4, color='darkmagenta', markeredgecolor='darkmagenta', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_fin_nps_m, 'p-', markersize=4, color='darkcyan', markerfacecolor='cyan', markeredgecolor='darkcyan', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Baseline')
ax.plot(DEU.year, EUR4_cf_trd_share_fin, 'p--', markersize=4, color='darkcyan', markeredgecolor='darkcyan', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Counterfactual')
ax.plot(DEU.year, EUR4_share_nps_nps_m, 'v-', markersize=4, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Baseline')
ax.plot(DEU.year, EUR4_cf_trd_share_nps, 'v--', markersize=4, color='saddlebrown', markeredgecolor='saddlebrown', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Counterfactual')
plt.title(r'$\texttt{trd}$', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid()

fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.025), ncol=3, fontsize=12)

plt.tight_layout()
plt.savefig('../output/figures/fig_3.pdf', bbox_inches="tight")
plt.close()
