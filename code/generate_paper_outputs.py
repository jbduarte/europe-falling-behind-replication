"""
=======================================================================================
Generate Paper Figures and Tables

Author: Joao B. Duarte
Last Modified: Feb 2026

This is the final step in the pipeline. It:
  1. Generates endogenous trade model outputs (Figure 6, Table 6, Figure A.4)
  2. Generates Figure 5 (trade openness by sector)
  3. Consolidates all pipeline outputs with paper-consistent naming

Requires: Steps 1-9 of master.py to have been run first.

Usage:
    cd code
    python generate_paper_outputs.py
=======================================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex=True)
rc("font", family="serif")

print("Importing endogenous model modules (this triggers calibration)...")

# US parameters
from model_calibration_USA import (
    sigma,
    eps_agr,
    eps_trd,
    eps_fin,
    eps_bss,
    eps_nps,
    eps_ser,
)

# US endogenous open-economy calibration
from model_calibration_USA_endogenous_open import (
    E,
    A_tot,
    A_tot_nps,
    A_agr,
    A_man,
    A_trd,
    A_bss,
    A_fin,
    A_nps,
    A_ser,
    share_agr,
    share_man,
    share_trd,
    share_bss,
    share_fin,
    share_nps,
    share_agr_nps,
    share_man_nps,
    share_trd_nps,
    share_bss_nps,
    share_fin_nps,
    share_nps_nps,
    share_agr_nps_closed,
    share_man_nps_closed,
    share_trd_nps_closed,
    share_bss_nps_closed,
    share_fin_nps_closed,
    share_nps_nps_closed,
    GDP_ph,
)

# European endogenous calibration
from model_test_europe_endogenous_xn import (
    model_country,
    EUR4_h_tot,
    EURCORE_h_tot,
    EURPERI_h_tot,
    EUR13_h_tot,
    EUR4_A_tot,
    EURCORE_A_tot,
    EURPERI_A_tot,
    EUR13_A_tot,
    EUR4_rel_A_tot,
    EUR13_rel_A_tot,
    EUR4_E,
    EUR13_E,
    EUR4_share_agr,
    EUR4_share_man,
    EUR4_share_trd,
    EUR4_share_bss,
    EUR4_share_fin,
    EUR4_share_nps,
    EUR4_share_ser,
    EUR4_share_agr_nps_m,
    EUR4_share_man_nps_m,
    EUR4_share_trd_nps_m,
    EUR4_share_bss_nps_m,
    EUR4_share_fin_nps_m,
    EUR4_share_nps_nps_m,
    EUR4_share_agr_nps_m_closed,
    EUR4_share_man_nps_m_closed,
    EUR4_share_trd_nps_m_closed,
    EUR4_share_bss_nps_m_closed,
    EUR4_share_fin_nps_m_closed,
    EUR4_share_nps_nps_m_closed,
    EUR13_share_agr,
    EUR13_share_man,
    EUR13_share_trd,
    EUR13_share_bss,
    EUR13_share_fin,
    EUR13_share_nps,
    EUR13_share_agr_nps_m,
    EUR13_share_man_nps_m,
    EUR13_share_trd_nps_m,
    EUR13_share_bss_nps_m,
    EUR13_share_fin_nps_m,
    EUR13_share_nps_nps_m,
)

# Import EUR13 closed shares and individual country objects
from model_test_europe_endogenous_xn import (
    EUR13_share_agr_nps_m_closed,
    EUR13_share_man_nps_m_closed,
    EUR13_share_trd_nps_m_closed,
    EUR13_share_bss_nps_m_closed,
    EUR13_share_fin_nps_m_closed,
    EUR13_share_nps_nps_m_closed,
)

# Country objects for computing aggregate productivity
from model_test_europe_endogenous_xn import (
    AUT,
    BEL,
    DEU,
    DNK,
    ESP,
    FRA,
    GBR,
    GRC,
    ITA,
    NLD,
    PRT,
    SWE,
)
from model_test_europe_endogenous_xn import FIN as FIN_cou

print("Imports complete. Computing derived variables...")


# ==============================================================================
# COMPUTE DERIVED VARIABLES
# ==============================================================================

# --- US closed-economy aggregate productivity (NPS model) ---
A_tot_nps_closed = [
    sum(x)
    for x in zip(
        [a * b for a, b in zip(share_agr_nps_closed, A_agr)],
        [a * b for a, b in zip(share_man_nps_closed, A_man)],
        [a * b for a, b in zip(share_trd_nps_closed, A_trd)],
        [a * b for a, b in zip(share_bss_nps_closed, A_bss)],
        [a * b for a, b in zip(share_fin_nps_closed, A_fin)],
        [a * b for a, b in zip(share_nps_nps_closed, A_nps)],
    )
]

# --- EUR4 open-economy aggregate productivity (NPS model) ---
EUR4_A_tot_nps_open = (
    np.array(DEU.A_tot_nps).flatten() * np.array(DEU.h_tot).flatten()
    + np.array(FRA.A_tot_nps).flatten() * np.array(FRA.h_tot).flatten()
    + np.array(GBR.A_tot_nps).flatten() * np.array(GBR.h_tot).flatten()
    + np.array(ITA.A_tot_nps).flatten() * np.array(ITA.h_tot).flatten()
) / EUR4_h_tot

# --- EUR4 closed-economy aggregate productivity (NPS model) ---
EUR4_A_tot_nps_closed = (
    np.array(DEU.A_tot_nps_closed).flatten() * np.array(DEU.h_tot).flatten()
    + np.array(FRA.A_tot_nps_closed).flatten() * np.array(FRA.h_tot).flatten()
    + np.array(GBR.A_tot_nps_closed).flatten() * np.array(GBR.h_tot).flatten()
    + np.array(ITA.A_tot_nps_closed).flatten() * np.array(ITA.h_tot).flatten()
) / EUR4_h_tot

# --- EUR13 open-economy aggregate productivity (NPS model) ---
EUR13_A_tot_nps_open = (
    np.array(AUT.A_tot_nps).flatten() * np.array(AUT.h_tot).flatten()
    + np.array(BEL.A_tot_nps).flatten() * np.array(BEL.h_tot).flatten()
    + np.array(DEU.A_tot_nps).flatten() * np.array(DEU.h_tot).flatten()
    + np.array(DNK.A_tot_nps).flatten() * np.array(DNK.h_tot).flatten()
    + np.array(ESP.A_tot_nps).flatten() * np.array(ESP.h_tot).flatten()
    + np.array(FIN_cou.A_tot_nps).flatten() * np.array(FIN_cou.h_tot).flatten()
    + np.array(FRA.A_tot_nps).flatten() * np.array(FRA.h_tot).flatten()
    + np.array(GBR.A_tot_nps).flatten() * np.array(GBR.h_tot).flatten()
    + np.array(GRC.A_tot_nps).flatten() * np.array(GRC.h_tot).flatten()
    + np.array(ITA.A_tot_nps).flatten() * np.array(ITA.h_tot).flatten()
    + np.array(NLD.A_tot_nps).flatten() * np.array(NLD.h_tot).flatten()
    + np.array(PRT.A_tot_nps).flatten() * np.array(PRT.h_tot).flatten()
    + np.array(SWE.A_tot_nps).flatten() * np.array(SWE.h_tot).flatten()
) / EUR13_h_tot

# --- EUR13 closed-economy aggregate productivity (NPS model) ---
EUR13_A_tot_nps_closed = (
    np.array(AUT.A_tot_nps_closed).flatten() * np.array(AUT.h_tot).flatten()
    + np.array(BEL.A_tot_nps_closed).flatten() * np.array(BEL.h_tot).flatten()
    + np.array(DEU.A_tot_nps_closed).flatten() * np.array(DEU.h_tot).flatten()
    + np.array(DNK.A_tot_nps_closed).flatten() * np.array(DNK.h_tot).flatten()
    + np.array(ESP.A_tot_nps_closed).flatten() * np.array(ESP.h_tot).flatten()
    + np.array(FIN_cou.A_tot_nps_closed).flatten()
    * np.array(FIN_cou.h_tot).flatten()
    + np.array(FRA.A_tot_nps_closed).flatten() * np.array(FRA.h_tot).flatten()
    + np.array(GBR.A_tot_nps_closed).flatten() * np.array(GBR.h_tot).flatten()
    + np.array(GRC.A_tot_nps_closed).flatten() * np.array(GRC.h_tot).flatten()
    + np.array(ITA.A_tot_nps_closed).flatten() * np.array(ITA.h_tot).flatten()
    + np.array(NLD.A_tot_nps_closed).flatten() * np.array(NLD.h_tot).flatten()
    + np.array(PRT.A_tot_nps_closed).flatten() * np.array(PRT.h_tot).flatten()
    + np.array(SWE.A_tot_nps_closed).flatten() * np.array(SWE.h_tot).flatten()
) / EUR13_h_tot

years = DEU.year  # common year vector


# ==============================================================================
# FIGURE 1: fig_2_open_endo.pdf
# Employment share scatter (data vs model) + EUR4/US productivity path
# ==============================================================================
print("Generating fig_2_open_endo.pdf...")

fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

# --- Left panel: Employment shares scatter ---
ax = plt.subplot(1, 2, 1)

# Sectors to plot (nps commented out to match exogenous version)
sectors = ["agr", "man", "trd", "bss", "fin"]
# US data shares (x-axis for US points)
us_data = {
    "agr": np.array(share_agr)[-1],
    "man": np.array(share_man)[-1],
    "trd": np.array(share_trd)[-1],
    "bss": np.array(share_bss)[-1],
    "fin": np.array(share_fin)[-1],
}
# US open-economy model shares (y-axis)
us_open = {
    "agr": share_agr_nps[-1],
    "man": share_man_nps[-1],
    "trd": share_trd_nps[-1],
    "bss": share_bss_nps[-1],
    "fin": share_fin_nps[-1],
}
# US closed-economy model shares (y-axis)
us_closed = {
    "agr": share_agr_nps_closed[-1],
    "man": share_man_nps_closed[-1],
    "trd": share_trd_nps_closed[-1],
    "bss": share_bss_nps_closed[-1],
    "fin": share_fin_nps_closed[-1],
}
# EUR4 data shares (x-axis for EUR4 points)
eur4_data = {
    "agr": EUR4_share_agr[-1],
    "man": EUR4_share_man[-1],
    "trd": EUR4_share_trd[-1],
    "bss": EUR4_share_bss[-1],
    "fin": EUR4_share_fin[-1],
}
# EUR4 open-economy model shares (y-axis)
eur4_open = {
    "agr": EUR4_share_agr_nps_m[-1],
    "man": EUR4_share_man_nps_m[-1],
    "trd": EUR4_share_trd_nps_m[-1],
    "bss": EUR4_share_bss_nps_m[-1],
    "fin": EUR4_share_fin_nps_m[-1],
}
# EUR4 closed-economy model shares (y-axis)
eur4_closed = {
    "agr": EUR4_share_agr_nps_m_closed[-1],
    "man": EUR4_share_man_nps_m_closed[-1],
    "trd": EUR4_share_trd_nps_m_closed[-1],
    "bss": EUR4_share_bss_nps_m_closed[-1],
    "fin": EUR4_share_fin_nps_m_closed[-1],
}

for i, sec in enumerate(sectors):
    label_us_open = "Open: U.S." if i == 0 else None
    label_us_closed = "Closed: U.S." if i == 0 else None
    label_eur_open = "Open: Europe" if i == 0 else None
    label_eur_closed = "Closed: Europe" if i == 0 else None

    # US open
    ax.plot(
        us_data[sec],
        us_open[sec],
        "D",
        markerfacecolor="lightcoral",
        markeredgecolor="darkred",
        markersize=6,
        alpha=0.75,
        label=label_us_open,
    )
    # US closed
    ax.plot(
        us_data[sec],
        us_closed[sec],
        "D",
        markerfacecolor="none",
        markeredgecolor="darkred",
        markersize=6,
        alpha=0.75,
        label=label_us_closed,
    )
    # EUR4 open
    ax.plot(
        eur4_data[sec],
        eur4_open[sec],
        "o",
        markerfacecolor="lightskyblue",
        markeredgecolor="darkblue",
        markersize=6,
        alpha=0.75,
        label=label_eur_open,
    )
    # EUR4 closed
    ax.plot(
        eur4_data[sec],
        eur4_closed[sec],
        "o",
        markerfacecolor="none",
        markeredgecolor="darkblue",
        markersize=6,
        alpha=0.75,
        label=label_eur_closed,
    )
    # Annotations
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (us_data[sec], us_open[sec]),
        alpha=0.75,
        fontsize=16,
    )
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (us_data[sec], us_closed[sec]),
        alpha=0.75,
        fontsize=16,
    )
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (eur4_data[sec], eur4_open[sec]),
        alpha=0.75,
        fontsize=16,
    )
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (eur4_data[sec], eur4_closed[sec]),
        alpha=0.75,
        fontsize=16,
    )

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.title("Employment Shares in 2019", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid()

# --- Right panel: EUR4/US aggregate productivity ---
ax = plt.subplot(1, 2, 2)
ax.plot(years, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
ax.plot(
    years,
    EUR4_A_tot_nps_open / np.array(A_tot_nps),
    "s-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markevery=2,
    alpha=0.95,
    label="Open Economy Model",
)
plt.plot(
    years,
    EUR4_A_tot_nps_closed / np.array(A_tot_nps_closed),
    "D-",
    markersize=6,
    color="darkred",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markevery=2,
    alpha=0.95,
    label="Closed Economy Model",
)
plt.title("Labor Productivity (Relative to U.S.)", fontsize=16)
plt.axis([1993, 2021, 0.85, 1.05])
plt.xticks(fontsize=14)
plt.yticks(
    [0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14
)
plt.legend(loc="upper right", fontsize=14)
plt.grid()

plt.tight_layout()
plt.savefig("../output/figures/fig_2_open_endo.pdf", bbox_inches="tight")
plt.close()
print("  Saved: output/figures/fig_2_open_endo.pdf")


# ==============================================================================
# FIGURE 2: fig_test_EUR_endo.pdf
# 6-panel: EU4 prod + scatter, GBR prod + scatter, EU13 prod + scatter
# ==============================================================================
print("Generating fig_test_EUR_endo.pdf...")

fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(6)
fig.set_figwidth(6)

# --- Panel 1: EU4 productivity ---
ax = plt.subplot(3, 2, 1)
plt.plot(years, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
plt.plot(
    years,
    EUR4_A_tot_nps_open / np.array(A_tot_nps).flatten(),
    "H-",
    markersize=5,
    markevery=2,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
    label="Model",
)
plt.axis([1993, 2021, 0.73, 1.05])
plt.title("EU4", fontsize=12)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.grid()

# --- Panel 2: EU4 scatter ---
ax = plt.subplot(3, 2, 2)
eur4_sec_model = {
    "agr": EUR4_share_agr_nps_m,
    "man": EUR4_share_man_nps_m,
    "trd": EUR4_share_trd_nps_m,
    "bss": EUR4_share_bss_nps_m,
    "fin": EUR4_share_fin_nps_m,
    "nps": EUR4_share_nps_nps_m,
}
eur4_sec_data = {
    "agr": EUR4_share_agr,
    "man": EUR4_share_man,
    "trd": EUR4_share_trd,
    "bss": EUR4_share_bss,
    "fin": EUR4_share_fin,
    "nps": EUR4_share_nps,
}
for i, sec in enumerate(["agr", "man", "trd", "bss", "fin", "nps"]):
    label = "Model" if i == 0 else None
    ax.plot(
        eur4_sec_model[sec][-1],
        eur4_sec_data[sec][-1],
        "H",
        markerfacecolor="lime",
        markeredgecolor="darkgreen",
        markersize=6,
        alpha=0.75,
        label=label,
    )
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (eur4_sec_model[sec][-1], eur4_sec_data[sec][-1]),
        alpha=0.75,
        fontsize=14,
    )
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("EU4", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.grid()

# --- Panel 3: GBR productivity ---
ax = plt.subplot(3, 2, 3)
plt.plot(
    GBR.year,
    GBR.A_tot / np.array(A_tot),
    "b-",
    alpha=0.95,
    label=r"$\frac{Y_t}{L_t}$: Data",
)
plt.plot(
    GBR.year,
    np.array(GBR.A_tot_nps).flatten() / np.array(A_tot_nps).flatten(),
    "H-",
    markersize=5,
    markevery=2,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
    label="Model",
)
plt.axis([1993, 2021, 0.73, 1.05])
plt.title("GBR", fontsize=12)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.grid()

# --- Panel 4: GBR scatter ---
ax = plt.subplot(3, 2, 4)
gbr_sec_model = {
    "agr": GBR.share_agr_nps_m,
    "man": GBR.share_man_nps_m,
    "trd": GBR.share_trd_nps_m,
    "bss": GBR.share_bss_nps_m,
    "fin": GBR.share_fin_nps_m,
    "nps": GBR.share_nps_nps_m,
}
gbr_sec_data = {
    "agr": GBR.share_agr,
    "man": GBR.share_man,
    "trd": GBR.share_trd,
    "bss": GBR.share_bss,
    "fin": GBR.share_fin,
    "nps": GBR.share_nps,
}
for i, sec in enumerate(["agr", "man", "trd", "bss", "fin", "nps"]):
    label = "Model" if i == 0 else None
    model_val = gbr_sec_model[sec][-1]
    data_val = gbr_sec_data[sec]
    # Handle both list and pandas Series
    if hasattr(data_val, "values"):
        data_val = data_val.values[-1]
    else:
        data_val = np.array(data_val)[-1]
    ax.plot(
        model_val,
        data_val,
        "H",
        markerfacecolor="lime",
        markeredgecolor="darkgreen",
        markersize=6,
        alpha=0.75,
        label=label,
    )
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (model_val, data_val),
        alpha=0.75,
        fontsize=14,
    )
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("GBR", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.grid()

# --- Panel 5: EU13 productivity ---
ax = plt.subplot(3, 2, 5)
plt.plot(years, EUR13_rel_A_tot, "b-", alpha=0.95, label="Data")
plt.plot(
    years,
    EUR13_A_tot_nps_open / np.array(A_tot_nps).flatten(),
    "H-",
    markersize=5,
    markevery=2,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
    label="Model",
)
plt.axis([1993, 2021, 0.73, 1.05])
plt.title("EU15", fontsize=12)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=10)
plt.grid()

# --- Panel 6: EU13 scatter ---
ax = plt.subplot(3, 2, 6)
eur13_sec_model = {
    "agr": EUR13_share_agr_nps_m,
    "man": EUR13_share_man_nps_m,
    "trd": EUR13_share_trd_nps_m,
    "bss": EUR13_share_bss_nps_m,
    "fin": EUR13_share_fin_nps_m,
    "nps": EUR13_share_nps_nps_m,
}
eur13_sec_data = {
    "agr": EUR13_share_agr,
    "man": EUR13_share_man,
    "trd": EUR13_share_trd,
    "bss": EUR13_share_bss,
    "fin": EUR13_share_fin,
    "nps": EUR13_share_nps,
}
for i, sec in enumerate(["agr", "man", "trd", "bss", "fin", "nps"]):
    label = "Model" if i == 0 else None
    ax.plot(
        eur13_sec_model[sec][-1],
        eur13_sec_data[sec][-1],
        "H",
        markerfacecolor="lime",
        markeredgecolor="darkgreen",
        markersize=6,
        alpha=0.75,
        label=label,
    )
    ax.annotate(
        r"$\texttt{" + sec + r"}$",
        (eur13_sec_model[sec][-1], eur13_sec_data[sec][-1]),
        alpha=0.75,
        fontsize=14,
    )
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("EU15", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.grid()

plt.tight_layout()
plt.savefig("../output/figures/fig_test_EUR_endo.pdf", bbox_inches="tight")
plt.close()
print("  Saved: output/figures/fig_test_EUR_endo.pdf")


# ==============================================================================
# TABLE: tab_trade_cf_endo (LaTeX + Excel)
# CF2 catch-up ratios + trade cure net exports
# ==============================================================================
print("Generating trade counterfactual table...")

# --- Panel A: CF2 catch-up ratios ---
# Read endogenous CF2 results from Excel files
# Format: row 0 = country headers, rows 1-6 = sector data (agr, man, trd, fin, bss, nps)
# EU4 is a COLUMN (not a row)
cf2_model_file = "../output/figures/Counterfactual_2_catch_nps_endo_trade.xlsx"
cf2_ss_file = "../output/figures/Counterfactual_2_catch_nps_ss_endo_trade.xlsx"

if os.path.exists(cf2_model_file) and os.path.exists(cf2_ss_file):
    cf2_model = pd.read_excel(cf2_model_file, header=None)
    cf2_ss = pd.read_excel(cf2_ss_file, header=None)

    # Find EU4 column index from header row
    header_row = [str(x).strip() for x in cf2_model.iloc[0, :].values]
    eu4_col = None
    for j, h in enumerate(header_row):
        if h in ("EU4", "EUR4"):
            eu4_col = j
            break

    # Build sector-to-row mapping from column 0
    sec_rows = {}
    for i in range(1, len(cf2_model)):
        sec_name = str(cf2_model.iloc[i, 0]).strip().lower()
        sec_rows[sec_name] = i

    if eu4_col is not None:
        cf2_man_model = float(cf2_model.iloc[sec_rows["man"], eu4_col])
        cf2_trd_model = float(cf2_model.iloc[sec_rows["trd"], eu4_col])
        cf2_bss_model = float(cf2_model.iloc[sec_rows["bss"], eu4_col])
        cf2_man_ss = float(cf2_ss.iloc[sec_rows["man"], eu4_col])
        cf2_trd_ss = float(cf2_ss.iloc[sec_rows["trd"], eu4_col])
        cf2_bss_ss = float(cf2_ss.iloc[sec_rows["bss"], eu4_col])
        print(f"  CF2 Model (EU4): man={cf2_man_model:.2f}, trd={cf2_trd_model:.2f}, bss={cf2_bss_model:.2f}")
        print(f"  CF2 SS (EU4):    man={cf2_man_ss:.2f}, trd={cf2_trd_ss:.2f}, bss={cf2_bss_ss:.2f}")
    else:
        print("  WARNING: EU4 column not found in CF2 Excel files")
        cf2_man_model = cf2_trd_model = cf2_bss_model = np.nan
        cf2_man_ss = cf2_trd_ss = cf2_bss_ss = np.nan
else:
    print(
        f"  WARNING: CF2 Excel files not found. Run trade_counterfactuals_endogenous.py first."
    )
    cf2_man_model = cf2_trd_model = cf2_bss_model = np.nan
    cf2_man_ss = cf2_trd_ss = cf2_bss_ss = np.nan

# --- Panel B: Trade cure net exports ---
# Try reading from saved Excel file first (from trade_counterfactuals_endogenous.py)
trade_cure_file = "../output/figures/trade_cure_nx_endo.xlsx"
trade_cure_available = False

if os.path.exists(trade_cure_file):
    tc_df = pd.read_excel(trade_cure_file)
    tc_dict = dict(zip(tc_df["Sector"], zip(tc_df["Observed_NX"], tc_df["Counterfactual_NX"])))
    EUR4_nx_agr_E, EUR4_nx_cf_agr = tc_dict["agr"]
    EUR4_nx_man_E, EUR4_nx_cf_man = tc_dict["man"]
    EUR4_nx_trd_E, EUR4_nx_cf_trd = tc_dict["trd"]
    EUR4_nx_bss_E, EUR4_nx_cf_bss = tc_dict["bss"]
    EUR4_nx_fin_E, EUR4_nx_cf_fin = tc_dict["fin"]
    EUR4_nx_nps_E, EUR4_nx_cf_nps = tc_dict["nps"]
    trade_cure_available = True
    print(f"  Trade cure NX loaded from {trade_cure_file}")
elif hasattr(DEU, "nx_cf_agr"):
    trade_cure_available = True
else:
    trade_cure_available = False

if trade_cure_available and not os.path.exists(trade_cure_file):
    # Compute from country objects (find_nx was run in this session)
    EUR4_nx_agr_E = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_agr_E)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_agr_E)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_agr_E)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_agr_E)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_man_E = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_man_E)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_man_E)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_man_E)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_man_E)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_trd_E = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_trd_E)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_trd_E)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_trd_E)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_trd_E)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_bss_E = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_bss_E)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_bss_E)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_bss_E)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_bss_E)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_fin_E = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_fin_E)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_fin_E)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_fin_E)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_fin_E)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_nps_E = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_nps_E)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_nps_E)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_nps_E)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_nps_E)[-1]
    ) / EUR4_h_tot[-1]

    # Counterfactual net exports
    EUR4_nx_cf_agr = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_cf_agr)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_cf_agr)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_cf_agr)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_cf_agr)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_cf_man = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_cf_man)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_cf_man)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_cf_man)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_cf_man)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_cf_trd = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_cf_trd)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_cf_trd)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_cf_trd)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_cf_trd)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_cf_bss = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_cf_bss)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_cf_bss)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_cf_bss)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_cf_bss)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_cf_fin = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_cf_fin)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_cf_fin)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_cf_fin)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_cf_fin)[-1]
    ) / EUR4_h_tot[-1]
    EUR4_nx_cf_nps = (
        np.array(DEU.h_tot)[-1] * np.array(DEU.nx_cf_nps)[-1]
        + np.array(GBR.h_tot)[-1] * np.array(GBR.nx_cf_nps)[-1]
        + np.array(FRA.h_tot)[-1] * np.array(FRA.nx_cf_nps)[-1]
        + np.array(ITA.h_tot)[-1] * np.array(ITA.nx_cf_nps)[-1]
    ) / EUR4_h_tot[-1]

    print(f"  Trade cure observed NX:  agr={EUR4_nx_agr_E:.4f}, man={EUR4_nx_man_E:.4f}, trd={EUR4_nx_trd_E:.4f}, bss={EUR4_nx_bss_E:.4f}, fin={EUR4_nx_fin_E:.4f}, nps={EUR4_nx_nps_E:.4f}")
    print(f"  Trade cure CF NX:        agr={EUR4_nx_cf_agr:.4f}, man={EUR4_nx_cf_man:.4f}, trd={EUR4_nx_cf_trd:.4f}, bss={EUR4_nx_cf_bss:.4f}, fin={EUR4_nx_cf_fin:.4f}, nps={EUR4_nx_cf_nps:.4f}")

if not trade_cure_available:
    print(
        "  NOTE: Trade cure (find_nx) results not available."
    )
    print(
        "  Run trade_counterfactuals_endogenous.py first to compute find_nx."
    )
    print("  Using NaN placeholders for trade cure table.")
    EUR4_nx_agr_E = EUR4_nx_man_E = EUR4_nx_trd_E = np.nan
    EUR4_nx_bss_E = EUR4_nx_fin_E = EUR4_nx_nps_E = np.nan
    EUR4_nx_cf_agr = EUR4_nx_cf_man = EUR4_nx_cf_trd = np.nan
    EUR4_nx_cf_bss = EUR4_nx_cf_fin = EUR4_nx_cf_nps = np.nan

# Total observed and counterfactual net exports
EUR4_nx_tot_E = (
    EUR4_nx_agr_E
    + EUR4_nx_man_E
    + EUR4_nx_trd_E
    + EUR4_nx_bss_E
    + EUR4_nx_fin_E
    + EUR4_nx_nps_E
)
EUR4_nx_tot_cf = (
    EUR4_nx_cf_agr
    + EUR4_nx_cf_man
    + EUR4_nx_cf_trd
    + EUR4_nx_cf_bss
    + EUR4_nx_cf_fin
    + EUR4_nx_cf_nps
)


def fmt_val(v, decimals=2):
    """Format a value for the table, handling NaN."""
    if np.isnan(v):
        return "---"
    return f"{v:.{decimals}f}"


def fmt_pct(v):
    """Format a percentage value."""
    if np.isnan(v):
        return "---"
    return f"{100*v:.2f}"


# --- Generate LaTeX table ---
latex = r"""\begin{table}[htbp]
\centering
\caption{Open Economy Counterfactuals -- Endogenous Trade Model}
\label{tab:trade_cf_endo}
\begin{tabular}{lcccccc}
\hline\hline
"""

# Panel A: Catch-up counterfactual
latex += r"& \multicolumn{6}{c}{\textit{Panel A: Catch-Up Counterfactual (CF2)}} \\" + "\n"
latex += r"\cmidrule(lr){2-7}" + "\n"
latex += r"& man & trd & bss & Total & & \\" + "\n"
latex += r"\hline" + "\n"
latex += (
    r"Model & "
    + fmt_val(cf2_man_model)
    + r" & "
    + fmt_val(cf2_trd_model)
    + r" & "
    + fmt_val(cf2_bss_model)
    + r" & & & \\"
    + "\n"
)
latex += (
    r"Shift-Share & "
    + fmt_val(cf2_man_ss)
    + r" & "
    + fmt_val(cf2_trd_ss)
    + r" & "
    + fmt_val(cf2_bss_ss)
    + r" & & & \\"
    + "\n"
)
latex += r"\hline" + "\n"

# Panel B: Trade cure
latex += r"& \multicolumn{6}{c}{\textit{Panel B: Net Trade Adjustment (EUR4)}} \\" + "\n"
latex += r"\cmidrule(lr){2-7}" + "\n"
latex += r"& agr & man & trd & bss & fin & nps \\" + "\n"
latex += r"\hline" + "\n"
latex += (
    r"Observed $nx$ & "
    + fmt_pct(EUR4_nx_agr_E)
    + r"\% & "
    + fmt_pct(EUR4_nx_man_E)
    + r"\% & "
    + fmt_pct(EUR4_nx_trd_E)
    + r"\% & "
    + fmt_pct(EUR4_nx_bss_E)
    + r"\% & "
    + fmt_pct(EUR4_nx_fin_E)
    + r"\% & "
    + fmt_pct(EUR4_nx_nps_E)
    + r"\% \\"
    + "\n"
)
latex += (
    r"Counterfactual $nx$ & "
    + fmt_pct(EUR4_nx_cf_agr)
    + r"\% & "
    + fmt_pct(EUR4_nx_cf_man)
    + r"\% & "
    + fmt_pct(EUR4_nx_cf_trd)
    + r"\% & "
    + fmt_pct(EUR4_nx_cf_bss)
    + r"\% & "
    + fmt_pct(EUR4_nx_cf_fin)
    + r"\% & "
    + fmt_pct(EUR4_nx_cf_nps)
    + r"\% \\"
    + "\n"
)
latex += r"\hline\hline" + "\n"
latex += r"""\end{tabular}
\end{table}
"""

with open("../output/figures/tab_trade_cf_endo.tex", "w") as f:
    f.write(latex)
print("  Saved: output/tables/tab_trade_cf_endo.tex")

# --- Generate Excel table ---
table_data = {
    "Panel A: CF2 Catch-Up": ["", "man", "trd", "bss"],
    "Model": ["", fmt_val(cf2_man_model), fmt_val(cf2_trd_model), fmt_val(cf2_bss_model)],
    "Shift-Share": ["", fmt_val(cf2_man_ss), fmt_val(cf2_trd_ss), fmt_val(cf2_bss_ss)],
}
df_a = pd.DataFrame(table_data)

table_b = {
    "Panel B: Net Trade (EUR4)": ["Observed nx", "Counterfactual nx"],
    "agr": [EUR4_nx_agr_E, EUR4_nx_cf_agr],
    "man": [EUR4_nx_man_E, EUR4_nx_cf_man],
    "trd": [EUR4_nx_trd_E, EUR4_nx_cf_trd],
    "bss": [EUR4_nx_bss_E, EUR4_nx_cf_bss],
    "fin": [EUR4_nx_fin_E, EUR4_nx_cf_fin],
    "nps": [EUR4_nx_nps_E, EUR4_nx_cf_nps],
    "Total": [EUR4_nx_tot_E, EUR4_nx_tot_cf],
}
df_b = pd.DataFrame(table_b)

with pd.ExcelWriter("../output/figures/tab_trade_cf_endo.xlsx") as writer:
    df_a.to_excel(writer, sheet_name="CF2 Catch-Up", index=False)
    df_b.to_excel(writer, sheet_name="Trade Cure NX", index=False)
print("  Saved: output/tables/tab_trade_cf_endo.xlsx")

print("\nEndogenous trade model outputs generated.")


# ==============================================================================
# FIGURE 5: Trade Openness by Sector (US vs EUR4)
# NOTE: Now generated by generate_fig_opennes.py (Step 11 in master.py)
# ==============================================================================
print("Figure 5 generated by generate_fig_opennes.py (Step 11).")

if False:  # Old inline Figure 5 code — replaced by generate_fig_opennes.py
    import statsmodels.api as sm

    io_data = pd.read_excel("../data/io_panel.xlsx", index_col=[0, 1], engine="openpyxl")
    agg_data = pd.read_excel(
    "../data/exp_imp_aggregate_panel.xlsx", index_col=[0, 1], engine="openpyxl"
)

sectors_io = ["agr", "man", "trd", "bss", "fin", "nps"]
sector_styles = {
    "agr": ("D-", "darkgreen", "lime", "darkgreen"),
    "man": ("o-", "darkblue", "lightskyblue", "darkblue"),
    "trd": ("s-", "darkred", "lightcoral", "darkred"),
    "bss": ("^-", "darkmagenta", "violet", "darkmagenta"),
    "fin": ("p-", "darkcyan", "cyan", "darkcyan"),
    "nps": ("v-", "saddlebrown", "sandybrown", "saddlebrown"),
}


def compute_trade_openness(io_data, agg_data, country):
    """Compute (Exports + |Imports|)/GDP by sector, HP-filtered."""
    openness = {}
    for sec in sectors_io:
        sec_data = io_data.loc[country].loc[io_data.loc[country, "sec"] == sec]
        gdp = agg_data.loc[country, "gdp"]
        # impo is stored as negative, so (-1)*impo = |imports|
        raw = (sec_data["expo"].values + (-1) * sec_data["impo"].values) / gdp.values
        _, openness[sec] = sm.tsa.filters.hpfilter(raw, 100)
    return openness


try:
    us_openness = compute_trade_openness(io_data, agg_data, "USA")

    # EUR4: hours-weighted average of trade openness
    eur4_countries_io = {"DEU": DEU, "FRA": FRA, "GBR": GBR, "ITA": ITA}
    eur4_openness = {}
    for sec in sectors_io:
        weighted = None
        total_h = None
        for c_code, c_obj in eur4_countries_io.items():
            c_data = io_data.loc[c_code]
            c_data = c_data.loc[c_data["sec"] == sec]
            gdp_c = agg_data.loc[c_code, "gdp"]
            raw = (c_data["expo"].values + (-1) * c_data["impo"].values) / gdp_c.values
            _, hp = sm.tsa.filters.hpfilter(raw, 100)
            h_c = np.array(c_obj.h_tot).flatten()
            n = min(len(hp), len(h_c))
            hp_arr = np.array(hp)[:n]
            h_arr = h_c[:n]
            if weighted is None:
                weighted = hp_arr * h_arr
                total_h = h_arr.copy()
            else:
                weighted[:n] += hp_arr * h_arr[:n]
                total_h[:n] += h_arr[:n]
        eur4_openness[sec] = weighted / np.where(total_h > 0, total_h, 1)

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # Left panel: US
    ax = plt.subplot(1, 2, 1)
    for sec in sectors_io:
        marker, color, mfc, mec = sector_styles[sec]
        plt.plot(
            us_openness[sec],
            marker,
            markersize=8,
            color=color,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markevery=3,
            alpha=0.95,
            label=r"$\texttt{" + sec + r"}$",
        )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("(Exports + Imports)/GDP", fontsize=14)
    plt.title("United States", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([1994, 2020, 0, 0.33])
    plt.grid()

    # Right panel: EUR4
    ax = plt.subplot(1, 2, 2)
    for sec in sectors_io:
        marker, color, mfc, mec = sector_styles[sec]
        plt.plot(
            eur4_openness[sec],
            marker,
            markersize=8,
            color=color,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markevery=3,
            alpha=0.95,
            label=r"$\texttt{" + sec + r"}$",
        )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("(Exports + Imports)/GDP", fontsize=14)
    plt.title("Europe", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([1994, 2020, 0, 0.33])
    plt.grid()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=14
    )
    plt.tight_layout()
    plt.savefig("../output/figures/figure_5.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: output/figures/figure_5.pdf")
except Exception as e:
    print(f"  WARNING: Could not generate Figure 5: {e}")


# ==============================================================================
# FIGURE 3: Counterfactual decomposition of productivity gap
# (Requires Counterfactual_ts.xlsx from Step 3)
# ==============================================================================
print("\nGenerating Figure 3 (counterfactual decomposition)...")
try:
    from model_test_europe import DEU
    data_cfs = pd.read_excel('../output/figures/Counterfactual_ts.xlsx')

    fig, ax = plt.subplots(1, 1)
    ax.plot(DEU.year, data_cfs['cf3'], 'D--', markersize=6, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha=0.95, label='CF1: U.S. labor reallocation after 1990')
    ax.plot(DEU.year, data_cfs['cf2'], 'o--', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha=0.95, label='CF2: U.S. sectoral productivity after 1990')
    ax.plot(DEU.year, data_cfs['cf3_init'], 'v--', markersize=6, color='orange', markerfacecolor='orange', markeredgecolor='darkorange', markevery=7, alpha=0.95, label='CF3: No labor reallocation after 1990')
    ax.plot(DEU.year, data_cfs['obs'], 's-', markersize=6, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label='Model baseline')
    plt.title('Labor Productivity (Relative to U.S.)', fontsize=16)
    ax.axis([1968, 2021, 0.68, 1.04])
    plt.xticks(fontsize=14)
    plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig('../output/figures/fig_cfs.pdf', bbox_inches="tight")
    plt.close()
    print("  Saved: output/figures/fig_cfs.pdf")
except Exception as e:
    print(f"  WARNING: Could not generate Figure 3: {e}")


# ==============================================================================
# CONSOLIDATION: Copy all outputs to paper-consistent names
# ==============================================================================
print("\nConsolidating outputs with paper-consistent naming...")

import shutil

os.makedirs("../output/figures", exist_ok=True)
os.makedirs("../output/tables", exist_ok=True)

# Figure mapping: source -> paper name
figure_map = {
    # Main figures
    "fig_1.pdf": "figure_1.pdf",
    "fig_2.pdf": "figure_2.pdf",
    "fig_cfs.pdf": "figure_3.pdf",
    "fig_opennes.pdf": "figure_5.pdf",
    "fig_2_open_endo.pdf": "figure_6.pdf",
    "fig_test_EUR_endo.pdf": "figure_A4.pdf",
    # Appendix figures
    "fig_USA_Ai_appendix.pdf": "figure_A1.pdf",
    "fig_calibration_USA.pdf": "figure_A2.pdf",
    "fig_3.pdf": "figure_4.pdf",
    "fig_2_ams.pdf": "figure_A3.pdf",
    "fig_test_EUR_appendix.pdf": "figure_A4.pdf",
    "fig_test_EUR_2_appendix.pdf": "figure_A5.pdf",
}

# Table mapping: source path -> paper name
table_map = {
    # Main tables
    "../output/tables/table1_ss.xlsx": "table_1.xlsx",
    "../output/data/table_cf1_EU4.tex": "table_2a.tex",
    "../output/data/table_cf2_EU4.tex": "table_2b.tex",
    "../output/tables/table_3.tex": "table_3.tex",
    "../output/tables/table_3.xlsx": "table_3.xlsx",
    "../output/figures/corr_lp_tfp_klems.tex": "table_4.tex",
    "../output/data/beta_last_period_results.xlsx": "table_5.xlsx",
    "../output/figures/tab_trade_cf_endo.tex": "table_6.tex",
    "../output/figures/tab_trade_cf_endo.xlsx": "table_6.xlsx",
    # Appendix tables
    "../output/data/table_cf1_ams_EU4.tex": "table_A4a.tex",
    "../output/data/table_cf2_ams_EU4.tex": "table_A4b.tex",
    "../output/tables/table_c2_new.tex": "table_A6.tex",
    "../output/tables/table_c3_new.tex": "table_A7.tex",
    "../output/data/table_cf1_EU15.tex": "table_A8a.tex",
    "../output/data/table_cf2_EU15.tex": "table_A8b.tex",
    "../output/data/table_cf1_GBR.tex": "table_A8c.tex",
    "../output/data/table_cf2_GBR.tex": "table_A8d.tex",
    "../output/tables/table_c2_core_vs_periphery_new.tex": "table_A9.tex",
    "../output/tables/table_c3_core_vs_periphery_new.tex": "table_A10.tex",
}

# Copy figures
copied = 0
for src_name, dst_name in figure_map.items():
    src = os.path.join("../output/figures", src_name)
    dst = os.path.join("../output/figures", dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  {src_name} -> {dst_name}")
        copied += 1

# Copy tables
for src_path, dst_name in table_map.items():
    dst = os.path.join("../output/tables", dst_name)
    if os.path.exists(src_path) and os.path.abspath(src_path) != os.path.abspath(dst):
        shutil.copy2(src_path, dst)
        print(f"  {os.path.basename(src_path)} -> {dst_name}")
        copied += 1

print(f"\nConsolidation complete: {copied} files copied with paper-consistent names.")

# Clean up: remove intermediate files, keep only paper-named outputs
print("\nCleaning up intermediate files...")
for folder in ["../output/figures", "../output/tables"]:
    for f in os.listdir(folder):
        if f.startswith("figure_") or f.startswith("table_"):
            continue  # keep paper-named files
        fpath = os.path.join(folder, f)
        if os.path.isfile(fpath):
            os.remove(fpath)

print("Done! All paper outputs generated.")
