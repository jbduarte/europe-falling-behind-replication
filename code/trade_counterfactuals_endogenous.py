"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        trade_counterfactuals_endogenous.py
Purpose:     Run ENDOGENOUS-trade counterfactual experiments. This is the
             headline quantitative exercise: each country's sectoral
             productivities are replaced with US-style paths, and trade flows
             are re-generated endogenously via the calibrated xi_i elasticities.
             Produces Table 6 decomposition (incl. CF4), the 2-panel
             open-vs-closed comparison (Figure 2), and the trade-cure rows.
             Supports optional cache loading (dill) to skip the ~long
             endogenous calibration on re-runs.
Pipeline:    Step 9/19 — Endogenous-trade counterfactuals.
Inputs:      Either (i) the cached US calibration at
             ../output/data/calibration_cache_endogenous.pkl, or (ii) direct
             imports from model_calibration_USA.py, model_calibration_USA_open.py,
             and model_calibration_USA_endogenous_open.py. Also imports European
             objects from model_test_europe_open.py (note: the endogenous EU
             module is wired in downstream of the open-economy EU object).
Outputs:     ../output/figures/Counterfactual_*_trade.xlsx (endogenous CF1 and
             CF2 catch-up AMS/NPS), trade_cure_data table, comparison_data
             for Figure 2, and ../output/figures/fig_2_open_comparison.pdf.
Dependencies: model_calibration_USA.py (Step 1), model_calibration_USA_open.py
              (Step 4), model_calibration_USA_endogenous_open.py (Step 7),
              model_test_europe_open.py (Step 5). Optional cache file speeds
              re-execution.
"""

import matplotlib
matplotlib.use("Agg")

import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import fsolve

rc("text", usetex=True)
rc("font", family="serif")

# --------------------------------------------------------------------------
# Calibration loading: try cache first, fall back to full (re-)calibration.
#
# The endogenous-trade calibration takes ~60 minutes end-to-end (model_country
# instantiation + xi_i calibration + European aggregates) and is deterministic in
# inputs, so we persist the full calibration object to a dill pickle the first
# time it runs. Subsequent runs unpack from the cache in seconds.
#
# Cache path:  ../output/data/calibration_cache_endogenous.pkl
# Created by:  save_calibration_endogenous.py (Step 8/19)
# Consumed by: this file (USE_CACHE branch below)
#
# To force a full recalibration (e.g., after changing model_test_europe_open.py,
# model_calibration_USA_endogenous_open.py, or any xi_i input):
#     rm ../output/data/calibration_cache_endogenous.pkl
# Then re-run save_calibration_endogenous.py (or just this script; the fallback
# branch will regenerate everything, but the cache file will not be rewritten
# unless save_calibration_endogenous.py is run).
#
# dill (not vanilla pickle) is required because model_country instances carry
# bound methods and closures that pickle cannot serialise.
# --------------------------------------------------------------------------
USE_CACHE = os.path.exists("../output/data/calibration_cache_endogenous.pkl")

if USE_CACHE:
    import dill
    print("Loading calibration from cache...")
    with open("../output/data/calibration_cache_endogenous.pkl", "rb") as _f:
        _cache = dill.load(_f)
    # Unpack US parameters
    sigma = _cache["sigma"]
    eps_agr = _cache["eps_agr"]
    eps_trd = _cache["eps_trd"]
    eps_fin = _cache["eps_fin"]
    eps_bss = _cache["eps_bss"]
    eps_nps = _cache["eps_nps"]
    eps_ser = _cache["eps_ser"]
    # Unpack US calibration
    GDP_ph = _cache["GDP_ph"]
    E = _cache["E"]
    A_tot = _cache["A_tot"]
    share_agr = _cache["share_agr"]
    share_man = _cache["share_man"]
    share_trd = _cache["share_trd"]
    share_bss = _cache["share_bss"]
    share_fin = _cache["share_fin"]
    share_nps = _cache["share_nps"]
    share_ser = _cache["share_ser"]
    share_agr_ams = _cache["share_agr_ams"]
    share_man_ams = _cache["share_man_ams"]
    share_ser_ams = _cache["share_ser_ams"]
    A_tot_ams = _cache["A_tot_ams"]
    share_agr_nps = _cache["share_agr_nps"]
    share_man_nps = _cache["share_man_nps"]
    share_trd_nps = _cache["share_trd_nps"]
    share_bss_nps = _cache["share_bss_nps"]
    share_fin_nps = _cache["share_fin_nps"]
    share_nps_nps = _cache["share_nps_nps"]
    A_tot_nps = _cache["A_tot_nps"]
    xi_agr = _cache["xi_agr"]
    xi_man = _cache["xi_man"]
    xi_trd = _cache["xi_trd"]
    xi_bss = _cache["xi_bss"]
    xi_fin = _cache["xi_fin"]
    xi_nps_usa = _cache["xi_nps"]
    xi_ser = _cache["xi_ser"]
    x_agr_q_index_usa = _cache["x_agr_q_index"]
    x_man_q_index_usa = _cache["x_man_q_index"]
    x_trd_q_index_usa = _cache["x_trd_q_index"]
    x_bss_q_index_usa = _cache["x_bss_q_index"]
    x_fin_q_index_usa = _cache["x_fin_q_index"]
    x_nps_q_index_usa = _cache["x_nps_q_index"]
    x_ser_q_index_usa = _cache["x_ser_q_index"]
    x_agr_index_usa = _cache["x_agr_index"]
    x_man_index_usa = _cache["x_man_index"]
    x_trd_index_usa = _cache["x_trd_index"]
    x_bss_index_usa = _cache["x_bss_index"]
    x_fin_index_usa = _cache["x_fin_index"]
    x_nps_index_usa = _cache["x_nps_index"]
    x_ser_index_usa = _cache["x_ser_index"]
    M_agr_E_usa = _cache["M_agr_E"]
    M_man_E_usa = _cache["M_man_E"]
    M_trd_E_usa = _cache["M_trd_E"]
    M_bss_E_usa = _cache["M_bss_E"]
    M_fin_E_usa = _cache["M_fin_E"]
    M_nps_E_usa = _cache["M_nps_E"]
    M_ser_E_usa = _cache["M_ser_E"]
    GDP_ph_USA = GDP_ph
    E_USA = E
    # Unpack country instances
    model_country = _cache["model_country"]
    AUT = _cache["countries"]["AUT"]
    BEL = _cache["countries"]["BEL"]
    DEU = _cache["countries"]["DEU"]
    DNK = _cache["countries"]["DNK"]
    ESP = _cache["countries"]["ESP"]
    FIN = _cache["countries"]["FIN"]
    FRA = _cache["countries"]["FRA"]
    GBR = _cache["countries"]["GBR"]
    GRC = _cache["countries"]["GRC"]
    ITA = _cache["countries"]["ITA"]
    NLD = _cache["countries"]["NLD"]
    PRT = _cache["countries"]["PRT"]
    SWE = _cache["countries"]["SWE"]
    USA = _cache["countries"]["USA"]
    # Unpack European aggregates
    EUR4_h_tot = _cache["EUR4_h_tot"]
    EURCORE_h_tot = _cache["EURCORE_h_tot"]
    EURPERI_h_tot = _cache["EURPERI_h_tot"]
    EUR13_h_tot = _cache["EUR13_h_tot"]
    EUR4_A_tot = _cache["EUR4_A_tot"]
    EURCORE_A_tot = _cache["EURCORE_A_tot"]
    EURPERI_A_tot = _cache["EURPERI_A_tot"]
    EUR13_A_tot = _cache["EUR13_A_tot"]
    EUR4_rel_A_tot = _cache["EUR4_rel_A_tot"]
    EUR13_rel_A_tot = _cache["EUR13_rel_A_tot"]
    EUR4_E = _cache["EUR4_E"]
    EUR13_E = _cache["EUR13_E"]
    EUR4_rel_E = _cache["EUR4_rel_E"]
    EUR13_rel_E = _cache["EUR13_rel_E"]
    EUR4_share_agr = _cache["EUR4_share_agr"]
    EUR13_share_agr = _cache["EUR13_share_agr"]
    EUR4_share_man = _cache["EUR4_share_man"]
    EUR13_share_man = _cache["EUR13_share_man"]
    EUR4_share_ser = _cache["EUR4_share_ser"]
    EUR13_share_ser = _cache["EUR13_share_ser"]
    EUR4_share_trd = _cache["EUR4_share_trd"]
    EUR13_share_trd = _cache["EUR13_share_trd"]
    EUR4_share_bss = _cache["EUR4_share_bss"]
    EUR13_share_bss = _cache["EUR13_share_bss"]
    EUR4_share_fin = _cache["EUR4_share_fin"]
    EUR13_share_fin = _cache["EUR13_share_fin"]
    EUR4_share_nps = _cache["EUR4_share_nps"]
    EUR13_share_nps = _cache["EUR13_share_nps"]
    EUR4_share_agr_ams_m = _cache["EUR4_share_agr_ams_m"]
    EUR13_share_agr_ams_m = _cache["EUR13_share_agr_ams_m"]
    EUR4_share_agr_nps_m = _cache["EUR4_share_agr_nps_m"]
    EUR13_share_agr_nps_m = _cache["EUR13_share_agr_nps_m"]
    EUR4_share_man_ams_m = _cache["EUR4_share_man_ams_m"]
    EUR13_share_man_ams_m = _cache["EUR13_share_man_ams_m"]
    EUR4_share_man_nps_m = _cache["EUR4_share_man_nps_m"]
    EUR13_share_man_nps_m = _cache["EUR13_share_man_nps_m"]
    EUR4_share_ser_ams_m = _cache["EUR4_share_ser_ams_m"]
    EUR13_share_ser_ams_m = _cache["EUR13_share_ser_ams_m"]
    EUR4_share_trd_nps_m = _cache["EUR4_share_trd_nps_m"]
    EUR13_share_trd_nps_m = _cache["EUR13_share_trd_nps_m"]
    EUR4_share_bss_nps_m = _cache["EUR4_share_bss_nps_m"]
    EUR13_share_bss_nps_m = _cache["EUR13_share_bss_nps_m"]
    EUR4_share_fin_nps_m = _cache["EUR4_share_fin_nps_m"]
    EUR13_share_fin_nps_m = _cache["EUR13_share_fin_nps_m"]
    EUR4_share_nps_nps_m = _cache["EUR4_share_nps_nps_m"]
    EUR13_share_nps_nps_m = _cache["EUR13_share_nps_nps_m"]
    EUR4_A_tot_ams = _cache["EUR4_A_tot_ams"]
    EUR13_A_tot_ams = _cache["EUR13_A_tot_ams"]
    EUR4_A_tot_nps = _cache["EUR4_A_tot_nps"]
    EURCORE_A_tot_nps = _cache["EURCORE_A_tot_nps"]
    EURPERI_A_tot_nps = _cache["EURPERI_A_tot_nps"]
    EUR13_A_tot_nps = _cache["EUR13_A_tot_nps"]
    del _cache
    print("Cache loaded successfully.")

if not USE_CACHE:
    from model_calibration_USA import (
        sigma,
        eps_agr,
        eps_trd,
        eps_fin,
        eps_bss,
        eps_nps,
        eps_ser,
    )

if not USE_CACHE:
    from model_calibration_USA_endogenous_open import (
        GDP_ph,
        E,
        A_tot_nps,
    )
    # Rename US Aggregates
    GDP_ph_USA, E_USA = GDP_ph, E

if not USE_CACHE:
    from model_test_europe_endogenous_xn import (
        model_country,
        EUR4_h_tot,
        EURCORE_h_tot,
        EURPERI_h_tot,
        EUR13_h_tot,
        EUR4_rel_A_tot,
        EUR4_share_agr,
        EUR4_share_man,
        EUR4_share_nps,
        EUR4_share_agr_nps_m,
        EUR4_share_man_nps_m,
        EUR4_share_nps_nps_m,
        EUR4_A_tot_nps,
    )


if not USE_CACHE:
    AUT = model_country("AUT")
    AUT.productivity_series()
    AUT.predictions_ams()
    AUT.predictions_nps()

    BEL = model_country("BEL")
    BEL.productivity_series()
    BEL.predictions_ams()
    BEL.predictions_nps()

    DEU = model_country("DEU")
    DEU.productivity_series()
    DEU.predictions_ams()
    DEU.predictions_nps()

    DNK = model_country("DNK")
    DNK.productivity_series()
    DNK.predictions_ams()
    DNK.predictions_nps()

    ESP = model_country("ESP")
    ESP.productivity_series()
    ESP.predictions_ams()
    ESP.predictions_nps()

    FIN = model_country("FIN")
    FIN.productivity_series()
    FIN.predictions_ams()
    FIN.predictions_nps()

    FRA = model_country("FRA")
    FRA.productivity_series()
    FRA.predictions_ams()
    FRA.predictions_nps()

    GBR = model_country("GBR")
    GBR.productivity_series()
    GBR.predictions_ams()
    GBR.predictions_nps()

    GRC = model_country("GRC")
    GRC.productivity_series()
    GRC.predictions_ams()
    GRC.predictions_nps()

    # IRL = model_country('IRL')
    # IRL.productivity_series()
    # IRL.predictions_ams()
    # IRL.predictions_nps()

    ITA = model_country("ITA")
    ITA.productivity_series()
    ITA.predictions_ams()
    ITA.predictions_nps()

    # LUX = model_country('LUX')
    # LUX.productivity_series()
    # LUX.predictions_ams()
    # LUX.predictions_nps()

    NLD = model_country("NLD")
    NLD.productivity_series()
    NLD.predictions_ams()
    NLD.predictions_nps()

    PRT = model_country("PRT")
    PRT.productivity_series()
    PRT.predictions_ams()
    PRT.predictions_nps()

    SWE = model_country("SWE")
    SWE.productivity_series()
    SWE.predictions_ams()
    SWE.predictions_nps()

    # UNITED STATES
    USA = model_country("USA")
    USA.productivity_series()
    USA.predictions_ams()
    USA.predictions_nps()


# Pre-calibrated country cache for fast deep-copy in the counterfactual class.
# copy.deepcopy on an already-calibrated instance is ~1000x cheaper than calling
# model_country(code).productivity_series() from scratch (which refits all
# auxiliary regressions), and it guarantees each counterfactual starts from the
# same calibrated primitives without mutating the originals across CF experiments.
_CALIBRATED = {
    "AUT": AUT, "BEL": BEL, "DEU": DEU, "DNK": DNK,
    "ESP": ESP, "FIN": FIN, "FRA": FRA, "GBR": GBR,
    "GRC": GRC, "ITA": ITA, "NLD": NLD, "PRT": PRT,
    "SWE": SWE, "USA": USA,
}


class counterfactual:
    "Counterfactuals"

    def __init__(self, country_code):
        # Deep-copy from _CALIBRATED: counterfactuals mutate self.cou (productivity
        # paths, A_i_catch, growth-rate overrides). Without deepcopy, running a
        # second counterfactual for the same country would start from the previous
        # CF's perturbed state rather than the baseline calibration.
        self.country_code = country_code
        self.cou = copy.deepcopy(_CALIBRATED[country_code])

    "Baseline"

    def baseline(self):
        "Shift-share baseline"
        self.ss_A_base_ams_init = (
            np.array(self.cou.share_agr)[0] * self.cou.A_agr[-1]
            + np.array(self.cou.share_man)[0] * self.cou.A_man[-1]
            + np.array(self.cou.share_ser)[0] * self.cou.A_ser[-1]
        )
        self.ss_A_base_ams = (
            self.cou.share_agr * self.cou.A_agr
            + self.cou.share_man * self.cou.A_man
            + self.cou.share_ser * self.cou.A_ser
        )

        self.ss_A_base_nps_init = (
            np.array(self.cou.share_agr)[0] * self.cou.A_agr[-1]
            + np.array(self.cou.share_man)[0] * self.cou.A_man[-1]
            + np.array(self.cou.share_trd)[0] * self.cou.A_trd[-1]
            + np.array(self.cou.share_bss)[0] * self.cou.A_bss[-1]
            + np.array(self.cou.share_fin)[0] * self.cou.A_fin[-1]
            + np.array(self.cou.share_nps)[0] * self.cou.A_nps[-1]
        )
        self.ss_A_base_nps = (
            self.cou.share_agr * self.cou.A_agr
            + self.cou.share_man * self.cou.A_man
            + self.cou.share_trd * self.cou.A_trd
            + self.cou.share_bss * self.cou.A_bss
            + self.cou.share_fin * self.cou.A_fin
            + self.cou.share_nps * self.cou.A_nps
        )

        "Baseline C and E from endogenous open-economy model"
        self.C_baseline = list(self.cou.C)
        self.E_baseline = list(self.cou.E)

        # Closed-economy C baseline (ams model): solve the market-clearing identity
        # L_t^(1-sigma) = sum_i om_i * C_t^eps_i * A_it^(sigma-1) for C_t, year by
        # year. This is the closed-economy counterpart used only to form the ratio
        # C_closed_cf / C_closed_baseline in _compute_cf_C — the open-economy C
        # stays anchored to self.C_baseline. Initial guess L_t (same order of
        # magnitude as C) converges in fewer than 10 iterations at default xtol.
        "Closed-economy C baseline (ams model)"
        C_lev_E_ams = []
        for i in range(len(self.cou.h_tot)):   # time loop over years (1970-2019)
            L_t = np.array(self.cou.h_tot)[i]
            A_agr_t = np.array(self.cou.A_agr)[i]
            A_man_t = np.array(self.cou.A_man)[i]
            A_ser_t = np.array(self.cou.A_ser)[i]

            def C_exp_ams(C):
                return L_t ** (1 - sigma) - (
                    self.cou.om_agr_ams
                    * (C**eps_agr)
                    * (A_agr_t ** (sigma - 1))
                    + self.cou.om_man_ams * C * (A_man_t ** (sigma - 1))
                    + (1 - self.cou.om_agr_ams - self.cou.om_man_ams)
                    * (C**eps_ser)
                    * (A_ser_t ** (sigma - 1))
                )

            C_lev_E_ams.append(fsolve(C_exp_ams, L_t).item())
        C_level_E_ams = np.array(C_lev_E_ams)
        g_C_E_ams = C_level_E_ams[1:] / C_level_E_ams[:-1] - 1
        self.C_E_ams_baseline = [
            float(np.array(self.cou.GDP_ph)[0] / np.array(GDP_ph_USA)[0])
        ]
        for i in range(len(g_C_E_ams)):
            self.C_E_ams_baseline.append(
                float((1 + g_C_E_ams[i]) * self.C_E_ams_baseline[i])
            )

        "Closed-economy C baseline (nps model)"
        C_lev_E_nps = []
        for i in range(len(self.cou.h_tot)):
            L_t = np.array(self.cou.h_tot)[i]
            A_agr_t = np.array(self.cou.A_agr)[i]
            A_man_t = np.array(self.cou.A_man)[i]
            A_trd_t = np.array(self.cou.A_trd)[i]
            A_bss_t = np.array(self.cou.A_bss)[i]
            A_fin_t = np.array(self.cou.A_fin)[i]
            A_nps_t = np.array(self.cou.A_nps)[i]

            def C_exp_nps(C):
                return L_t ** (1 - sigma) - (
                    self.cou.om_agr_nps
                    * (C**eps_agr)
                    * (A_agr_t ** (sigma - 1))
                    + self.cou.om_man_nps * C * (A_man_t ** (sigma - 1))
                    + self.cou.om_trd_nps
                    * (C**eps_trd)
                    * (A_trd_t ** (sigma - 1))
                    + self.cou.om_bss_nps
                    * (C**eps_bss)
                    * (A_bss_t ** (sigma - 1))
                    + self.cou.om_fin_nps
                    * (C**eps_fin)
                    * (A_fin_t ** (sigma - 1))
                    + (
                        1
                        - self.cou.om_agr_nps
                        - self.cou.om_man_nps
                        - self.cou.om_trd_nps
                        - self.cou.om_bss_nps
                        - self.cou.om_fin_nps
                    )
                    * (C**eps_nps)
                    * (A_nps_t ** (sigma - 1))
                )

            C_lev_E_nps.append(fsolve(C_exp_nps, L_t).item())
        C_level_E_nps = np.array(C_lev_E_nps)
        g_C_E_nps = C_level_E_nps[1:] / C_level_E_nps[:-1] - 1
        self.C_E_nps_baseline = [
            float(np.array(self.cou.GDP_ph)[0] / np.array(GDP_ph_USA)[0])
        ]
        for i in range(len(g_C_E_nps)):
            self.C_E_nps_baseline.append(
                float((1 + g_C_E_nps[i]) * self.C_E_nps_baseline[i])
            )

    def _compute_cf_C(self):
        """Compute counterfactual utility index C using a closed-economy ratio.

        Rationale:
          Under endogenous trade, the open-economy C_t depends on the full fixed
          point (C, E, net exports, labor shares). Re-solving that fixed point at
          every counterfactual grid point would be prohibitively costly. Instead,
          we exploit the fact that the CF change in productivity (A_i^cf vs A_i)
          propagates into C_t similarly in open and closed economy when the trade
          structure is held fixed, and we scale the observed open-economy C by
          the closed-economy ratio:

              C_cf(t) = C_baseline(t) * [C_closed_cf(t) / C_closed_baseline(t)]

          where C_closed_* is obtained by fsolve of the closed-economy market-
          clearing residual (same residual as in baseline(), with CF A values).
          E (aggregate expenditure index) is NOT perturbed here — it tracks the
          exogenous-trade specification so that cross-model comparisons (Fig. 2)
          differ only through the endogenous net-export channel.

          Side effect: sets self.C_cf_ams and self.C_cf_nps as python lists of
          length equal to self.cou.h_tot.
        """
        cou = self.cou

        "Closed-economy C with counterfactual A (ams)"
        C_lev_E_ams = []
        for i in range(len(cou.h_tot)):
            L_t = np.array(cou.h_tot)[i]
            A_agr_t = np.array(cou.A_agr)[i]
            A_man_t = np.array(cou.A_man)[i]
            A_ser_t = np.array(cou.A_ser)[i]

            def C_exp_ams(C):
                return L_t ** (1 - sigma) - (
                    cou.om_agr_ams
                    * (C**eps_agr)
                    * (A_agr_t ** (sigma - 1))
                    + cou.om_man_ams * C * (A_man_t ** (sigma - 1))
                    + (1 - cou.om_agr_ams - cou.om_man_ams)
                    * (C**eps_ser)
                    * (A_ser_t ** (sigma - 1))
                )

            C_lev_E_ams.append(fsolve(C_exp_ams, L_t).item())
        C_level_E_ams = np.array(C_lev_E_ams)
        g_C_E_ams = C_level_E_ams[1:] / C_level_E_ams[:-1] - 1
        C_E_ams_cf = [float(np.array(cou.GDP_ph)[0] / np.array(GDP_ph_USA)[0])]
        for i in range(len(g_C_E_ams)):
            C_E_ams_cf.append(
                float((1 + g_C_E_ams[i]) * C_E_ams_cf[i])
            )

        "Apply ratio: C_cf = C_data * (C_closed_cf / C_closed_baseline)"
        self.C_cf_ams = [
            self.C_baseline[t]
            * (C_E_ams_cf[t] / self.C_E_ams_baseline[t])
            for t in range(len(C_E_ams_cf))
        ]

        "Closed-economy C with counterfactual A (nps)"
        C_lev_E_nps = []
        for i in range(len(cou.h_tot)):
            L_t = np.array(cou.h_tot)[i]
            A_agr_t = np.array(cou.A_agr)[i]
            A_man_t = np.array(cou.A_man)[i]
            A_trd_t = np.array(cou.A_trd)[i]
            A_bss_t = np.array(cou.A_bss)[i]
            A_fin_t = np.array(cou.A_fin)[i]
            A_nps_t = np.array(cou.A_nps)[i]

            def C_exp_nps(C):
                return L_t ** (1 - sigma) - (
                    cou.om_agr_nps
                    * (C**eps_agr)
                    * (A_agr_t ** (sigma - 1))
                    + cou.om_man_nps * C * (A_man_t ** (sigma - 1))
                    + cou.om_trd_nps
                    * (C**eps_trd)
                    * (A_trd_t ** (sigma - 1))
                    + cou.om_bss_nps
                    * (C**eps_bss)
                    * (A_bss_t ** (sigma - 1))
                    + cou.om_fin_nps
                    * (C**eps_fin)
                    * (A_fin_t ** (sigma - 1))
                    + (
                        1
                        - cou.om_agr_nps
                        - cou.om_man_nps
                        - cou.om_trd_nps
                        - cou.om_bss_nps
                        - cou.om_fin_nps
                    )
                    * (C**eps_nps)
                    * (A_nps_t ** (sigma - 1))
                )

            C_lev_E_nps.append(fsolve(C_exp_nps, L_t).item())
        C_level_E_nps = np.array(C_lev_E_nps)
        g_C_E_nps = C_level_E_nps[1:] / C_level_E_nps[:-1] - 1
        C_E_nps_cf = [float(np.array(cou.GDP_ph)[0] / np.array(GDP_ph_USA)[0])]
        for i in range(len(g_C_E_nps)):
            C_E_nps_cf.append(
                float((1 + g_C_E_nps[i]) * C_E_nps_cf[i])
            )

        "Apply ratio: C_cf = C_data * (C_closed_cf / C_closed_baseline)"
        self.C_cf_nps = [
            self.C_baseline[t]
            * (C_E_nps_cf[t] / self.C_E_nps_baseline[t])
            for t in range(len(C_E_nps_cf))
        ]

    def feed_US_productivity_growth(self, init_year, sec):
        "Baseline"
        self.baseline()

        "Feeding US productivity growth into sectors"
        if sec == "agr":
            self.cou.g_y_l_agr[init_year:] = USA.g_y_l_agr[init_year:]
        if sec == "man":
            self.cou.g_y_l_man[init_year:] = USA.g_y_l_man[init_year:]
        if sec == "trd":
            self.cou.g_y_l_trd[init_year:] = USA.g_y_l_trd[init_year:]
        if sec == "bss":
            self.cou.g_y_l_bss[init_year:] = USA.g_y_l_bss[init_year:]
        if sec == "fin":
            self.cou.g_y_l_fin[init_year:] = USA.g_y_l_fin[init_year:]
        if sec == "nps":
            self.cou.g_y_l_nps[init_year:] = USA.g_y_l_nps[init_year:]
        if sec == "ser":
            self.cou.g_y_l_ser[init_year:] = USA.g_y_l_ser[init_year:]
        if sec == "prs":
            self.cou.g_y_l_trd[init_year:] = USA.g_y_l_trd[init_year:]
            self.cou.g_y_l_bss[init_year:] = USA.g_y_l_bss[init_year:]
            self.cou.g_y_l_fin[init_year:] = USA.g_y_l_fin[init_year:]
            self.cou.g_y_l_ser[init_year:] = USA.g_y_l_ser[init_year:]
        if sec == "all":
            self.cou.g_y_l_agr[init_year:] = USA.g_y_l_agr[init_year:]
            self.cou.g_y_l_man[init_year:] = USA.g_y_l_man[init_year:]
            self.cou.g_y_l_trd[init_year:] = USA.g_y_l_trd[init_year:]
            self.cou.g_y_l_bss[init_year:] = USA.g_y_l_bss[init_year:]
            self.cou.g_y_l_fin[init_year:] = USA.g_y_l_fin[init_year:]
            self.cou.g_y_l_nps[init_year:] = USA.g_y_l_nps[init_year:]
            self.cou.g_y_l_ser[init_year:] = USA.g_y_l_ser[init_year:]

        "Generate counterfactual series"
        self.cou.productivity_series()

        "Compute counterfactual C using closed-economy ratio"
        self._compute_cf_C()

        "ams"
        self.cou.C = self.C_cf_ams
        self.cou.predictions_ams()

        "nps"
        self.cou.C = self.C_cf_nps
        self.C_nps = self.C_cf_nps
        self.cou.predictions_nps()

        self.A_agr = self.cou.A_agr
        self.A_man = self.cou.A_man
        self.A_trd = self.cou.A_trd
        self.A_bss = self.cou.A_bss
        self.A_fin = self.cou.A_fin
        self.A_nps = self.cou.A_nps

        self.share_agr_nps_m = self.cou.share_agr_nps_m
        self.share_man_nps_m = self.cou.share_man_nps_m
        self.share_trd_nps_m = self.cou.share_trd_nps_m
        self.share_bss_nps_m = self.cou.share_bss_nps_m
        self.share_fin_nps_m = self.cou.share_fin_nps_m
        self.share_nps_nps_m = self.cou.share_nps_nps_m

        self.A_tot_ams = self.cou.A_tot_ams
        self.A_tot_nps = self.cou.A_tot_nps

    def feed_catch_up_growth(self, init_year, sec):
        # CF2 "catch-up" counterfactual (endogenous-trade variant).
        # Algebra identical to counterfactuals.py::feed_catch_up_growth: pick the
        # end-of-sample A_i that, holding the other five sectors' A_j fixed, lifts
        # aggregate labor productivity to the observed US level E_USA[-1]. Then
        # back out the constant growth rate g that takes A_i[0] to A_i_catch over
        # ts_length years and overwrite the growth-rate path from init_year onward.
        #
        # Infeasibility: the algebra produces A_i_catch < 0 when sum_{j!=i} h_j*A_j
        # alone already exceeds E_USA[-1]. This happens for small sectors where
        # closing the gap would require negative productivity. We emit a
        # UserWarning tagged with country/sector and let the caller decide: the
        # subsequent g computation via (A/A_0)^(1/T) returns a complex number,
        # which propagates into the counterfactual productivity series. For
        # reporting (Table 6) only the numerically well-defined entries are used.
        "Baseline"
        self.baseline()

        if sec == "agr":
            A_agr_catch = (
                E_USA[-1] / np.array(self.cou.share_agr)[-1]
                - (
                    np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1]
                    + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]
                    + np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1]
                    + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1]
                    + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1]
                )
                / np.array(self.cou.h_agr)[-1]
            )
            if A_agr_catch < 0:
                import warnings
                warnings.warn(f"CF2: {self.country_code} agr catch-up infeasible (A_catch={A_agr_catch:.2f})")
            g_y_l_agr_catch = (A_agr_catch / np.array(self.cou.A_agr)[0]) ** (
                1 / self.cou.ts_length
            ) - 1
            g_catch = np.empty(int(self.cou.ts_length) + 1)
            g_catch.fill(g_y_l_agr_catch)
            self.cou.g_y_l_agr[init_year:] = g_catch[init_year:]
            self.A_agr_catch = A_agr_catch
            self.cou.A_agr = A_agr_catch
            self.g_y_l_agr_catch = g_y_l_agr_catch

        if sec == "man":
            A_man_catch = (
                E_USA[-1] / np.array(self.cou.share_man)[-1]
                - (
                    np.array(self.cou.h_agr)[-1] * np.array(self.cou.A_agr)[-1]
                    + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]
                    + np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1]
                    + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1]
                    + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1]
                )
                / np.array(self.cou.h_man)[-1]
            )
            if A_man_catch < 0:
                import warnings
                warnings.warn(f"CF2: {self.country_code} man catch-up infeasible (A_catch={A_man_catch:.2f})")
            g_y_l_man_catch = (A_man_catch / np.array(self.cou.A_man)[0]) ** (
                1 / self.cou.ts_length
            ) - 1
            g_catch = np.empty(int(self.cou.ts_length) + 1)
            g_catch.fill(g_y_l_man_catch)
            self.cou.g_y_l_man[init_year:] = g_catch[init_year:]
            self.A_man_catch = A_man_catch
            self.cou.A_man = A_man_catch
            self.g_y_l_man_catch = g_y_l_man_catch

        if sec == "trd":
            A_trd_catch = (
                E_USA[-1] / np.array(self.cou.share_trd)[-1]
                - (
                    np.array(self.cou.h_agr)[-1] * np.array(self.cou.A_agr)[-1]
                    + np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1]
                    + np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1]
                    + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1]
                    + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1]
                )
                / np.array(self.cou.h_trd)[-1]
            )
            if A_trd_catch < 0:
                import warnings
                warnings.warn(f"CF2: {self.country_code} trd catch-up infeasible (A_catch={A_trd_catch:.2f})")
            g_y_l_trd_catch = (A_trd_catch / np.array(self.cou.A_trd)[0]) ** (
                1 / self.cou.ts_length
            ) - 1
            g_catch = np.empty(int(self.cou.ts_length) + 1)
            g_catch.fill(g_y_l_trd_catch)
            self.cou.g_y_l_trd[init_year:] = g_catch[init_year:]
            self.A_trd_catch = A_trd_catch
            self.cou.A_trd = A_trd_catch
            self.g_y_l_trd_catch = g_y_l_trd_catch

        if sec == "bss":
            A_bss_catch = (
                E_USA[-1] / np.array(self.cou.share_bss)[-1]
                - (
                    np.array(self.cou.h_agr)[-1] * np.array(self.cou.A_agr)[-1]
                    + np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1]
                    + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]
                    + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1]
                    + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1]
                )
                / np.array(self.cou.h_bss)[-1]
            )
            if A_bss_catch < 0:
                import warnings
                warnings.warn(f"CF2: {self.country_code} bss catch-up infeasible (A_catch={A_bss_catch:.2f})")
            g_y_l_bss_catch = (A_bss_catch / np.array(self.cou.A_bss)[0]) ** (
                1 / self.cou.ts_length
            ) - 1
            g_catch = np.empty(int(self.cou.ts_length) + 1)
            g_catch.fill(g_y_l_bss_catch)
            self.cou.g_y_l_bss[init_year:] = g_catch[init_year:]
            self.A_bss_catch = A_bss_catch
            self.cou.A_bss = A_bss_catch
            self.g_y_l_bss_catch = g_y_l_bss_catch

        if sec == "fin":
            A_fin_catch = (
                E_USA[-1] / np.array(self.cou.share_fin)[-1]
                - (
                    np.array(self.cou.h_agr)[-1] * np.array(self.cou.A_agr)[-1]
                    + np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1]
                    + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]
                    + np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1]
                    + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1]
                )
                / np.array(self.cou.h_fin)[-1]
            )
            if A_fin_catch < 0:
                import warnings
                warnings.warn(f"CF2: {self.country_code} fin catch-up infeasible (A_catch={A_fin_catch:.2f})")
            g_y_l_fin_catch = (A_fin_catch / np.array(self.cou.A_fin)[0]) ** (
                1 / self.cou.ts_length
            ) - 1
            g_catch = np.empty(int(self.cou.ts_length) + 1)
            g_catch.fill(g_y_l_fin_catch)
            self.cou.g_y_l_fin[init_year:] = g_catch[init_year:]
            self.A_fin_catch = A_fin_catch
            self.cou.A_fin = A_fin_catch
            self.g_y_l_fin_catch = g_y_l_fin_catch

        if sec == "nps":
            A_nps_catch = (
                E_USA[-1] / np.array(self.cou.share_nps)[-1]
                - (
                    np.array(self.cou.h_agr)[-1] * np.array(self.cou.A_agr)[-1]
                    + np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1]
                    + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]
                    + np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1]
                    + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1]
                )
                / np.array(self.cou.h_nps)[-1]
            )
            if A_nps_catch < 0:
                import warnings
                warnings.warn(f"CF2: {self.country_code} nps catch-up infeasible (A_catch={A_nps_catch:.2f})")
            g_y_l_nps_catch = (A_nps_catch / np.array(self.cou.A_nps)[0]) ** (
                1 / self.cou.ts_length
            ) - 1
            g_catch = np.empty(int(self.cou.ts_length) + 1)
            g_catch.fill(g_y_l_nps_catch)
            self.cou.g_y_l_nps[init_year:] = g_catch[init_year:]
            self.A_nps_catch = A_nps_catch
            self.cou.A_nps = A_nps_catch
            self.g_y_l_nps_catch = g_y_l_nps_catch

        "Generate counterfactual series"
        self.cou.productivity_series()

        "Compute counterfactual C using closed-economy ratio"
        self._compute_cf_C()

        # 'ams'
        # self.cou.C = self.C_cf_ams
        # self.cou.predictions_ams()
        #
        # self.A_agr = self.cou.A_agr
        # self.A_man = self.cou.A_man
        # self.A_ser = self.cou.A_ser
        #
        # self.share_agr_ams_m = self.cou.share_agr_ams_m
        # self.share_man_ams_m = self.cou.share_man_ams_m
        # self.share_ser_ams_m = self.cou.share_ser_ams_m
        #
        # self.A_tot_ams = self.cou.A_tot_ams

        "nps"
        self.cou.C = self.C_cf_nps
        self.C_nps = self.C_cf_nps

        self.A_agr = self.cou.A_agr
        self.A_man = self.cou.A_man
        self.A_trd = self.cou.A_trd
        self.A_bss = self.cou.A_bss
        self.A_fin = self.cou.A_fin
        self.A_nps = self.cou.A_nps

        self.cou.predictions_nps()

        self.share_agr_nps_m = self.cou.share_agr_nps_m
        self.share_man_nps_m = self.cou.share_man_nps_m
        self.share_trd_nps_m = self.cou.share_trd_nps_m
        self.share_bss_nps_m = self.cou.share_bss_nps_m
        self.share_fin_nps_m = self.cou.share_fin_nps_m
        self.share_nps_nps_m = self.cou.share_nps_nps_m

        self.A_tot_nps = self.cou.A_tot_nps

    def shift_share(self, sec):
        "ams"
        self.ss_A_ams_init = (
            np.array(self.cou.share_agr)[0] * self.cou.A_agr[-1]
            + np.array(self.cou.share_man)[0] * self.cou.A_man[-1]
            + np.array(self.cou.share_ser)[0] * self.cou.A_ser[-1]
        )
        self.ss_A_ams = (
            self.cou.share_agr * self.cou.A_agr
            + self.cou.share_man * self.cou.A_man
            + self.cou.share_ser * self.cou.A_ser
        )

        "nps"
        self.ss_A_nps_init = (
            np.array(self.cou.share_agr)[0] * self.cou.A_agr[-1]
            + np.array(self.cou.share_man)[0] * self.cou.A_man[-1]
            + np.array(self.cou.share_trd)[0] * self.cou.A_trd[-1]
            + np.array(self.cou.share_bss)[0] * self.cou.A_bss[-1]
            + np.array(self.cou.share_fin)[0] * self.cou.A_fin[-1]
            + np.array(self.cou.share_nps)[0] * self.cou.A_nps[-1]
        )
        self.ss_A_nps = (
            self.cou.share_agr * self.cou.A_agr
            + self.cou.share_man * self.cou.A_man
            + self.cou.share_trd * self.cou.A_trd
            + self.cou.share_bss * self.cou.A_bss
            + self.cou.share_fin * self.cou.A_fin
            + self.cou.share_nps * self.cou.A_nps
        )


"""
----------------------------------------------------------------------------------------
	Counterfactual 1: Each sector keeping the pace with the US for the entire period
----------------------------------------------------------------------------------------
"""

"ams"
cf_1_ams_ss_init = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EU13",
    ]
]
cf_1_ams_ss = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EU13",
    ]
]
cf_1_ams = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EU13",
    ]
]

for sec in ["agr", "man", "ser"]:
    AUT_cf = counterfactual("AUT")
    BEL_cf = counterfactual("BEL")
    DEU_cf = counterfactual("DEU")
    DNK_cf = counterfactual("DNK")
    ESP_cf = counterfactual("ESP")
    FIN_cf = counterfactual("FIN")
    FRA_cf = counterfactual("FRA")
    GBR_cf = counterfactual("GBR")
    GRC_cf = counterfactual("GRC")
    ITA_cf = counterfactual("ITA")
    NLD_cf = counterfactual("NLD")
    PRT_cf = counterfactual("PRT")
    SWE_cf = counterfactual("SWE")

    AUT_cf.baseline()
    BEL_cf.baseline()
    DEU_cf.baseline()
    DNK_cf.baseline()
    ESP_cf.baseline()
    FIN_cf.baseline()
    FRA_cf.baseline()
    GBR_cf.baseline()
    GRC_cf.baseline()
    ITA_cf.baseline()
    NLD_cf.baseline()
    PRT_cf.baseline()
    SWE_cf.baseline()
    AUT_cf.baseline()

    AUT_cf.feed_US_productivity_growth(0, sec)
    BEL_cf.feed_US_productivity_growth(0, sec)
    DEU_cf.feed_US_productivity_growth(0, sec)
    DNK_cf.feed_US_productivity_growth(0, sec)
    ESP_cf.feed_US_productivity_growth(0, sec)
    FIN_cf.feed_US_productivity_growth(0, sec)
    FRA_cf.feed_US_productivity_growth(0, sec)
    GBR_cf.feed_US_productivity_growth(0, sec)
    GRC_cf.feed_US_productivity_growth(0, sec)
    ITA_cf.feed_US_productivity_growth(0, sec)
    NLD_cf.feed_US_productivity_growth(0, sec)
    PRT_cf.feed_US_productivity_growth(0, sec)
    SWE_cf.feed_US_productivity_growth(0, sec)

    AUT_cf.shift_share(sec)
    BEL_cf.shift_share(sec)
    DEU_cf.shift_share(sec)
    DNK_cf.shift_share(sec)
    ESP_cf.shift_share(sec)
    FIN_cf.shift_share(sec)
    FRA_cf.shift_share(sec)
    GBR_cf.shift_share(sec)
    GRC_cf.shift_share(sec)
    ITA_cf.shift_share(sec)
    NLD_cf.shift_share(sec)
    PRT_cf.shift_share(sec)
    SWE_cf.shift_share(sec)

    cf_1_sec_ams_init_ss = [
        sec,
        (AUT_cf.ss_A_ams_init / AUT_cf.ss_A_base_ams_init - 1) * 100,
        (BEL_cf.ss_A_ams_init / BEL_cf.ss_A_base_ams_init - 1) * 100,
        (DEU_cf.ss_A_ams_init / DEU_cf.ss_A_base_ams_init - 1) * 100,
        (DNK_cf.ss_A_ams_init / DNK_cf.ss_A_base_ams_init - 1) * 100,
        (ESP_cf.ss_A_ams_init / ESP_cf.ss_A_base_ams_init - 1) * 100,
        (FIN_cf.ss_A_ams_init / FIN_cf.ss_A_base_ams_init - 1) * 100,
        (FRA_cf.ss_A_ams_init / FRA_cf.ss_A_base_ams_init - 1) * 100,
        (GBR_cf.ss_A_ams_init / GBR_cf.ss_A_base_ams_init - 1) * 100,
        (GRC_cf.ss_A_ams_init / GRC_cf.ss_A_base_ams_init - 1) * 100,
        (ITA_cf.ss_A_ams_init / ITA_cf.ss_A_base_ams_init - 1) * 100,
        (NLD_cf.ss_A_ams_init / NLD_cf.ss_A_base_ams_init - 1) * 100,
        (PRT_cf.ss_A_ams_init / PRT_cf.ss_A_base_ams_init - 1) * 100,
        (SWE_cf.ss_A_ams_init / SWE_cf.ss_A_base_ams_init - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[0]
                    * (DEU_cf.ss_A_ams_init / DEU_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(FRA.h_tot)[0]
                    * (FRA_cf.ss_A_ams_init / FRA_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(ITA.h_tot)[0]
                    * (ITA_cf.ss_A_ams_init / ITA_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(GBR.h_tot)[0]
                    * (GBR_cf.ss_A_ams_init / GBR_cf.ss_A_base_ams_init - 1)
                    * 100
                )
                / EUR4_h_tot[0]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[0]
                    * (AUT_cf.ss_A_ams_init / AUT_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(BEL.h_tot)[0]
                    * (BEL_cf.ss_A_ams_init / BEL_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(DEU.h_tot)[0]
                    * (DEU_cf.ss_A_ams_init / DEU_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(DNK.h_tot)[0]
                    * (DNK_cf.ss_A_ams_init / DNK_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(ESP.h_tot)[0]
                    * (ESP_cf.ss_A_ams_init / ESP_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(FIN.h_tot)[0]
                    * (FIN_cf.ss_A_ams_init / FIN_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(FRA.h_tot)[0]
                    * (FRA_cf.ss_A_ams_init / FRA_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(GBR.h_tot)[0]
                    * (GBR_cf.ss_A_ams_init / GBR_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(GRC.h_tot)[0]
                    * (GRC_cf.ss_A_ams_init / GRC_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(ITA.h_tot)[0]
                    * (ITA_cf.ss_A_ams_init / ITA_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(NLD.h_tot)[0]
                    * (NLD_cf.ss_A_ams_init / NLD_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(PRT.h_tot)[0]
                    * (PRT_cf.ss_A_ams_init / PRT_cf.ss_A_base_ams_init - 1)
                    * 100
                    + np.array(SWE.h_tot)[0]
                    * (SWE_cf.ss_A_ams_init / SWE_cf.ss_A_base_ams_init - 1)
                    * 100
                )
                / EUR13_h_tot[0]
            ]
        ).item(),
    ]

    cf_1_sec_ams_init_ss[1:] = ["%.1f" % elem for elem in cf_1_sec_ams_init_ss[1:]]
    cf_1_ams_ss_init.append(cf_1_sec_ams_init_ss)

    cf_1_sec_ams_ss = [
        sec,
        (np.array(AUT_cf.ss_A_ams)[-1] / np.array(AUT_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(BEL_cf.ss_A_ams)[-1] / np.array(BEL_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(DEU_cf.ss_A_ams)[-1] / np.array(DEU_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(DNK_cf.ss_A_ams)[-1] / np.array(DNK_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(ESP_cf.ss_A_ams)[-1] / np.array(ESP_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(FIN_cf.ss_A_ams)[-1] / np.array(FIN_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(FRA_cf.ss_A_ams)[-1] / np.array(FRA_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(GBR_cf.ss_A_ams)[-1] / np.array(GBR_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(GRC_cf.ss_A_ams)[-1] / np.array(GRC_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(ITA_cf.ss_A_ams)[-1] / np.array(ITA_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(NLD_cf.ss_A_ams)[-1] / np.array(NLD_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(PRT_cf.ss_A_ams)[-1] / np.array(PRT_cf.ss_A_base_ams)[-1] - 1) * 100,
        (np.array(SWE_cf.ss_A_ams)[-1] / np.array(SWE_cf.ss_A_base_ams)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf.ss_A_ams)[-1]
                        / np.array(DEU_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf.ss_A_ams)[-1]
                        / np.array(FRA_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf.ss_A_ams)[-1]
                        / np.array(ITA_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf.ss_A_ams)[-1]
                        / np.array(GBR_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (
                        np.array(AUT_cf.ss_A_ams)[-1]
                        / np.array(AUT_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (
                        np.array(BEL_cf.ss_A_ams)[-1]
                        / np.array(BEL_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf.ss_A_ams)[-1]
                        / np.array(DEU_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf.ss_A_ams)[-1]
                        / np.array(DNK_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf.ss_A_ams)[-1]
                        / np.array(ESP_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (
                        np.array(FIN_cf.ss_A_ams)[-1]
                        / np.array(FIN_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf.ss_A_ams)[-1]
                        / np.array(FRA_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf.ss_A_ams)[-1]
                        / np.array(GBR_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf.ss_A_ams)[-1]
                        / np.array(GRC_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf.ss_A_ams)[-1]
                        / np.array(ITA_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf.ss_A_ams)[-1]
                        / np.array(NLD_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf.ss_A_ams)[-1]
                        / np.array(PRT_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (
                        np.array(SWE_cf.ss_A_ams)[-1]
                        / np.array(SWE_cf.ss_A_base_ams)[-1]
                        - 1
                    )
                    * 100
                )
                / EUR13_h_tot[-1]
            ]
        ).item(),
    ]

    cf_1_sec_ams_ss[1:] = ["%.1f" % elem for elem in cf_1_sec_ams_ss[1:]]
    cf_1_ams_ss.append(cf_1_sec_ams_ss)

    cf_1_sec_ams = [
        sec,
        (np.array(AUT_cf.A_tot_ams)[-1] / np.array(AUT.A_tot_ams)[-1] - 1) * 100,
        (np.array(BEL_cf.A_tot_ams)[-1] / np.array(BEL.A_tot_ams)[-1] - 1) * 100,
        (np.array(DEU_cf.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[-1] - 1) * 100,
        (np.array(DNK_cf.A_tot_ams)[-1] / np.array(DNK.A_tot_ams)[-1] - 1) * 100,
        (np.array(ESP_cf.A_tot_ams)[-1] / np.array(ESP.A_tot_ams)[-1] - 1) * 100,
        (np.array(FIN_cf.A_tot_ams)[-1] / np.array(FIN.A_tot_ams)[-1] - 1) * 100,
        (np.array(FRA_cf.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[-1] - 1) * 100,
        (np.array(GBR_cf.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[-1] - 1) * 100,
        (np.array(GRC_cf.A_tot_ams)[-1] / np.array(GRC.A_tot_ams)[-1] - 1) * 100,
        (np.array(ITA_cf.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[-1] - 1) * 100,
        (np.array(NLD_cf.A_tot_ams)[-1] / np.array(NLD.A_tot_ams)[-1] - 1) * 100,
        (np.array(PRT_cf.A_tot_ams)[-1] / np.array(PRT.A_tot_ams)[-1] - 1) * 100,
        (np.array(SWE_cf.A_tot_ams)[-1] / np.array(SWE.A_tot_ams)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (np.array(DEU_cf.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (np.array(FRA_cf.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (np.array(ITA_cf.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (np.array(GBR_cf.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[-1] - 1)
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (np.array(AUT_cf.A_tot_ams)[-1] / np.array(AUT.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (np.array(BEL_cf.A_tot_ams)[-1] / np.array(BEL.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (np.array(DEU_cf.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (np.array(DNK_cf.A_tot_ams)[-1] / np.array(DNK.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (np.array(ESP_cf.A_tot_ams)[-1] / np.array(ESP.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (np.array(FIN_cf.A_tot_ams)[-1] / np.array(FIN.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (np.array(FRA_cf.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (np.array(GBR_cf.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (np.array(GRC_cf.A_tot_ams)[-1] / np.array(GRC.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (np.array(ITA_cf.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (np.array(NLD_cf.A_tot_ams)[-1] / np.array(NLD.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (np.array(PRT_cf.A_tot_ams)[-1] / np.array(PRT.A_tot_ams)[-1] - 1)
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (np.array(SWE_cf.A_tot_ams)[-1] / np.array(SWE.A_tot_ams)[-1] - 1)
                    * 100
                )
                / EUR13_h_tot[-1]
            ]
        ).item(),
    ]
    cf_1_sec_ams[1:] = ["%.1f" % elem for elem in cf_1_sec_ams[1:]]
    cf_1_ams.append(cf_1_sec_ams)

pd.DataFrame(cf_1_ams_ss_init).to_excel(
    "../output/figures/Counterfactual_1_ams_ss_init_endo_trade.xlsx", index=False, header=False
)
pd.DataFrame(cf_1_ams_ss).to_excel(
    "../output/figures/Counterfactual_1_ams_ss_endo_trade.xlsx", index=False, header=False
)
pd.DataFrame(cf_1_ams).to_excel(
    "../output/figures/Counterfactual_1_ams_endo_trade.xlsx", index=False, header=False
)


"nps"
cf_1_nps_ss_init = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]
cf_1_nps_ss = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]
cf_1_nps = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]

for sec in ["agr", "man", "trd", "fin", "bss", "prs", "nps"]:
    AUT_cf = counterfactual("AUT")
    BEL_cf = counterfactual("BEL")
    DEU_cf = counterfactual("DEU")
    DNK_cf = counterfactual("DNK")
    ESP_cf = counterfactual("ESP")
    FIN_cf = counterfactual("FIN")
    FRA_cf = counterfactual("FRA")
    GBR_cf = counterfactual("GBR")
    GRC_cf = counterfactual("GRC")
    ITA_cf = counterfactual("ITA")
    NLD_cf = counterfactual("NLD")
    PRT_cf = counterfactual("PRT")
    SWE_cf = counterfactual("SWE")

    AUT_cf.baseline()
    BEL_cf.baseline()
    DEU_cf.baseline()
    DNK_cf.baseline()
    ESP_cf.baseline()
    FIN_cf.baseline()
    FRA_cf.baseline()
    GBR_cf.baseline()
    GRC_cf.baseline()
    ITA_cf.baseline()
    NLD_cf.baseline()
    PRT_cf.baseline()
    SWE_cf.baseline()
    AUT_cf.baseline()

    AUT_cf.feed_US_productivity_growth(0, sec)
    BEL_cf.feed_US_productivity_growth(0, sec)
    DEU_cf.feed_US_productivity_growth(0, sec)
    DNK_cf.feed_US_productivity_growth(0, sec)
    ESP_cf.feed_US_productivity_growth(0, sec)
    FIN_cf.feed_US_productivity_growth(0, sec)
    FRA_cf.feed_US_productivity_growth(0, sec)
    GBR_cf.feed_US_productivity_growth(0, sec)
    GRC_cf.feed_US_productivity_growth(0, sec)
    ITA_cf.feed_US_productivity_growth(0, sec)
    NLD_cf.feed_US_productivity_growth(0, sec)
    PRT_cf.feed_US_productivity_growth(0, sec)
    SWE_cf.feed_US_productivity_growth(0, sec)

    AUT_cf.shift_share(sec)
    BEL_cf.shift_share(sec)
    DEU_cf.shift_share(sec)
    DNK_cf.shift_share(sec)
    ESP_cf.shift_share(sec)
    FIN_cf.shift_share(sec)
    FRA_cf.shift_share(sec)
    GBR_cf.shift_share(sec)
    GRC_cf.shift_share(sec)
    ITA_cf.shift_share(sec)
    NLD_cf.shift_share(sec)
    PRT_cf.shift_share(sec)
    SWE_cf.shift_share(sec)

    cf_1_sec_nps_init_ss = [
        sec,
        (AUT_cf.ss_A_nps_init / AUT_cf.ss_A_base_nps_init - 1) * 100,
        (BEL_cf.ss_A_nps_init / BEL_cf.ss_A_base_nps_init - 1) * 100,
        (DEU_cf.ss_A_nps_init / DEU_cf.ss_A_base_nps_init - 1) * 100,
        (DNK_cf.ss_A_nps_init / DNK_cf.ss_A_base_nps_init - 1) * 100,
        (ESP_cf.ss_A_nps_init / ESP_cf.ss_A_base_nps_init - 1) * 100,
        (FIN_cf.ss_A_nps_init / FIN_cf.ss_A_base_nps_init - 1) * 100,
        (FRA_cf.ss_A_nps_init / FRA_cf.ss_A_base_nps_init - 1) * 100,
        (GBR_cf.ss_A_nps_init / GBR_cf.ss_A_base_nps_init - 1) * 100,
        (GRC_cf.ss_A_nps_init / GRC_cf.ss_A_base_nps_init - 1) * 100,
        (ITA_cf.ss_A_nps_init / ITA_cf.ss_A_base_nps_init - 1) * 100,
        (NLD_cf.ss_A_nps_init / NLD_cf.ss_A_base_nps_init - 1) * 100,
        (PRT_cf.ss_A_nps_init / PRT_cf.ss_A_base_nps_init - 1) * 100,
        (SWE_cf.ss_A_nps_init / SWE_cf.ss_A_base_nps_init - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[0]
                    * (DEU_cf.ss_A_nps_init / DEU_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(FRA.h_tot)[0]
                    * (FRA_cf.ss_A_nps_init / FRA_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(ITA.h_tot)[0]
                    * (ITA_cf.ss_A_nps_init / ITA_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(GBR.h_tot)[0]
                    * (GBR_cf.ss_A_nps_init / GBR_cf.ss_A_base_nps_init - 1)
                    * 100
                )
                / EUR4_h_tot[0]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[0]
                    * (DEU_cf.ss_A_nps_init / DEU_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(FRA.h_tot)[0]
                    * (FRA_cf.ss_A_nps_init / FRA_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(BEL.h_tot)[0]
                    * (BEL_cf.ss_A_nps_init / BEL_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(NLD.h_tot)[0]
                    * (NLD_cf.ss_A_nps_init / NLD_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(DNK.h_tot)[0]
                    * (DNK_cf.ss_A_nps_init / DNK_cf.ss_A_base_nps_init - 1)
                    * 100
                )
                / EURCORE_h_tot[0]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[0]
                    * (GRC_cf.ss_A_nps_init / GRC_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(PRT.h_tot)[0]
                    * (PRT_cf.ss_A_nps_init / PRT_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(ESP.h_tot)[0]
                    * (ESP_cf.ss_A_nps_init / ESP_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(ITA.h_tot)[0]
                    * (ITA_cf.ss_A_nps_init / ITA_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(GBR.h_tot)[0]
                    * (GBR_cf.ss_A_nps_init / GBR_cf.ss_A_base_nps_init - 1)
                    * 100
                )
                / EURPERI_h_tot[0]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[0]
                    * (AUT_cf.ss_A_nps_init / AUT_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(BEL.h_tot)[0]
                    * (BEL_cf.ss_A_nps_init / BEL_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(DEU.h_tot)[0]
                    * (DEU_cf.ss_A_nps_init / DEU_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(DNK.h_tot)[0]
                    * (DNK_cf.ss_A_nps_init / DNK_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(ESP.h_tot)[0]
                    * (ESP_cf.ss_A_nps_init / ESP_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(FIN.h_tot)[0]
                    * (FIN_cf.ss_A_nps_init / FIN_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(FRA.h_tot)[0]
                    * (FRA_cf.ss_A_nps_init / FRA_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(GBR.h_tot)[0]
                    * (GBR_cf.ss_A_nps_init / GBR_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(GRC.h_tot)[0]
                    * (GRC_cf.ss_A_nps_init / GRC_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(ITA.h_tot)[0]
                    * (ITA_cf.ss_A_nps_init / ITA_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(NLD.h_tot)[0]
                    * (NLD_cf.ss_A_nps_init / NLD_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(PRT.h_tot)[0]
                    * (PRT_cf.ss_A_nps_init / PRT_cf.ss_A_base_nps_init - 1)
                    * 100
                    + np.array(SWE.h_tot)[0]
                    * (SWE_cf.ss_A_nps_init / SWE_cf.ss_A_base_nps_init - 1)
                    * 100
                )
                / EUR13_h_tot[0]
            ]
        ).item(),
    ]

    cf_1_sec_nps_init_ss[1:] = ["%.1f" % elem for elem in cf_1_sec_nps_init_ss[1:]]
    cf_1_nps_ss_init.append(cf_1_sec_nps_init_ss)

    cf_1_sec_nps_ss = [
        sec,
        (np.array(AUT_cf.ss_A_nps)[-1] / np.array(AUT_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(BEL_cf.ss_A_nps)[-1] / np.array(BEL_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(DEU_cf.ss_A_nps)[-1] / np.array(DEU_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(DNK_cf.ss_A_nps)[-1] / np.array(DNK_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(ESP_cf.ss_A_nps)[-1] / np.array(ESP_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(FIN_cf.ss_A_nps)[-1] / np.array(FIN_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(FRA_cf.ss_A_nps)[-1] / np.array(FRA_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(GBR_cf.ss_A_nps)[-1] / np.array(GBR_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(GRC_cf.ss_A_nps)[-1] / np.array(GRC_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(ITA_cf.ss_A_nps)[-1] / np.array(ITA_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(NLD_cf.ss_A_nps)[-1] / np.array(NLD_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(PRT_cf.ss_A_nps)[-1] / np.array(PRT_cf.ss_A_base_nps)[-1] - 1) * 100,
        (np.array(SWE_cf.ss_A_nps)[-1] / np.array(SWE_cf.ss_A_base_nps)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf.ss_A_nps)[-1]
                        / np.array(DEU_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf.ss_A_nps)[-1]
                        / np.array(FRA_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf.ss_A_nps)[-1]
                        / np.array(ITA_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf.ss_A_nps)[-1]
                        / np.array(GBR_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf.ss_A_nps)[-1]
                        / np.array(DEU_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf.ss_A_nps)[-1]
                        / np.array(FRA_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (
                        np.array(BEL_cf.ss_A_nps)[-1]
                        / np.array(BEL_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf.ss_A_nps)[-1]
                        / np.array(NLD_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf.ss_A_nps)[-1]
                        / np.array(DNK_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EURCORE_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf.ss_A_nps)[-1]
                        / np.array(GRC_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf.ss_A_nps)[-1]
                        / np.array(PRT_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf.ss_A_nps)[-1]
                        / np.array(ESP_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf.ss_A_nps)[-1]
                        / np.array(ITA_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf.ss_A_nps)[-1]
                        / np.array(GBR_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EURPERI_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (
                        np.array(AUT_cf.ss_A_nps)[-1]
                        / np.array(AUT_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (
                        np.array(BEL_cf.ss_A_nps)[-1]
                        / np.array(BEL_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf.ss_A_nps)[-1]
                        / np.array(DEU_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf.ss_A_nps)[-1]
                        / np.array(DNK_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf.ss_A_nps)[-1]
                        / np.array(ESP_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (
                        np.array(FIN_cf.ss_A_nps)[-1]
                        / np.array(FIN_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf.ss_A_nps)[-1]
                        / np.array(FRA_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf.ss_A_nps)[-1]
                        / np.array(GBR_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf.ss_A_nps)[-1]
                        / np.array(GRC_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf.ss_A_nps)[-1]
                        / np.array(ITA_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf.ss_A_nps)[-1]
                        / np.array(NLD_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf.ss_A_nps)[-1]
                        / np.array(PRT_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (
                        np.array(SWE_cf.ss_A_nps)[-1]
                        / np.array(SWE_cf.ss_A_base_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EUR13_h_tot[-1]
            ]
        ).item(),
    ]

    cf_1_sec_nps_ss[1:] = ["%.1f" % elem for elem in cf_1_sec_nps_ss[1:]]
    cf_1_nps_ss.append(cf_1_sec_nps_ss)

    cf_1_sec_nps = [
        sec,
        (np.array(AUT_cf.A_tot_nps)[-1] / np.array(AUT.A_tot_nps)[-1] - 1) * 100,
        (np.array(BEL_cf.A_tot_nps)[-1] / np.array(BEL.A_tot_nps)[-1] - 1) * 100,
        (np.array(DEU_cf.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1) * 100,
        (np.array(DNK_cf.A_tot_nps)[-1] / np.array(DNK.A_tot_nps)[-1] - 1) * 100,
        (np.array(ESP_cf.A_tot_nps)[-1] / np.array(ESP.A_tot_nps)[-1] - 1) * 100,
        (np.array(FIN_cf.A_tot_nps)[-1] / np.array(FIN.A_tot_nps)[-1] - 1) * 100,
        (np.array(FRA_cf.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1) * 100,
        (np.array(GBR_cf.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1) * 100,
        (np.array(GRC_cf.A_tot_nps)[-1] / np.array(GRC.A_tot_nps)[-1] - 1) * 100,
        (np.array(ITA_cf.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1) * 100,
        (np.array(NLD_cf.A_tot_nps)[-1] / np.array(NLD.A_tot_nps)[-1] - 1) * 100,
        (np.array(PRT_cf.A_tot_nps)[-1] / np.array(PRT.A_tot_nps)[-1] - 1) * 100,
        (np.array(SWE_cf.A_tot_nps)[-1] / np.array(SWE.A_tot_nps)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (np.array(DEU_cf.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (np.array(FRA_cf.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (np.array(ITA_cf.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (np.array(GBR_cf.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1)
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (np.array(DEU_cf.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (np.array(FRA_cf.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (np.array(BEL_cf.A_tot_nps)[-1] / np.array(BEL.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (np.array(NLD_cf.A_tot_nps)[-1] / np.array(NLD.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (np.array(DNK_cf.A_tot_nps)[-1] / np.array(DNK.A_tot_nps)[-1] - 1)
                    * 100
                )
                / EURCORE_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[-1]
                    * (np.array(GRC_cf.A_tot_nps)[-1] / np.array(GRC.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (np.array(PRT_cf.A_tot_nps)[-1] / np.array(PRT.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (np.array(ESP_cf.A_tot_nps)[-1] / np.array(ESP.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (np.array(ITA_cf.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (np.array(GBR_cf.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1)
                    * 100
                )
                / EURPERI_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (np.array(AUT_cf.A_tot_nps)[-1] / np.array(AUT.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (np.array(BEL_cf.A_tot_nps)[-1] / np.array(BEL.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (np.array(DEU_cf.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (np.array(DNK_cf.A_tot_nps)[-1] / np.array(DNK.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (np.array(ESP_cf.A_tot_nps)[-1] / np.array(ESP.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (np.array(FIN_cf.A_tot_nps)[-1] / np.array(FIN.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (np.array(FRA_cf.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (np.array(GBR_cf.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (np.array(GRC_cf.A_tot_nps)[-1] / np.array(GRC.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (np.array(ITA_cf.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (np.array(NLD_cf.A_tot_nps)[-1] / np.array(NLD.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (np.array(PRT_cf.A_tot_nps)[-1] / np.array(PRT.A_tot_nps)[-1] - 1)
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (np.array(SWE_cf.A_tot_nps)[-1] / np.array(SWE.A_tot_nps)[-1] - 1)
                    * 100
                )
                / EUR13_h_tot[-1]
            ]
        ).item(),
    ]
    cf_1_sec_nps[1:] = ["%.1f" % elem for elem in cf_1_sec_nps[1:]]
    cf_1_nps.append(cf_1_sec_nps)

pd.DataFrame(cf_1_nps_ss_init).to_excel(
    "../output/figures/Counterfactual_1_nps_ss_init_endo_trade.xlsx", index=False, header=False
)
pd.DataFrame(cf_1_nps_ss).to_excel(
    "../output/figures/Counterfactual_1_nps_ss_endo_trade.xlsx", index=False, header=False
)
pd.DataFrame(cf_1_nps).to_excel(
    "../output/figures/Counterfactual_1_nps_endo_trade.xlsx", index=False, header=False
)


"""
------------------------------------------------------------
	Counterfactual 2: Catch up Productivity with the US
------------------------------------------------------------
"""

"ams"
cf_2_catch_ams_ss = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]
cf_2_catch_ams = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]
for sec in ["agr", "man", "ser"]:
    AUT_cf2_catch = counterfactual("AUT")
    BEL_cf2_catch = counterfactual("BEL")
    DEU_cf2_catch = counterfactual("DEU")
    DNK_cf2_catch = counterfactual("DNK")
    ESP_cf2_catch = counterfactual("ESP")
    FIN_cf2_catch = counterfactual("FIN")
    FRA_cf2_catch = counterfactual("FRA")
    GBR_cf2_catch = counterfactual("GBR")
    GRC_cf2_catch = counterfactual("GRC")
    ITA_cf2_catch = counterfactual("ITA")
    NLD_cf2_catch = counterfactual("NLD")
    PRT_cf2_catch = counterfactual("PRT")
    SWE_cf2_catch = counterfactual("SWE")

    AUT_cf2_catch.baseline()
    BEL_cf2_catch.baseline()
    DEU_cf2_catch.baseline()
    DNK_cf2_catch.baseline()
    ESP_cf2_catch.baseline()
    FIN_cf2_catch.baseline()
    FRA_cf2_catch.baseline()
    GBR_cf2_catch.baseline()
    GRC_cf2_catch.baseline()
    ITA_cf2_catch.baseline()
    NLD_cf2_catch.baseline()
    PRT_cf2_catch.baseline()
    SWE_cf2_catch.baseline()
    AUT_cf2_catch.baseline()

    AUT_cf2_catch.feed_catch_up_growth(0, sec)
    BEL_cf2_catch.feed_catch_up_growth(0, sec)
    DEU_cf2_catch.feed_catch_up_growth(0, sec)
    DNK_cf2_catch.feed_catch_up_growth(0, sec)
    ESP_cf2_catch.feed_catch_up_growth(0, sec)
    FIN_cf2_catch.feed_catch_up_growth(0, sec)
    FRA_cf2_catch.feed_catch_up_growth(0, sec)
    GBR_cf2_catch.feed_catch_up_growth(0, sec)
    GRC_cf2_catch.feed_catch_up_growth(0, sec)
    ITA_cf2_catch.feed_catch_up_growth(0, sec)
    NLD_cf2_catch.feed_catch_up_growth(0, sec)
    PRT_cf2_catch.feed_catch_up_growth(0, sec)
    SWE_cf2_catch.feed_catch_up_growth(0, sec)

    cf_2_catch_sec_ams_ss = [
        sec,
        (E_USA[-1] / np.array(AUT_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(BEL_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(DNK_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(ESP_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(FIN_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(GRC_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(NLD_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(PRT_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(SWE_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (E_USA[-1] / np.array(NLD_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (E_USA[-1] / np.array(DNK_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / (EURCORE_h_tot[-1] - np.array(BEL.h_tot)[-1])
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[-1]
                    * (E_USA[-1] / np.array(GRC_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (E_USA[-1] / np.array(PRT_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (E_USA[-1] / np.array(ESP_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / EURPERI_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (E_USA[-1] / np.array(AUT_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (E_USA[-1] / np.array(BEL_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (E_USA[-1] / np.array(DNK_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (E_USA[-1] / np.array(ESP_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (E_USA[-1] / np.array(FIN_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (E_USA[-1] / np.array(GRC_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (E_USA[-1] / np.array(NLD_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (E_USA[-1] / np.array(PRT_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (E_USA[-1] / np.array(SWE_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / EUR13_h_tot[-1]
            ]
        ).item(),
    ]

    cf_2_catch_sec_ams_ss[1:] = ["%.1f" % elem if not np.isnan(float(elem)) else "---" for elem in cf_2_catch_sec_ams_ss[1:]]
    cf_2_catch_ams_ss.append(cf_2_catch_sec_ams_ss)

    cf_2_catch_sec_ams = [
        sec,
        (np.array(AUT_cf2_catch.A_tot_nps)[-1] / np.array(AUT.A_tot_nps)[-1] - 1) * 100,
        (np.array(BEL_cf2_catch.A_tot_nps)[-1] / np.array(BEL.A_tot_nps)[-1] - 1) * 100,
        (np.array(DEU_cf2_catch.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1) * 100,
        (np.array(DNK_cf2_catch.A_tot_nps)[-1] / np.array(DNK.A_tot_nps)[-1] - 1) * 100,
        (np.array(ESP_cf2_catch.A_tot_nps)[-1] / np.array(ESP.A_tot_nps)[-1] - 1) * 100,
        (np.array(FIN_cf2_catch.A_tot_nps)[-1] / np.array(FIN.A_tot_nps)[-1] - 1) * 100,
        (np.array(FRA_cf2_catch.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1) * 100,
        (np.array(GBR_cf2_catch.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1) * 100,
        (np.array(GRC_cf2_catch.A_tot_nps)[-1] / np.array(GRC.A_tot_nps)[-1] - 1) * 100,
        (np.array(ITA_cf2_catch.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1) * 100,
        (np.array(NLD_cf2_catch.A_tot_nps)[-1] / np.array(NLD.A_tot_nps)[-1] - 1) * 100,
        (np.array(PRT_cf2_catch.A_tot_nps)[-1] / np.array(PRT.A_tot_nps)[-1] - 1) * 100,
        (np.array(SWE_cf2_catch.A_tot_nps)[-1] / np.array(SWE.A_tot_nps)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf2_catch.A_tot_nps)[-1]
                        / np.array(DEU.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf2_catch.A_tot_nps)[-1]
                        / np.array(FRA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf2_catch.A_tot_nps)[-1]
                        / np.array(GBR.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf2_catch.A_tot_nps)[-1]
                        / np.array(ITA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf2_catch.A_tot_nps)[-1]
                        / np.array(DEU.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf2_catch.A_tot_nps)[-1]
                        / np.array(FRA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf2_catch.A_tot_nps)[-1]
                        / np.array(NLD.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf2_catch.A_tot_nps)[-1]
                        / np.array(DNK.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / (EURCORE_h_tot[-1] - np.array(BEL.h_tot)[-1])
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf2_catch.A_tot_nps)[-1]
                        / np.array(GRC.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf2_catch.A_tot_nps)[-1]
                        / np.array(PRT.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf2_catch.A_tot_nps)[-1]
                        / np.array(ESP.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf2_catch.A_tot_nps)[-1]
                        / np.array(ITA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf2_catch.A_tot_nps)[-1]
                        / np.array(GBR.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EURPERI_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (
                        np.array(AUT_cf2_catch.A_tot_nps)[-1]
                        / np.array(AUT.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf2_catch.A_tot_nps)[-1]
                        / np.array(DEU.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf2_catch.A_tot_nps)[-1]
                        / np.array(DNK.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf2_catch.A_tot_nps)[-1]
                        / np.array(ESP.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (
                        np.array(FIN_cf2_catch.A_tot_nps)[-1]
                        / np.array(FIN.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf2_catch.A_tot_nps)[-1]
                        / np.array(FRA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf2_catch.A_tot_nps)[-1]
                        / np.array(GBR.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf2_catch.A_tot_nps)[-1]
                        / np.array(GRC.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf2_catch.A_tot_nps)[-1]
                        / np.array(ITA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf2_catch.A_tot_nps)[-1]
                        / np.array(NLD.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf2_catch.A_tot_nps)[-1]
                        / np.array(PRT.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (
                        np.array(SWE_cf2_catch.A_tot_nps)[-1]
                        / np.array(SWE.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / (EUR13_h_tot[-1] - np.array(BEL.h_tot)[-1])
            ]
        ).item(),
    ]

    cf_2_catch_sec_ams[1:] = ["%.1f" % elem if not np.isnan(float(elem)) else "---" for elem in cf_2_catch_sec_ams[1:]]
    cf_2_catch_ams.append(cf_2_catch_sec_ams)

pd.DataFrame(cf_2_catch_ams_ss).to_excel(
    "../output/figures/Counterfactual_2_catch_ams_ss_endo_trade.xlsx", index=False, header=False
)
pd.DataFrame(cf_2_catch_ams).to_excel(
    "../output/figures/Counterfactual_2_catch_ams_endo_trade.xlsx", index=False, header=False
)


"nps"
cf_2_catch_nps_ss = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]
cf_2_catch_nps = [
    [
        "",
        "AUT",
        "BEL",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "ITA",
        "NLD",
        "PRT",
        "SWE",
        "EU4",
        "EUCORE",
        "EUPERI",
        "EU13",
    ]
]
for sec in ["agr", "man", "trd", "fin", "bss", "nps"]:
    AUT_cf2_catch = counterfactual("AUT")
    BEL_cf2_catch = counterfactual("BEL")
    DEU_cf2_catch = counterfactual("DEU")
    DNK_cf2_catch = counterfactual("DNK")
    ESP_cf2_catch = counterfactual("ESP")
    FIN_cf2_catch = counterfactual("FIN")
    FRA_cf2_catch = counterfactual("FRA")
    GBR_cf2_catch = counterfactual("GBR")
    GRC_cf2_catch = counterfactual("GRC")
    ITA_cf2_catch = counterfactual("ITA")
    NLD_cf2_catch = counterfactual("NLD")
    PRT_cf2_catch = counterfactual("PRT")
    SWE_cf2_catch = counterfactual("SWE")

    AUT_cf2_catch.baseline()
    BEL_cf2_catch.baseline()
    DEU_cf2_catch.baseline()
    DNK_cf2_catch.baseline()
    ESP_cf2_catch.baseline()
    FIN_cf2_catch.baseline()
    FRA_cf2_catch.baseline()
    GBR_cf2_catch.baseline()
    GRC_cf2_catch.baseline()
    ITA_cf2_catch.baseline()
    NLD_cf2_catch.baseline()
    PRT_cf2_catch.baseline()
    SWE_cf2_catch.baseline()
    AUT_cf2_catch.baseline()

    AUT_cf2_catch.feed_catch_up_growth(0, sec)
    BEL_cf2_catch.feed_catch_up_growth(0, sec)
    DEU_cf2_catch.feed_catch_up_growth(0, sec)
    DNK_cf2_catch.feed_catch_up_growth(0, sec)
    ESP_cf2_catch.feed_catch_up_growth(0, sec)
    FIN_cf2_catch.feed_catch_up_growth(0, sec)
    FRA_cf2_catch.feed_catch_up_growth(0, sec)
    GBR_cf2_catch.feed_catch_up_growth(0, sec)
    GRC_cf2_catch.feed_catch_up_growth(0, sec)
    ITA_cf2_catch.feed_catch_up_growth(0, sec)
    NLD_cf2_catch.feed_catch_up_growth(0, sec)
    PRT_cf2_catch.feed_catch_up_growth(0, sec)
    SWE_cf2_catch.feed_catch_up_growth(0, sec)

    cf_2_catch_sec_nps_ss = [
        sec,
        (E_USA[-1] / np.array(AUT_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(BEL_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(DNK_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(ESP_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(FIN_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(GRC_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(NLD_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(PRT_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        (E_USA[-1] / np.array(SWE_cf2_catch.ss_A_base_nps)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (E_USA[-1] / np.array(NLD_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (E_USA[-1] / np.array(DNK_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / (EURCORE_h_tot[-1] - np.array(BEL.h_tot)[-1])
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[-1]
                    * (E_USA[-1] / np.array(GRC_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (E_USA[-1] / np.array(PRT_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (E_USA[-1] / np.array(ESP_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / EURPERI_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (E_USA[-1] / np.array(AUT_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(BEL.h_tot)[-1]
                    * (E_USA[-1] / np.array(BEL_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (E_USA[-1] / np.array(DNK_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (E_USA[-1] / np.array(ESP_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (E_USA[-1] / np.array(FIN_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (E_USA[-1] / np.array(GRC_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (E_USA[-1] / np.array(NLD_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (E_USA[-1] / np.array(PRT_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (E_USA[-1] / np.array(SWE_cf2_catch.ss_A_base_nps)[-1] - 1)
                    * 100
                )
                / EUR13_h_tot[-1]
            ]
        ).item(),
    ]

    cf_2_catch_sec_nps_ss[1:] = ["%.1f" % elem if not np.isnan(float(elem)) else "---" for elem in cf_2_catch_sec_nps_ss[1:]]
    cf_2_catch_nps_ss.append(cf_2_catch_sec_nps_ss)

    cf_2_catch_sec_nps = [
        sec,
        (np.array(AUT_cf2_catch.A_tot_nps)[-1] / np.array(AUT.A_tot_nps)[-1] - 1) * 100,
        (np.array(BEL_cf2_catch.A_tot_nps)[-1] / np.array(BEL.A_tot_nps)[-1] - 1) * 100,
        (np.array(DEU_cf2_catch.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1) * 100,
        (np.array(DNK_cf2_catch.A_tot_nps)[-1] / np.array(DNK.A_tot_nps)[-1] - 1) * 100,
        (np.array(ESP_cf2_catch.A_tot_nps)[-1] / np.array(ESP.A_tot_nps)[-1] - 1) * 100,
        (np.array(FIN_cf2_catch.A_tot_nps)[-1] / np.array(FIN.A_tot_nps)[-1] - 1) * 100,
        (np.array(FRA_cf2_catch.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1) * 100,
        (np.array(GBR_cf2_catch.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1) * 100,
        (np.array(GRC_cf2_catch.A_tot_nps)[-1] / np.array(GRC.A_tot_nps)[-1] - 1) * 100,
        (np.array(ITA_cf2_catch.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1) * 100,
        (np.array(NLD_cf2_catch.A_tot_nps)[-1] / np.array(NLD.A_tot_nps)[-1] - 1) * 100,
        (np.array(PRT_cf2_catch.A_tot_nps)[-1] / np.array(PRT.A_tot_nps)[-1] - 1) * 100,
        (np.array(SWE_cf2_catch.A_tot_nps)[-1] / np.array(SWE.A_tot_nps)[-1] - 1) * 100,
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf2_catch.A_tot_nps)[-1]
                        / np.array(DEU.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf2_catch.A_tot_nps)[-1]
                        / np.array(FRA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf2_catch.A_tot_nps)[-1]
                        / np.array(GBR.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf2_catch.A_tot_nps)[-1]
                        / np.array(ITA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EUR4_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf2_catch.A_tot_nps)[-1]
                        / np.array(DEU.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf2_catch.A_tot_nps)[-1]
                        / np.array(FRA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf2_catch.A_tot_nps)[-1]
                        / np.array(NLD.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf2_catch.A_tot_nps)[-1]
                        / np.array(DNK.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / (EURCORE_h_tot[-1] - np.array(BEL.h_tot)[-1])
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf2_catch.A_tot_nps)[-1]
                        / np.array(GRC.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf2_catch.A_tot_nps)[-1]
                        / np.array(PRT.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf2_catch.A_tot_nps)[-1]
                        / np.array(ESP.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf2_catch.A_tot_nps)[-1]
                        / np.array(ITA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf2_catch.A_tot_nps)[-1]
                        / np.array(GBR.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / EURPERI_h_tot[-1]
            ]
        ).item(),
        np.array(
            [
                (
                    np.array(AUT.h_tot)[-1]
                    * (
                        np.array(AUT_cf2_catch.A_tot_nps)[-1]
                        / np.array(AUT.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DEU.h_tot)[-1]
                    * (
                        np.array(DEU_cf2_catch.A_tot_nps)[-1]
                        / np.array(DEU.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(DNK.h_tot)[-1]
                    * (
                        np.array(DNK_cf2_catch.A_tot_nps)[-1]
                        / np.array(DNK.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ESP.h_tot)[-1]
                    * (
                        np.array(ESP_cf2_catch.A_tot_nps)[-1]
                        / np.array(ESP.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FIN.h_tot)[-1]
                    * (
                        np.array(FIN_cf2_catch.A_tot_nps)[-1]
                        / np.array(FIN.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(FRA.h_tot)[-1]
                    * (
                        np.array(FRA_cf2_catch.A_tot_nps)[-1]
                        / np.array(FRA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GBR.h_tot)[-1]
                    * (
                        np.array(GBR_cf2_catch.A_tot_nps)[-1]
                        / np.array(GBR.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(GRC.h_tot)[-1]
                    * (
                        np.array(GRC_cf2_catch.A_tot_nps)[-1]
                        / np.array(GRC.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(ITA.h_tot)[-1]
                    * (
                        np.array(ITA_cf2_catch.A_tot_nps)[-1]
                        / np.array(ITA.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(NLD.h_tot)[-1]
                    * (
                        np.array(NLD_cf2_catch.A_tot_nps)[-1]
                        / np.array(NLD.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(PRT.h_tot)[-1]
                    * (
                        np.array(PRT_cf2_catch.A_tot_nps)[-1]
                        / np.array(PRT.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                    + np.array(SWE.h_tot)[-1]
                    * (
                        np.array(SWE_cf2_catch.A_tot_nps)[-1]
                        / np.array(SWE.A_tot_nps)[-1]
                        - 1
                    )
                    * 100
                )
                / (EUR13_h_tot[-1] - np.array(BEL.h_tot)[-1])
            ]
        ).item(),
    ]

    cf_2_catch_sec_nps[1:] = ["%.1f" % elem if not np.isnan(float(elem)) else "---" for elem in cf_2_catch_sec_nps[1:]]
    cf_2_catch_nps.append(cf_2_catch_sec_nps)


pd.DataFrame(cf_2_catch_nps_ss).to_excel(
    "../output/figures/Counterfactual_2_catch_nps_ss_endo_trade.xlsx", index=False, header=False
)
pd.DataFrame(cf_2_catch_nps).to_excel(
    "../output/figures/Counterfactual_2_catch_nps_endo_trade.xlsx", index=False, header=False
)


"Germany"
# Endogenous trade constants K_i are calibrated so that sectoral exports in the
# base year (t=0) match the data: x_{i,0} = K_i * A_{i,0}^(xi_i - 1) / E_0 = M_{i,0} * E_0
# implies K_i = E_0 * M_{i,0} / A_{i,0}^(xi_i - 1). This is the "x_0,i" anchoring
# discussed in the paper's Section 6.2 so that initial-period sectoral exports
# match observed values. xi_i are the trade-elasticity parameters calibrated in
# model_calibration_USA_endogenous_open.py (paper's EU4 xi values: agr=0.74,
# man=0.91, trd=1.26, bss=2.59, fin=2.01, nps=9.28).
#
# The "trade cure" question: given CF catch-up productivity A_i, what
# counterfactual net-export share NX_CF in sector `sec` would push aggregate
# labor productivity to the observed US level? Unlike trade_counterfactuals.py
# (exogenous trade, where NX in non-cure sectors are observed data), here the
# non-cure nx_j_E are themselves endogenous responses to the CF A: they are
# computed from K_j, xi_j and CF A_j via nx_j = K_j*A_j^(xi_j-1)/E - M_j[-1].
# This is what drives the sign reversals reported in Table 6 Panel B (e.g., the
# nps overshoot from +11.54 to -5.85).
K_agr_DEU = DEU.E[0] * DEU.M_agr_E[0] / (DEU.A_agr[0] ** (DEU.xi_agr - 1))
K_man_DEU = DEU.E[0] * DEU.M_man_E[0] / (DEU.A_man[0] ** (DEU.xi_man - 1))
K_trd_DEU = DEU.E[0] * DEU.M_trd_E[0] / (DEU.A_trd[0] ** (DEU.xi_trd - 1))
K_bss_DEU = DEU.E[0] * DEU.M_bss_E[0] / (DEU.A_bss[0] ** (DEU.xi_bss - 1))
K_fin_DEU = DEU.E[0] * DEU.M_fin_E[0] / (DEU.A_fin[0] ** (DEU.xi_fin - 1))
K_nps_DEU = DEU.E[0] * DEU.M_nps_E[0] / (DEU.A_nps[0] ** (DEU.xi_nps - 1))

for sec in ["agr", "man", "trd", "fin", "bss", "nps"]:
    DEU_cf2_catch_trade = counterfactual("DEU")
    DEU_cf2_catch_trade.baseline()
    DEU_cf2_catch_trade.feed_catch_up_growth(0, sec)

    A_agr = np.array(DEU_cf2_catch_trade.A_agr)[-1]
    A_man = np.array(DEU_cf2_catch_trade.A_man)[-1]
    A_trd = np.array(DEU_cf2_catch_trade.A_trd)[-1]
    A_fin = np.array(DEU_cf2_catch_trade.A_fin)[-1]
    A_bss = np.array(DEU_cf2_catch_trade.A_bss)[-1]
    A_nps = np.array(DEU_cf2_catch_trade.A_nps)[-1]

    C = DEU_cf2_catch_trade.C_nps[-1]

    weight_agr = DEU.om_agr_nps * (A_agr ** (sigma - 1)) * (C**eps_agr)
    weight_man = DEU.om_man_nps * (A_man ** (sigma - 1)) * C
    weight_trd = DEU.om_trd_nps * (A_trd ** (sigma - 1)) * (C**eps_trd)
    weight_bss = DEU.om_bss_nps * (A_bss ** (sigma - 1)) * (C**eps_bss)
    weight_fin = DEU.om_fin_nps * (A_fin ** (sigma - 1)) * (C**eps_fin)
    weight_nps = (
        (
            1
            - DEU.om_agr_nps
            - DEU.om_man_nps
            - DEU.om_trd_nps
            - DEU.om_bss_nps
            - DEU.om_fin_nps
        )
        * (A_nps ** (sigma - 1))
        * (C**eps_nps)
    )

    E = (
        weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps
    ) ** (1 / (1 - sigma))

    # Endogenous net exports given counterfactual productivity
    nx_agr_E = K_agr_DEU * A_agr ** (DEU.xi_agr - 1) / E - DEU.M_agr_E[-1]
    nx_man_E = K_man_DEU * A_man ** (DEU.xi_man - 1) / E - DEU.M_man_E[-1]
    nx_trd_E = K_trd_DEU * A_trd ** (DEU.xi_trd - 1) / E - DEU.M_trd_E[-1]
    nx_bss_E = K_bss_DEU * A_bss ** (DEU.xi_bss - 1) / E - DEU.M_bss_E[-1]
    nx_fin_E = K_fin_DEU * A_fin ** (DEU.xi_fin - 1) / E - DEU.M_fin_E[-1]
    nx_nps_E = K_nps_DEU * A_nps ** (DEU.xi_nps - 1) / E - DEU.M_nps_E[-1]

    if sec == "agr":

        def find_nx(NX_CF):
            L = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_man = (
                DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess 0 for agr NX: observed agr net-export share is small and
        # close to zero, so seeding at 0 gives the cleanest Jacobian scaling. Note:
        # occasional "iteration is not making good progress" RuntimeWarnings are
        # expected when the residual is already near machine precision at x0 — they
        # do not indicate failure, and fsolve still returns the converged root.
        nx_cf = fsolve(find_nx, 0)
        print(
            "Counterfactual trade in agr is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_agr_E)
        )
        print("Counterfactual trade in agr is " + str(nx_cf / nx_agr_E) + " fold")
        print("Counterfactual productivity in agr is " + str(A_agr) + " fold")
        print("Current productivity in agr is " + str(DEU.A_agr[-1]) + " fold")
        DEU.nx_cf_agr = nx_cf

    if sec == "man":

        def find_nx(NX_CF):
            L = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                DEU.om_man_nps * C * (A_man ** (sigma - 1)) + E ** (1 - sigma) * NX_CF
            ) / L
            share_trd = (
                DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess = observed (endogenously recomputed) man NX. Under
        # endogenous trade the CF root is typically nearby because nx_man_E already
        # reflects the CF productivity response through K_man and xi_man.
        nx_cf = fsolve(find_nx, nx_man_E)
        print(
            "Counterfactual trade in man is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_man_E)
        )
        print("Counterfactual trade in man is " + str(nx_cf / nx_man_E) + " fold")
        print("Counterfactual productivity in man is " + str(A_man) + " fold")
        print("Current productivity in man is " + str(DEU.A_man[-1]) + " fold")
        DEU.nx_cf_man = nx_cf

    if sec == "trd":

        def find_nx(NX_CF):
            L = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_bss = (
                DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_trd_E)
        print(
            "Counterfactual trade in trd is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_trd_E)
        )
        print("Counterfactual trade in trd is " + str(nx_cf / nx_trd_E) + " fold")
        print("Counterfactual productivity in trd is " + str(A_trd) + " fold")
        print("Current productivity in trd is " + str(DEU.A_trd[-1]) + " fold")
        DEU.nx_cf_trd = nx_cf

    if sec == "bss":

        def find_nx(NX_CF):
            L = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_fin = (
                DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_bss_E)
        print(
            "Counterfactual trade in bss is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_bss_E)
        )
        print("Counterfactual trade in bss is " + str(nx_cf / nx_bss_E) + " fold")
        print("Counterfactual productivity in bss is " + str(A_bss) + " fold")
        print("Current productivity in bss is " + str(DEU.A_bss[-1]) + " fold")
        DEU.nx_cf_bss = nx_cf

    if sec == "fin":

        def find_nx(NX_CF):
            L = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_nps = (
                (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_fin_E)
        print(
            "Counterfactual trade in fin is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_fin_E)
        )
        print("Counterfactual trade in fin is " + str(nx_cf / nx_fin_E) + " fold")
        print("Counterfactual productivity in fin is " + str(A_fin) + " fold")
        print("Current productivity in fin is " + str(DEU.A_fin[-1]) + " fold")
        DEU.nx_cf_fin = nx_cf

    if sec == "nps":

        def find_nx(NX_CF):
            L = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
            )

            share_agr = (
                DEU.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                DEU.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                DEU.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                DEU.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                DEU.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - DEU.om_agr_nps
                    - DEU.om_man_nps
                    - DEU.om_trd_nps
                    - DEU.om_bss_nps
                    - DEU.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_nps_E)
        print(
            "Counterfactual trade in nps is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_nps_E)
        )
        print("Counterfactual trade in nps is " + str(nx_cf / nx_nps_E) + " fold")
        print("Counterfactual productivity in nps is " + str(A_nps) + " fold")
        print("Current productivity in nps is " + str(DEU.A_nps[-1]) + " fold")
        DEU.nx_cf_nps = nx_cf

# France: identical trade-cure structure as the Germany block above (see header
# comment there for the algebra). K_* constants rebuilt per country so that each
# country's t=0 sectoral exports match its own data.
"France"
# Endogenous trade constants: K_i = E[0]*M_i_E[0] / A_i[0]^(xi_i-1)
K_agr_FRA = FRA.E[0] * FRA.M_agr_E[0] / (FRA.A_agr[0] ** (FRA.xi_agr - 1))
K_man_FRA = FRA.E[0] * FRA.M_man_E[0] / (FRA.A_man[0] ** (FRA.xi_man - 1))
K_trd_FRA = FRA.E[0] * FRA.M_trd_E[0] / (FRA.A_trd[0] ** (FRA.xi_trd - 1))
K_bss_FRA = FRA.E[0] * FRA.M_bss_E[0] / (FRA.A_bss[0] ** (FRA.xi_bss - 1))
K_fin_FRA = FRA.E[0] * FRA.M_fin_E[0] / (FRA.A_fin[0] ** (FRA.xi_fin - 1))
K_nps_FRA = FRA.E[0] * FRA.M_nps_E[0] / (FRA.A_nps[0] ** (FRA.xi_nps - 1))

for sec in ["agr", "man", "trd", "fin", "bss", "nps"]:
    FRA_cf2_catch_trade = counterfactual("FRA")
    FRA_cf2_catch_trade.baseline()
    FRA_cf2_catch_trade.feed_catch_up_growth(0, sec)

    A_agr = np.array(FRA_cf2_catch_trade.A_agr)[-1]
    A_man = np.array(FRA_cf2_catch_trade.A_man)[-1]
    A_trd = np.array(FRA_cf2_catch_trade.A_trd)[-1]
    A_fin = np.array(FRA_cf2_catch_trade.A_fin)[-1]
    A_bss = np.array(FRA_cf2_catch_trade.A_bss)[-1]
    A_nps = np.array(FRA_cf2_catch_trade.A_nps)[-1]

    C = FRA_cf2_catch_trade.C_nps[-1]

    weight_agr = FRA.om_agr_nps * (A_agr ** (sigma - 1)) * (C**eps_agr)
    weight_man = FRA.om_man_nps * (A_man ** (sigma - 1)) * C
    weight_trd = FRA.om_trd_nps * (A_trd ** (sigma - 1)) * (C**eps_trd)
    weight_bss = FRA.om_bss_nps * (A_bss ** (sigma - 1)) * (C**eps_bss)
    weight_fin = FRA.om_fin_nps * (A_fin ** (sigma - 1)) * (C**eps_fin)
    weight_nps = (
        (
            1
            - FRA.om_agr_nps
            - FRA.om_man_nps
            - FRA.om_trd_nps
            - FRA.om_bss_nps
            - FRA.om_fin_nps
        )
        * (A_nps ** (sigma - 1))
        * (C**eps_nps)
    )

    E = (
        weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps
    ) ** (1 / (1 - sigma))

    # Endogenous net exports given counterfactual productivity
    nx_agr_E = K_agr_FRA * A_agr ** (FRA.xi_agr - 1) / E - FRA.M_agr_E[-1]
    nx_man_E = K_man_FRA * A_man ** (FRA.xi_man - 1) / E - FRA.M_man_E[-1]
    nx_trd_E = K_trd_FRA * A_trd ** (FRA.xi_trd - 1) / E - FRA.M_trd_E[-1]
    nx_bss_E = K_bss_FRA * A_bss ** (FRA.xi_bss - 1) / E - FRA.M_bss_E[-1]
    nx_fin_E = K_fin_FRA * A_fin ** (FRA.xi_fin - 1) / E - FRA.M_fin_E[-1]
    nx_nps_E = K_nps_FRA * A_nps ** (FRA.xi_nps - 1) / E - FRA.M_nps_E[-1]

    if sec == "agr":

        def find_nx(NX_CF):
            L = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_man = (
                FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess 0 for agr NX: observed agr net-export share is small and
        # close to zero, so seeding at 0 gives the cleanest Jacobian scaling. Note:
        # occasional "iteration is not making good progress" RuntimeWarnings are
        # expected when the residual is already near machine precision at x0 — they
        # do not indicate failure, and fsolve still returns the converged root.
        nx_cf = fsolve(find_nx, 0)
        print(
            "Counterfactual trade in agr is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_agr_E)
        )
        print("Counterfactual trade in agr is " + str(nx_cf / nx_agr_E) + " fold")
        print("Counterfactual productivity in agr is " + str(A_agr) + " fold")
        print("Current productivity in agr is " + str(FRA.A_agr[-1]) + " fold")
        FRA.nx_cf_agr = nx_cf

    if sec == "man":

        def find_nx(NX_CF):
            L = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                FRA.om_man_nps * C * (A_man ** (sigma - 1)) + E ** (1 - sigma) * NX_CF
            ) / L
            share_trd = (
                FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess = observed (endogenously recomputed) man NX. Under
        # endogenous trade the CF root is typically nearby because nx_man_E already
        # reflects the CF productivity response through K_man and xi_man.
        nx_cf = fsolve(find_nx, nx_man_E)
        print(
            "Counterfactual trade in man is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_man_E)
        )
        print("Counterfactual trade in man is " + str(nx_cf / nx_man_E) + " fold")
        print("Counterfactual productivity in man is " + str(A_man) + " fold")
        print("Current productivity in man is " + str(FRA.A_man[-1]) + " fold")
        FRA.nx_cf_man = nx_cf

    if sec == "trd":

        def find_nx(NX_CF):
            L = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_bss = (
                FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_trd_E)
        print(
            "Counterfactual trade in trd is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_trd_E)
        )
        print("Counterfactual trade in trd is " + str(nx_cf / nx_trd_E) + " fold")
        print("Counterfactual productivity in trd is " + str(A_trd) + " fold")
        print("Current productivity in trd is " + str(FRA.A_trd[-1]) + " fold")
        FRA.nx_cf_trd = nx_cf

    if sec == "bss":

        def find_nx(NX_CF):
            L = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_fin = (
                FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_bss_E)
        print(
            "Counterfactual trade in bss is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_bss_E)
        )
        print("Counterfactual trade in bss is " + str(nx_cf / nx_bss_E) + " fold")
        print("Counterfactual productivity in bss is " + str(A_bss) + " fold")
        print("Current productivity in bss is " + str(FRA.A_bss[-1]) + " fold")
        FRA.nx_cf_bss = nx_cf

    if sec == "fin":

        def find_nx(NX_CF):
            L = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_nps = (
                (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_fin_E)
        print(
            "Counterfactual trade in fin is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_fin_E)
        )
        print("Counterfactual trade in fin is " + str(nx_cf / nx_fin_E) + " fold")
        print("Counterfactual productivity in fin is " + str(A_fin) + " fold")
        print("Current productivity in fin is " + str(FRA.A_fin[-1]) + " fold")
        FRA.nx_cf_fin = nx_cf

    if sec == "nps":

        def find_nx(NX_CF):
            L = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
            )

            share_agr = (
                FRA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                FRA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                FRA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                FRA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                FRA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - FRA.om_agr_nps
                    - FRA.om_man_nps
                    - FRA.om_trd_nps
                    - FRA.om_bss_nps
                    - FRA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_nps_E)
        print(
            "Counterfactual trade in nps is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_nps_E)
        )
        print("Counterfactual trade in nps is " + str(nx_cf / nx_nps_E) + " fold")
        print("Counterfactual productivity in nps is " + str(A_nps) + " fold")
        print("Current productivity in nps is " + str(FRA.A_nps[-1]) + " fold")
        FRA.nx_cf_nps = nx_cf

# Great Britain: identical trade-cure structure (see Germany block for the
# algebra). Note GBR never adopted the euro — discussed in robustness paragraph
# of sec_introduction.tex — so the country-specific K_* keep GBR's trade elasticities
# separate from the euro-area countries.
"Great Britain"
# Endogenous trade constants: K_i = E[0]*M_i_E[0] / A_i[0]^(xi_i-1)
K_agr_GBR = GBR.E[0] * GBR.M_agr_E[0] / (GBR.A_agr[0] ** (GBR.xi_agr - 1))
K_man_GBR = GBR.E[0] * GBR.M_man_E[0] / (GBR.A_man[0] ** (GBR.xi_man - 1))
K_trd_GBR = GBR.E[0] * GBR.M_trd_E[0] / (GBR.A_trd[0] ** (GBR.xi_trd - 1))
K_bss_GBR = GBR.E[0] * GBR.M_bss_E[0] / (GBR.A_bss[0] ** (GBR.xi_bss - 1))
K_fin_GBR = GBR.E[0] * GBR.M_fin_E[0] / (GBR.A_fin[0] ** (GBR.xi_fin - 1))
K_nps_GBR = GBR.E[0] * GBR.M_nps_E[0] / (GBR.A_nps[0] ** (GBR.xi_nps - 1))

for sec in ["agr", "man", "trd", "fin", "bss", "nps"]:
    GBR_cf2_catch_trade = counterfactual("GBR")
    GBR_cf2_catch_trade.baseline()
    GBR_cf2_catch_trade.feed_catch_up_growth(0, sec)

    A_agr = np.array(GBR_cf2_catch_trade.A_agr)[-1]
    A_man = np.array(GBR_cf2_catch_trade.A_man)[-1]
    A_trd = np.array(GBR_cf2_catch_trade.A_trd)[-1]
    A_fin = np.array(GBR_cf2_catch_trade.A_fin)[-1]
    A_bss = np.array(GBR_cf2_catch_trade.A_bss)[-1]
    A_nps = np.array(GBR_cf2_catch_trade.A_nps)[-1]

    C = GBR_cf2_catch_trade.C_nps[-1]

    weight_agr = GBR.om_agr_nps * (A_agr ** (sigma - 1)) * (C**eps_agr)
    weight_man = GBR.om_man_nps * (A_man ** (sigma - 1)) * C
    weight_trd = GBR.om_trd_nps * (A_trd ** (sigma - 1)) * (C**eps_trd)
    weight_bss = GBR.om_bss_nps * (A_bss ** (sigma - 1)) * (C**eps_bss)
    weight_fin = GBR.om_fin_nps * (A_fin ** (sigma - 1)) * (C**eps_fin)
    weight_nps = (
        (
            1
            - GBR.om_agr_nps
            - GBR.om_man_nps
            - GBR.om_trd_nps
            - GBR.om_bss_nps
            - GBR.om_fin_nps
        )
        * (A_nps ** (sigma - 1))
        * (C**eps_nps)
    )

    E = (
        weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps
    ) ** (1 / (1 - sigma))

    # Endogenous net exports given counterfactual productivity
    nx_agr_E = K_agr_GBR * A_agr ** (GBR.xi_agr - 1) / E - GBR.M_agr_E[-1]
    nx_man_E = K_man_GBR * A_man ** (GBR.xi_man - 1) / E - GBR.M_man_E[-1]
    nx_trd_E = K_trd_GBR * A_trd ** (GBR.xi_trd - 1) / E - GBR.M_trd_E[-1]
    nx_bss_E = K_bss_GBR * A_bss ** (GBR.xi_bss - 1) / E - GBR.M_bss_E[-1]
    nx_fin_E = K_fin_GBR * A_fin ** (GBR.xi_fin - 1) / E - GBR.M_fin_E[-1]
    nx_nps_E = K_nps_GBR * A_nps ** (GBR.xi_nps - 1) / E - GBR.M_nps_E[-1]

    if sec == "agr":

        def find_nx(NX_CF):
            L = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_man = (
                GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess 0 for agr NX: observed agr net-export share is small and
        # close to zero, so seeding at 0 gives the cleanest Jacobian scaling. Note:
        # occasional "iteration is not making good progress" RuntimeWarnings are
        # expected when the residual is already near machine precision at x0 — they
        # do not indicate failure, and fsolve still returns the converged root.
        nx_cf = fsolve(find_nx, 0)
        print(
            "Counterfactual trade in agr is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_agr_E)
        )
        print("Counterfactual trade in agr is " + str(nx_cf / nx_agr_E) + " fold")
        print("Counterfactual productivity in agr is " + str(A_agr) + " fold")
        print("Current productivity in agr is " + str(GBR.A_agr[-1]) + " fold")
        GBR.nx_cf_agr = nx_cf

    if sec == "man":

        def find_nx(NX_CF):
            L = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                GBR.om_man_nps * C * (A_man ** (sigma - 1)) + E ** (1 - sigma) * NX_CF
            ) / L
            share_trd = (
                GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess = observed (endogenously recomputed) man NX. Under
        # endogenous trade the CF root is typically nearby because nx_man_E already
        # reflects the CF productivity response through K_man and xi_man.
        nx_cf = fsolve(find_nx, nx_man_E)
        print(
            "Counterfactual trade in man is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_man_E)
        )
        print("Counterfactual trade in man is " + str(nx_cf / nx_man_E) + " fold")
        print("Counterfactual productivity in man is " + str(A_man) + " fold")
        print("Current productivity in man is " + str(GBR.A_man[-1]) + " fold")
        GBR.nx_cf_man = nx_cf

    if sec == "trd":

        def find_nx(NX_CF):
            L = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_bss = (
                GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_trd_E)
        print(
            "Counterfactual trade in trd is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_trd_E)
        )
        print("Counterfactual trade in trd is " + str(nx_cf / nx_trd_E) + " fold")
        print("Counterfactual productivity in trd is " + str(A_trd) + " fold")
        print("Current productivity in trd is " + str(GBR.A_trd[-1]) + " fold")
        GBR.nx_cf_trd = nx_cf

    if sec == "bss":

        def find_nx(NX_CF):
            L = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_fin = (
                GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_bss_E)
        print(
            "Counterfactual trade in bss is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_bss_E)
        )
        print("Counterfactual trade in bss is " + str(nx_cf / nx_bss_E) + " fold")
        print("Counterfactual productivity in bss is " + str(A_bss) + " fold")
        print("Current productivity in bss is " + str(GBR.A_bss[-1]) + " fold")
        GBR.nx_cf_bss = nx_cf

    if sec == "fin":

        def find_nx(NX_CF):
            L = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_nps = (
                (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_fin_E)
        print(
            "Counterfactual trade in fin is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_fin_E)
        )
        print("Counterfactual trade in fin is " + str(nx_cf / nx_fin_E) + " fold")
        print("Counterfactual productivity in fin is " + str(A_fin) + " fold")
        print("Current productivity in fin is " + str(GBR.A_fin[-1]) + " fold")
        GBR.nx_cf_fin = nx_cf

    if sec == "nps":

        def find_nx(NX_CF):
            L = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
            )

            share_agr = (
                GBR.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                GBR.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                GBR.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                GBR.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                GBR.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - GBR.om_agr_nps
                    - GBR.om_man_nps
                    - GBR.om_trd_nps
                    - GBR.om_bss_nps
                    - GBR.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_nps_E)
        print(
            "Counterfactual trade in nps is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_nps_E)
        )
        print("Counterfactual trade in nps is " + str(nx_cf / nx_nps_E) + " fold")
        print("Counterfactual productivity in nps is " + str(A_nps) + " fold")
        print("Current productivity in nps is " + str(GBR.A_nps[-1]) + " fold")
        GBR.nx_cf_nps = nx_cf
# Italy: identical trade-cure structure (see Germany block). EU4 aggregate
# trade-cure values are built below by h_tot-weighted aggregation over DEU, FRA,
# GBR, ITA.
"Italy"
# Endogenous trade constants: K_i = E[0]*M_i_E[0] / A_i[0]^(xi_i-1)
K_agr_ITA = ITA.E[0] * ITA.M_agr_E[0] / (ITA.A_agr[0] ** (ITA.xi_agr - 1))
K_man_ITA = ITA.E[0] * ITA.M_man_E[0] / (ITA.A_man[0] ** (ITA.xi_man - 1))
K_trd_ITA = ITA.E[0] * ITA.M_trd_E[0] / (ITA.A_trd[0] ** (ITA.xi_trd - 1))
K_bss_ITA = ITA.E[0] * ITA.M_bss_E[0] / (ITA.A_bss[0] ** (ITA.xi_bss - 1))
K_fin_ITA = ITA.E[0] * ITA.M_fin_E[0] / (ITA.A_fin[0] ** (ITA.xi_fin - 1))
K_nps_ITA = ITA.E[0] * ITA.M_nps_E[0] / (ITA.A_nps[0] ** (ITA.xi_nps - 1))

for sec in ["agr", "man", "trd", "fin", "bss", "nps"]:
    ITA_cf2_catch_trade = counterfactual("ITA")
    ITA_cf2_catch_trade.baseline()
    ITA_cf2_catch_trade.feed_catch_up_growth(0, sec)

    A_agr = np.array(ITA_cf2_catch_trade.A_agr)[-1]
    A_man = np.array(ITA_cf2_catch_trade.A_man)[-1]
    A_trd = np.array(ITA_cf2_catch_trade.A_trd)[-1]
    A_fin = np.array(ITA_cf2_catch_trade.A_fin)[-1]
    A_bss = np.array(ITA_cf2_catch_trade.A_bss)[-1]
    A_nps = np.array(ITA_cf2_catch_trade.A_nps)[-1]

    C = ITA_cf2_catch_trade.C_nps[-1]

    weight_agr = ITA.om_agr_nps * (A_agr ** (sigma - 1)) * (C**eps_agr)
    weight_man = ITA.om_man_nps * (A_man ** (sigma - 1)) * C
    weight_trd = ITA.om_trd_nps * (A_trd ** (sigma - 1)) * (C**eps_trd)
    weight_bss = ITA.om_bss_nps * (A_bss ** (sigma - 1)) * (C**eps_bss)
    weight_fin = ITA.om_fin_nps * (A_fin ** (sigma - 1)) * (C**eps_fin)
    weight_nps = (
        (
            1
            - ITA.om_agr_nps
            - ITA.om_man_nps
            - ITA.om_trd_nps
            - ITA.om_bss_nps
            - ITA.om_fin_nps
        )
        * (A_nps ** (sigma - 1))
        * (C**eps_nps)
    )

    E = (
        weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps
    ) ** (1 / (1 - sigma))

    # Endogenous net exports given counterfactual productivity
    nx_agr_E = K_agr_ITA * A_agr ** (ITA.xi_agr - 1) / E - ITA.M_agr_E[-1]
    nx_man_E = K_man_ITA * A_man ** (ITA.xi_man - 1) / E - ITA.M_man_E[-1]
    nx_trd_E = K_trd_ITA * A_trd ** (ITA.xi_trd - 1) / E - ITA.M_trd_E[-1]
    nx_bss_E = K_bss_ITA * A_bss ** (ITA.xi_bss - 1) / E - ITA.M_bss_E[-1]
    nx_fin_E = K_fin_ITA * A_fin ** (ITA.xi_fin - 1) / E - ITA.M_fin_E[-1]
    nx_nps_E = K_nps_ITA * A_nps ** (ITA.xi_nps - 1) / E - ITA.M_nps_E[-1]

    if sec == "agr":

        def find_nx(NX_CF):
            L = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_man = (
                ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess 0 for agr NX: observed agr net-export share is small and
        # close to zero, so seeding at 0 gives the cleanest Jacobian scaling. Note:
        # occasional "iteration is not making good progress" RuntimeWarnings are
        # expected when the residual is already near machine precision at x0 — they
        # do not indicate failure, and fsolve still returns the converged root.
        nx_cf = fsolve(find_nx, 0)
        print(
            "Counterfactual trade in agr is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_agr_E)
        )
        print("Counterfactual trade in agr is " + str(nx_cf / nx_agr_E) + " fold")
        print("Counterfactual productivity in agr is " + str(A_agr) + " fold")
        print("Current productivity in agr is " + str(ITA.A_agr[-1]) + " fold")
        ITA.nx_cf_agr = nx_cf

    if sec == "man":

        def find_nx(NX_CF):
            L = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                ITA.om_man_nps * C * (A_man ** (sigma - 1)) + E ** (1 - sigma) * NX_CF
            ) / L
            share_trd = (
                ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        # Initial guess = observed (endogenously recomputed) man NX. Under
        # endogenous trade the CF root is typically nearby because nx_man_E already
        # reflects the CF productivity response through K_man and xi_man.
        nx_cf = fsolve(find_nx, nx_man_E)
        print(
            "Counterfactual trade in man is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_man_E)
        )
        print("Counterfactual trade in man is " + str(nx_cf / nx_man_E) + " fold")
        print("Counterfactual productivity in man is " + str(A_man) + " fold")
        print("Current productivity in man is " + str(ITA.A_man[-1]) + " fold")
        ITA.nx_cf_man = nx_cf

    if sec == "trd":

        def find_nx(NX_CF):
            L = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_bss = (
                ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_trd_E)
        print(
            "Counterfactual trade in trd is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_trd_E)
        )
        print("Counterfactual trade in trd is " + str(nx_cf / nx_trd_E) + " fold")
        print("Counterfactual productivity in trd is " + str(A_trd) + " fold")
        print("Current productivity in trd is " + str(ITA.A_trd[-1]) + " fold")
        ITA.nx_cf_trd = nx_cf

    if sec == "bss":

        def find_nx(NX_CF):
            L = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_fin = (
                ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_bss_E)
        print(
            "Counterfactual trade in bss is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_bss_E)
        )
        print("Counterfactual trade in bss is " + str(nx_cf / nx_bss_E) + " fold")
        print("Counterfactual productivity in bss is " + str(A_bss) + " fold")
        print("Current productivity in bss is " + str(ITA.A_bss[-1]) + " fold")
        ITA.nx_cf_bss = nx_cf

    if sec == "fin":

        def find_nx(NX_CF):
            L = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
                + (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_nps_E
            )

            share_agr = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L
            share_nps = (
                (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * nx_nps_E
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_fin_E)
        print(
            "Counterfactual trade in fin is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_fin_E)
        )
        print("Counterfactual trade in fin is " + str(nx_cf / nx_fin_E) + " fold")
        print("Counterfactual productivity in fin is " + str(A_fin) + " fold")
        print("Current productivity in fin is " + str(ITA.A_fin[-1]) + " fold")
        ITA.nx_cf_fin = nx_cf

    if sec == "nps":

        def find_nx(NX_CF):
            L = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_agr_E
                + ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_man_E
                + ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_trd_E
                + ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_bss_E
                + ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + (E ** (1 - sigma)) * nx_fin_E
                + (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + (E ** (1 - sigma)) * NX_CF
            )

            share_agr = (
                ITA.om_agr_nps * (C**eps_agr) * (A_agr ** (sigma - 1))
                + E ** (1 - sigma) * nx_agr_E
            ) / L
            share_man = (
                ITA.om_man_nps * C * (A_man ** (sigma - 1))
                + E ** (1 - sigma) * nx_man_E
            ) / L
            share_trd = (
                ITA.om_trd_nps * (C**eps_trd) * (A_trd ** (sigma - 1))
                + E ** (1 - sigma) * nx_trd_E
            ) / L
            share_bss = (
                ITA.om_bss_nps * (C**eps_bss) * (A_bss ** (sigma - 1))
                + E ** (1 - sigma) * nx_bss_E
            ) / L
            share_fin = (
                ITA.om_fin_nps * (C**eps_fin) * (A_fin ** (sigma - 1))
                + E ** (1 - sigma) * nx_fin_E
            ) / L
            share_nps = (
                (
                    1
                    - ITA.om_agr_nps
                    - ITA.om_man_nps
                    - ITA.om_trd_nps
                    - ITA.om_bss_nps
                    - ITA.om_fin_nps
                )
                * (C**eps_nps)
                * (A_nps ** (sigma - 1))
                + E ** (1 - sigma) * NX_CF
            ) / L

            return E_USA[-1] - (
                share_agr * A_agr
                + share_man * A_man
                + share_trd * A_trd
                + share_bss * A_bss
                + share_fin * A_fin
                + share_nps * A_nps
            )

        nx_cf = fsolve(find_nx, nx_nps_E)
        print(
            "Counterfactual trade in nps is "
            + str(nx_cf)
            + " while current net trade is "
            + str(nx_nps_E)
        )
        print("Counterfactual trade in nps is " + str(nx_cf / nx_nps_E) + " fold")
        print("Counterfactual productivity in nps is " + str(A_nps) + " fold")
        print("Current productivity in nps is " + str(ITA.A_nps[-1]) + " fold")
        ITA.nx_cf_nps = nx_cf


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

print(EUR4_nx_agr_E)
print(EUR4_nx_man_E)
print(EUR4_nx_trd_E)
print(EUR4_nx_bss_E)
print(EUR4_nx_fin_E)
print(EUR4_nx_nps_E)

print(EUR4_nx_cf_agr)
print(EUR4_nx_cf_man)
print(EUR4_nx_cf_trd)
print(EUR4_nx_cf_bss)
print(EUR4_nx_cf_fin)
print(EUR4_nx_cf_nps)

# Save trade-cure net exports to Excel for consumption by
# generate_endogenous_paper_outputs.py (Step 11/19), which builds Table 6 Panel B
# "exogenous trade cure" column from Observed_NX and the endogenous-trade
# "Counterfactual_NX" column. Values are h_tot-weighted EU4 averages of country
# nx shares (see EUR4_nx_cf_* construction above).
trade_cure_data = {
    "Sector": ["agr", "man", "trd", "bss", "fin", "nps"],
    "Observed_NX": [EUR4_nx_agr_E, EUR4_nx_man_E, EUR4_nx_trd_E,
                    EUR4_nx_bss_E, EUR4_nx_fin_E, EUR4_nx_nps_E],
    "Counterfactual_NX": [EUR4_nx_cf_agr, EUR4_nx_cf_man, EUR4_nx_cf_trd,
                          EUR4_nx_cf_bss, EUR4_nx_cf_fin, EUR4_nx_cf_nps],
}
pd.DataFrame(trade_cure_data).to_excel(
    "../output/figures/trade_cure_nx_endo.xlsx", index=False
)
print("Trade cure NX saved to ../output/figures/trade_cure_nx_endo.xlsx")

ss = np.array(
    [
        (
            np.array(DEU.h_tot)[-1]
            * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
            + np.array(FRA.h_tot)[-1]
            * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
            + np.array(GBR.h_tot)[-1]
            * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
            + np.array(ITA.h_tot)[-1]
            * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
        )
        / EUR4_h_tot[-1]
    ]
)
ss_base = np.array(
    [
        (
            np.array(DEU.h_tot)[-1]
            * (E_USA[-1] / np.array(DEU_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
            + np.array(FRA.h_tot)[-1]
            * (E_USA[-1] / np.array(FRA_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
            + np.array(GBR.h_tot)[-1]
            * (E_USA[-1] / np.array(GBR_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
            + np.array(ITA.h_tot)[-1]
            * (E_USA[-1] / np.array(ITA_cf2_catch.ss_A_base_nps)[-1] - 1)
            * 100
        )
        / EUR4_h_tot[-1]
    ]
)
cf = np.array(
    [
        (
            np.array(DEU.h_tot)[-1]
            * (np.array(DEU_cf2_catch.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1)
            * 100
            + np.array(FRA.h_tot)[-1]
            * (np.array(FRA_cf2_catch.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1)
            * 100
            + np.array(GBR.h_tot)[-1]
            * (np.array(GBR_cf2_catch.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1)
            * 100
            + np.array(ITA.h_tot)[-1]
            * (np.array(ITA_cf2_catch.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1)
            * 100
        )
        / EUR4_h_tot[-1]
    ]
)
cf_base = np.array(
    [
        (
            np.array(DEU.h_tot)[-1]
            * (np.array(DEU_cf2_catch.A_tot_nps)[-1] / np.array(DEU.A_tot_nps)[-1] - 1)
            * 100
            + np.array(FRA.h_tot)[-1]
            * (np.array(FRA_cf2_catch.A_tot_nps)[-1] / np.array(FRA.A_tot_nps)[-1] - 1)
            * 100
            + np.array(GBR.h_tot)[-1]
            * (np.array(GBR_cf2_catch.A_tot_nps)[-1] / np.array(GBR.A_tot_nps)[-1] - 1)
            * 100
            + np.array(ITA.h_tot)[-1]
            * (np.array(ITA_cf2_catch.A_tot_nps)[-1] / np.array(ITA.A_tot_nps)[-1] - 1)
            * 100
        )
        / EUR4_h_tot[-1]
    ]
)


"""
========================================================================================
    Comparison: Exogenous vs Endogenous Trade Model
========================================================================================
"""

# Late import of exogenous-trade model predictions. Deferred to this point (not
# at top of file) so that the endogenous-trade calibration can finish without the
# exogenous namespace polluting the endogenous `share_*` / `A_tot_*` variables
# used above. `_exo` suffix is applied on import to prevent accidental shadowing.
from model_test_europe_open import (
    EUR4_share_agr_nps_m as EUR4_share_agr_nps_m_exo,
    EUR4_share_man_nps_m as EUR4_share_man_nps_m_exo,
    EUR4_share_nps_nps_m as EUR4_share_nps_nps_m_exo,
    EUR4_A_tot_nps as EUR4_A_tot_nps_exo,
)

# ------- Comparison Table: Counterfactual results (exogenous vs endogenous) -------
# Read exogenous CF results from existing Excel files
exo_cf1_nps = pd.read_excel(
    "../output/figures/Counterfactual_1_nps_trade.xlsx", header=None
)
endo_cf1_nps = pd.read_excel(
    "../output/figures/Counterfactual_1_nps_endo_trade.xlsx", header=None
)
exo_cf2_nps = pd.read_excel(
    "../output/figures/Counterfactual_2_catch_nps_trade.xlsx", header=None
)
endo_cf2_nps = pd.read_excel(
    "../output/figures/Counterfactual_2_catch_nps_endo_trade.xlsx", header=None
)
exo_cf1_nps_ss = pd.read_excel(
    "../output/figures/Counterfactual_1_nps_ss_trade.xlsx", header=None
)
endo_cf1_nps_ss = pd.read_excel(
    "../output/figures/Counterfactual_1_nps_ss_endo_trade.xlsx", header=None
)
exo_cf2_nps_ss = pd.read_excel(
    "../output/figures/Counterfactual_2_catch_nps_ss_trade.xlsx", header=None
)
endo_cf2_nps_ss = pd.read_excel(
    "../output/figures/Counterfactual_2_catch_nps_ss_endo_trade.xlsx", header=None
)

# Build side-by-side comparison
def _build_comparison_block(title, exo_df, endo_df):
    """Build a comparison block with proper row/column alignment."""
    rows = []
    rows.append([title])
    exo_ncols = exo_df.shape[1]
    endo_ncols = endo_df.shape[1]
    # Header: country names from row 0
    rows.append(
        ["Sector"] + list(exo_df.iloc[0, 1:]) + [""] + list(endo_df.iloc[0, 1:])
    )
    rows.append(
        [""] + ["Exogenous"] * (exo_ncols - 1) + [""] + ["Endogenous"] * (endo_ncols - 1)
    )
    target_len = 1 + (exo_ncols - 1) + 1 + (endo_ncols - 1)
    n_rows = max(len(exo_df), len(endo_df))
    for i in range(1, n_rows):
        row_exo = list(exo_df.iloc[i]) if i < len(exo_df) else [""] * exo_ncols
        row_endo = list(endo_df.iloc[i]) if i < len(endo_df) else [""] * endo_ncols
        # Pad to expected column count
        while len(row_exo) < exo_ncols:
            row_exo.append("")
        while len(row_endo) < endo_ncols:
            row_endo.append("")
        combined = row_exo + [""] + row_endo[1:]
        # Pad or trim to target length
        while len(combined) < target_len:
            combined.append("")
        rows.append(combined[:target_len])
    return rows

comparison_data = []
for title, exo_df, endo_df in [
    ("CF1 NPS: Feed US Productivity Growth (Model)", exo_cf1_nps, endo_cf1_nps),
    ("CF1 NPS: Feed US Productivity Growth (Shift-Share)", exo_cf1_nps_ss, endo_cf1_nps_ss),
    ("CF2 NPS: Catch-Up Growth (Model)", exo_cf2_nps, endo_cf2_nps),
    ("CF2 NPS: Catch-Up Growth (Shift-Share)", exo_cf2_nps_ss, endo_cf2_nps_ss),
]:
    if comparison_data:
        comparison_data.append([])
    comparison_data.extend(_build_comparison_block(title, exo_df, endo_df))

pd.DataFrame(comparison_data).to_excel(
    "../output/figures/Counterfactual_comparison_exo_vs_endo.xlsx",
    index=False,
    header=False,
)

# ------- Comparison Figure: EUR4 Employment Shares & Productivity -------
# Figure 2 in the paper. Two-panel side-by-side:
#   (Left)  EUR4 NPS employment shares (agr/man/nps) — data vs. exogenous-trade
#           model vs. endogenous-trade model, 1970-2019.
#   (Right) EUR4 aggregate labor productivity relative to the US — same three
#           series. Visualises that the endogenous-trade model captures the
#           2000s slowdown observed in the data that the exogenous model misses.
# Endogenous predictions (already imported from model_test_europe_endogenous_xn)
# Exogenous predictions (imported above with _exo suffix)

years = DEU.year  # common year vector

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Employment shares (NPS model, EUR4)
ax = axes[0]
# Data
ax.plot(years, EUR4_share_agr, "k-", linewidth=1.5, label="Data: Agr")
ax.plot(years, EUR4_share_man, "k--", linewidth=1.5, label="Data: Man")
ax.plot(years, EUR4_share_nps, "k:", linewidth=1.5, label="Data: NPS")
# Exogenous model
ax.plot(years, EUR4_share_agr_nps_m_exo, "b-", linewidth=1, alpha=0.7, label="Exo: Agr")
ax.plot(years, EUR4_share_man_nps_m_exo, "b--", linewidth=1, alpha=0.7, label="Exo: Man")
ax.plot(years, EUR4_share_nps_nps_m_exo, "b:", linewidth=1, alpha=0.7, label="Exo: NPS")
# Endogenous model
ax.plot(years, EUR4_share_agr_nps_m, "r-", linewidth=1, alpha=0.7, label="Endo: Agr")
ax.plot(years, EUR4_share_man_nps_m, "r--", linewidth=1, alpha=0.7, label="Endo: Man")
ax.plot(years, EUR4_share_nps_nps_m, "r:", linewidth=1, alpha=0.7, label="Endo: NPS")
ax.set_xlabel("Year")
ax.set_ylabel("Employment Share")
ax.set_title("EUR4 Employment Shares (NPS)")
ax.legend(fontsize=7, ncol=3)

# Right panel: Aggregate productivity relative to US
ax = axes[1]
ax.plot(years, EUR4_rel_A_tot, "k-", linewidth=1.5, label="Data")
exo_rel_nps = np.array(EUR4_A_tot_nps_exo) / np.array(A_tot_nps)
endo_rel_nps = np.array(EUR4_A_tot_nps) / np.array(A_tot_nps)
ax.plot(years, exo_rel_nps, "b-", linewidth=1, alpha=0.7, label="Exogenous (NPS)")
ax.plot(years, endo_rel_nps, "r-", linewidth=1, alpha=0.7, label="Endogenous (NPS)")
ax.set_xlabel("Year")
ax.set_ylabel("Relative to US")
ax.set_title("EUR4 Aggregate Productivity")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("../output/figures/fig_2_open_comparison.pdf", bbox_inches="tight")
plt.close()

print("Comparison table saved to ../output/figures/Counterfactual_comparison_exo_vs_endo.xlsx")
print("Comparison figure saved to ../output/figures/fig_2_open_comparison.pdf")
