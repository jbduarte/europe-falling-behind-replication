"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        model_test_europe_endogenous_xn.py
Purpose:     Apply the US-calibrated endogenous-trade model to European
             economies. Each country's sectoral exports are generated inside
             the model via the trade elasticities xi_i calibrated on US data,
             so net-export paths are model-implied (not data-fed). This is the
             main specification used to evaluate the role of trade in the EU4
             structural-change gap (Table 6, Figure 6).
Pipeline:    Step 8/19 — Endogenous-trade model test on Europe.
Inputs:      ../data/euklems_2023.csv (EUKLEMS 2023 VA_Q, H by country-sector)
             ../data/io_panel.xlsx and ../data/exp_imp_aggregate_panel.xlsx
             (OECD ICIO sectoral and aggregate trade flows, used to back out
             country-specific export quantity indices)
             ../data/raw/OECD_GDP_ph.xlsx (OECD GDP per hour, PPP USD)
             Preference parameters (sigma, eps_*) from model_calibration_USA.py
             and the endogenous-trade calibration (A_*, E, share_*, xi_*, year)
             from model_calibration_USA_endogenous_open.py.
Outputs:     The model_country class in its endogenous-trade form plus country
             instances and EU aggregates EUR4_*, EUR13_*, EURCORE_*, EURPERI_*
             consumed by trade_counterfactuals_endogenous.py and
             generate_paper_outputs.py.
Dependencies: model_calibration_USA.py (Step 1),
              model_calibration_USA_open.py (Step 4, used for "year" alignment),
              model_calibration_USA_endogenous_open.py (Step 7).
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import statsmodels.api as sm
from scipy.optimize import fsolve
import pandas as pd
from scipy.optimize import minimize_scalar


# Import calibrated preference parameters from Step 1 and the US endogenous-trade
# calibration from Step 7. US GDP and productivity aggregates are needed to
# measure European sectoral productivity in comparable units.
from model_calibration_USA import (
    sigma,
    eps_agr,
    eps_trd,
    eps_fin,
    eps_bss,
    eps_nps,
    eps_ser,
)
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
    GDP_ph as USA_GDP_ph,
    share_agr,
    share_man,
    share_trd,
    share_bss,
    share_fin,
    share_agr_nps,
    share_man_nps,
    share_trd_nps,
    share_bss_nps,
    share_fin_nps,
    A_tot_nps,
    share_agr_nps_closed,
    share_man_nps_closed,
    share_trd_nps_closed,
    share_bss_nps_closed,
    share_fin_nps_closed,
    A_tot_nps_closed,
)

rc("text", usetex=True)
rc("font", family="serif")


"""
----------------------------
       Model Economy
----------------------------
"""


class model_country:
    """Country-level open-economy model (ENDOGENOUS trade). Sectoral exports
    follow x_i,t = x_0,i * A_i,t^xi_i (equivalently x_i = x_0,i * p_i^{-xi_i}
    since p_i = 1/A_i under linear-in-labor production). Preferences (sigma,
    eps_*) are fixed at the US-calibrated values. The trade elasticities
    xi_i are re-fit country by country (xi_fit_sector below) to hit each
    economy's observed terminal-period real export level; net-export paths
    are therefore model OUTPUTS that respond to productivity. This is the
    specification used in Section 6, Table 6 and Figure 6 of the paper."""

    def __init__(
        self,
        country_code,
        sigma=sigma,
        eps_agr=eps_agr,
        eps_trd=eps_trd,
        eps_fin=eps_fin,
        eps_bss=eps_bss,
        eps_nps=eps_nps,
        eps_ser=eps_ser,
    ):
        "Initialize the Parameters"
        (
            self.cou,
            self.sigma,
            self.eps_agr,
            self.eps_trd,
            self.eps_fin,
            self.eps_bss,
            self.eps_nps,
            self.eps_ser,
        ) = country_code, sigma, eps_agr, eps_trd, eps_fin, eps_bss, eps_nps, eps_ser

        """
        -----------
            Data
        -----------
        """

        "KLEMS"
        data = pd.read_csv("../data/euklems_2023.csv", index_col=[0, 1])
        data.rename(index={"AT": "AUT"}, inplace=True)
        data.rename(index={"BE": "BEL"}, inplace=True)
        data.rename(index={"DE": "DEU"}, inplace=True)
        data.rename(index={"DK": "DNK"}, inplace=True)
        data.rename(index={"ES": "ESP"}, inplace=True)
        data.rename(index={"FI": "FIN"}, inplace=True)
        data.rename(index={"FR": "FRA"}, inplace=True)
        data.rename(index={"GB": "GBR"}, inplace=True)
        data.rename(index={"GR": "GRC"}, inplace=True)
        data.rename(index={"IE": "IRL"}, inplace=True)
        data.rename(index={"IT": "ITA"}, inplace=True)
        data.rename(index={"LU": "LUX"}, inplace=True)
        data.rename(index={"NL": "NLD"}, inplace=True)
        data.rename(index={"PT": "PRT"}, inplace=True)
        data.rename(index={"SE": "SWE"}, inplace=True)
        data.rename(index={"US": "USA"}, inplace=True)
        data.rename(columns={"sector": "sec"}, inplace=True)

        "OECD"
        if self.cou == "EU15":
            GDP_ph = pd.read_excel(
                "../data/raw/OECD_GDP_ph_EU15.xlsx",
                index_col=[0, 5],
                engine="openpyxl",
            )  # Measured in USD (constant prices 2010 and PPPs).
            GDP_ph = GDP_ph[GDP_ph["MEASURE"] == "USD"]
            GDP_ph.index.rename(["country", "year"], inplace=True)
        else:
            GDP_ph = pd.read_excel(
                "../data/raw/OECD_GDP_ph.xlsx", index_col=[0, 5], engine="openpyxl"
            )  # Measured in USD (constant prices 2010 and PPPs).
            GDP_ph = GDP_ph[GDP_ph["MEASURE"] == "USD"]
            GDP_ph.index.rename(["country", "year"], inplace=True)

        "OECD-IO"
        data_NX_sec = pd.read_excel(
            "../data/io_panel.xlsx", index_col=[0, 1], engine="openpyxl"
        )
        data_NX_sec["nx"] = data_NX_sec["expo"] + data_NX_sec["impo"]
        # data_NX_sec['nx'] = 0
        data_NX_agg = pd.read_excel(
            "../data/exp_imp_aggregate_panel.xlsx",
            index_col=[0, 1],
            engine="openpyxl",
        )

        "Country data"
        data = data.loc[self.cou]
        data = data[data.index >= 1995]

        GDP_ph = GDP_ph.loc[self.cou]
        GDP_ph = GDP_ph[GDP_ph.index >= 1995]

        data_NX_sec = data_NX_sec.loc[self.cou]
        data_NX_sec = data_NX_sec[data_NX_sec.index >= 1995]

        data_NX_agg = data_NX_agg.loc[self.cou]
        data_NX_agg = data_NX_agg[data_NX_agg.index >= 1995]

        "Labor Productivity"
        data["y_l"] = (data["VA_Q"] / data["H"]) * 100

        "Sectoral Data"
        # Ensure indexes are aligned for merging
        data_NX_sec = data_NX_sec.reset_index()
        data = data.reset_index()

        # Merge sectoral data
        data = pd.merge(data, data_NX_sec, on=["year", "sec"], how="left")
        data.set_index("year", inplace=True)

        data_agr = data[data["sec"] == "agr"]
        data_man = data[data["sec"] == "man"]
        data_trd = data[data["sec"] == "trd"]
        data_bss = data[data["sec"] == "bss"]
        data_fin = data[data["sec"] == "fin"]
        data_nps = data[data["sec"] == "nps"]
        data_ser = data[data["sec"] == "ser"]
        data_tot = data[data["sec"] == "tot"]

        # Adding trade in services
        data_ser.loc[:, "expo"] = (
            data_trd.loc[:, "expo"]
            + data_bss.loc[:, "expo"]
            + data_fin.loc[:, "expo"]
            + data_nps.loc[:, "expo"]
        )
        data_ser.loc[:, "impo"] = (
            data_trd.loc[:, "impo"]
            + data_bss.loc[:, "impo"]
            + data_fin.loc[:, "impo"]
            + data_nps.loc[:, "impo"]
        )
        data_ser.loc[:, "gdp"] = (
            data_trd.loc[:, "gdp"]
            + data_bss.loc[:, "gdp"]
            + data_fin.loc[:, "gdp"]
            + data_nps.loc[:, "gdp"]
        )
        data_ser.loc[:, "nx"] = (
            data_trd.loc[:, "nx"]
            + data_bss.loc[:, "nx"]
            + data_fin.loc[:, "nx"]
            + data_nps.loc[:, "nx"]
        )

        """
        -------------------------------
            Time Series (Filtering)
        -------------------------------
        """

        "GDP"
        c_GDP_ph, self.GDP_ph = sm.tsa.filters.hpfilter(GDP_ph["Value"], 100)

        "GDP Growth"
        self.g_GDP_ph = np.array(
            self.GDP_ph / self.GDP_ph.shift(1) - 1
        ).flatten()  # GDP Growth from OECD

        "Nominal Expenditures"
        nom_exp_c, self.nom_exp = sm.tsa.filters.hpfilter(
            (data_tot["VA"] - data_tot["nx"]), 100
        )
        self.g_nom_exp = np.array(self.nom_exp / self.nom_exp.shift(1) - 1).flatten()

        "Employment hours"
        h_agr_c, self.h_agr = sm.tsa.filters.hpfilter(data_agr["H"], 100)
        h_man_c, self.h_man = sm.tsa.filters.hpfilter(data_man["H"], 100)
        h_trd_c, self.h_trd = sm.tsa.filters.hpfilter(data_trd["H"], 100)
        h_bss_c, self.h_bss = sm.tsa.filters.hpfilter(data_bss["H"], 100)
        h_fin_c, self.h_fin = sm.tsa.filters.hpfilter(data_fin["H"], 100)
        h_nps_c, self.h_nps = sm.tsa.filters.hpfilter(data_nps["H"], 100)
        h_ser_c, self.h_ser = sm.tsa.filters.hpfilter(data_ser["H"], 100)
        h_tot_c, self.h_tot = sm.tsa.filters.hpfilter(data_tot["H"], 100)

        "Labor Productivity"
        y_l_agr_c, y_l_agr = sm.tsa.filters.hpfilter(data_agr["y_l"], 100)
        y_l_man_c, y_l_man = sm.tsa.filters.hpfilter(data_man["y_l"], 100)
        y_l_trd_c, y_l_trd = sm.tsa.filters.hpfilter(data_trd["y_l"], 100)
        y_l_bss_c, y_l_bss = sm.tsa.filters.hpfilter(data_bss["y_l"], 100)
        y_l_fin_c, y_l_fin = sm.tsa.filters.hpfilter(data_fin["y_l"], 100)
        y_l_nps_c, y_l_nps = sm.tsa.filters.hpfilter(data_nps["y_l"], 100)
        y_l_ser_c, y_l_ser = sm.tsa.filters.hpfilter(data_ser["y_l"], 100)
        y_l_tot_c, y_l_tot = sm.tsa.filters.hpfilter(data_tot["y_l"], 100)

        "Labor Productivity Growth"
        self.g_y_l_agr = np.array(y_l_agr / y_l_agr.shift(1) - 1)
        self.g_y_l_man = np.array(y_l_man / y_l_man.shift(1) - 1)
        self.g_y_l_trd = np.array(y_l_trd / y_l_trd.shift(1) - 1)
        self.g_y_l_bss = np.array(y_l_bss / y_l_bss.shift(1) - 1)
        self.g_y_l_fin = np.array(y_l_fin / y_l_fin.shift(1) - 1)
        self.g_y_l_nps = np.array(y_l_nps / y_l_nps.shift(1) - 1)
        self.g_y_l_ser = np.array(y_l_ser / y_l_ser.shift(1) - 1)
        self.g_y_l_tot = np.array(y_l_tot / y_l_tot.shift(1) - 1)

        "Prices"
        p_agr_c, self.p_agr = sm.tsa.filters.hpfilter(
            data_agr["VA"] / data_agr["VA_Q"], 100
        )
        p_man_c, self.p_man = sm.tsa.filters.hpfilter(
            data_man["VA"] / data_man["VA_Q"], 100
        )
        p_trd_c, self.p_trd = sm.tsa.filters.hpfilter(
            data_trd["VA"] / data_trd["VA_Q"], 100
        )
        p_bss_c, self.p_bss = sm.tsa.filters.hpfilter(
            data_bss["VA"] / data_bss["VA_Q"], 100
        )
        p_fin_c, self.p_fin = sm.tsa.filters.hpfilter(
            data_fin["VA"] / data_fin["VA_Q"], 100
        )
        p_nps_c, self.p_nps = sm.tsa.filters.hpfilter(
            data_nps["VA"] / data_nps["VA_Q"], 100
        )
        p_ser_c, self.p_ser = sm.tsa.filters.hpfilter(
            data_ser["VA"] / data_ser["VA_Q"], 100
        )
        p_tot_c, self.p_tot = sm.tsa.filters.hpfilter(
            data_tot["VA"] / data_tot["VA_Q"], 100
        )

        "Employment Shares"
        share_c_agr, self.share_agr = sm.tsa.filters.hpfilter(
            (
                self.h_agr
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )
        share_c_man, self.share_man = sm.tsa.filters.hpfilter(
            (
                self.h_man
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )
        share_c_trd, self.share_trd = sm.tsa.filters.hpfilter(
            (
                self.h_trd
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )
        share_c_bss, self.share_bss = sm.tsa.filters.hpfilter(
            (
                self.h_bss
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )
        share_c_fin, self.share_fin = sm.tsa.filters.hpfilter(
            (
                self.h_fin
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )
        share_c_nps, self.share_nps = sm.tsa.filters.hpfilter(
            (
                self.h_nps
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )
        share_c_ser, self.share_ser = sm.tsa.filters.hpfilter(
            (
                self.h_ser
                / (
                    self.h_agr
                    + self.h_man
                    + self.h_trd
                    + self.h_bss
                    + self.h_fin
                    + self.h_nps
                )
            ),
            100,
        )

        "Employment Shares Without Manufacturing (Weights of C)"
        share_c_agr_no_man, self.share_agr_no_man = sm.tsa.filters.hpfilter(
            self.h_agr
            / (self.h_agr + self.h_trd + self.h_bss + self.h_fin + self.h_nps),
            100,
        )
        share_c_trd_no_man, self.share_trd_no_man = sm.tsa.filters.hpfilter(
            self.h_trd
            / (self.h_agr + self.h_trd + self.h_bss + self.h_fin + self.h_nps),
            100,
        )
        share_c_bss_no_man, self.share_bss_no_man = sm.tsa.filters.hpfilter(
            self.h_bss
            / (self.h_agr + self.h_trd + self.h_bss + self.h_fin + self.h_nps),
            100,
        )
        share_c_fin_no_man, self.share_fin_no_man = sm.tsa.filters.hpfilter(
            self.h_fin
            / (self.h_agr + self.h_trd + self.h_bss + self.h_fin + self.h_nps),
            100,
        )
        share_c_nps_no_man, self.share_nps_no_man = sm.tsa.filters.hpfilter(
            self.h_nps
            / (self.h_agr + self.h_trd + self.h_bss + self.h_fin + self.h_nps),
            100,
        )
        share_c_ser_no_man, self.share_ser_no_man = sm.tsa.filters.hpfilter(
            self.h_ser
            / (self.h_agr + self.h_trd + self.h_bss + self.h_fin + self.h_nps),
            100,
        )

        share_c_trd_no_agm, self.share_trd_no_agm = sm.tsa.filters.hpfilter(
            self.h_trd / (self.h_trd + self.h_bss + self.h_fin + self.h_nps), 100
        )
        share_c_bss_no_agm, self.share_bss_no_agm = sm.tsa.filters.hpfilter(
            self.h_bss / (self.h_trd + self.h_bss + self.h_fin + self.h_nps), 100
        )
        share_c_fin_no_agm, self.share_fin_no_agm = sm.tsa.filters.hpfilter(
            self.h_fin / (self.h_trd + self.h_bss + self.h_fin + self.h_nps), 100
        )
        share_c_nps_no_agm, self.share_nps_no_agm = sm.tsa.filters.hpfilter(
            self.h_nps / (self.h_trd + self.h_bss + self.h_fin + self.h_nps), 100
        )
        share_c_ser_no_agm, self.share_ser_no_agm = sm.tsa.filters.hpfilter(
            self.h_ser / (self.h_trd + self.h_bss + self.h_fin + self.h_nps), 100
        )

        #        'Net Exports as a Share of Credited Expenditures'
        #        nx_c_agr_E, self.nx_agr_E = sm.tsa.filters.hpfilter(data_agr['nx']/(data_tot['VA']-data_tot['nx']),100)
        #        nx_c_man_E, self.nx_man_E = sm.tsa.filters.hpfilter(data_man['nx']/(data_tot['VA']-data_tot['nx']),100)
        #        nx_c_trd_E, self.nx_trd_E = sm.tsa.filters.hpfilter(data_trd['nx']/(data_tot['VA']-data_tot['nx']),100)
        #        nx_c_bss_E, self.nx_bss_E = sm.tsa.filters.hpfilter(data_bss['nx']/(data_tot['VA']-data_tot['nx']),100)
        #        nx_c_fin_E, self.nx_fin_E = sm.tsa.filters.hpfilter(data_fin['nx']/(data_tot['VA']-data_tot['nx']),100)
        #        nx_c_nps_E, self.nx_nps_E = sm.tsa.filters.hpfilter(data_nps['nx']/(data_tot['VA']-data_tot['nx']),100)
        #        self.nx_ser_E = self.nx_trd_E + self.nx_bss_E + self.nx_fin_E + self.nx_nps_E
        #        nx_c_tot_E, self.nx_tot_E = sm.tsa.filters.hpfilter(data_tot['nx']/(data_tot['VA']-data_tot['nx']),100)

        "Real Net Exports"
        nx_c_agr_q, self.nx_agr_q = sm.tsa.filters.hpfilter(
            data_agr["nx"] / (data_agr["VA"] / data_agr["VA_Q"]), 100
        )
        nx_c_man_q, self.nx_man_q = sm.tsa.filters.hpfilter(
            data_man["nx"] / (data_man["VA"] / data_man["VA_Q"]), 100
        )
        nx_c_trd_q, self.nx_trd_q = sm.tsa.filters.hpfilter(
            data_trd["nx"] / (data_trd["VA"] / data_trd["VA_Q"]), 100
        )
        nx_c_bss_q, self.nx_bss_q = sm.tsa.filters.hpfilter(
            data_bss["nx"] / (data_bss["VA"] / data_bss["VA_Q"]), 100
        )
        nx_c_fin_q, self.nx_fin_q = sm.tsa.filters.hpfilter(
            data_fin["nx"] / (data_fin["VA"] / data_fin["VA_Q"]), 100
        )
        nx_c_nps_q, self.nx_nps_q = sm.tsa.filters.hpfilter(
            data_nps["nx"] / (data_nps["VA"] / data_nps["VA_Q"]), 100
        )
        self.nx_ser_q = self.nx_trd_q + self.nx_bss_q + self.nx_fin_q + self.nx_nps_q
        nx_c_tot_q, self.nx_tot_q = sm.tsa.filters.hpfilter(
            data_tot["nx"] / (data_tot["VA"] / data_tot["VA_Q"]), 100
        )

        "Real Exports"
        x_c_agr_q, self.x_agr_q = sm.tsa.filters.hpfilter(
            data_agr["expo"] / (data_agr["VA"] / data_agr["VA_Q"]), 100
        )
        x_c_man_q, self.x_man_q = sm.tsa.filters.hpfilter(
            data_man["expo"] / (data_man["VA"] / data_man["VA_Q"]), 100
        )
        x_c_trd_q, self.x_trd_q = sm.tsa.filters.hpfilter(
            data_trd["expo"] / (data_trd["VA"] / data_trd["VA_Q"]), 100
        )
        x_c_bss_q, self.x_bss_q = sm.tsa.filters.hpfilter(
            data_bss["expo"] / (data_bss["VA"] / data_bss["VA_Q"]), 100
        )
        x_c_fin_q, self.x_fin_q = sm.tsa.filters.hpfilter(
            data_fin["expo"] / (data_fin["VA"] / data_fin["VA_Q"]), 100
        )
        x_c_nps_q, self.x_nps_q = sm.tsa.filters.hpfilter(
            data_nps["expo"] / (data_nps["VA"] / data_nps["VA_Q"]), 100
        )
        self.x_ser_q = self.x_trd_q + self.x_bss_q + self.x_fin_q + self.x_nps_q
        x_c_tot_q, self.x_tot_q = sm.tsa.filters.hpfilter(
            data_tot["expo"] / (data_tot["VA"] / data_tot["VA_Q"]), 100
        )

        "Real Imports"
        m_c_agr_q, self.m_agr_q = sm.tsa.filters.hpfilter(
            (-1) * data_agr["impo"] / (data_agr["VA"] / data_agr["VA_Q"]), 100
        )
        m_c_man_q, self.m_man_q = sm.tsa.filters.hpfilter(
            (-1) * data_man["impo"] / (data_man["VA"] / data_man["VA_Q"]), 100
        )
        m_c_trd_q, self.m_trd_q = sm.tsa.filters.hpfilter(
            (-1) * data_trd["impo"] / (data_trd["VA"] / data_trd["VA_Q"]), 100
        )
        m_c_bss_q, self.m_bss_q = sm.tsa.filters.hpfilter(
            (-1) * data_bss["impo"] / (data_bss["VA"] / data_bss["VA_Q"]), 100
        )
        m_c_fin_q, self.m_fin_q = sm.tsa.filters.hpfilter(
            (-1) * data_fin["impo"] / (data_fin["VA"] / data_fin["VA_Q"]), 100
        )
        m_c_nps_q, self.m_nps_q = sm.tsa.filters.hpfilter(
            (-1) * data_nps["impo"] / (data_nps["VA"] / data_nps["VA_Q"]), 100
        )
        self.m_ser_q = (-1) * (
            self.m_trd_q + self.m_bss_q + self.m_fin_q + self.m_nps_q
        )
        m_c_tot_q, self.m_tot_q = sm.tsa.filters.hpfilter(
            (-1) * data_tot["impo"] / (data_tot["VA"] / data_tot["VA_Q"]), 100
        )

        "Nominal Net Exports as a Share of Credited Expenditures"
        nx_c_agr_E, self.nx_agr_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_agr["y_l"])
                * (data_agr["nx"] / (data_agr["VA"] / data_agr["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )
        nx_c_man_E, self.nx_man_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_man["y_l"])
                * (data_man["nx"] / (data_man["VA"] / data_man["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )
        nx_c_trd_E, self.nx_trd_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_trd["y_l"])
                * (data_trd["nx"] / (data_trd["VA"] / data_trd["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )
        nx_c_bss_E, self.nx_bss_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_bss["y_l"])
                * (data_bss["nx"] / (data_bss["VA"] / data_bss["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )
        nx_c_fin_E, self.nx_fin_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_fin["y_l"])
                * (data_fin["nx"] / (data_fin["VA"] / data_fin["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )
        nx_c_nps_E, self.nx_nps_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_nps["y_l"])
                * (data_nps["nx"] / (data_nps["VA"] / data_nps["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )
        self.nx_ser_E = self.nx_trd_E + self.nx_bss_E + self.nx_fin_E + self.nx_nps_E
        nx_c_tot_E, self.nx_tot_E = sm.tsa.filters.hpfilter(
            (
                (1 / data_tot["y_l"])
                * (data_tot["nx"] / (data_tot["VA"] / data_tot["VA_Q"]))
            )
            / (data_tot["VA"] - data_tot["nx"]),
            100,
        )

        "Exports as a Share of Credited Expenditures"
        x_c_agr_E, self.x_agr_E = sm.tsa.filters.hpfilter(
            data_agr["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        x_c_man_E, self.x_man_E = sm.tsa.filters.hpfilter(
            data_man["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        x_c_trd_E, self.x_trd_E = sm.tsa.filters.hpfilter(
            data_trd["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        x_c_bss_E, self.x_bss_E = sm.tsa.filters.hpfilter(
            data_bss["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        x_c_fin_E, self.x_fin_E = sm.tsa.filters.hpfilter(
            data_fin["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        x_c_nps_E, self.x_nps_E = sm.tsa.filters.hpfilter(
            data_nps["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        self.x_ser_E = self.x_trd_E + self.x_bss_E + self.x_fin_E + self.x_nps_E
        x_c_tot_E, self.x_tot_E = sm.tsa.filters.hpfilter(
            data_tot["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )

        "Imports as a Share of Credited Expenditures"
        m_c_agr_E, self.m_agr_E = sm.tsa.filters.hpfilter(
            data_agr["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        m_c_man_E, self.m_man_E = sm.tsa.filters.hpfilter(
            data_man["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        m_c_trd_E, self.m_trd_E = sm.tsa.filters.hpfilter(
            data_trd["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        m_c_bss_E, self.m_bss_E = sm.tsa.filters.hpfilter(
            data_bss["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        m_c_fin_E, self.m_fin_E = sm.tsa.filters.hpfilter(
            data_fin["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        m_c_nps_E, self.m_nps_E = sm.tsa.filters.hpfilter(
            data_nps["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )
        self.m_ser_E = self.m_trd_E + self.m_bss_E + self.m_fin_E + self.m_nps_E
        m_c_tot_E, self.m_tot_E = sm.tsa.filters.hpfilter(
            data_tot["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
        )

        "Net Exports as a Share of GDP"
        nx_c_agr_Y, self.nx_agr_Y = sm.tsa.filters.hpfilter(
            data_agr["nx"] / (data_tot["gdp"]), 100
        )
        nx_c_man_Y, self.nx_man_Y = sm.tsa.filters.hpfilter(
            data_man["nx"] / (data_tot["gdp"]), 100
        )
        nx_c_trd_Y, self.nx_trd_Y = sm.tsa.filters.hpfilter(
            data_trd["nx"] / (data_tot["gdp"]), 100
        )
        nx_c_bss_Y, self.nx_bss_Y = sm.tsa.filters.hpfilter(
            data_bss["nx"] / (data_tot["gdp"]), 100
        )
        nx_c_fin_Y, self.nx_fin_Y = sm.tsa.filters.hpfilter(
            data_fin["nx"] / (data_tot["gdp"]), 100
        )
        nx_c_nps_Y, self.nx_nps_Y = sm.tsa.filters.hpfilter(
            data_nps["nx"] / (data_tot["gdp"]), 100
        )
        self.nx_ser_Y = self.nx_trd_Y + self.nx_bss_Y + self.nx_fin_Y + self.nx_nps_Y
        nx_c_tot_Y, self.nx_tot_Y = sm.tsa.filters.hpfilter(
            data_tot["nx"] / (data_tot["gdp"]), 100
        )

        "Exports as a Share of GDP"
        x_c_agr_Y, self.x_agr_Y = sm.tsa.filters.hpfilter(
            data_agr["expo"] / (data_tot["gdp"]), 100
        )
        x_c_man_Y, self.x_man_Y = sm.tsa.filters.hpfilter(
            data_man["expo"] / (data_tot["gdp"]), 100
        )
        x_c_trd_Y, self.x_trd_Y = sm.tsa.filters.hpfilter(
            data_trd["expo"] / (data_tot["gdp"]), 100
        )
        x_c_bss_Y, self.x_bss_Y = sm.tsa.filters.hpfilter(
            data_bss["expo"] / (data_tot["gdp"]), 100
        )
        x_c_fin_Y, self.x_fin_Y = sm.tsa.filters.hpfilter(
            data_fin["expo"] / (data_tot["gdp"]), 100
        )
        x_c_nps_Y, self.x_nps_Y = sm.tsa.filters.hpfilter(
            data_nps["expo"] / (data_tot["gdp"]), 100
        )
        self.x_ser_Y = self.x_trd_Y + self.x_bss_Y + self.x_fin_Y + self.x_nps_Y
        x_c_tot_Y, self.x_tot_Y = sm.tsa.filters.hpfilter(
            data_tot["expo"] / (data_tot["gdp"]), 100
        )

        "Imports as a Share of GDP"
        m_c_agr_Y, self.m_agr_Y = sm.tsa.filters.hpfilter(
            data_agr["impo"] / (data_tot["gdp"]), 100
        )
        m_c_man_Y, self.m_man_Y = sm.tsa.filters.hpfilter(
            data_man["impo"] / (data_tot["gdp"]), 100
        )
        m_c_trd_Y, self.m_trd_Y = sm.tsa.filters.hpfilter(
            data_trd["impo"] / (data_tot["gdp"]), 100
        )
        m_c_bss_Y, self.m_bss_Y = sm.tsa.filters.hpfilter(
            data_bss["impo"] / (data_tot["gdp"]), 100
        )
        m_c_fin_Y, self.m_fin_Y = sm.tsa.filters.hpfilter(
            data_fin["impo"] / (data_tot["gdp"]), 100
        )
        m_c_nps_Y, self.m_nps_Y = sm.tsa.filters.hpfilter(
            data_nps["impo"] / (data_tot["gdp"]), 100
        )
        self.m_ser_Y = self.m_trd_Y + self.m_bss_Y + self.m_fin_Y + self.m_nps_Y
        m_c_tot_Y, self.m_tot_Y = sm.tsa.filters.hpfilter(
            data_tot["impo"] / (data_tot["gdp"]), 100
        )

        sector_list = ["agr", "man", "trd", "bss", "fin", "nps", "ser", "tot"]

        # Prepare observed series for each sector without eval or self
        x_Y_series = {
            "agr": np.array(self.x_agr_Y),
            "man": np.array(self.x_man_Y),
            "trd": np.array(self.x_trd_Y),
            "bss": np.array(self.x_bss_Y),
            "fin": np.array(self.x_fin_Y),
            "nps": np.array(self.x_nps_Y),
            "ser": np.array(self.x_ser_Y),  # sum of trd, bss, fin, nps
            "tot": np.array(self.x_tot_Y),
        }
        m_Y_series = {
            "agr": np.array(self.m_agr_Y),
            "man": np.array(self.m_man_Y),
            "trd": np.array(self.m_trd_Y),
            "bss": np.array(self.m_bss_Y),
            "fin": np.array(self.m_fin_Y),
            "nps": np.array(self.m_nps_Y),
            "ser": np.array(self.m_ser_Y),  # sum of trd, bss, fin, nps
            "tot": np.array(self.m_tot_Y),
        }

        im_share_index_all = {}

        for sec in sector_list:
            # Initial value: exports as share of GDP in first period
            im_share_idx = [x_Y_series[sec][0]]
            m_arr = m_Y_series[sec]
            for i in range(1, len(m_arr)):
                g = m_arr[i] / m_arr[i - 1] if m_arr[i - 1] != 0 else 1
                im_share_idx.append(im_share_idx[-1] * g)
            im_share_index_all[sec] = np.array(im_share_idx)

        # Compute net export series as x_Y_series - m_Y_series for all sectors
        nx_Y_series = {
            sec: x_Y_series[sec] - im_share_index_all[sec] for sec in sector_list
        }

        # Store each sector's net export series (adjusted) independently
        self.nx_agr_Y_adj = nx_Y_series["agr"]
        self.nx_man_Y_adj = nx_Y_series["man"]
        self.nx_trd_Y_adj = nx_Y_series["trd"]
        self.nx_bss_Y_adj = nx_Y_series["bss"]
        self.nx_fin_Y_adj = nx_Y_series["fin"]
        self.nx_nps_Y_adj = nx_Y_series["nps"]
        self.nx_ser_Y_adj = nx_Y_series["ser"]
        self.nx_tot_Y_adj = nx_Y_series["tot"]

        # Compute real exports index for all sectors using growth rates from x_{sec}_q  starting at initial period of x_{sec}_E
        sectors = ["agr", "man", "trd", "bss", "fin", "nps", "ser"]

        # Prepare initial values for exports and imports for each sector
        x_0 = {
            "agr": np.array(self.x_agr_Y)[0],
            "man": np.array(self.x_man_Y)[0],
            "trd": np.array(self.x_trd_Y)[0],
            "bss": np.array(self.x_bss_Y)[0],
            "fin": np.array(self.x_fin_Y)[0],
            "nps": np.array(self.x_nps_Y)[0],
            "ser": np.array(self.x_ser_Y)[0],
        }

        # Prepare observed real exports and imports series for each sector
        x_q_series = {
            "agr": np.array(self.x_agr_q),
            "man": np.array(self.x_man_q),
            "trd": np.array(self.x_trd_q),
            "bss": np.array(self.x_bss_q),
            "fin": np.array(self.x_fin_q),
            "nps": np.array(self.x_nps_q),
            "ser": np.array(self.x_ser_q),
        }

        # Compute real exports and imports index for each sector
        x_index = {}
        for sec in sectors:
            # Exports
            x_idx = [x_0[sec]]
            x_q = x_q_series[sec]
            for i in range(1, len(x_q)):
                g = x_q[i] / x_q[i - 1] if x_q[i - 1] != 0 else 1
                x_idx.append(x_idx[-1] * g)
            x_index[sec] = np.array(x_idx)

        self.x_agr_index = x_index["agr"]
        self.x_man_index = x_index["man"]
        self.x_trd_index = x_index["trd"]
        self.x_bss_index = x_index["bss"]
        self.x_fin_index = x_index["fin"]
        self.x_nps_index = x_index["nps"]
        self.x_ser_index = x_index["ser"]

        sector_list = ["agr", "man", "trd", "bss", "fin", "nps", "ser"]
        M_E = {}

        x_index_map = {
            "agr": self.x_agr_index,
            "man": self.x_man_index,
            "trd": self.x_trd_index,
            "bss": self.x_bss_index,
            "fin": self.x_fin_index,
            "nps": self.x_nps_index,
            "ser": self.x_ser_index,
        }
        m_Y_map = {
            "agr": self.m_agr_Y,
            "man": self.m_man_Y,
            "trd": self.m_trd_Y,
            "bss": self.m_bss_Y,
            "fin": self.m_fin_Y,
            "nps": self.m_nps_Y,
            "ser": self.m_ser_Y,
        }

        for sec in sector_list:
            x_index_sec = x_index_map[sec]
            m_Y_sec = m_Y_map[sec]

            M_E_sec = [np.array(x_index_sec)[0]]
            for i in range(1, len(m_Y_sec)):
                g = (
                    np.array(m_Y_sec)[i] / np.array(m_Y_sec)[i - 1]
                    if np.array(m_Y_sec)[i - 1] != 0
                    else 1
                )
                M_E_sec.append(M_E_sec[-1] * g)
            M_E[sec] = np.array(M_E_sec)

        # Unpack into individual variables for backward compatibility
        self.M_agr_E = M_E["agr"]
        self.M_man_E = M_E["man"]
        self.M_trd_E = M_E["trd"]
        self.M_bss_E = M_E["bss"]
        self.M_fin_E = M_E["fin"]
        self.M_nps_E = M_E["nps"]
        self.M_ser_E = M_E["ser"]

        """
        ---------------------------------------------
                Time Series (Inputs of the Model)
        ---------------------------------------------
        """
        E_level = (
            np.array(self.h_agr)
            + np.array(self.h_man)
            + np.array(self.h_ser)
            - np.array(self.nx_tot_q)
        )
        g_E_level = np.array(pd.Series(E_level) / pd.Series(E_level).shift(1) - 1)
        self.E = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        for i in range(len(g_E_level) - 1):
            self.E.append((1 + g_E_level[i + 1]) * self.E[i])

        t_0 = np.array(data.index)[0]
        self.Y_ph = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.year = [t_0]
        self.ts_length = np.array(data.index)[-1] - np.array(data.index)[0]
        for i in range(int(self.ts_length)):
            self.Y_ph.append((1 + self.g_GDP_ph[i + 1]) * self.Y_ph[i])
            self.year.append(t_0 + i + 1)

        "Calibration ams"

        def ces_weights_ams(p):
            omega_agr, omega_man = p
            GAP_init = np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]
            l_agr = (
                omega_agr * (GAP_init**self.eps_agr) * (GAP_init ** (self.sigma - 1))
            )
            l_man = omega_man * GAP_init * (GAP_init ** (self.sigma - 1))
            l_ser = (
                (1 - omega_agr - omega_man)
                * (GAP_init**self.eps_ser)
                * (GAP_init ** (self.sigma - 1))
            )
            omegas = (
                l_agr / (l_agr + l_man + l_ser) - np.array(self.share_agr)[0],
                l_man / (l_agr + l_man + l_ser) - np.array(self.share_man)[0],
            )
            return np.reshape(omegas, (2,))

        self.om_agr_ams, self.om_man_ams = fsolve(ces_weights_ams, (0.5, 0.5))
        self.om_ser_ams = 1 - self.om_agr_ams - self.om_man_ams

        "Calibration nps"

        def ces_weights_nps(p):
            omega_agr, omega_man, omega_trd, omega_bss, omega_fin = p
            GAP_init = np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]
            l_agr = (
                omega_agr * (GAP_init**self.eps_agr) * (GAP_init ** (self.sigma - 1))
            )
            l_man = omega_man * GAP_init * (GAP_init ** (self.sigma - 1))
            l_trd = (
                omega_trd * (GAP_init**self.eps_trd) * (GAP_init ** (self.sigma - 1))
            )
            l_bss = (
                omega_bss * (GAP_init**self.eps_bss) * (GAP_init ** (self.sigma - 1))
            )
            l_fin = (
                omega_fin * (GAP_init**self.eps_fin) * (GAP_init ** (self.sigma - 1))
            )
            l_nps = (
                (1 - omega_agr - omega_man - omega_trd - omega_bss - omega_fin)
                * (GAP_init**self.eps_nps)
                * (GAP_init ** (self.sigma - 1))
            )
            omegas = (
                l_agr / (l_agr + l_man + l_trd + l_bss + l_fin + l_nps)
                - np.array(self.share_agr)[0],
                l_man / (l_agr + l_man + l_trd + l_bss + l_fin + l_nps)
                - np.array(self.share_man)[0],
                l_trd / (l_agr + l_man + l_trd + l_bss + l_fin + l_nps)
                - np.array(self.share_trd)[0],
                l_bss / (l_agr + l_man + l_trd + l_bss + l_fin + l_nps)
                - np.array(self.share_bss)[0],
                l_fin / (l_agr + l_man + l_trd + l_bss + l_fin + l_nps)
                - np.array(self.share_fin)[0],
            )
            return np.reshape(omegas, (5,))

        (
            self.om_agr_nps,
            self.om_man_nps,
            self.om_trd_nps,
            self.om_bss_nps,
            self.om_fin_nps,
        ) = fsolve(ces_weights_nps, (0.5, 0.5, 0.05, 0.05, 0.05))
        self.om_nps_nps = (
            1
            - self.om_agr_nps
            - self.om_man_nps
            - self.om_trd_nps
            - self.om_bss_nps
            - self.om_fin_nps
        )

        # Compute x_man_q_index from growth rates of x_man_q, starting at x_man_E[0]
        sector_list = ["agr", "man", "trd", "bss", "fin", "nps", "ser", "tot"]

        # Build x_q_series and x_E_series from stored attributes
        x_q_series = {
            "agr": np.array(self.x_agr_q),
            "man": np.array(self.x_man_q),
            "trd": np.array(self.x_trd_q),
            "bss": np.array(self.x_bss_q),
            "fin": np.array(self.x_fin_q),
            "nps": np.array(self.x_nps_q),
            "ser": np.array(self.x_ser_q),
            "tot": np.array(self.x_tot_q),
        }
        x_E_series = {
            "agr": np.array(self.x_agr_E),
            "man": np.array(self.x_man_E),
            "trd": np.array(self.x_trd_E),
            "bss": np.array(self.x_bss_E),
            "fin": np.array(self.x_fin_E),
            "nps": np.array(self.x_nps_E),
            "ser": np.array(self.x_ser_E),
            "tot": np.array(self.x_tot_E),
        }

        x_q_index = {}
        for sec in sector_list:
            x_q_arr = x_q_series[sec]
            x_E0 = x_E_series[sec][0]
            idx = [x_E0]
            for i in range(1, len(x_q_arr)):
                g = x_q_arr[i] / x_q_arr[i - 1] if x_q_arr[i - 1] != 0 else 1
                idx.append(idx[-1] * g)
            x_q_index[sec] = np.array(idx)

        # Compute m_{sec}_q_index from growth rates of m_{sec}_q, starting at x_{sec}_E[0]
        m_q_series = {
            "agr": np.array(self.m_agr_q),
            "man": np.array(self.m_man_q),
            "trd": np.array(self.m_trd_q),
            "bss": np.array(self.m_bss_q),
            "fin": np.array(self.m_fin_q),
            "nps": np.array(self.m_nps_q),
            "ser": np.array(self.m_ser_q),
            "tot": np.array(self.m_tot_q),
        }

        m_q_index = {}
        for sec in sector_list:
            m_q_arr = m_q_series[sec]
            x_E0 = x_E_series[sec][0]
            idx = [x_E0]
            for i in range(1, len(m_q_arr)):
                g = m_q_arr[i] / m_q_arr[i - 1] if m_q_arr[i - 1] != 0 else 1
                idx.append(idx[-1] * g)
            m_q_index[sec] = np.array(idx)

        self.x_agr_q_index = np.array(x_q_index["agr"])
        self.x_man_q_index = np.array(x_q_index["man"])
        self.x_trd_q_index = np.array(x_q_index["trd"])
        self.x_bss_q_index = np.array(x_q_index["bss"])
        self.x_fin_q_index = np.array(x_q_index["fin"])
        self.x_nps_q_index = np.array(x_q_index["nps"])
        self.x_ser_q_index = np.array(x_q_index["ser"])

        self.m_agr_q_index = np.array(m_q_index["agr"])
        self.m_man_q_index = np.array(m_q_index["man"])
        self.m_trd_q_index = np.array(m_q_index["trd"])
        self.m_bss_q_index = np.array(m_q_index["bss"])
        self.m_fin_q_index = np.array(m_q_index["fin"])
        self.m_nps_q_index = np.array(m_q_index["nps"])
        self.m_ser_q_index = np.array(m_q_index["ser"])

        self.nx_agr_q_index = self.x_agr_q_index - self.m_agr_q_index
        self.nx_man_q_index = self.x_man_q_index - self.m_man_q_index
        self.nx_trd_q_index = self.x_trd_q_index - self.m_trd_q_index
        self.nx_bss_q_index = self.x_bss_q_index - self.m_bss_q_index
        self.nx_fin_q_index = self.x_fin_q_index - self.m_fin_q_index
        self.nx_nps_q_index = self.x_nps_q_index - self.m_nps_q_index
        self.nx_ser_q_index = self.x_ser_q_index - self.m_ser_q_index

        def xi_fit_sector(x_q_index, A):
            """Country-specific recalibration of xi_i. Exactly-identified fit:
            given the initial-period real export level x_0 and the sectoral
            productivity path A_i, pick xi_i so that x_0 * A_i,T^xi_i equals
            the terminal-period data export level. Matches Section 6 of the
            paper (A is the US productivity series imported from Step 7 and
            used uniformly across countries; xi_i therefore absorbs
            country-specific terminal export levels)."""

            def objective(xi):
                model_last = x_q_index[0] * A[-1] ** xi
                data_last = x_q_index[-1]
                return (model_last - data_last) ** 2  # squared error

            res = minimize_scalar(objective, method="bounded", bounds=(0, 10))
            return res.x  # returns xi

        self.xi_agr = xi_fit_sector(self.x_agr_q_index, A_agr)
        self.xi_man = xi_fit_sector(self.x_man_q_index, A_man)
        self.xi_trd = xi_fit_sector(self.x_trd_q_index, A_trd)
        self.xi_bss = xi_fit_sector(self.x_bss_q_index, A_bss)
        self.xi_fin = xi_fit_sector(self.x_fin_q_index, A_fin)
        self.xi_nps = xi_fit_sector(self.x_nps_q_index, A_nps)
        self.xi_ser = xi_fit_sector(self.x_ser_q_index, A_ser)

        "Country-level non-homothetic CES index (endogenous-trade form)."
        "Uses (l_i - nx_i)/(l_m - nx_m), i.e. relative domestic absorption"
        "rather than relative employment, to neutralize the trade wedge in"
        "the relative-labor identity. Initialized at the US-relative GDP"
        "level to keep C and the productivity A_i paths in comparable units."

        def C_index(om_i, li, xn_i, lm, xn_m, pi_pm, sigma, epsilon_i):
            C_level = (
                (self.om_man_ams / om_i)
                * ((li - xn_i) / (lm - xn_m))
                * (pi_pm ** (sigma - 1))
            ) ** (1 / (epsilon_i - 1))
            g_C = np.array(C_level / C_level.shift(1) - 1)
            C = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
            for i in range(len(g_C) - 1):
                C.append((1 + g_C[i + 1]) * C[i])
            return C

        self.C = C_index(
            (1 - self.om_agr_ams - self.om_man_ams),
            self.h_ser,
            self.nx_ser_q_index,
            self.h_man,
            self.nx_man_q_index,
            self.p_ser / self.p_man,
            self.sigma,
            self.eps_ser,
        )

        self.C_closed = C_index(
            (1 - self.om_agr_ams - self.om_man_ams),
            self.h_ser,
            0,
            self.h_man,
            0,
            self.p_ser / self.p_man,
            self.sigma,
            self.eps_ser,
        )

    """
    -------------------------------
            Open Economy Models
    -------------------------------
    """

    def labor_agr(self, C, E, A_agr, M_agr_E):
        "Labor Demand in Agriculture"
        return self.om_agr_ams * (C**self.eps_agr) * (A_agr ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.E[0] * self.M_agr_E[0] / (self.A_agr[0] ** (self.xi_agr - 1)))
            * (A_agr ** (self.xi_agr - 1))
            / E
            - M_agr_E
        )

    def labor_man(self, C, E, A_man, M_man_E):
        "Labor Demand in Manufacturing"
        return self.om_man_ams * C * (A_man ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.E[0] * self.M_man_E[0] / (self.A_man[0] ** (self.xi_man - 1)))
            * (A_man ** (self.xi_man - 1))
            / E
            - M_man_E
        )

    def labor_ser(self, C, E, A_ser, M_ser_E):
        "Labor Demand in Services"
        return self.om_ser_ams * (C**self.eps_ser) * (A_ser ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.E[0] * self.M_ser_E[0] / (self.A_ser[0] ** (self.xi_ser - 1)))
            * (A_ser ** (self.xi_ser - 1))
            / E
            - M_ser_E
        )

    def agg_labor_demand_ams(
        self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
    ):
        "Aggregate Labor Demand"
        return (
            self.labor_agr(C, E, A_agr, M_agr_E)
            + self.labor_man(C, E, A_man, M_man_E)
            + self.labor_ser(C, E, A_ser, M_ser_E)
        )

    def share_agr_ams(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Employment Share in Agriculture"
        return self.labor_agr(C, E, A_agr, M_agr_E) / self.agg_labor_demand_ams(
            C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
        )

    def share_man_ams(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Employment Share in Manufacturing"
        return self.labor_man(C, E, A_man, M_man_E) / self.agg_labor_demand_ams(
            C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
        )

    def share_ser_ams(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Employment Share in Services"
        return self.labor_ser(C, E, A_ser, M_ser_E) / self.agg_labor_demand_ams(
            C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
        )

    def labor_trd(self, C, E, A_trd, M_trd_E):
        "Labor Demand in Trade Services"
        return self.om_trd_nps * (C**self.eps_trd) * (A_trd ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.M_trd_E[0] / (self.A_trd[0] ** (self.xi_trd - 1)))
            * (A_trd ** (self.xi_trd - 1))
            / E
            - M_trd_E
        )

    def labor_bss(self, C, E, A_bss, M_bss_E):
        "Labor Demand in Business Services"
        return self.om_bss_nps * (C**self.eps_bss) * (A_bss ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.E[0] * self.M_bss_E[0] / (self.A_bss[0] ** (self.xi_bss - 1)))
            * (A_bss ** (self.xi_bss - 1))
            / E
            - M_bss_E
        )

    def labor_fin(self, C, E, A_fin, M_fin_E):
        "Labor Demand in Financial Services"
        return self.om_fin_nps * (C**self.eps_fin) * (A_fin ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.E[0] * self.M_fin_E[0] / (self.A_fin[0] ** (self.xi_fin - 1)))
            * (A_fin ** (self.xi_fin - 1))
            / E
            - M_fin_E
        )

    def labor_nps(self, C, E, A_nps, M_nps_E):
        "Labor Demand in Non-Progressive Services"
        return self.om_nps_nps * (C**self.eps_nps) * (A_nps ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * (
            (self.E[0] * self.M_nps_E[0] / (self.A_nps[0] ** (self.xi_nps - 1)))
            * (A_nps ** (self.xi_nps - 1))
            / E
            - M_nps_E
        )

    def agg_labor_demand_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Aggregate Labor Demand"
        return (
            self.labor_agr(C, E, A_agr, M_agr_E)
            + self.labor_man(C, E, A_man, M_man_E)
            + self.labor_trd(C, E, A_trd, M_trd_E)
            + self.labor_bss(C, E, A_bss, M_bss_E)
            + self.labor_fin(C, E, A_fin, M_fin_E)
            + self.labor_nps(C, E, A_nps, M_nps_E)
        )

    def share_agr_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Employment Share in Agriculture"
        return self.labor_agr(C, E, A_agr, M_agr_E) / self.agg_labor_demand_nps(
            C,
            E,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
            M_agr_E,
            M_man_E,
            M_trd_E,
            M_bss_E,
            M_fin_E,
            M_nps_E,
        )

    def share_man_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Employment Share in Manufacturing"
        return self.labor_man(C, E, A_man, M_man_E) / self.agg_labor_demand_nps(
            C,
            E,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
            M_agr_E,
            M_man_E,
            M_trd_E,
            M_bss_E,
            M_fin_E,
            M_nps_E,
        )

    def share_trd_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Employment Share in Trade Services"
        return self.labor_trd(C, E, A_trd, M_trd_E) / self.agg_labor_demand_nps(
            C,
            E,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
            M_agr_E,
            M_man_E,
            M_trd_E,
            M_bss_E,
            M_fin_E,
            M_nps_E,
        )

    def share_bss_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Employment Share in Business Services"
        return self.labor_bss(C, E, A_bss, M_bss_E) / self.agg_labor_demand_nps(
            C,
            E,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
            M_agr_E,
            M_man_E,
            M_trd_E,
            M_bss_E,
            M_fin_E,
            M_nps_E,
        )

    def share_fin_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Employment Share in Financial Services"
        return self.labor_fin(C, E, A_fin, M_fin_E) / self.agg_labor_demand_nps(
            C,
            E,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
            M_agr_E,
            M_man_E,
            M_trd_E,
            M_bss_E,
            M_fin_E,
            M_nps_E,
        )

    def share_nps_nps(
        self,
        C,
        E,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
        M_agr_E,
        M_man_E,
        M_trd_E,
        M_bss_E,
        M_fin_E,
        M_nps_E,
    ):
        "Employment Share in Non-Progressive Services"
        return self.labor_nps(C, E, A_nps, M_nps_E) / self.agg_labor_demand_nps(
            C,
            E,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
            M_agr_E,
            M_man_E,
            M_trd_E,
            M_bss_E,
            M_fin_E,
            M_nps_E,
        )

    """
    -------------------------------
        Closed Economy Models
    -------------------------------
    """

    def labor_agr_closed(self, C, A_agr):
        "Labor Demand in Agriculture"
        return self.om_agr_ams * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))

    def labor_man_closed(self, C, A_man):
        "Labor Demand in Manufacturing"
        return self.om_man_ams * C * (A_man ** (self.sigma - 1))

    def labor_ser_closed(self, C, A_ser):
        "Labor Demand in Services"
        return self.om_ser_ams * (C**self.eps_ser) * (A_ser ** (self.sigma - 1))

    def agg_labor_demand_ams_closed(self, C, A_agr, A_man, A_ser):
        "Aggregate Labor Demand"
        return (
            self.labor_agr_closed(C, A_agr)
            + self.labor_man_closed(C, A_man)
            + self.labor_ser_closed(C, A_ser)
        )

    def share_agr_ams_closed(self, C, A_agr, A_man, A_ser):
        "Employment Share in Agriculture"
        return self.labor_agr_closed(C, A_agr) / self.agg_labor_demand_ams_closed(
            C, A_agr, A_man, A_ser
        )

    def share_man_ams_closed(self, C, A_agr, A_man, A_ser):
        "Employment Share in Manufacturing"
        return self.labor_man_closed(C, A_man) / self.agg_labor_demand_ams_closed(
            C, A_agr, A_man, A_ser
        )

    def share_ser_ams_closed(self, C, A_agr, A_man, A_ser):
        "Employment Share in Services"
        return self.labor_ser_closed(C, A_ser) / self.agg_labor_demand_ams_closed(
            C, A_agr, A_man, A_ser
        )

    def labor_trd_closed(self, C, A_trd):
        "Labor Demand in Trade Services"
        return self.om_trd_nps * (C**self.eps_trd) * (A_trd ** (self.sigma - 1))

    def labor_bss_closed(self, C, A_bss):
        "Labor Demand in Business Services"
        return self.om_bss_nps * (C**self.eps_bss) * (A_bss ** (self.sigma - 1))

    def labor_fin_closed(self, C, A_fin):
        "Labor Demand in Financial Services"
        return self.om_fin_nps * (C**self.eps_fin) * (A_fin ** (self.sigma - 1))

    def labor_nps_closed(self, C, A_nps):
        "Labor Demand in Non-Progressive Services"
        return self.om_nps_nps * (C**self.eps_nps) * (A_nps ** (self.sigma - 1))

    def agg_labor_demand_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Aggregate Labor Demand"
        return (
            self.labor_agr_closed(C, A_agr)
            + self.labor_man_closed(C, A_man)
            + self.labor_trd_closed(C, A_trd)
            + self.labor_bss_closed(C, A_bss)
            + self.labor_fin_closed(C, A_fin)
            + self.labor_nps_closed(C, A_nps)
        )

    def share_agr_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Employment Share in Agriculture"
        return self.labor_agr_closed(C, A_agr) / self.agg_labor_demand_nps_closed(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_man_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Employment Share in Manufacturing"
        return self.labor_man_closed(C, A_man) / self.agg_labor_demand_nps_closed(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_trd_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Employment Share in Trade Services"
        return self.labor_trd_closed(C, A_trd) / self.agg_labor_demand_nps_closed(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_bss_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Employment Share in Business Services"
        return self.labor_bss_closed(C, A_bss) / self.agg_labor_demand_nps_closed(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_fin_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Employment Share in Financial Services"
        return self.labor_fin_closed(C, A_fin) / self.agg_labor_demand_nps_closed(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_nps_nps_closed(
        self,
        C,
        A_agr,
        A_man,
        A_trd,
        A_bss,
        A_fin,
        A_nps,
    ):
        "Employment Share in Non-Progressive Services"
        return self.labor_nps_closed(C, A_nps) / self.agg_labor_demand_nps_closed(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    """
    ------------------------
        Model Predictions
    ------------------------
    """

    "Productivity Time Indexes"

    def productivity_series(self):
        "First period. Normalization"
        self.A_agr = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_man = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_trd = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_bss = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_fin = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_nps = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_ser = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]
        self.A_tot = [np.array(self.GDP_ph)[0] / np.array(USA_GDP_ph)[0]]

        "Productivity and Real Expenditure Growth"
        for i in range(int(self.ts_length)):
            self.A_agr.append((1 + self.g_y_l_agr[i + 1]) * self.A_agr[i])
            self.A_man.append((1 + self.g_y_l_man[i + 1]) * self.A_man[i])
            self.A_trd.append((1 + self.g_y_l_trd[i + 1]) * self.A_trd[i])
            self.A_bss.append((1 + self.g_y_l_bss[i + 1]) * self.A_bss[i])
            self.A_fin.append((1 + self.g_y_l_fin[i + 1]) * self.A_fin[i])
            self.A_nps.append((1 + self.g_y_l_nps[i + 1]) * self.A_nps[i])
            self.A_ser.append((1 + self.g_y_l_ser[i + 1]) * self.A_ser[i])
            self.A_tot.append((1 + self.g_y_l_tot[i + 1]) * self.A_tot[i])

    def predictions_ams(self):
        "Model Implied Employment Shares: Open Economy"
        self.share_agr_ams_m = [
            self.share_agr_ams(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_ser[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_ser_E[0],
            )
        ]
        self.share_man_ams_m = [
            self.share_man_ams(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_ser[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_ser_E[0],
            )
        ]
        self.share_ser_ams_m = [
            self.share_ser_ams(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_ser[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_ser_E[0],
            )
        ]
        for i in range(int(self.ts_length)):
            self.share_agr_ams_m.append(
                self.share_agr_ams(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_ser[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_ser_E[i + 1],
                )
            )
            self.share_man_ams_m.append(
                self.share_man_ams(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_ser[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_ser_E[i + 1],
                )
            )
            self.share_ser_ams_m.append(
                self.share_ser_ams(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_ser[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_ser_E[i + 1],
                )
            )

        "Aggregate Productivity: Open Economy"
        weighted_ams_A_agr = [a * b for a, b in zip(self.share_agr_ams_m, self.A_agr)]
        weighted_ams_A_man = [a * b for a, b in zip(self.share_man_ams_m, self.A_man)]
        weighted_ams_A_ser = [a * b for a, b in zip(self.share_ser_ams_m, self.A_ser)]
        self.A_tot_ams = [
            sum(x)
            for x in zip(weighted_ams_A_agr, weighted_ams_A_man, weighted_ams_A_ser)
        ]

        "Model Implied Employment Shares: Closed Economy"
        self.share_agr_ams_m_closed = [
            self.share_agr_ams_closed(
                self.C_closed[0], self.A_agr[0], self.A_man[0], self.A_ser[0]
            )
        ]
        self.share_man_ams_m_closed = [
            self.share_man_ams_closed(
                self.C_closed[0], self.A_agr[0], self.A_man[0], self.A_ser[0]
            )
        ]
        self.share_ser_ams_m_closed = [
            self.share_ser_ams_closed(
                self.C_closed[0], self.A_agr[0], self.A_man[0], self.A_ser[0]
            )
        ]
        for i in range(int(self.ts_length)):
            self.share_agr_ams_m_closed.append(
                self.share_agr_ams_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_ser[i + 1],
                )
            )
            self.share_man_ams_m_closed.append(
                self.share_man_ams_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_ser[i + 1],
                )
            )
            self.share_ser_ams_m_closed.append(
                self.share_ser_ams_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_ser[i + 1],
                )
            )

        "Aggregate Productivity: Closed Economy"
        weighted_ams_A_agr_closed = [
            a * b for a, b in zip(self.share_agr_ams_m_closed, self.A_agr)
        ]
        weighted_ams_A_man_closed = [
            a * b for a, b in zip(self.share_man_ams_m_closed, self.A_man)
        ]
        weighted_ams_A_ser_closed = [
            a * b for a, b in zip(self.share_ser_ams_m_closed, self.A_ser)
        ]
        self.A_tot_ams_closed = [
            sum(x)
            for x in zip(
                weighted_ams_A_agr_closed,
                weighted_ams_A_man_closed,
                weighted_ams_A_ser_closed,
            )
        ]

    "nps"

    def predictions_nps(self):
        "Model Implied Emloyment Shares: Open Economy"
        self.share_agr_nps_m = [
            self.share_agr_nps(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_trd_E[0],
                self.M_bss_E[0],
                self.M_fin_E[0],
                self.M_nps_E[0],
            )
        ]
        self.share_man_nps_m = [
            self.share_man_nps(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_trd_E[0],
                self.M_bss_E[0],
                self.M_fin_E[0],
                self.M_nps_E[0],
            )
        ]
        self.share_trd_nps_m = [
            self.share_trd_nps(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_trd_E[0],
                self.M_bss_E[0],
                self.M_fin_E[0],
                self.M_nps_E[0],
            )
        ]
        self.share_bss_nps_m = [
            self.share_bss_nps(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_trd_E[0],
                self.M_bss_E[0],
                self.M_fin_E[0],
                self.M_nps_E[0],
            )
        ]
        self.share_fin_nps_m = [
            self.share_fin_nps(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_trd_E[0],
                self.M_bss_E[0],
                self.M_fin_E[0],
                self.M_nps_E[0],
            )
        ]
        self.share_nps_nps_m = [
            self.share_nps_nps(
                self.C[0],
                self.E[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
                self.M_agr_E[0],
                self.M_man_E[0],
                self.M_trd_E[0],
                self.M_bss_E[0],
                self.M_fin_E[0],
                self.M_nps_E[0],
            )
        ]
        for i in range(int(self.ts_length)):
            self.share_agr_nps_m.append(
                self.share_agr_nps(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_trd_E[i + 1],
                    self.M_bss_E[i + 1],
                    self.M_fin_E[i + 1],
                    self.M_nps_E[i + 1],
                )
            )
            self.share_man_nps_m.append(
                self.share_man_nps(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_trd_E[i + 1],
                    self.M_bss_E[i + 1],
                    self.M_fin_E[i + 1],
                    self.M_nps_E[i + 1],
                )
            )
            self.share_trd_nps_m.append(
                self.share_trd_nps(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_trd_E[i + 1],
                    self.M_bss_E[i + 1],
                    self.M_fin_E[i + 1],
                    self.M_nps_E[i + 1],
                )
            )
            self.share_bss_nps_m.append(
                self.share_bss_nps(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_trd_E[i + 1],
                    self.M_bss_E[i + 1],
                    self.M_fin_E[i + 1],
                    self.M_nps_E[i + 1],
                )
            )
            self.share_fin_nps_m.append(
                self.share_fin_nps(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_trd_E[i + 1],
                    self.M_bss_E[i + 1],
                    self.M_fin_E[i + 1],
                    self.M_nps_E[i + 1],
                )
            )
            self.share_nps_nps_m.append(
                self.share_nps_nps(
                    self.C[i + 1],
                    self.E[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                    self.M_agr_E[i + 1],
                    self.M_man_E[i + 1],
                    self.M_trd_E[i + 1],
                    self.M_bss_E[i + 1],
                    self.M_fin_E[i + 1],
                    self.M_nps_E[i + 1],
                )
            )

        "Aggregate Productivity: Open Economy"
        weighted_nps_A_agr = [a * b for a, b in zip(self.share_agr_nps_m, self.A_agr)]
        weighted_nps_A_man = [a * b for a, b in zip(self.share_man_nps_m, self.A_man)]
        weighted_nps_A_trd = [a * b for a, b in zip(self.share_trd_nps_m, self.A_trd)]
        weighted_nps_A_bss = [a * b for a, b in zip(self.share_bss_nps_m, self.A_bss)]
        weighted_nps_A_fin = [a * b for a, b in zip(self.share_fin_nps_m, self.A_fin)]
        weighted_nps_A_nps = [a * b for a, b in zip(self.share_nps_nps_m, self.A_nps)]
        self.A_tot_nps = [
            sum(x)
            for x in zip(
                weighted_nps_A_agr,
                weighted_nps_A_man,
                weighted_nps_A_trd,
                weighted_nps_A_bss,
                weighted_nps_A_fin,
                weighted_nps_A_nps,
            )
        ]

        "Model Implied Emloyment Shares: Closed Economy"
        self.share_agr_nps_m_closed = [
            self.share_agr_nps_closed(
                self.C_closed[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
            )
        ]
        self.share_man_nps_m_closed = [
            self.share_man_nps_closed(
                self.C_closed[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
            )
        ]
        self.share_trd_nps_m_closed = [
            self.share_trd_nps_closed(
                self.C_closed[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
            )
        ]
        self.share_bss_nps_m_closed = [
            self.share_bss_nps_closed(
                self.C_closed[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
            )
        ]
        self.share_fin_nps_m_closed = [
            self.share_fin_nps_closed(
                self.C_closed[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
            )
        ]
        self.share_nps_nps_m_closed = [
            self.share_nps_nps_closed(
                self.C_closed[0],
                self.A_agr[0],
                self.A_man[0],
                self.A_trd[0],
                self.A_bss[0],
                self.A_fin[0],
                self.A_nps[0],
            )
        ]
        for i in range(int(self.ts_length)):
            self.share_agr_nps_m_closed.append(
                self.share_agr_nps_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                )
            )
            self.share_man_nps_m_closed.append(
                self.share_man_nps_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                )
            )
            self.share_trd_nps_m_closed.append(
                self.share_trd_nps_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                )
            )
            self.share_bss_nps_m_closed.append(
                self.share_bss_nps_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                )
            )
            self.share_fin_nps_m_closed.append(
                self.share_fin_nps_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                )
            )
            self.share_nps_nps_m_closed.append(
                self.share_nps_nps_closed(
                    self.C_closed[i + 1],
                    self.A_agr[i + 1],
                    self.A_man[i + 1],
                    self.A_trd[i + 1],
                    self.A_bss[i + 1],
                    self.A_fin[i + 1],
                    self.A_nps[i + 1],
                )
            )

        "Aggregate Productivity: Closed Economy"
        weighted_nps_A_agr_closed = [
            a * b for a, b in zip(self.share_agr_nps_m_closed, self.A_agr)
        ]
        weighted_nps_A_man_closed = [
            a * b for a, b in zip(self.share_man_nps_m_closed, self.A_man)
        ]
        weighted_nps_A_trd_closed = [
            a * b for a, b in zip(self.share_trd_nps_m_closed, self.A_trd)
        ]
        weighted_nps_A_bss_closed = [
            a * b for a, b in zip(self.share_bss_nps_m_closed, self.A_bss)
        ]
        weighted_nps_A_fin_closed = [
            a * b for a, b in zip(self.share_fin_nps_m_closed, self.A_fin)
        ]
        weighted_nps_A_nps_closed = [
            a * b for a, b in zip(self.share_nps_nps_m_closed, self.A_nps)
        ]
        self.A_tot_nps_closed = [
            sum(x)
            for x in zip(
                weighted_nps_A_agr_closed,
                weighted_nps_A_man_closed,
                weighted_nps_A_trd_closed,
                weighted_nps_A_bss_closed,
                weighted_nps_A_fin_closed,
                weighted_nps_A_nps_closed,
            )
        ]


"""
-----------------------------------------
    Predictions for European Countries
-----------------------------------------
"""

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

IRL = model_country("IRL")
IRL.productivity_series()
IRL.predictions_ams()
IRL.predictions_nps()

ITA = model_country("ITA")
ITA.productivity_series()
ITA.predictions_ams()
ITA.predictions_nps()

LUX = model_country("LUX")
LUX.productivity_series()
LUX.predictions_ams()
LUX.predictions_nps()

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

plt.plot(AUT.A_tot)
plt.plot(AUT.A_tot_nps, "--")
plt.plot(AUT.A_tot_nps_closed, "-.")
plt.title("AUT")
plt.show()

plt.plot(BEL.A_tot)
plt.plot(BEL.A_tot_nps, "--")
plt.plot(BEL.A_tot_nps_closed, "-.")
plt.title("BEL")
plt.show()

plt.plot(DEU.A_tot)
plt.plot(DEU.A_tot_nps, "--")
plt.plot(DEU.A_tot_nps_closed, "-.")
plt.title("DEU")
plt.show()

plt.plot(DNK.A_tot)
plt.plot(DNK.A_tot_nps, "--")
plt.plot(DNK.A_tot_nps_closed, "-.")
plt.title("DNK")
plt.show()

plt.plot(ESP.A_tot)
plt.plot(ESP.A_tot_nps, "--")
plt.plot(ESP.A_tot_nps_closed, "-.")
plt.title("ESP")
plt.show()

plt.plot(FIN.A_tot)
plt.plot(FIN.A_tot_nps, "--")
plt.plot(FIN.A_tot_nps_closed, "-.")
plt.title("FIN")
plt.show()

plt.plot(FRA.A_tot)
plt.plot(FRA.A_tot_nps, "--")
plt.plot(FRA.A_tot_nps_closed, "-.")
plt.title("FRA")
plt.show()

plt.plot(GBR.A_tot)
plt.plot(GBR.A_tot_nps, "--")
plt.plot(GBR.A_tot_nps_closed, "-.")
plt.title("GBR")
plt.show()

plt.plot(GRC.A_tot)
plt.plot(GRC.A_tot_nps, "--")
plt.plot(GRC.A_tot_nps_closed, "-.")
plt.title("GRC")
plt.show()

plt.plot(IRL.A_tot)
plt.plot(IRL.A_tot_nps, "--")
plt.plot(IRL.A_tot_nps_closed, "-.")
plt.title("IRL")
plt.show()

plt.plot(ITA.A_tot)
plt.plot(ITA.A_tot_nps, "--")
plt.plot(ITA.A_tot_nps_closed, "-.")
plt.title("ITA")
plt.show()

plt.plot(LUX.A_tot)
plt.plot(LUX.A_tot_nps, "--")
plt.plot(LUX.A_tot_nps_closed, "-.")
plt.title("LUX")
plt.show()

plt.plot(NLD.year, NLD.A_tot)
plt.plot(NLD.year, NLD.A_tot_nps, "--")
plt.plot(NLD.year, NLD.A_tot_nps_closed, "-.")
plt.title("NLD")
plt.show()

plt.plot(PRT.A_tot)
plt.plot(PRT.A_tot_nps, "--")
plt.plot(PRT.A_tot_nps_closed, "-.")
plt.title("PRT")
plt.show()

plt.plot(SWE.A_tot)
plt.plot(SWE.A_tot_nps, "--")
plt.plot(SWE.A_tot_nps_closed, "-.")
plt.title("SWE")
plt.show()


plt.plot(DEU.year, DEU.share_agr, "g-", label="agr")
plt.plot(DEU.year, DEU.share_agr_ams_m, "g--")
plt.plot(DEU.year, DEU.share_agr_ams_m_closed, "g-.")

plt.plot(DEU.year, DEU.share_man, "b-", label="man")
plt.plot(DEU.year, DEU.share_man_ams_m, "b--")
plt.plot(DEU.year, DEU.share_man_ams_m_closed, "b-.")

plt.plot(DEU.year, DEU.share_ser, "r-", label="ser")
plt.plot(DEU.year, DEU.share_ser_ams_m, "r--")
plt.plot(DEU.year, DEU.share_ser_ams_m_closed, "r-.")

plt.title("DEU")
plt.legend()
plt.grid()
plt.show()

plt.plot(DEU.year, DEU.share_agr, "g-", label="agr")
plt.plot(DEU.year, DEU.share_agr_nps_m, "g--")
plt.plot(DEU.year, DEU.share_agr_nps_m_closed, "g-.")

plt.plot(DEU.year, DEU.share_man, "b-", label="man")
plt.plot(DEU.year, DEU.share_man_nps_m, "b--")
plt.plot(DEU.year, DEU.share_man_nps_m_closed, "b-.")

plt.plot(DEU.year, DEU.share_trd, "r-", label="trd")
plt.plot(DEU.year, DEU.share_trd_nps_m, "r--")
plt.plot(DEU.year, DEU.share_trd_nps_m_closed, "r-.")

plt.plot(DEU.year, DEU.share_bss, "c-", label="bss")
plt.plot(DEU.year, DEU.share_bss_nps_m, "c--")
plt.plot(DEU.year, DEU.share_bss_nps_m_closed, "c-.")

plt.plot(DEU.year, DEU.share_fin, "m-", label="fin")
plt.plot(DEU.year, DEU.share_fin_nps_m, "m--")
plt.plot(DEU.year, DEU.share_fin_nps_m_closed, "m-.")


plt.plot(DEU.year, DEU.share_nps, "y-", label="nps")
plt.plot(DEU.year, DEU.share_nps_nps_m, "y--")
plt.plot(DEU.year, DEU.share_nps_nps_m_closed, "y-.")

plt.title("DEU")
plt.legend()
plt.grid()
plt.show()

plt.plot(PRT.year, PRT.share_agr, "g-", label="agr")
plt.plot(PRT.year, PRT.share_agr_ams_m, "g--")
plt.plot(PRT.year, PRT.share_agr_ams_m_closed, "g-.")

plt.plot(PRT.year, PRT.share_man, "b-", label="man")
plt.plot(PRT.year, PRT.share_man_ams_m, "b--")
plt.plot(PRT.year, PRT.share_man_ams_m_closed, "b-.")

plt.plot(PRT.year, PRT.share_ser, "r-", label="ser")
plt.plot(PRT.year, PRT.share_ser_ams_m, "r--")
plt.plot(PRT.year, PRT.share_ser_ams_m_closed, "r-.")

plt.title("PRT")
plt.legend()
plt.grid()
plt.show()


plt.plot(PRT.year, PRT.share_agr, "g-", label="agr")
plt.plot(PRT.year, PRT.share_agr_nps_m, "g--")
plt.plot(PRT.year, PRT.share_agr_nps_m_closed, "g-.")

plt.plot(PRT.year, PRT.share_man, "b-", label="man")
plt.plot(PRT.year, PRT.share_man_nps_m, "b--")
plt.plot(PRT.year, PRT.share_man_nps_m_closed, "b-.")

plt.plot(PRT.year, PRT.share_trd, "r-", label="trd")
plt.plot(PRT.year, PRT.share_trd_nps_m, "r--")
plt.plot(PRT.year, PRT.share_trd_nps_m_closed, "r-.")

plt.plot(PRT.year, PRT.share_bss, "c-", label="bss")
plt.plot(PRT.year, PRT.share_bss_nps_m, "c--")
plt.plot(PRT.year, PRT.share_bss_nps_m_closed, "c-.")

plt.plot(PRT.year, PRT.share_fin, "m-", label="fin")
plt.plot(PRT.year, PRT.share_fin_nps_m, "m--")
plt.plot(PRT.year, PRT.share_fin_nps_m_closed, "m-.")


plt.plot(PRT.year, PRT.share_nps, "y-", label="nps")
plt.plot(PRT.year, PRT.share_nps_nps_m, "y--")
plt.plot(PRT.year, PRT.share_nps_nps_m_closed, "y-.")

plt.title("PRT")
plt.legend()
plt.grid()
plt.show()

plt.plot(GBR.year, GBR.share_agr, "g-", label="agr")
plt.plot(GBR.year, GBR.share_agr_ams_m, "g--")
plt.plot(GBR.year, GBR.share_agr_ams_m_closed, "g-.")

plt.plot(GBR.year, GBR.share_man, "b-", label="man")
plt.plot(GBR.year, GBR.share_man_ams_m, "b--")
plt.plot(GBR.year, GBR.share_man_ams_m_closed, "b-.")

plt.plot(GBR.year, GBR.share_ser, "r-", label="ser")
plt.plot(GBR.year, GBR.share_ser_ams_m, "r--")
plt.plot(GBR.year, GBR.share_ser_ams_m_closed, "r-.")

plt.title("GBR")
plt.legend()
plt.grid()
plt.show()


plt.plot(GBR.year, GBR.share_agr, "g-", label="agr")
plt.plot(GBR.year, GBR.share_agr_nps_m, "g--")
plt.plot(GBR.year, GBR.share_agr_nps_m_closed, "g-.")

plt.plot(GBR.year, GBR.share_man, "b-", label="man")
plt.plot(GBR.year, GBR.share_man_nps_m, "b--")
plt.plot(GBR.year, GBR.share_man_nps_m_closed, "b-.")

plt.plot(GBR.year, GBR.share_trd, "r-", label="trd")
plt.plot(GBR.year, GBR.share_trd_nps_m, "r--")
plt.plot(GBR.year, GBR.share_trd_nps_m_closed, "r-.")

plt.plot(GBR.year, GBR.share_bss, "c-", label="bss")
plt.plot(GBR.year, GBR.share_bss_nps_m, "c--")
plt.plot(GBR.year, GBR.share_bss_nps_m_closed, "c-.")

plt.plot(GBR.year, GBR.share_fin, "m-", label="fin")
plt.plot(GBR.year, GBR.share_fin_nps_m, "m--")
plt.plot(GBR.year, GBR.share_fin_nps_m_closed, "m-.")


plt.plot(GBR.year, GBR.share_nps, "y-", label="nps")
plt.plot(GBR.year, GBR.share_nps_nps_m, "y--")
plt.plot(GBR.year, GBR.share_nps_nps_m_closed, "y-.")

plt.title("GBR")
plt.legend()
plt.grid()
plt.show()


"Productivity and GDP in Europe"
# We get rid of Louxembourg and Ireland in these aggregates.
EUR4_h_tot = (
    np.array(DEU.h_tot)
    + np.array(FRA.h_tot)
    + np.array(GBR.h_tot)
    + np.array(ITA.h_tot)
)
EURCORE_h_tot = (
    np.array(DEU.h_tot)
    + np.array(FRA.h_tot)
    + np.array(BEL.h_tot)
    + np.array(NLD.h_tot)
    + np.array(DNK.h_tot)
)
EURPERI_h_tot = (
    np.array(GRC.h_tot)
    + np.array(PRT.h_tot)
    + np.array(ESP.h_tot)
    + np.array(ITA.h_tot)
    + np.array(GBR.h_tot)
)
EUR13_h_tot = (
    np.array(AUT.h_tot)
    + np.array(BEL.h_tot)
    + np.array(DEU.h_tot)
    + np.array(DNK.h_tot)
    + np.array(ESP.h_tot)
    + np.array(FIN.h_tot)
    + np.array(FRA.h_tot)
    + np.array(GBR.h_tot)
    + np.array(GRC.h_tot)
    + np.array(ITA.h_tot)
    + np.array(NLD.h_tot)
    + np.array(PRT.h_tot)
    + np.array(SWE.h_tot)
)

EUR4_A_tot = (
    np.array(DEU.h_tot) * np.array(DEU.A_tot)
    + np.array(FRA.h_tot) * np.array(FRA.A_tot)
    + np.array(GBR.h_tot) * np.array(GBR.A_tot)
    + np.array(ITA.h_tot) * np.array(ITA.A_tot)
) / EUR4_h_tot
EURCORE_A_tot = (
    np.array(DEU.h_tot) * np.array(DEU.A_tot)
    + np.array(FRA.h_tot) * np.array(FRA.A_tot)
    + np.array(BEL.h_tot) * np.array(BEL.A_tot)
    + np.array(NLD.h_tot) * np.array(NLD.A_tot)
    + np.array(DNK.h_tot) * np.array(DNK.A_tot)
) / EURCORE_h_tot
EURPERI_A_tot = (
    np.array(GRC.h_tot) * np.array(GRC.A_tot)
    + np.array(PRT.h_tot) * np.array(PRT.A_tot)
    + np.array(ESP.h_tot) * np.array(ESP.A_tot)
    + np.array(ITA.h_tot) * np.array(ITA.A_tot)
    + np.array(GBR.h_tot) * np.array(GBR.A_tot)
) / EURPERI_h_tot
EUR13_A_tot = (
    np.array(AUT.h_tot) * np.array(AUT.A_tot)
    + np.array(BEL.h_tot) * np.array(BEL.A_tot)
    + np.array(DEU.h_tot) * np.array(DEU.A_tot)
    + np.array(DNK.h_tot) * np.array(DNK.A_tot)
    + np.array(ESP.h_tot) * np.array(ESP.A_tot)
    + np.array(FIN.h_tot) * np.array(FIN.A_tot)
    + np.array(FRA.h_tot) * np.array(FRA.A_tot)
    + np.array(GBR.h_tot) * np.array(GBR.A_tot)
    + np.array(GRC.h_tot) * np.array(GRC.A_tot)
    + np.array(ITA.h_tot) * np.array(ITA.A_tot)
    + np.array(NLD.h_tot) * np.array(NLD.A_tot)
    + np.array(PRT.h_tot) * np.array(PRT.A_tot)
    + np.array(SWE.h_tot) * np.array(SWE.A_tot)
) / EUR13_h_tot

EUR4_rel_A_tot = EUR4_A_tot / A_tot
EURCORE_rel_A_tot = EURCORE_A_tot / A_tot
EURPERI_rel_A_tot = EURPERI_A_tot / A_tot
EUR13_rel_A_tot = EUR13_A_tot / A_tot

EUR4_E = (
    np.array(DEU.E) * np.array(DEU.h_tot)
    + np.array(FRA.E) * np.array(FRA.h_tot)
    + np.array(GBR.E) * np.array(GBR.h_tot)
    + np.array(ITA.E) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_E = (
    np.array(DEU.E) * np.array(DEU.h_tot)
    + np.array(FRA.E) * np.array(FRA.h_tot)
    + np.array(BEL.E) * np.array(BEL.h_tot)
    + np.array(NLD.E) * np.array(NLD.h_tot)
    + np.array(DNK.E) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_E = (
    np.array(GRC.E) * np.array(GRC.h_tot)
    + np.array(PRT.E) * np.array(PRT.h_tot)
    + np.array(ESP.E) * np.array(ESP.h_tot)
    + np.array(ITA.E) * np.array(ITA.h_tot)
    + np.array(GBR.E) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_E = (
    np.array(AUT.E) * np.array(AUT.h_tot)
    + np.array(BEL.E) * np.array(BEL.h_tot)
    + np.array(DEU.E) * np.array(DEU.h_tot)
    + np.array(DNK.E) * np.array(DNK.h_tot)
    + np.array(ESP.E) * np.array(ESP.h_tot)
    + np.array(FIN.E) * np.array(FIN.h_tot)
    + np.array(FRA.E) * np.array(FRA.h_tot)
    + np.array(GBR.E) * np.array(GBR.h_tot)
    + np.array(GRC.E) * np.array(GRC.h_tot)
    + np.array(ITA.E) * np.array(ITA.h_tot)
    + np.array(NLD.E) * np.array(NLD.h_tot)
    + np.array(PRT.E) * np.array(PRT.h_tot)
    + np.array(SWE.E) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_rel_E = (
    (np.array(DEU.E) / np.array(E)) * np.array(DEU.h_tot)
    + (np.array(FRA.E) / np.array(E)) * np.array(FRA.h_tot)
    + (np.array(GBR.E) / np.array(E)) * np.array(GBR.h_tot)
    + (np.array(ITA.E) / np.array(E)) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_rel_E = (
    (np.array(DEU.E) / np.array(E)) * np.array(DEU.h_tot)
    + (np.array(FRA.E) / np.array(E)) * np.array(FRA.h_tot)
    + (np.array(BEL.E) / np.array(E)) * np.array(BEL.h_tot)
    + (np.array(NLD.E) / np.array(E)) * np.array(NLD.h_tot)
    + (np.array(DNK.E) / np.array(E)) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_rel_E = (
    (np.array(GRC.E) / np.array(E)) * np.array(GRC.h_tot)
    + (np.array(PRT.E) / np.array(E)) * np.array(PRT.h_tot)
    + (np.array(ESP.E) / np.array(E)) * np.array(ESP.h_tot)
    + (np.array(ITA.E) / np.array(E)) * np.array(ITA.h_tot)
    + (np.array(GBR.E) / np.array(E)) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_rel_E = (
    (np.array(AUT.E) / np.array(E)) * np.array(AUT.h_tot)
    + (np.array(BEL.E) / np.array(E)) * np.array(BEL.h_tot)
    + (np.array(DEU.E) / np.array(E)) * np.array(DEU.h_tot)
    + (np.array(DNK.E) / np.array(E)) * np.array(DNK.h_tot)
    + (np.array(ESP.E) / np.array(E)) * np.array(ESP.h_tot)
    + (np.array(FIN.E) / np.array(E)) * np.array(FIN.h_tot)
    + (np.array(FRA.E) / np.array(E)) * np.array(FRA.h_tot)
    + (np.array(GBR.E) / np.array(E)) * np.array(GBR.h_tot)
    + (np.array(GRC.E) / np.array(E)) * np.array(GRC.h_tot)
    + (np.array(ITA.E) / np.array(E)) * np.array(ITA.h_tot)
    + (np.array(NLD.E) / np.array(E)) * np.array(NLD.h_tot)
    + (np.array(PRT.E) / np.array(E)) * np.array(PRT.h_tot)
    + (np.array(SWE.E) / np.array(E)) * np.array(SWE.h_tot)
) / EUR13_h_tot

"Predictions for Aggregate Productivity"
"ams"
AUT_A_tot_ams_h_tot = np.array(AUT.A_tot_ams).flatten() * np.array(AUT.h_tot).flatten()
BEL_A_tot_ams_h_tot = np.array(BEL.A_tot_ams).flatten() * np.array(BEL.h_tot).flatten()
DEU_A_tot_ams_h_tot = np.array(DEU.A_tot_ams).flatten() * np.array(DEU.h_tot).flatten()
DNK_A_tot_ams_h_tot = np.array(DNK.A_tot_ams).flatten() * np.array(DNK.h_tot).flatten()
ESP_A_tot_ams_h_tot = np.array(ESP.A_tot_ams).flatten() * np.array(ESP.h_tot).flatten()
FIN_A_tot_ams_h_tot = np.array(FIN.A_tot_ams).flatten() * np.array(FIN.h_tot).flatten()
FRA_A_tot_ams_h_tot = np.array(FRA.A_tot_ams).flatten() * np.array(FRA.h_tot).flatten()
GBR_A_tot_ams_h_tot = np.array(GBR.A_tot_ams).flatten() * np.array(GBR.h_tot).flatten()
GRC_A_tot_ams_h_tot = np.array(GRC.A_tot_ams).flatten() * np.array(GRC.h_tot).flatten()
ITA_A_tot_ams_h_tot = np.array(ITA.A_tot_ams).flatten() * np.array(ITA.h_tot).flatten()
NLD_A_tot_ams_h_tot = np.array(NLD.A_tot_ams).flatten() * np.array(NLD.h_tot).flatten()
PRT_A_tot_ams_h_tot = np.array(PRT.A_tot_ams).flatten() * np.array(PRT.h_tot).flatten()
SWE_A_tot_ams_h_tot = np.array(SWE.A_tot_ams).flatten() * np.array(SWE.h_tot).flatten()

EUR4_A_tot_ams = (
    DEU_A_tot_ams_h_tot
    + FRA_A_tot_ams_h_tot
    + GBR_A_tot_ams_h_tot
    + ITA_A_tot_ams_h_tot
) / EUR4_h_tot
EURCORE_A_tot_ams = (
    DEU_A_tot_ams_h_tot
    + FRA_A_tot_ams_h_tot
    + BEL_A_tot_ams_h_tot
    + NLD_A_tot_ams_h_tot
    + DNK_A_tot_ams_h_tot
) / EURCORE_h_tot
EURPERI_A_tot_ams = (
    GRC_A_tot_ams_h_tot
    + PRT_A_tot_ams_h_tot
    + ESP_A_tot_ams_h_tot
    + ITA_A_tot_ams_h_tot
    + GBR_A_tot_ams_h_tot
) / EURPERI_h_tot
EUR13_A_tot_ams = (
    AUT_A_tot_ams_h_tot
    + BEL_A_tot_ams_h_tot
    + DEU_A_tot_ams_h_tot
    + DNK_A_tot_ams_h_tot
    + ESP_A_tot_ams_h_tot
    + FIN_A_tot_ams_h_tot
    + FRA_A_tot_ams_h_tot
    + GBR_A_tot_ams_h_tot
    + GRC_A_tot_ams_h_tot
    + ITA_A_tot_ams_h_tot
    + NLD_A_tot_ams_h_tot
    + PRT_A_tot_ams_h_tot
    + SWE_A_tot_ams_h_tot
) / EUR13_h_tot

"ams: closed"
AUT_A_tot_ams_h_tot_closed = (
    np.array(AUT.A_tot_ams_closed).flatten() * np.array(AUT.h_tot).flatten()
)
BEL_A_tot_ams_h_tot_closed = (
    np.array(BEL.A_tot_ams_closed).flatten() * np.array(BEL.h_tot).flatten()
)
DEU_A_tot_ams_h_tot_closed = (
    np.array(DEU.A_tot_ams_closed).flatten() * np.array(DEU.h_tot).flatten()
)
DNK_A_tot_ams_h_tot_closed = (
    np.array(DNK.A_tot_ams_closed).flatten() * np.array(DNK.h_tot).flatten()
)
ESP_A_tot_ams_h_tot_closed = (
    np.array(ESP.A_tot_ams_closed).flatten() * np.array(ESP.h_tot).flatten()
)
FIN_A_tot_ams_h_tot_closed = (
    np.array(FIN.A_tot_ams_closed).flatten() * np.array(FIN.h_tot).flatten()
)
FRA_A_tot_ams_h_tot_closed = (
    np.array(FRA.A_tot_ams_closed).flatten() * np.array(FRA.h_tot).flatten()
)
GBR_A_tot_ams_h_tot_closed = (
    np.array(GBR.A_tot_ams_closed).flatten() * np.array(GBR.h_tot).flatten()
)
GRC_A_tot_ams_h_tot_closed = (
    np.array(GRC.A_tot_ams_closed).flatten() * np.array(GRC.h_tot).flatten()
)
ITA_A_tot_ams_h_tot_closed = (
    np.array(ITA.A_tot_ams_closed).flatten() * np.array(ITA.h_tot).flatten()
)
NLD_A_tot_ams_h_tot_closed = (
    np.array(NLD.A_tot_ams_closed).flatten() * np.array(NLD.h_tot).flatten()
)
PRT_A_tot_ams_h_tot_closed = (
    np.array(PRT.A_tot_ams_closed).flatten() * np.array(PRT.h_tot).flatten()
)
SWE_A_tot_ams_h_tot_closed = (
    np.array(SWE.A_tot_ams_closed).flatten() * np.array(SWE.h_tot).flatten()
)

EUR4_A_tot_ams_closed = (
    DEU_A_tot_ams_h_tot_closed
    + FRA_A_tot_ams_h_tot_closed
    + GBR_A_tot_ams_h_tot_closed
    + ITA_A_tot_ams_h_tot_closed
) / EUR4_h_tot
EURCORE_A_tot_ams_closed = (
    DEU_A_tot_ams_h_tot_closed
    + FRA_A_tot_ams_h_tot_closed
    + BEL_A_tot_ams_h_tot_closed
    + NLD_A_tot_ams_h_tot_closed
    + DNK_A_tot_ams_h_tot_closed
) / EURCORE_h_tot
EURPERI_A_tot_ams_closed = (
    GRC_A_tot_ams_h_tot_closed
    + PRT_A_tot_ams_h_tot_closed
    + ESP_A_tot_ams_h_tot_closed
    + ITA_A_tot_ams_h_tot_closed
    + GBR_A_tot_ams_h_tot_closed
) / EURPERI_h_tot
EUR13_A_tot_ams_closed = (
    AUT_A_tot_ams_h_tot_closed
    + BEL_A_tot_ams_h_tot_closed
    + DEU_A_tot_ams_h_tot_closed
    + DNK_A_tot_ams_h_tot_closed
    + ESP_A_tot_ams_h_tot_closed
    + FIN_A_tot_ams_h_tot_closed
    + FRA_A_tot_ams_h_tot_closed
    + GBR_A_tot_ams_h_tot_closed
    + GRC_A_tot_ams_h_tot_closed
    + ITA_A_tot_ams_h_tot_closed
    + NLD_A_tot_ams_h_tot_closed
    + PRT_A_tot_ams_h_tot_closed
    + SWE_A_tot_ams_h_tot_closed
) / EUR13_h_tot

"nps"
AUT_A_tot_nps_h_tot = np.array(AUT.A_tot_nps).flatten() * np.array(AUT.h_tot).flatten()
BEL_A_tot_nps_h_tot = np.array(BEL.A_tot_nps).flatten() * np.array(BEL.h_tot).flatten()
DEU_A_tot_nps_h_tot = np.array(DEU.A_tot_nps).flatten() * np.array(DEU.h_tot).flatten()
DNK_A_tot_nps_h_tot = np.array(DNK.A_tot_nps).flatten() * np.array(DNK.h_tot).flatten()
ESP_A_tot_nps_h_tot = np.array(ESP.A_tot_nps).flatten() * np.array(ESP.h_tot).flatten()
FIN_A_tot_nps_h_tot = np.array(FIN.A_tot_nps).flatten() * np.array(FIN.h_tot).flatten()
FRA_A_tot_nps_h_tot = np.array(FRA.A_tot_nps).flatten() * np.array(FRA.h_tot).flatten()
GBR_A_tot_nps_h_tot = np.array(GBR.A_tot_nps).flatten() * np.array(GBR.h_tot).flatten()
GRC_A_tot_nps_h_tot = np.array(GRC.A_tot_nps).flatten() * np.array(GRC.h_tot).flatten()
ITA_A_tot_nps_h_tot = np.array(ITA.A_tot_nps).flatten() * np.array(ITA.h_tot).flatten()
NLD_A_tot_nps_h_tot = np.array(NLD.A_tot_nps).flatten() * np.array(NLD.h_tot).flatten()
PRT_A_tot_nps_h_tot = np.array(PRT.A_tot_nps).flatten() * np.array(PRT.h_tot).flatten()
SWE_A_tot_nps_h_tot = np.array(SWE.A_tot_nps).flatten() * np.array(SWE.h_tot).flatten()

EUR4_A_tot_nps = (
    DEU_A_tot_nps_h_tot
    + FRA_A_tot_nps_h_tot
    + GBR_A_tot_nps_h_tot
    + ITA_A_tot_nps_h_tot
) / EUR4_h_tot
EURCORE_A_tot_nps = (
    DEU_A_tot_nps_h_tot
    + FRA_A_tot_nps_h_tot
    + BEL_A_tot_nps_h_tot
    + NLD_A_tot_nps_h_tot
    + DNK_A_tot_nps_h_tot
) / EURCORE_h_tot
EURPERI_A_tot_nps = (
    GRC_A_tot_nps_h_tot
    + PRT_A_tot_nps_h_tot
    + ESP_A_tot_nps_h_tot
    + ITA_A_tot_nps_h_tot
    + GBR_A_tot_nps_h_tot
) / EURPERI_h_tot
EUR13_A_tot_nps = (
    AUT_A_tot_nps_h_tot
    + BEL_A_tot_nps_h_tot
    + DEU_A_tot_nps_h_tot
    + DNK_A_tot_nps_h_tot
    + ESP_A_tot_nps_h_tot
    + FIN_A_tot_nps_h_tot
    + FRA_A_tot_nps_h_tot
    + GBR_A_tot_nps_h_tot
    + GRC_A_tot_nps_h_tot
    + ITA_A_tot_nps_h_tot
    + NLD_A_tot_nps_h_tot
    + PRT_A_tot_nps_h_tot
    + SWE_A_tot_nps_h_tot
) / EUR13_h_tot

"nps: closed"
AUT_A_tot_nps_h_tot_closed = (
    np.array(AUT.A_tot_nps_closed).flatten() * np.array(AUT.h_tot).flatten()
)
BEL_A_tot_nps_h_tot_closed = (
    np.array(BEL.A_tot_nps_closed).flatten() * np.array(BEL.h_tot).flatten()
)
DEU_A_tot_nps_h_tot_closed = (
    np.array(DEU.A_tot_nps_closed).flatten() * np.array(DEU.h_tot).flatten()
)
DNK_A_tot_nps_h_tot_closed = (
    np.array(DNK.A_tot_nps_closed).flatten() * np.array(DNK.h_tot).flatten()
)
ESP_A_tot_nps_h_tot_closed = (
    np.array(ESP.A_tot_nps_closed).flatten() * np.array(ESP.h_tot).flatten()
)
FIN_A_tot_nps_h_tot_closed = (
    np.array(FIN.A_tot_nps_closed).flatten() * np.array(FIN.h_tot).flatten()
)
FRA_A_tot_nps_h_tot_closed = (
    np.array(FRA.A_tot_nps_closed).flatten() * np.array(FRA.h_tot).flatten()
)
GBR_A_tot_nps_h_tot_closed = (
    np.array(GBR.A_tot_nps_closed).flatten() * np.array(GBR.h_tot).flatten()
)
GRC_A_tot_nps_h_tot_closed = (
    np.array(GRC.A_tot_nps_closed).flatten() * np.array(GRC.h_tot).flatten()
)
ITA_A_tot_nps_h_tot_closed = (
    np.array(ITA.A_tot_nps_closed).flatten() * np.array(ITA.h_tot).flatten()
)
NLD_A_tot_nps_h_tot_closed = (
    np.array(NLD.A_tot_nps_closed).flatten() * np.array(NLD.h_tot).flatten()
)
PRT_A_tot_nps_h_tot_closed = (
    np.array(PRT.A_tot_nps_closed).flatten() * np.array(PRT.h_tot).flatten()
)
SWE_A_tot_nps_h_tot_closed = (
    np.array(SWE.A_tot_nps_closed).flatten() * np.array(SWE.h_tot).flatten()
)

EUR4_A_tot_nps = (
    DEU_A_tot_nps_h_tot_closed
    + FRA_A_tot_nps_h_tot_closed
    + GBR_A_tot_nps_h_tot_closed
    + ITA_A_tot_nps_h_tot_closed
) / EUR4_h_tot
EURCORE_A_tot_nps = (
    DEU_A_tot_nps_h_tot_closed
    + FRA_A_tot_nps_h_tot_closed
    + BEL_A_tot_nps_h_tot_closed
    + NLD_A_tot_nps_h_tot_closed
    + DNK_A_tot_nps_h_tot_closed
) / EURCORE_h_tot
EURPERI_A_tot_nps = (
    GRC_A_tot_nps_h_tot_closed
    + PRT_A_tot_nps_h_tot_closed
    + ESP_A_tot_nps_h_tot_closed
    + ITA_A_tot_nps_h_tot_closed
    + GBR_A_tot_nps_h_tot_closed
) / EURPERI_h_tot
EUR13_A_tot_nps = (
    AUT_A_tot_nps_h_tot_closed
    + BEL_A_tot_nps_h_tot_closed
    + DEU_A_tot_nps_h_tot_closed
    + DNK_A_tot_nps_h_tot_closed
    + ESP_A_tot_nps_h_tot_closed
    + FIN_A_tot_nps_h_tot_closed
    + FRA_A_tot_nps_h_tot_closed
    + GBR_A_tot_nps_h_tot_closed
    + GRC_A_tot_nps_h_tot_closed
    + ITA_A_tot_nps_h_tot_closed
    + NLD_A_tot_nps_h_tot_closed
    + PRT_A_tot_nps_h_tot_closed
    + SWE_A_tot_nps_h_tot_closed
) / EUR13_h_tot


"Euroshares (data)"
EUR4_share_agr = (
    np.array(DEU.share_agr) * np.array(DEU.h_tot)
    + np.array(FRA.share_agr) * np.array(FRA.h_tot)
    + np.array(GBR.share_agr) * np.array(GBR.h_tot)
    + np.array(ITA.share_agr) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_agr = (
    np.array(DEU.share_agr) * np.array(DEU.h_tot)
    + np.array(FRA.share_agr) * np.array(FRA.h_tot)
    + np.array(BEL.share_agr) * np.array(BEL.h_tot)
    + np.array(NLD.share_agr) * np.array(NLD.h_tot)
    + np.array(DNK.share_agr) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_agr = (
    np.array(GRC.share_agr) * np.array(GRC.h_tot)
    + np.array(PRT.share_agr) * np.array(PRT.h_tot)
    + np.array(ESP.share_agr) * np.array(ESP.h_tot)
    + np.array(ITA.share_agr) * np.array(ITA.h_tot)
    + np.array(GBR.share_agr) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_agr = (
    np.array(AUT.share_agr) * np.array(AUT.h_tot)
    + np.array(BEL.share_agr) * np.array(BEL.h_tot)
    + np.array(DEU.share_agr) * np.array(DEU.h_tot)
    + np.array(DNK.share_agr) * np.array(DNK.h_tot)
    + np.array(ESP.share_agr) * np.array(ESP.h_tot)
    + np.array(FIN.share_agr) * np.array(FIN.h_tot)
    + np.array(FRA.share_agr) * np.array(FRA.h_tot)
    + np.array(GBR.share_agr) * np.array(GBR.h_tot)
    + np.array(GRC.share_agr) * np.array(GRC.h_tot)
    + np.array(ITA.share_agr) * np.array(ITA.h_tot)
    + np.array(NLD.share_agr) * np.array(NLD.h_tot)
    + np.array(PRT.share_agr) * np.array(PRT.h_tot)
    + np.array(SWE.share_agr) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_man = (
    np.array(DEU.share_man) * np.array(DEU.h_tot)
    + np.array(FRA.share_man) * np.array(FRA.h_tot)
    + np.array(GBR.share_man) * np.array(GBR.h_tot)
    + np.array(ITA.share_man) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_man = (
    np.array(DEU.share_man) * np.array(DEU.h_tot)
    + np.array(FRA.share_man) * np.array(FRA.h_tot)
    + np.array(BEL.share_man) * np.array(BEL.h_tot)
    + np.array(NLD.share_man) * np.array(NLD.h_tot)
    + np.array(DNK.share_man) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_man = (
    np.array(GRC.share_man) * np.array(GRC.h_tot)
    + np.array(PRT.share_man) * np.array(PRT.h_tot)
    + np.array(ESP.share_man) * np.array(ESP.h_tot)
    + np.array(ITA.share_man) * np.array(ITA.h_tot)
    + np.array(GBR.share_man) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_man = (
    np.array(AUT.share_man) * np.array(AUT.h_tot)
    + np.array(BEL.share_man) * np.array(BEL.h_tot)
    + np.array(DEU.share_man) * np.array(DEU.h_tot)
    + np.array(DNK.share_man) * np.array(DNK.h_tot)
    + np.array(ESP.share_man) * np.array(ESP.h_tot)
    + np.array(FIN.share_man) * np.array(FIN.h_tot)
    + np.array(FRA.share_man) * np.array(FRA.h_tot)
    + np.array(GBR.share_man) * np.array(GBR.h_tot)
    + np.array(GRC.share_man) * np.array(GRC.h_tot)
    + np.array(ITA.share_man) * np.array(ITA.h_tot)
    + np.array(NLD.share_man) * np.array(NLD.h_tot)
    + np.array(PRT.share_man) * np.array(PRT.h_tot)
    + np.array(SWE.share_man) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_ser = (
    np.array(DEU.share_ser) * np.array(DEU.h_tot)
    + np.array(FRA.share_ser) * np.array(FRA.h_tot)
    + np.array(GBR.share_ser) * np.array(GBR.h_tot)
    + np.array(ITA.share_ser) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_ser = (
    np.array(DEU.share_ser) * np.array(DEU.h_tot)
    + np.array(FRA.share_ser) * np.array(FRA.h_tot)
    + np.array(BEL.share_ser) * np.array(BEL.h_tot)
    + np.array(NLD.share_ser) * np.array(NLD.h_tot)
    + np.array(DNK.share_ser) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_ser = (
    np.array(GRC.share_ser) * np.array(GRC.h_tot)
    + np.array(PRT.share_ser) * np.array(PRT.h_tot)
    + np.array(ESP.share_ser) * np.array(ESP.h_tot)
    + np.array(ITA.share_ser) * np.array(ITA.h_tot)
    + np.array(GBR.share_ser) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_ser = (
    np.array(AUT.share_ser) * np.array(AUT.h_tot)
    + np.array(BEL.share_ser) * np.array(BEL.h_tot)
    + np.array(DEU.share_ser) * np.array(DEU.h_tot)
    + np.array(DNK.share_ser) * np.array(DNK.h_tot)
    + np.array(ESP.share_ser) * np.array(ESP.h_tot)
    + np.array(FIN.share_ser) * np.array(FIN.h_tot)
    + np.array(FRA.share_ser) * np.array(FRA.h_tot)
    + np.array(GBR.share_ser) * np.array(GBR.h_tot)
    + np.array(GRC.share_ser) * np.array(GRC.h_tot)
    + np.array(ITA.share_ser) * np.array(ITA.h_tot)
    + np.array(NLD.share_ser) * np.array(NLD.h_tot)
    + np.array(PRT.share_ser) * np.array(PRT.h_tot)
    + np.array(SWE.share_ser) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_trd = (
    np.array(DEU.share_trd) * np.array(DEU.h_tot)
    + np.array(FRA.share_trd) * np.array(FRA.h_tot)
    + np.array(GBR.share_trd) * np.array(GBR.h_tot)
    + np.array(ITA.share_trd) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_trd = (
    np.array(DEU.share_trd) * np.array(DEU.h_tot)
    + np.array(FRA.share_trd) * np.array(FRA.h_tot)
    + np.array(BEL.share_trd) * np.array(BEL.h_tot)
    + np.array(NLD.share_trd) * np.array(NLD.h_tot)
    + np.array(DNK.share_trd) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_trd = (
    np.array(GRC.share_trd) * np.array(GRC.h_tot)
    + np.array(PRT.share_trd) * np.array(PRT.h_tot)
    + np.array(ESP.share_trd) * np.array(ESP.h_tot)
    + np.array(ITA.share_trd) * np.array(ITA.h_tot)
    + np.array(GBR.share_trd) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_trd = (
    np.array(AUT.share_trd) * np.array(AUT.h_tot)
    + np.array(BEL.share_trd) * np.array(BEL.h_tot)
    + np.array(DEU.share_trd) * np.array(DEU.h_tot)
    + np.array(DNK.share_trd) * np.array(DNK.h_tot)
    + np.array(ESP.share_trd) * np.array(ESP.h_tot)
    + np.array(FIN.share_trd) * np.array(FIN.h_tot)
    + np.array(FRA.share_trd) * np.array(FRA.h_tot)
    + np.array(GBR.share_trd) * np.array(GBR.h_tot)
    + np.array(GRC.share_trd) * np.array(GRC.h_tot)
    + np.array(ITA.share_trd) * np.array(ITA.h_tot)
    + np.array(NLD.share_trd) * np.array(NLD.h_tot)
    + np.array(PRT.share_trd) * np.array(PRT.h_tot)
    + np.array(SWE.share_trd) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_bss = (
    np.array(DEU.share_bss) * np.array(DEU.h_tot)
    + np.array(FRA.share_bss) * np.array(FRA.h_tot)
    + np.array(GBR.share_bss) * np.array(GBR.h_tot)
    + np.array(ITA.share_bss) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_bss = (
    np.array(DEU.share_bss) * np.array(DEU.h_tot)
    + np.array(FRA.share_bss) * np.array(FRA.h_tot)
    + np.array(BEL.share_bss) * np.array(BEL.h_tot)
    + np.array(NLD.share_bss) * np.array(NLD.h_tot)
    + np.array(DNK.share_bss) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_bss = (
    np.array(GRC.share_bss) * np.array(GRC.h_tot)
    + np.array(PRT.share_bss) * np.array(PRT.h_tot)
    + np.array(ESP.share_bss) * np.array(ESP.h_tot)
    + np.array(ITA.share_bss) * np.array(ITA.h_tot)
    + np.array(GBR.share_bss) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_bss = (
    np.array(AUT.share_bss) * np.array(AUT.h_tot)
    + np.array(BEL.share_bss) * np.array(BEL.h_tot)
    + np.array(DEU.share_bss) * np.array(DEU.h_tot)
    + np.array(DNK.share_bss) * np.array(DNK.h_tot)
    + np.array(ESP.share_bss) * np.array(ESP.h_tot)
    + np.array(FIN.share_bss) * np.array(FIN.h_tot)
    + np.array(FRA.share_bss) * np.array(FRA.h_tot)
    + np.array(GBR.share_bss) * np.array(GBR.h_tot)
    + np.array(GRC.share_bss) * np.array(GRC.h_tot)
    + np.array(ITA.share_bss) * np.array(ITA.h_tot)
    + np.array(NLD.share_bss) * np.array(NLD.h_tot)
    + np.array(PRT.share_bss) * np.array(PRT.h_tot)
    + np.array(SWE.share_bss) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_fin = (
    np.array(DEU.share_fin) * np.array(DEU.h_tot)
    + np.array(FRA.share_fin) * np.array(FRA.h_tot)
    + np.array(GBR.share_fin) * np.array(GBR.h_tot)
    + np.array(ITA.share_fin) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_fin = (
    np.array(DEU.share_fin) * np.array(DEU.h_tot)
    + np.array(FRA.share_fin) * np.array(FRA.h_tot)
    + np.array(BEL.share_fin) * np.array(BEL.h_tot)
    + np.array(NLD.share_fin) * np.array(NLD.h_tot)
    + np.array(DNK.share_fin) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_fin = (
    np.array(GRC.share_fin) * np.array(GRC.h_tot)
    + np.array(PRT.share_fin) * np.array(PRT.h_tot)
    + np.array(ESP.share_fin) * np.array(ESP.h_tot)
    + np.array(ITA.share_fin) * np.array(ITA.h_tot)
    + np.array(GBR.share_fin) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_fin = (
    np.array(AUT.share_fin) * np.array(AUT.h_tot)
    + np.array(BEL.share_fin) * np.array(BEL.h_tot)
    + np.array(DEU.share_fin) * np.array(DEU.h_tot)
    + np.array(DNK.share_fin) * np.array(DNK.h_tot)
    + np.array(ESP.share_fin) * np.array(ESP.h_tot)
    + np.array(FIN.share_fin) * np.array(FIN.h_tot)
    + np.array(FRA.share_fin) * np.array(FRA.h_tot)
    + np.array(GBR.share_fin) * np.array(GBR.h_tot)
    + np.array(GRC.share_fin) * np.array(GRC.h_tot)
    + np.array(ITA.share_fin) * np.array(ITA.h_tot)
    + np.array(NLD.share_fin) * np.array(NLD.h_tot)
    + np.array(PRT.share_fin) * np.array(PRT.h_tot)
    + np.array(SWE.share_fin) * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_nps = (
    np.array(DEU.share_nps) * np.array(DEU.h_tot)
    + np.array(FRA.share_nps) * np.array(FRA.h_tot)
    + np.array(GBR.share_nps) * np.array(GBR.h_tot)
    + np.array(ITA.share_nps) * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_nps = (
    np.array(DEU.share_nps) * np.array(DEU.h_tot)
    + np.array(FRA.share_nps) * np.array(FRA.h_tot)
    + np.array(BEL.share_nps) * np.array(BEL.h_tot)
    + np.array(NLD.share_nps) * np.array(NLD.h_tot)
    + np.array(DNK.share_nps) * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_nps = (
    np.array(GRC.share_nps) * np.array(GRC.h_tot)
    + np.array(PRT.share_nps) * np.array(PRT.h_tot)
    + np.array(ESP.share_nps) * np.array(ESP.h_tot)
    + np.array(ITA.share_nps) * np.array(ITA.h_tot)
    + np.array(GBR.share_nps) * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_nps = (
    np.array(AUT.share_nps) * np.array(AUT.h_tot)
    + np.array(BEL.share_nps) * np.array(BEL.h_tot)
    + np.array(DEU.share_nps) * np.array(DEU.h_tot)
    + np.array(DNK.share_nps) * np.array(DNK.h_tot)
    + np.array(ESP.share_nps) * np.array(ESP.h_tot)
    + np.array(FIN.share_nps) * np.array(FIN.h_tot)
    + np.array(FRA.share_nps) * np.array(FRA.h_tot)
    + np.array(GBR.share_nps) * np.array(GBR.h_tot)
    + np.array(GRC.share_nps) * np.array(GRC.h_tot)
    + np.array(ITA.share_nps) * np.array(ITA.h_tot)
    + np.array(NLD.share_nps) * np.array(NLD.h_tot)
    + np.array(PRT.share_nps) * np.array(PRT.h_tot)
    + np.array(SWE.share_nps) * np.array(SWE.h_tot)
) / EUR13_h_tot

"Euroshares (model)"
EUR4_share_agr_ams_m = (
    np.array(DEU.share_agr_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_ams_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_agr_ams_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_agr_ams_m = (
    np.array(DEU.share_agr_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_agr_ams_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_agr_ams_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_agr_ams_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_agr_ams_m = (
    np.array(GRC.share_agr_ams_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_agr_ams_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_agr_ams_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_agr_ams_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_agr_ams_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_agr_ams_m = (
    np.array(AUT.share_agr_ams_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_agr_ams_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_agr_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_agr_ams_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_agr_ams_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_agr_ams_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_agr_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_ams_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_agr_ams_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_agr_ams_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_agr_ams_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_agr_ams_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_agr_ams_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_agr_nps_m = (
    np.array(DEU.share_agr_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_agr_nps_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_agr_nps_m = (
    np.array(DEU.share_agr_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_agr_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_agr_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_agr_nps_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_agr_nps_m = (
    np.array(GRC.share_agr_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_agr_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_agr_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_agr_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_agr_nps_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_agr_nps_m = (
    np.array(AUT.share_agr_nps_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_agr_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_agr_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_agr_nps_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_agr_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_agr_nps_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_agr_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_agr_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_agr_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_agr_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_agr_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_agr_nps_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_man_ams_m = (
    np.array(DEU.share_man_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_ams_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_man_ams_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_man_ams_m = (
    np.array(DEU.share_man_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_man_ams_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_man_ams_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_man_ams_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_man_ams_m = (
    np.array(GRC.share_man_ams_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_man_ams_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_man_ams_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_man_ams_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_man_ams_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_man_ams_m = (
    np.array(AUT.share_man_ams_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_man_ams_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_man_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_man_ams_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_man_ams_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_man_ams_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_man_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_ams_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_man_ams_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_man_ams_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_man_ams_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_man_ams_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_man_ams_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_man_nps_m = (
    np.array(DEU.share_man_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_man_nps_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_man_nps_m = (
    np.array(DEU.share_man_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_man_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_man_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_man_nps_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_man_nps_m = (
    np.array(GRC.share_man_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_man_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_man_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_man_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_man_nps_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_man_nps_m = (
    np.array(AUT.share_man_nps_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_man_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_man_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_man_nps_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_man_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_man_nps_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_man_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_man_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_man_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_man_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_man_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_man_nps_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_ser_ams_m = (
    np.array(DEU.share_ser_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_ser_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_ser_ams_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_ser_ams_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_ser_ams_m = (
    np.array(DEU.share_ser_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_ser_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_ser_ams_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_ser_ams_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_ser_ams_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_ser_ams_m = (
    np.array(GRC.share_ser_ams_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_ser_ams_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_ser_ams_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_ser_ams_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_ser_ams_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_ser_ams_m = (
    np.array(AUT.share_ser_ams_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_ser_ams_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_ser_ams_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_ser_ams_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_ser_ams_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_ser_ams_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_ser_ams_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_ser_ams_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_ser_ams_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_ser_ams_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_ser_ams_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_ser_ams_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_ser_ams_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_trd_nps_m = (
    np.array(DEU.share_trd_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_trd_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_trd_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_trd_nps_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_trd_nps_m = (
    np.array(DEU.share_trd_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_trd_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_trd_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_trd_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_trd_nps_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_trd_nps_m = (
    np.array(GRC.share_trd_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_trd_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_trd_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_trd_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_trd_nps_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_trd_nps_m = (
    np.array(AUT.share_trd_nps_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_trd_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_trd_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_trd_nps_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_trd_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_trd_nps_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_trd_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_trd_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_trd_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_trd_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_trd_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_trd_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_trd_nps_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_bss_nps_m = (
    np.array(DEU.share_bss_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_bss_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_bss_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_bss_nps_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_bss_nps_m = (
    np.array(DEU.share_bss_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_bss_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_bss_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_bss_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_bss_nps_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_bss_nps_m = (
    np.array(GRC.share_bss_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_bss_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_bss_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_bss_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_bss_nps_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_bss_nps_m = (
    np.array(AUT.share_bss_nps_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_bss_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_bss_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_bss_nps_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_bss_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_bss_nps_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_bss_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_bss_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_bss_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_bss_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_bss_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_bss_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_bss_nps_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_fin_nps_m = (
    np.array(DEU.share_fin_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_fin_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_fin_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_fin_nps_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_fin_nps_m = (
    np.array(DEU.share_fin_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_fin_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_fin_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_fin_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_fin_nps_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_fin_nps_m = (
    np.array(GRC.share_fin_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_fin_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_fin_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_fin_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_fin_nps_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_fin_nps_m = (
    np.array(AUT.share_fin_nps_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_fin_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_fin_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_fin_nps_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_fin_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_fin_nps_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_fin_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_fin_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_fin_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_fin_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_fin_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_fin_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_fin_nps_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_nps_nps_m = (
    np.array(DEU.share_nps_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_nps_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_nps_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_nps_nps_m).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_nps_nps_m = (
    np.array(DEU.share_nps_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_nps_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_nps_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_nps_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_nps_nps_m).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_nps_nps_m = (
    np.array(GRC.share_nps_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_nps_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_nps_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_nps_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_nps_nps_m).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_nps_nps_m = (
    np.array(AUT.share_nps_nps_m).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_nps_nps_m).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_nps_nps_m).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_nps_nps_m).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_nps_nps_m).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_nps_nps_m).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_nps_nps_m).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_nps_nps_m).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_nps_nps_m).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_nps_nps_m).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_nps_nps_m).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_nps_nps_m).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_nps_nps_m).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

"Euroshares (model: closed)"
EUR4_share_agr_ams_m_closed = (
    np.array(DEU.share_agr_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_ams_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_agr_ams_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_agr_ams_m_closed = (
    np.array(DEU.share_agr_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_agr_ams_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_agr_ams_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_agr_ams_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_agr_ams_m_closed = (
    np.array(GRC.share_agr_ams_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_agr_ams_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_agr_ams_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_agr_ams_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_agr_ams_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_agr_ams_m_closed = (
    np.array(AUT.share_agr_ams_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_agr_ams_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_agr_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_agr_ams_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_agr_ams_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_agr_ams_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_agr_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_ams_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_agr_ams_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_agr_ams_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_agr_ams_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_agr_ams_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_agr_ams_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_agr_nps_m_closed = (
    np.array(DEU.share_agr_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_agr_nps_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_agr_nps_m_closed = (
    np.array(DEU.share_agr_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_agr_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_agr_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_agr_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_agr_nps_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_agr_nps_m_closed = (
    np.array(GRC.share_agr_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_agr_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_agr_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_agr_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_agr_nps_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_agr_nps_m_closed = (
    np.array(AUT.share_agr_nps_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_agr_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_agr_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_agr_nps_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_agr_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_agr_nps_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_agr_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_agr_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_agr_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_agr_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_agr_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_agr_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_agr_nps_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_man_ams_m_closed = (
    np.array(DEU.share_man_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_ams_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_man_ams_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_man_ams_m_closed = (
    np.array(DEU.share_man_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_man_ams_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_man_ams_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_man_ams_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_man_ams_m_closed = (
    np.array(GRC.share_man_ams_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_man_ams_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_man_ams_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_man_ams_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_man_ams_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_man_ams_m_closed = (
    np.array(AUT.share_man_ams_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_man_ams_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_man_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_man_ams_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_man_ams_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_man_ams_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_man_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_ams_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_man_ams_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_man_ams_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_man_ams_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_man_ams_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_man_ams_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_man_nps_m_closed = (
    np.array(DEU.share_man_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_man_nps_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_man_nps_m_closed = (
    np.array(DEU.share_man_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_man_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_man_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_man_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_man_nps_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_man_nps_m_closed = (
    np.array(GRC.share_man_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_man_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_man_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_man_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_man_nps_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_man_nps_m_closed = (
    np.array(AUT.share_man_nps_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_man_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_man_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_man_nps_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_man_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_man_nps_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_man_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_man_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_man_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_man_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_man_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_man_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_man_nps_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_ser_ams_m_closed = (
    np.array(DEU.share_ser_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_ser_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_ser_ams_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_ser_ams_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_ser_ams_m_closed = (
    np.array(DEU.share_ser_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_ser_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_ser_ams_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_ser_ams_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_ser_ams_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_ser_ams_m_closed = (
    np.array(GRC.share_ser_ams_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_ser_ams_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_ser_ams_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_ser_ams_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_ser_ams_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_ser_ams_m_closed = (
    np.array(AUT.share_ser_ams_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_ser_ams_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_ser_ams_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_ser_ams_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_ser_ams_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_ser_ams_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_ser_ams_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_ser_ams_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_ser_ams_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_ser_ams_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_ser_ams_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_ser_ams_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_ser_ams_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_trd_nps_m_closed = (
    np.array(DEU.share_trd_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_trd_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_trd_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_trd_nps_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_trd_nps_m_closed = (
    np.array(DEU.share_trd_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_trd_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_trd_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_trd_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_trd_nps_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_trd_nps_m_closed = (
    np.array(GRC.share_trd_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_trd_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_trd_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_trd_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_trd_nps_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_trd_nps_m_closed = (
    np.array(AUT.share_trd_nps_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_trd_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_trd_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_trd_nps_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_trd_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_trd_nps_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_trd_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_trd_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_trd_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_trd_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_trd_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_trd_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_trd_nps_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_bss_nps_m_closed = (
    np.array(DEU.share_bss_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_bss_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_bss_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_bss_nps_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_bss_nps_m_closed = (
    np.array(DEU.share_bss_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_bss_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_bss_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_bss_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_bss_nps_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_bss_nps_m_closed = (
    np.array(GRC.share_bss_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_bss_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_bss_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_bss_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_bss_nps_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_bss_nps_m_closed = (
    np.array(AUT.share_bss_nps_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_bss_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_bss_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_bss_nps_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_bss_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_bss_nps_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_bss_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_bss_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_bss_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_bss_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_bss_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_bss_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_bss_nps_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_fin_nps_m_closed = (
    np.array(DEU.share_fin_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_fin_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_fin_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_fin_nps_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_fin_nps_m_closed = (
    np.array(DEU.share_fin_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_fin_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_fin_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_fin_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_fin_nps_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_fin_nps_m_closed = (
    np.array(GRC.share_fin_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_fin_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_fin_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_fin_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_fin_nps_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_fin_nps_m_closed = (
    np.array(AUT.share_fin_nps_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_fin_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_fin_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_fin_nps_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_fin_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_fin_nps_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_fin_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_fin_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_fin_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_fin_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_fin_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_fin_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_fin_nps_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot

EUR4_share_nps_nps_m_closed = (
    np.array(DEU.share_nps_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_nps_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_nps_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(ITA.share_nps_nps_m_closed).flatten() * np.array(ITA.h_tot)
) / EUR4_h_tot
EURCORE_share_nps_nps_m_closed = (
    np.array(DEU.share_nps_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(FRA.share_nps_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(BEL.share_nps_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(NLD.share_nps_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(DNK.share_nps_nps_m_closed).flatten() * np.array(DNK.h_tot)
) / EURCORE_h_tot
EURPERI_share_nps_nps_m_closed = (
    np.array(GRC.share_nps_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(PRT.share_nps_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(ESP.share_nps_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(ITA.share_nps_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(GBR.share_nps_nps_m_closed).flatten() * np.array(GBR.h_tot)
) / EURPERI_h_tot
EUR13_share_nps_nps_m_closed = (
    np.array(AUT.share_nps_nps_m_closed).flatten() * np.array(AUT.h_tot)
    + np.array(BEL.share_nps_nps_m_closed).flatten() * np.array(BEL.h_tot)
    + np.array(DEU.share_nps_nps_m_closed).flatten() * np.array(DEU.h_tot)
    + np.array(DNK.share_nps_nps_m_closed).flatten() * np.array(DNK.h_tot)
    + np.array(ESP.share_nps_nps_m_closed).flatten() * np.array(ESP.h_tot)
    + np.array(FIN.share_nps_nps_m_closed).flatten() * np.array(FIN.h_tot)
    + np.array(FRA.share_nps_nps_m_closed).flatten() * np.array(FRA.h_tot)
    + np.array(GBR.share_nps_nps_m_closed).flatten() * np.array(GBR.h_tot)
    + np.array(GRC.share_nps_nps_m_closed).flatten() * np.array(GRC.h_tot)
    + np.array(ITA.share_nps_nps_m_closed).flatten() * np.array(ITA.h_tot)
    + np.array(NLD.share_nps_nps_m_closed).flatten() * np.array(NLD.h_tot)
    + np.array(PRT.share_nps_nps_m_closed).flatten() * np.array(PRT.h_tot)
    + np.array(SWE.share_nps_nps_m_closed).flatten() * np.array(SWE.h_tot)
) / EUR13_h_tot


# FIGURE 2
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
ax.plot(
    np.array(share_agr)[-1],
    share_agr_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
    label="Open: U.S.",
)
ax.plot(
    np.array(share_agr)[-1],
    share_agr_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
    label="Closed: U.S.",
)
ax.plot(
    EUR4_share_agr[-1],
    EUR4_share_agr_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
    label="Open: Europe",
)
ax.plot(
    EUR4_share_agr[-1],
    EUR4_share_agr_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
    label="Closed: Europe",
)
ax.annotate(
    r"$\texttt{agr}$",
    (np.array(share_agr)[-1], share_agr_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (np.array(share_agr)[-1], share_agr_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr[-1], EUR4_share_agr_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr[-1], EUR4_share_agr_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_man)[-1],
    share_man_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_man)[-1],
    share_man_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man[-1],
    EUR4_share_man_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man[-1],
    EUR4_share_man_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{man}$",
    (np.array(share_man)[-1], share_man_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (np.array(share_man)[-1], share_man_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man[-1], EUR4_share_man_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man[-1], EUR4_share_man_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_trd)[-1],
    share_trd_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_trd)[-1],
    share_trd_nps[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_trd[-1],
    EUR4_share_trd_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_trd[-1],
    EUR4_share_trd_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (np.array(share_trd)[-1], share_trd_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (np.array(share_trd)[-1], share_trd_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd[-1], EUR4_share_trd_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd[-1], EUR4_share_trd_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_bss)[-1],
    share_bss_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_bss)[-1],
    share_bss_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_bss[-1],
    EUR4_share_bss_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_bss[-1],
    EUR4_share_bss_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (np.array(share_bss)[-1], share_bss_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (np.array(share_bss)[-1], share_bss_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss[-1], EUR4_share_bss_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss[-1], EUR4_share_bss_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_fin)[-1],
    share_fin_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_fin)[-1],
    share_fin_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_fin[-1],
    EUR4_share_fin_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_fin[-1],
    EUR4_share_fin_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (np.array(share_fin)[-1], share_fin_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (np.array(share_fin)[-1], share_fin_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin[-1], EUR4_share_fin_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin[-1], EUR4_share_fin_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

# ax.plot(np.array(share_nps)[-1], share_nps_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6,alpha=0.75)
# ax.plot(np.array(share_nps)[-1], share_nps_nps_closed[-1], 'D', markerfacecolor='none', markeredgecolor='darkred', markersize=6,alpha=0.75)
# ax.plot(EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
# ax.plot(EUR4_share_nps[-1], EUR4_share_nps_nps_m_closed[-1], 'o', markerfacecolor='none', markeredgecolor='darkblue', markersize=6, alpha=0.75)
# ax.annotate(r'$\texttt{nps}$', (np.array(share_nps)[-1], share_nps_nps[-1]), alpha=0.75, fontsize=16)
# ax.annotate(r'$\texttt{nps}$', (np.array(share_nps)[-1], share_nps_nps_closed[-1]), alpha=0.75, fontsize=16)
# ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1]), alpha=0.75, fontsize=16)
# ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps[-1], EUR4_share_nps_nps_m_closed[-1]), alpha=0.75, fontsize=16)

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
# plt.axis([0, 0.26, 0, 0.26])
plt.title("Employment Shares in 2019", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)

ax = plt.subplot(1, 2, 2)
# ax.plot(DEU.year, EUR4_rel_E, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha = 0.95, label = 'Data: OECD')
ax.plot(DEU.year, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
ax.plot(
    DEU.year,
    EUR4_A_tot_nps / A_tot_nps,
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
    DEU.year,
    EUR4_A_tot_nps / A_tot_nps_closed,
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
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14)
plt.legend(loc="upper right", fontsize=14)
plt.grid()

plt.tight_layout()
plt.savefig("../output/figures/fig_2_open_endogenous.pdf", bbox_inches="tight")
plt.close()

"""
# FIGURE 2
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
ax.plot(
    np.array(share_agr)[-1],
    share_agr_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
    label="Open: U.S.",
)
ax.plot(
    np.array(share_agr)[-1],
    share_agr_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
    label="Closed: U.S.",
)
ax.plot(
    EUR4_share_agr[-1],
    EUR4_share_agr_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
    label="Open: Europe",
)
ax.plot(
    EUR4_share_agr[-1],
    EUR4_share_agr_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
    label="Closed: Europe",
)
ax.annotate(
    r"$\texttt{agr}$",
    (np.array(share_agr)[-1], share_agr_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (np.array(share_agr)[-1], share_agr_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr[-1], EUR4_share_agr_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr[-1], EUR4_share_agr_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_man)[-1],
    share_man_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_man)[-1],
    share_man_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man[-1],
    EUR4_share_man_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man[-1],
    EUR4_share_man_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{man}$",
    (np.array(share_man)[-1], share_man_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (np.array(share_man)[-1], share_man_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man[-1], EUR4_share_man_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man[-1], EUR4_share_man_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_trd)[-1],
    share_trd_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_trd)[-1],
    share_trd_nps[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_trd[-1],
    EUR4_share_trd_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_trd[-1],
    EUR4_share_trd_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (np.array(share_trd)[-1], share_trd_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (np.array(share_trd)[-1], share_trd_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd[-1], EUR4_share_trd_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd[-1], EUR4_share_trd_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_bss)[-1],
    share_bss_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_bss)[-1],
    share_bss_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_bss[-1],
    EUR4_share_bss_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_bss[-1],
    EUR4_share_bss_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (np.array(share_bss)[-1], share_bss_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (np.array(share_bss)[-1], share_bss_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss[-1], EUR4_share_bss_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss[-1], EUR4_share_bss_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

ax.plot(
    np.array(share_fin)[-1],
    share_fin_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(share_fin)[-1],
    share_fin_nps_closed[-1],
    "D",
    markerfacecolor="none",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_fin[-1],
    EUR4_share_fin_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_fin[-1],
    EUR4_share_fin_nps_m_closed[-1],
    "o",
    markerfacecolor="none",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (np.array(share_fin)[-1], share_fin_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (np.array(share_fin)[-1], share_fin_nps_closed[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin[-1], EUR4_share_fin_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin[-1], EUR4_share_fin_nps_m_closed[-1]),
    alpha=0.75,
    fontsize=16,
)

# ax.plot(np.array(share_nps)[-1], share_nps_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6,alpha=0.75)
# ax.plot(np.array(share_nps)[-1], share_nps_nps_closed[-1], 'D', markerfacecolor='none', markeredgecolor='darkred', markersize=6,alpha=0.75)
# ax.plot(EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
# ax.plot(EUR4_share_nps[-1], EUR4_share_nps_nps_m_closed[-1], 'o', markerfacecolor='none', markeredgecolor='darkblue', markersize=6, alpha=0.75)
# ax.annotate(r'$\texttt{nps}$', (np.array(share_nps)[-1], share_nps_nps[-1]), alpha=0.75, fontsize=16)
# ax.annotate(r'$\texttt{nps}$', (np.array(share_nps)[-1], share_nps_nps_closed[-1]), alpha=0.75, fontsize=16)
# ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1]), alpha=0.75, fontsize=16)
# ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps[-1], EUR4_share_nps_nps_m_closed[-1]), alpha=0.75, fontsize=16)

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
# plt.axis([0, 0.26, 0, 0.26])
plt.title("Employment Shares in 2019", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)

ax = plt.subplot(1, 2, 2)
# ax.plot(DEU.year, EUR4_rel_E, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha = 0.95, label = 'Data: OECD')
ax.plot(DEU.year, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
ax.plot(
    DEU.year,
    EUR4_A_tot_nps / A_tot_nps,
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
    DEU.year,
    EUR4_A_tot_nps / A_tot_nps_closed,
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
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14)
plt.legend(loc="upper right", fontsize=14)
plt.grid()

plt.tight_layout()
# plt.savefig('../output/figures/fig_2_open.pdf', bbox_inches="tight")
plt.close()
"""


"""
---------------------------
    Tests of the Theory
---------------------------
"""
"""
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(8)
fig.set_figwidth(8)

"Agriculture"
ax = plt.subplot(3, 3, 1)
ax.plot(
    np.array(AUT.share_agr)[-1],
    AUT.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(BEL.share_agr)[-1],
    BEL.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_agr)[-1],
    DEU.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_agr)[-1],
    DNK.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_agr)[-1],
    ESP.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_agr)[-1],
    FIN.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_agr)[-1],
    FRA.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_agr)[-1],
    GBR.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_agr)[-1],
    GRC.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_agr)[-1],
    ITA.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_agr)[-1],
    NLD.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_agr)[-1],
    PRT.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_agr)[-1],
    SWE.share_agr_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_agr_ams_m[-1],
    EUR4_share_agr[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=10,
    label="Model (1)",
)
ax.plot(
    EUR13_share_agr_ams_m[-1],
    EUR13_share_agr[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=10,
)

# ax.annotate('AUT', (np.array(AUT.share_agr)[-1], AUT.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('BEL', (np.array(BEL.share_agr)[-1], BEL.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('DEU', (np.array(DEU.share_agr)[-1], DEU.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('DNK', (np.array(DNK.share_agr)[-1], DNK.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('ESP', (np.array(ESP.share_agr)[-1], ESP.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('FIN', (np.array(FIN.share_agr)[-1], FIN.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('FRA', (np.array(FRA.share_agr)[-1], FRA.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('GBR', (np.array(GBR.share_agr)[-1], GBR.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('GRC', (np.array(GRC.share_agr)[-1], GRC.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('ITA', (np.array(ITA.share_agr)[-1], ITA.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('NLD', (np.array(NLD.share_agr)[-1], NLD.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('PRT', (np.array(PRT.share_agr)[-1], PRT.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('SWE', (np.array(SWE.share_agr)[-1], SWE.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('EU4', (EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1]), fontsize=20)
# ax.annotate('EU13', (EUR13_share_agr_ams_m[-1], EUR13_share_agr[-1]), fontsize=20)


ax.plot(
    np.array(AUT.share_agr)[-1],
    AUT.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(BEL.share_agr)[-1],
    BEL.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_agr)[-1],
    DEU.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_agr)[-1],
    DNK.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_agr)[-1],
    ESP.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_agr)[-1],
    FIN.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_agr)[-1],
    FRA.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_agr)[-1],
    GBR.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_agr)[-1],
    GRC.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_agr)[-1],
    ITA.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_agr)[-1],
    NLD.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_agr)[-1],
    PRT.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_agr)[-1],
    SWE.share_agr_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_agr_nps_m[-1],
    EUR4_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
    label="Model (2)",
)
ax.plot(
    EUR13_share_agr_nps_m[-1],
    EUR13_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_agr)[-1], AUT.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_agr)[-1], BEL.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_agr)[-1], DEU.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_agr)[-1], DNK.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_agr)[-1], ESP.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_agr)[-1], FIN.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_agr)[-1], FRA.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_agr)[-1], GBR.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_agr)[-1], GRC.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_agr)[-1], ITA.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_agr)[-1], NLD.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_agr)[-1], PRT.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_agr)[-1], SWE.share_agr_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_agr_nps_m[-1], EUR13_share_agr[-1]), fontsize=12)

# ax.legend(fontsize=20)
plt.title("Agriculture", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.01, 0.04, 0.07, 0.10, 0.13],fontsize=10)
# plt.yticks([0.01, 0.04, 0.07, 0.10, 0.13],fontsize=10)

handles, labels = ax.get_legend_handles_labels()

"Manufacturing"
ax = plt.subplot(3, 3, 2)
ax.plot(
    np.array(AUT.share_man)[-1],
    AUT.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
    label="Model (1)",
)
ax.plot(
    np.array(BEL.share_man)[-1],
    BEL.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_man)[-1],
    DEU.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_man)[-1],
    DNK.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_man)[-1],
    ESP.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_man)[-1],
    FIN.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_man)[-1],
    FRA.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_man)[-1],
    GBR.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_man)[-1],
    GRC.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_man)[-1],
    ITA.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_man)[-1],
    NLD.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_man)[-1],
    PRT.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_man)[-1],
    SWE.share_man_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man_ams_m[-1],
    EUR4_share_man[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=10,
)
ax.plot(
    EUR13_share_man_ams_m[-1],
    EUR13_share_man[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=10,
)

# ax.annotate('AUT', (np.array(AUT.share_man)[-1], AUT.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('BEL', (np.array(BEL.share_man)[-1], BEL.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('DEU', (np.array(DEU.share_man)[-1], DEU.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('DNK', (np.array(DNK.share_man)[-1], DNK.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('ESP', (np.array(ESP.share_man)[-1], ESP.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('FIN', (np.array(FIN.share_man)[-1], FIN.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('FRA', (np.array(FRA.share_man)[-1], FRA.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('GBR', (np.array(GBR.share_man)[-1], GBR.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('GRC', (np.array(GRC.share_man)[-1], GRC.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('ITA', (np.array(ITA.share_man)[-1], ITA.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('NLD', (np.array(NLD.share_man)[-1], NLD.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('PRT', (np.array(PRT.share_man)[-1], PRT.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('SWE', (np.array(SWE.share_man)[-1], SWE.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
# ax.annotate('EU4', (EUR4_share_man_ams_m[-1], EUR4_share_man[-1]), fontsize=20)
# ax.annotate('EU15', (EUR13_share_man_ams_m[-1], EUR13_share_man[-1]), fontsize=20)

ax.plot(
    np.array(AUT.share_man)[-1],
    AUT.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model (2)",
)
ax.plot(
    np.array(BEL.share_man)[-1],
    BEL.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_man)[-1],
    DEU.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_man)[-1],
    DNK.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_man)[-1],
    ESP.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_man)[-1],
    FIN.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_man)[-1],
    FRA.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_man)[-1],
    GBR.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_man)[-1],
    GRC.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_man)[-1],
    ITA.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_man)[-1],
    NLD.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_man)[-1],
    PRT.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_man)[-1],
    SWE.share_man_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man_nps_m[-1],
    EUR4_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)
ax.plot(
    EUR13_share_man_nps_m[-1],
    EUR13_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_man)[-1], AUT.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_man)[-1], BEL.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_man)[-1], DEU.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_man)[-1], DNK.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_man)[-1], ESP.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_man)[-1], FIN.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_man)[-1], FRA.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_man)[-1], GBR.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_man)[-1], GRC.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_man)[-1], ITA.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_man)[-1], NLD.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_man)[-1], PRT.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_man)[-1], SWE.share_man_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_man_nps_m[-1], EUR4_share_man[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_man_nps_m[-1], EUR13_share_man[-1]), fontsize=12)

plt.title("Manufacturing", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.17,0.20,0.23,0.26],fontsize=10)
# plt.yticks([0.17,0.20,0.23,0.26],fontsize=10)

"Services"
ax = plt.subplot(3, 3, 3)
ax.plot(
    np.array(AUT.share_ser)[-1],
    AUT.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
    label="Model (1)",
)
ax.plot(
    np.array(BEL.share_ser)[-1],
    BEL.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_ser)[-1],
    DEU.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_ser)[-1],
    DNK.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_ser)[-1],
    ESP.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_ser)[-1],
    FIN.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_ser)[-1],
    FRA.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_ser)[-1],
    GBR.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_ser)[-1],
    GRC.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_ser)[-1],
    ITA.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_ser)[-1],
    NLD.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_ser)[-1],
    PRT.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_ser)[-1],
    SWE.share_ser_ams_m[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_ser_ams_m[-1],
    EUR4_share_ser[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=10,
)
ax.plot(
    EUR13_share_ser_ams_m[-1],
    EUR13_share_ser[-1],
    "D",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_ser)[-1], AUT.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_ser)[-1], BEL.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_ser)[-1], DEU.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_ser)[-1], DNK.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_ser)[-1], ESP.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_ser)[-1], FIN.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_ser)[-1], FRA.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_ser)[-1], GBR.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_ser)[-1], GRC.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_ser)[-1], ITA.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_ser)[-1], NLD.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_ser)[-1], PRT.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_ser)[-1], SWE.share_ser_ams_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_ser_ams_m[-1], EUR13_share_ser[-1]), fontsize=12)

plt.title("Services", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.60,0.70,0.80],fontsize=10)
# plt.yticks([0.60,0.70,0.80],fontsize=10)

"Wholesale and Retail Trade"
ax = plt.subplot(3, 3, 4)
ax.plot(
    np.array(AUT.share_trd)[-1],
    AUT.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model (2)",
)
ax.plot(
    np.array(BEL.share_trd)[-1],
    BEL.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_trd)[-1],
    DEU.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_trd)[-1],
    DNK.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_trd)[-1],
    ESP.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_trd)[-1],
    FIN.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_trd)[-1],
    FRA.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_trd)[-1],
    GBR.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_trd)[-1],
    GRC.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_trd)[-1],
    ITA.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_trd)[-1],
    NLD.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_trd)[-1],
    PRT.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_trd)[-1],
    SWE.share_trd_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_trd_nps_m[-1],
    EUR4_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)
ax.plot(
    EUR13_share_trd_nps_m[-1],
    EUR13_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_trd)[-1], AUT.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_trd)[-1], BEL.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_trd)[-1], DEU.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_trd)[-1], DNK.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_trd)[-1], ESP.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_trd)[-1], FIN.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_trd)[-1], FRA.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_trd)[-1], GBR.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_trd)[-1], GRC.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_trd)[-1], ITA.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_trd)[-1], NLD.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_trd)[-1], PRT.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_trd)[-1], SWE.share_trd_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_trd_nps_m[-1], EUR13_share_trd[-1]), fontsize=12)

plt.title("Trade", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.10, 0.14, 0.18],fontsize=10)
# plt.yticks([0.10, 0.14, 0.18],fontsize=10)

"Business Services"
ax = plt.subplot(3, 3, 5)
ax.plot(
    np.array(AUT.share_bss)[-1],
    AUT.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model (2)",
)
ax.plot(
    np.array(BEL.share_bss)[-1],
    BEL.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_bss)[-1],
    DEU.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_bss)[-1],
    DNK.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_bss)[-1],
    ESP.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_bss)[-1],
    FIN.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_bss)[-1],
    FRA.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_bss)[-1],
    GBR.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_bss)[-1],
    GRC.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_bss)[-1],
    ITA.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_bss)[-1],
    NLD.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_bss)[-1],
    PRT.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_bss)[-1],
    SWE.share_bss_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_bss_nps_m[-1],
    EUR4_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)
ax.plot(
    EUR13_share_bss_nps_m[-1],
    EUR13_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_bss)[-1], AUT.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_bss)[-1], BEL.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_bss)[-1], DEU.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_bss)[-1], DNK.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_bss)[-1], ESP.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_bss)[-1], FIN.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_bss)[-1], FRA.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_bss)[-1], GBR.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_bss)[-1], GRC.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_bss)[-1], ITA.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_bss)[-1], NLD.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_bss)[-1], PRT.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_bss)[-1], SWE.share_bss_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_bss_nps_m[-1], EUR13_share_bss[-1]), fontsize=12)

plt.title("Business", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.10, 0.15, 0.20, 0.25],fontsize=10)
# plt.yticks([0.10, 0.15, 0.20, 0.25],fontsize=10)

"Finance"
ax = plt.subplot(3, 3, 6)
ax.plot(
    np.array(AUT.share_fin)[-1],
    AUT.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model (2)",
)
ax.plot(
    np.array(BEL.share_fin)[-1],
    BEL.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_fin)[-1],
    DEU.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_fin)[-1],
    DNK.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_fin)[-1],
    ESP.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_fin)[-1],
    FIN.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_fin)[-1],
    FRA.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_fin)[-1],
    GBR.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_fin)[-1],
    GRC.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_fin)[-1],
    ITA.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_fin)[-1],
    NLD.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_fin)[-1],
    PRT.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_fin)[-1],
    SWE.share_fin_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_fin_nps_m[-1],
    EUR4_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)
ax.plot(
    EUR13_share_fin_nps_m[-1],
    EUR13_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_fin)[-1], AUT.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_fin)[-1], BEL.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_fin)[-1], DEU.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_fin)[-1], DNK.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_fin)[-1], ESP.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_fin)[-1], FIN.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_fin)[-1], FRA.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_fin)[-1], GBR.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_fin)[-1], GRC.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_fin)[-1], ITA.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_fin)[-1], NLD.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_fin)[-1], PRT.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_fin)[-1], SWE.share_fin_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_fin_nps_m[-1], EUR13_share_fin[-1]), fontsize=12)

plt.title("Finance", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.01, 0.02, 0.03, 0.04],fontsize=10)
# plt.yticks([0.01, 0.02, 0.03, 0.04],fontsize=10)

"Non-Progressive Services"
ax = plt.subplot(3, 3, 7)
ax.plot(
    np.array(AUT.share_nps)[-1],
    AUT.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model (2)",
)
ax.plot(
    np.array(BEL.share_nps)[-1],
    BEL.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DEU.share_nps)[-1],
    DEU.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(DNK.share_nps)[-1],
    DNK.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ESP.share_nps)[-1],
    ESP.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FIN.share_nps)[-1],
    FIN.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(FRA.share_nps)[-1],
    FRA.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GBR.share_nps)[-1],
    GBR.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(GRC.share_nps)[-1],
    GRC.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(ITA.share_nps)[-1],
    ITA.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(NLD.share_nps)[-1],
    NLD.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(PRT.share_nps)[-1],
    PRT.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    np.array(SWE.share_nps)[-1],
    SWE.share_nps_nps_m[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_nps_nps_m[-1],
    EUR4_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)
ax.plot(
    EUR13_share_nps_nps_m[-1],
    EUR13_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=10,
)

ax.annotate(
    "AUT",
    (np.array(AUT.share_nps)[-1], AUT.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "BEL",
    (np.array(BEL.share_nps)[-1], BEL.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DEU",
    (np.array(DEU.share_nps)[-1], DEU.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "DNK",
    (np.array(DNK.share_nps)[-1], DNK.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ESP",
    (np.array(ESP.share_nps)[-1], ESP.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FIN",
    (np.array(FIN.share_nps)[-1], FIN.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "FRA",
    (np.array(FRA.share_nps)[-1], FRA.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GBR",
    (np.array(GBR.share_nps)[-1], GBR.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "GRC",
    (np.array(GRC.share_nps)[-1], GRC.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "ITA",
    (np.array(ITA.share_nps)[-1], ITA.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "NLD",
    (np.array(NLD.share_nps)[-1], NLD.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "PRT",
    (np.array(PRT.share_nps)[-1], PRT.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate(
    "SWE",
    (np.array(SWE.share_nps)[-1], SWE.share_nps_nps_m[-1]),
    alpha=0.6,
    fontsize=10,
)
ax.annotate("EU4", (EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1]), fontsize=12)
ax.annotate("EU15", (EUR13_share_nps_nps_m[-1], EUR13_share_nps[-1]), fontsize=12)

plt.title("Non-Progressive", fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel("Data", fontsize=10)
plt.ylabel("Model", fontsize=10)
# plt.xticks([0.30, 0.40, 0.50],fontsize=10)
# plt.yticks([0.30, 0.40, 0.50],fontsize=10)


fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
plt.tight_layout()
# plt.savefig('../output/figures/fig_test.pdf', bbox_inches="tight")
plt.show()


"Test of the Theory as Time Series"
rc("axes", linewidth=0.4)

fig = plt.figure()
ax = plt.subplot(4, 4, 1)
plt.plot(AUT.year, np.array(AUT.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    AUT.year, np.array(AUT.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    AUT.year, np.array(AUT.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Austria", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 2)
plt.plot(BEL.year, np.array(BEL.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    BEL.year, np.array(BEL.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    BEL.year, np.array(BEL.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Belgium", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 3)
plt.plot(DEU.year, np.array(DEU.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    DEU.year, np.array(DEU.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    DEU.year, np.array(DEU.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Germany", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 4)
plt.plot(DNK.year, np.array(DNK.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    DNK.year, np.array(DNK.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    DNK.year, np.array(DNK.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Denmark", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 5)
plt.plot(ESP.year, np.array(ESP.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    ESP.year, np.array(ESP.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    ESP.year, np.array(ESP.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Spain", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 6)
plt.plot(FIN.year, np.array(FIN.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    FIN.year, np.array(FIN.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    FIN.year, np.array(FIN.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Finland", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 7)
plt.plot(FRA.year, np.array(FRA.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    FRA.year, np.array(FRA.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    FRA.year, np.array(FRA.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"France", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 8)
plt.plot(GBR.year, np.array(GBR.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    GBR.year, np.array(GBR.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    GBR.year, np.array(GBR.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Great Britan", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 9)
plt.plot(GRC.year, np.array(GRC.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    GRC.year, np.array(GRC.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    GRC.year, np.array(GRC.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Greece", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4, 4, 10)
plt.plot(ITA.year, np.array(ITA.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    ITA.year, np.array(ITA.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    ITA.year, np.array(ITA.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Italy", fontsize=12, y=-0.05)
plt.xticks([])
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

ax = plt.subplot(4, 4, 11)
plt.plot(NLD.year, np.array(NLD.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    NLD.year, np.array(NLD.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    NLD.year, np.array(NLD.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"The Netherlands", fontsize=12, y=-0.05)
plt.xticks([1995, 2005, 2020], fontsize=10)
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

ax = plt.subplot(4, 4, 12)
plt.plot(PRT.year, np.array(PRT.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    PRT.year, np.array(PRT.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    PRT.year, np.array(PRT.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Portugal", fontsize=12, y=-0.05)
plt.xticks([1995, 2005, 2020], fontsize=10)
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

ax = plt.subplot(4, 4, 13)
plt.plot(SWE.year, np.array(SWE.A_tot) / A_tot, "b-", alpha=0.75)
plt.plot(
    SWE.year, np.array(SWE.A_tot_ams) / A_tot_ams, "--", color="saddlebrown", alpha=0.75
)
plt.plot(
    SWE.year, np.array(SWE.A_tot_nps) / A_tot_nps, "-.", color="darkgreen", alpha=0.75
)
# plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis="y", alpha=0.35)
plt.title(r"Sweeden", fontsize=12, y=-0.05)
plt.xticks([1995, 2005, 2020], fontsize=10)
# plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

plt.tight_layout()
# plt.savefig('../output/figures/fig_test_A_EUR_countries.pdf', bbox_inches="tight")
plt.show()


fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
plt.plot(DEU.year, EUR4_rel_A_tot, "b-", lw=2, alpha=0.95, label="EU4 (Data)")
plt.plot(
    DEU.year,
    EUR4_A_tot_nps / np.array(A_tot_nps).flatten(),
    "H-",
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=5,
    alpha=0.95,
    label="Baseline Model",
)
plt.plot(
    DEU.year,
    EUR4_A_tot_ams / np.array(A_tot_ams).flatten(),
    "D-",
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=5,
    alpha=0.95,
    label="Three-Sector Model",
)
plt.title("EU4", fontsize=18)
plt.axis([1993, 2021, 0.85, 1.05])
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14)
plt.legend(fontsize=14)
plt.grid()

ax = plt.subplot(1, 2, 2)
plt.plot(DEU.year, EUR13_rel_A_tot, "b-", lw=2, alpha=0.75, label="EU13 (Data)")
plt.plot(
    DEU.year,
    EUR13_A_tot_nps / np.array(A_tot_nps).flatten(),
    "H-",
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=5,
    alpha=0.95,
    label="Baseline Model",
)
plt.plot(
    DEU.year,
    EUR13_A_tot_ams / np.array(A_tot_ams).flatten(),
    "D-",
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markersize=5,
    alpha=0.95,
    label="Three-Sector Model",
)
plt.title("EU13", fontsize=18)
plt.axis([1993, 2021, 0.85, 1.05])
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14)
plt.legend(fontsize=14)
plt.grid()

plt.tight_layout()
# plt.savefig('../output/figures/fig_test_A_EUR.pdf', bbox_inches="tight")
plt.show()
"""
"""
---------------------------
    Plots for the paper
---------------------------
"""

"""
# FIGURE 1
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
# plt.plot(DEU.year, DEU.E/np.array(E), 'o-', markersize=4, color='darkgoldenrod', markerfacecolor='gold', markeredgecolor='darkgoldenrod', alpha=0.5, label='Germany')
# plt.plot(DEU.year, FRA.E/np.array(E), 'D-', markersize=4, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.5, label='France')
# plt.plot(DEU.year, GBR.E/np.array(E), 's-', markersize=4, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', alpha=0.5, label='Great Britain')
# plt.plot(DEU.year, ITA.E/np.array(E), '^-', markersize=4, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', alpha=0.5, label='Italy')
ax.plot(DEU.year, EUR4_rel_A_tot, "b-", lw=2, alpha=0.95, label="Europe")
plt.title("Labor Productivity Gap (Relative to the U.S.)", fontsize=16)
plt.axis([1993, 2021, 0.85, 1.05])
plt.xticks(fontsize=14)
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14)
plt.legend(fontsize=14)
plt.grid()

ax = plt.subplot(1, 2, 2)
ax.plot(
    DEU.year,
    share_nps,
    "^-",
    markersize=6,
    color="darkmagenta",
    mfc="none",
    markevery=2,
    alpha=0.8,
    label="USA: Non-Progressive",
)
ax.plot(
    DEU.year,
    EUR4_share_nps,
    "^-",
    markersize=6,
    color="darkmagenta",
    markerfacecolor="violet",
    markeredgecolor="darkmagenta",
    markevery=2,
    alpha=0.8,
    label="Europe: Non-Progressive",
)
ax.plot(
    DEU.year,
    share_trd,
    "o-",
    markersize=6,
    color="darkblue",
    mfc="none",
    markevery=2,
    alpha=0.8,
    label="USA: Trade",
)
ax.plot(
    DEU.year,
    EUR4_share_trd,
    "o-",
    markersize=6,
    color="darkblue",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markevery=2,
    alpha=0.8,
    label="Europe: Trade",
)
ax.plot(
    DEU.year,
    share_bss,
    "D-",
    markersize=6,
    color="darkgreen",
    mfc="none",
    markevery=2,
    alpha=0.8,
    label="USA: Business",
)
ax.plot(
    DEU.year,
    EUR4_share_bss,
    "D-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markevery=2,
    alpha=0.8,
    label="Europe: Business",
)
ax.plot(
    DEU.year,
    share_fin,
    "s-",
    markersize=6,
    color="darkred",
    mfc="none",
    markevery=2,
    alpha=0.8,
    label="USA: Financial",
)
ax.plot(
    DEU.year,
    EUR4_share_fin,
    "s-",
    markersize=6,
    color="darkred",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markevery=2,
    alpha=0.8,
    label="Europe: Financial",
)
plt.title("Employment Shares Within Services", fontsize=16)
plt.axis([1993, 2021, 0, 0.5])
plt.xticks(fontsize=14)
plt.yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], fontsize=14)
plt.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=14)

plt.tight_layout()
# plt.savefig('../output/figures/fig_1.pdf', bbox_inches="tight")
plt.show()


# FIGURE 2
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
ax.plot(
    np.array(share_agr)[-1],
    share_agr_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
    label="United States",
)
ax.plot(
    EUR4_share_agr[-1],
    EUR4_share_agr_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
    label="Europe",
)
ax.annotate(
    r"$\texttt{agr}$",
    (np.array(share_agr)[-1], share_agr_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr[-1], EUR4_share_agr_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_man)[-1],
    share_man_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man[-1],
    EUR4_share_man_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{man}$",
    (np.array(share_man)[-1], share_man_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man[-1], EUR4_share_man_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_trd)[-1],
    share_trd_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_trd[-1],
    EUR4_share_trd_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (np.array(share_trd)[-1], share_trd_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd[-1], EUR4_share_trd_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_bss)[-1],
    share_bss_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_bss[-1],
    EUR4_share_bss_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (np.array(share_bss)[-1], share_bss_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss[-1], EUR4_share_bss_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_fin)[-1],
    share_fin_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_fin[-1],
    EUR4_share_fin_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (np.array(share_fin)[-1], share_fin_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin[-1], EUR4_share_fin_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_nps)[-1],
    share_nps_nps[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_nps[-1],
    EUR4_share_nps_nps_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (np.array(share_nps)[-1], share_nps_nps[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{nps}$",
    (EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1]),
    alpha=0.75,
    fontsize=16,
)

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
# plt.axis([0, 0.52, 0, 0.52])
plt.title("Employment Shares in 2019", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)

ax = plt.subplot(1, 2, 2)
# ax.plot(DEU.year, EUR4_rel_E, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha = 0.95, label = 'Data: OECD')
ax.plot(DEU.year, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
ax.plot(
    DEU.year,
    EUR4_A_tot_nps / A_tot_nps,
    "s-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markevery=2,
    alpha=0.95,
    label="Baseline Model",
)
plt.plot(
    DEU.year,
    EUR4_A_tot_ams / np.array(A_tot_ams).flatten(),
    "D-",
    markersize=6,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markevery=2,
    alpha=0.95,
    label="Three-Sector Model",
)
plt.title("Labor Productivity (Relative to U.S.)", fontsize=16)
plt.axis([1993, 2021, 0.85, 1.05])
plt.xticks(fontsize=14)
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04], fontsize=14)
plt.legend(loc="upper right", fontsize=14)
plt.grid()

plt.tight_layout()
# plt.savefig('../output/figures/fig_2.pdf', bbox_inches="tight")
plt.show()
"""
"""
---------------------------
    Plots for the paper
---------------------------
"""
"""
# FIG TEST EUROPE APPENDIX
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(6)
fig.set_figwidth(6)

ax = plt.subplot(3, 2, 1)
plt.plot(AUT.year, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
# plt.plot(AUT.year, EUR4_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(
    AUT.year,
    EUR4_A_tot_nps / np.array(A_tot_nps).flatten(),
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
# plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 2)
# ax.plot(EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(
    EUR4_share_agr_nps_m[-1],
    EUR4_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model",
)
# ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1]),
    alpha=0.75,
    fontsize=14,
)
# ax.plot(EUR4_share_man_ams_m[-1], EUR4_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(
    EUR4_share_man_nps_m[-1],
    EUR4_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
# ax.annotate(r'$\texttt{man}$', (EUR4_share_man_ams_m[-1], EUR4_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man_nps_m[-1], EUR4_share_man[-1]),
    alpha=0.75,
    fontsize=14,
)
# ax.plot(EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
# ax.annotate(r'$\texttt{ser}$', (EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(
    EUR4_share_trd_nps_m[-1],
    EUR4_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR4_share_bss_nps_m[-1],
    EUR4_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR4_share_fin_nps_m[-1],
    EUR4_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR4_share_nps_nps_m[-1],
    EUR4_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("EU4", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 3)
plt.plot(
    GBR.year,
    GBR.A_tot / np.array(A_tot),
    "b-",
    alpha=0.95,
    label=r"$\frac{Y_t}{L_t}$: Data",
)
# plt.plot(AUT.year, EUR13_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(
    GBR.year,
    GBR.A_tot_nps / np.array(A_tot_nps).flatten(),
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
# plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 4)
ax.plot(
    GBR.share_agr_nps_m[-1],
    GBR.share_agr.values[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model",
)
ax.annotate(
    r"$\texttt{agr}$",
    (GBR.share_agr_nps_m[-1], GBR.share_agr.values[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    GBR.share_man_nps_m[-1],
    GBR.share_man.values[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{man}$", (GBR.share_man_nps_m[-1], GBR.share_man.values[-1]), fontsize=14
)
ax.plot(
    GBR.share_trd_nps_m[-1],
    GBR.share_trd.values[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (GBR.share_trd_nps_m[-1], GBR.share_trd.values[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    GBR.share_bss_nps_m[-1],
    GBR.share_bss.values[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (GBR.share_bss_nps_m[-1], GBR.share_bss.values[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    GBR.share_fin_nps_m[-1],
    GBR.share_fin.values[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (GBR.share_fin_nps_m[-1], GBR.share_fin.values[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    GBR.share_nps_nps_m[-1],
    GBR.share_nps.values[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (GBR.share_nps_nps_m[-1], GBR.share_nps.values[-1]),
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
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 5)
plt.plot(DEU.year, EUR13_rel_A_tot, "b-", alpha=0.95, label=r"Data")
# plt.plot(AUT.year, EUR13_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(
    DEU.year,
    EUR13_A_tot_nps / np.array(A_tot_nps).flatten(),
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
# plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 6)
# ax.plot(EUR13_share_agr_ams_m[-1], EUR13_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(
    EUR13_share_agr_nps_m[-1],
    EUR13_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model",
)
# ax.annotate(r'$\texttt{agr}$', (EUR13_share_agr_ams_m[-1], EUR13_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR13_share_agr_nps_m[-1], EUR13_share_agr[-1]),
    alpha=0.75,
    fontsize=14,
)
# ax.plot(EUR13_share_man_ams_m[-1], EUR13_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(
    EUR13_share_man_nps_m[-1],
    EUR13_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
# ax.annotate(r'$\texttt{man}$', (EUR13_share_man_ams_m[-1], EUR13_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{man}$", (EUR13_share_man_nps_m[-1], EUR13_share_man[-1]), fontsize=14
)
# ax.plot(EUR13_share_ser_ams_m[-1], EUR13_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
# ax.annotate(r'$\texttt{ser}$', (EUR13_share_ser_ams_m[-1], EUR13_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(
    EUR13_share_trd_nps_m[-1],
    EUR13_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR13_share_trd_nps_m[-1], EUR13_share_trd[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR13_share_bss_nps_m[-1],
    EUR13_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR13_share_bss_nps_m[-1], EUR13_share_bss[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR13_share_fin_nps_m[-1],
    EUR13_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR13_share_fin_nps_m[-1], EUR13_share_fin[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR13_share_nps_nps_m[-1],
    EUR13_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (EUR13_share_nps_nps_m[-1], EUR13_share_nps[-1]),
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
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

plt.tight_layout()
# plt.savefig('../output/figures/fig_test_EUR_appendix.pdf', bbox_inches="tight")
plt.show()

# FIG TEST EUROPE APPENDIX
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(6)
fig.set_figwidth(6)

ax = plt.subplot(3, 2, 1)
plt.plot(AUT.year, EUR4_rel_A_tot, "b-", alpha=0.95, label="Data")
# plt.plot(AUT.year, EUR4_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(
    AUT.year,
    EUR4_A_tot_nps / np.array(A_tot_nps).flatten(),
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
# plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 2)
# ax.plot(EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(
    EUR4_share_agr_nps_m[-1],
    EUR4_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model",
)
# ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1]),
    alpha=0.75,
    fontsize=14,
)
# ax.plot(EUR4_share_man_ams_m[-1], EUR4_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(
    EUR4_share_man_nps_m[-1],
    EUR4_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
# ax.annotate(r'$\texttt{man}$', (EUR4_share_man_ams_m[-1], EUR4_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man_nps_m[-1], EUR4_share_man[-1]),
    alpha=0.75,
    fontsize=14,
)
# ax.plot(EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
# ax.annotate(r'$\texttt{ser}$', (EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(
    EUR4_share_trd_nps_m[-1],
    EUR4_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR4_share_bss_nps_m[-1],
    EUR4_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR4_share_fin_nps_m[-1],
    EUR4_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EUR4_share_nps_nps_m[-1],
    EUR4_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("EU4", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 3)
plt.plot(
    DEU.year,
    EURCORE_A_tot / np.array(A_tot),
    "b-",
    alpha=0.95,
    label=r"$\frac{Y_t}{L_t}$: Data",
)
# plt.plot(AUT.year, EUR13_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(
    DEU.year,
    EURCORE_A_tot_nps / np.array(A_tot_nps).flatten(),
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
plt.title("EU Core", fontsize=12)
# plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 4)
ax.plot(
    EURCORE_share_agr_nps_m[-1],
    EURCORE_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model",
)
ax.annotate(
    r"$\texttt{agr}$",
    (EURCORE_share_agr_nps_m[-1], EURCORE_share_agr[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURCORE_share_man_nps_m[-1],
    EURCORE_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{man}$", (EURCORE_share_man_nps_m[-1], EURCORE_share_man[-1]), fontsize=14
)
ax.plot(
    EURCORE_share_trd_nps_m[-1],
    EURCORE_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EURCORE_share_trd_nps_m[-1], EURCORE_share_trd[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURCORE_share_bss_nps_m[-1],
    EURCORE_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EURCORE_share_bss_nps_m[-1], EURCORE_share_bss[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURCORE_share_fin_nps_m[-1],
    EURCORE_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EURCORE_share_fin_nps_m[-1], EURCORE_share_fin[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURCORE_share_nps_nps_m[-1],
    EURCORE_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (EURCORE_share_nps_nps_m[-1], EURCORE_share_nps[-1]),
    alpha=0.75,
    fontsize=14,
)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("EU Core", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 5)
plt.plot(DEU.year, EURPERI_rel_A_tot, "b-", alpha=0.95, label=r"Data")
# plt.plot(AUT.year, EURPERI_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(
    DEU.year,
    EURPERI_A_tot_nps / np.array(A_tot_nps).flatten(),
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
plt.title("EU Periphery", fontsize=12)
# plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel("Year", fontsize=11)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1995, 2000, 2005, 2010, 2015, 2020], fontsize=10)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 6)
# ax.plot(EURPERI_share_agr_ams_m[-1], EURPERI_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(
    EURPERI_share_agr_nps_m[-1],
    EURPERI_share_agr[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
    label="Model",
)
# ax.annotate(r'$\texttt{agr}$', (EURPERI_share_agr_ams_m[-1], EURPERI_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{agr}$",
    (EURPERI_share_agr_nps_m[-1], EURPERI_share_agr[-1]),
    alpha=0.75,
    fontsize=14,
)
# ax.plot(EURPERI_share_man_ams_m[-1], EURPERI_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(
    EURPERI_share_man_nps_m[-1],
    EURPERI_share_man[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
# ax.annotate(r'$\texttt{man}$', (EURPERI_share_man_ams_m[-1], EURPERI_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(
    r"$\texttt{man}$", (EURPERI_share_man_nps_m[-1], EURPERI_share_man[-1]), fontsize=14
)
# ax.plot(EURPERI_share_ser_ams_m[-1], EURPERI_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
# ax.annotate(r'$\texttt{ser}$', (EURPERI_share_ser_ams_m[-1], EURPERI_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(
    EURPERI_share_trd_nps_m[-1],
    EURPERI_share_trd[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{trd}$",
    (EURPERI_share_trd_nps_m[-1], EURPERI_share_trd[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURPERI_share_bss_nps_m[-1],
    EURPERI_share_bss[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{bss}$",
    (EURPERI_share_bss_nps_m[-1], EURPERI_share_bss[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURPERI_share_fin_nps_m[-1],
    EURPERI_share_fin[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{fin}$",
    (EURPERI_share_fin_nps_m[-1], EURPERI_share_fin[-1]),
    alpha=0.75,
    fontsize=14,
)
ax.plot(
    EURPERI_share_nps_nps_m[-1],
    EURPERI_share_nps[-1],
    "H",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{nps}$",
    (EURPERI_share_nps_nps_m[-1], EURPERI_share_nps[-1]),
    alpha=0.75,
    fontsize=14,
)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.5, 0, 0.5])
plt.title("EU Periphery", fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

plt.tight_layout()
# plt.savefig('../output/figures/fig_test_EUR_2_appendix.pdf', bbox_inches="tight")
plt.show()
"""

"""
PLOTS WITH AMS MODEL
"""
"""
# FIGURE 2 with AMS
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
ax.plot(
    np.array(share_agr)[-1],
    share_agr_ams[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
    label="United States",
)
ax.plot(
    EUR4_share_agr[-1],
    EUR4_share_agr_ams_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
    label="Europe",
)
ax.annotate(
    r"$\texttt{agr}$",
    (np.array(share_agr)[-1], share_agr_ams[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{agr}$",
    (EUR4_share_agr[-1], EUR4_share_agr_ams_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_man)[-1],
    share_man_ams[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_man[-1],
    EUR4_share_man_ams_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{man}$",
    (np.array(share_man)[-1], share_man_ams[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{man}$",
    (EUR4_share_man[-1], EUR4_share_man_ams_m[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.plot(
    np.array(share_ser)[-1],
    share_ser_ams[-1],
    "D",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markersize=6,
    alpha=0.75,
)
ax.plot(
    EUR4_share_ser[-1],
    EUR4_share_ser_ams_m[-1],
    "o",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markersize=6,
    alpha=0.75,
)
ax.annotate(
    r"$\texttt{ser}$",
    (np.array(share_ser)[-1], share_ser_ams[-1]),
    alpha=0.75,
    fontsize=16,
)
ax.annotate(
    r"$\texttt{ser}$",
    (EUR4_share_ser[-1], EUR4_share_ser_ams_m[-1]),
    alpha=0.75,
    fontsize=16,
)

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.85, 0, 0.85])
plt.title("Employment Shares in 2019", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Model", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)

ax = plt.subplot(1, 2, 2)
ax.plot(
    DEU.year,
    EUR4_rel_E,
    "o-",
    markersize=6,
    color="darkblue",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markevery=7,
    alpha=0.95,
    label="Data: OECD",
)
ax.plot(
    DEU.year,
    EUR4_rel_A_tot,
    "D-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markevery=7,
    alpha=0.95,
    label="Data: KLEMS",
)
ax.plot(
    DEU.year,
    EUR4_A_tot_nps / A_tot_nps,
    "s-",
    markersize=6,
    color="darkred",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markevery=7,
    alpha=0.95,
    label="Baseline model",
)
ax.plot(
    DEU.year,
    EUR4_A_tot_ams / A_tot_ams,
    "^-",
    markersize=6,
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
    markevery=7,
    alpha=0.95,
    label="Three-sector model",
)
plt.title("Labor Productivity (Relative to U.S.)", fontsize=16)
plt.axis([1993, 2021, 0.73, 1.05])
plt.xticks(fontsize=14)
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.grid()

plt.tight_layout()
# plt.savefig('../output/figures/fig_2_ams.pdf', bbox_inches="tight")
plt.show()



def build_panel_nx_prod(countries, sectors):
    records = []
    for country in countries:
        for sector in sectors:
            # Get net exports as share of expenditure and productivity series
            nx_attr = f"nx_{sector}_E"
            prod_attr = f"A_{sector}"
            nx_series = getattr(country, nx_attr)
            prod_series = getattr(country, prod_attr)
            years = country.year
            for year, nx, prod in zip(years, nx_series, prod_series):
                records.append(
                    {
                        "year": year,
                        "country": country.cou,
                        "sector": sector,
                        "net_exports_share_expenditure": nx,
                        "productivity": prod,
                    }
                )
    panel = pd.DataFrame(records)
    panel.set_index("year", inplace=True)
    return panel


sectors = ["agr", "man", "trd", "bss", "fin", "nps", "ser"]
countries = [AUT, BEL, DEU, DNK, ESP, FIN, FRA, GBR, GRC, ITA, NLD, PRT, SWE]

nx_prod_panel = build_panel_nx_prod(countries, sectors)

# Add USA to the panel


class USAData:
    cou = "USA"
    year = list(USA_year)
    # These variables are imported from model_calibration_USA_open
    nx_agr_E = nx_agr_E
    nx_man_E = nx_man_E
    nx_trd_E = nx_trd_E
    nx_bss_E = nx_bss_E
    nx_fin_E = nx_fin_E
    nx_nps_E = nx_nps_E
    nx_ser_E = nx_ser_E
    A_agr = A_agr
    A_man = A_man
    A_trd = A_trd
    A_bss = A_bss
    A_fin = A_fin
    A_nps = A_nps
    A_ser = A_ser


USA = USAData()
countries_with_usa = countries + [USA]
nx_prod_panel = build_panel_nx_prod(countries_with_usa, sectors)
# nx_prod_panel.to_csv("nx_prod_panel.csv")


def fit_net_exports_endo(panel):
    # Fit one alpha and beta for all sectors (entire array)
    x_all = panel["productivity"].values
    y_all = panel["net_exports_share_expenditure"].values

    def objective(params):
        alpha, beta = params
        y_pred = alpha * x_all**beta
        return np.sum((y_all - y_pred) ** 2)

    result_all = minimize(objective, [1.0, 1.0], method="L-BFGS-B")
    alpha_all, beta_all = result_all.x
    print(f"All sectors (entire array): alpha = {alpha_all:.4f}, beta = {beta_all:.4f}")

    # Fit one alpha and beta for all sectors (last period only)
    last_year = panel.index.max()
    x_last = panel[panel.index == last_year]["productivity"].values
    y_last = panel[panel.index == last_year]["net_exports_share_expenditure"].values

    def objective_last(params):
        alpha, beta = params
        y_pred = alpha * x_last**beta
        return np.sum((y_last - y_pred) ** 2)

    result_last = minimize(objective_last, [1.0, 1.0], method="L-BFGS-B")
    alpha_last, beta_last = result_last.x
    print(
        f"All sectors (last period): alpha = {alpha_last:.4f}, beta = {beta_last:.4f}"
    )

    # Fit one alpha and beta per sector (entire array)
    for sector in sectors:
        df = panel[panel["sector"] == sector]
        x = df["productivity"].values
        y = df["net_exports_share_expenditure"].values

        def objective_sector(params):
            alpha, beta = params
            y_pred = alpha * x**beta
            return np.sum((y - y_pred) ** 2)

        result_sector = minimize(objective_sector, [1.0, 1.0], method="L-BFGS-B")
        alpha_sector, beta_sector = result_sector.x
        print(
            f"Sector: {sector}, alpha = {alpha_sector:.4f}, beta = {beta_sector:.4f} (entire array)"
        )

    # Fit one alpha and beta per sector (last period only)
    for sector in sectors:
        df = panel[(panel["sector"] == sector) & (panel.index == last_year)]
        x = df["productivity"].values
        y = df["net_exports_share_expenditure"].values

        def objective_sector_last(params):
            alpha, beta = params
            y_pred = alpha * x**beta
            return np.sum((y - y_pred) ** 2)

        result_sector_last = minimize(
            objective_sector_last, [1.0, 1.0], method="L-BFGS-B"
        )
        alpha_sector_last, beta_sector_last = result_sector_last.x
        print(
            f"Sector: {sector}, alpha = {alpha_sector_last:.4f}, beta = {beta_sector_last:.4f} (last period)"
        )


# Example usage: fit betas and print results for all sectors
fit_net_exports_endo(nx_prod_panel)


# Extract beta from "All sectors (last period)" fit
def get_beta_last_period(panel):
    last_year = panel.index.max()
    x_last = panel[panel.index == last_year]["productivity"].values
    y_last = panel[panel.index == last_year]["net_exports_share_expenditure"].values

    def objective_last(params):
        alpha, beta = params
        y_pred = alpha * x_last**beta
        return np.sum((y_last - y_pred) ** 2)

    result_last = minimize(objective_last, [1.0, 1.0], method="L-BFGS-B")
    _, beta_last = result_last.x
    return beta_last


beta_last_period = get_beta_last_period(nx_prod_panel)
print("beta (all sectors, last period):", beta_last_period)


beta = beta_last_period


# Find beta that minimizes the distance between model and data in last period for DEU, sector 'ser'
def fit_beta_ser_last_period(country):
    # Use DEU and 'ser' sector
    x = np.array(country.A_ser)
    y = np.array(country.nx_ser_E)
    # Only last period
    x_last = x[-1]
    y_last = y[-1]

    def objective(params):
        beta = params[0]
        xn_ser_endo = x**beta
        xn_ser_endo_plot = xn_ser_endo + y[0] - xn_ser_endo[0]
        # Only compare last period
        return (y_last - xn_ser_endo_plot[-1]) ** 2

    result = minimize(objective, [1.0], method="L-BFGS-B")
    return result.x[0]


beta_ser_last = fit_beta_ser_last_period(DEU)
xn_ser_endo = (np.array(DEU.A_ser)) ** beta_ser_last
xn_ser_endo_plot = xn_ser_endo + np.array(DEU.nx_ser_E)[0] - xn_ser_endo[0]
plt.plot(DEU.nx_ser_E)
plt.plot(DEU.nx_ser_E.index, xn_ser_endo_plot, "--")
plt.title(f"Best-fit beta (last period): {beta_ser_last:.4f}")
plt.show()

# Log-log regression: log(1 + net exports) vs log(productivity)


def fit_log_log_with_fixed_effects(panel):
    # Prepare data for fixed effects regression
    panel = panel.copy()
    panel["log_prod"] = np.log(panel["productivity"].values)
    panel["log_nx"] = np.log(1 + panel["net_exports_share_expenditure"].values)

    # Create dummy variables for country and year
    country_dummies = pd.get_dummies(panel["country"], prefix="country")
    year_dummies = pd.get_dummies(panel.index, prefix="year")
    X = pd.concat([panel["log_prod"], country_dummies, year_dummies], axis=1)
    y = panel["log_nx"]

    # Fit regression with fixed effects
    reg = LinearRegression().fit(X, y)
    print("All sectors (with country and year fixed effects):")
    print(f"Coefficient on log(productivity): {reg.coef_[0]:.4f}")

    # Each sector at a time
    for sector in sectors:
        df = panel[panel["sector"] == sector]
        country_dummies = pd.get_dummies(df["country"], prefix="country")
        year_dummies = pd.get_dummies(df.index, prefix="year")
        X_sec = pd.concat([df["log_prod"], country_dummies, year_dummies], axis=1)
        y_sec = df["log_nx"]
        reg_sec = LinearRegression().fit(X_sec, y_sec)
        print(f"Sector: {sector} (with fixed effects)")
        print(f"Coefficient on log(productivity): {reg_sec.coef_[0]:.4f}")


fit_log_log_with_fixed_effects(nx_prod_panel)


# Fit beta for each country and sector (last period) using beta * x instead of x ** beta
def fit_beta_last_period_country_sector(country, sector):
    x = np.array(getattr(country, f"A_{sector}"))
    y = np.array(getattr(country, f"nx_{sector}_E"))
    x_last = x[-1]
    y_last = y[-1]

    def objective(params):
        beta = params[0]
        xn_endo = beta * x
        xn_endo_plot = xn_endo + y[0] - xn_endo[0]
        return (y_last - xn_endo_plot[-1]) ** 2

    result = minimize(objective, [1.0], method="L-BFGS-B")
    return result.x[0]


# Collect results for all countries and sectors
results = []
for country in countries_with_usa:
    for sector in sectors:
        beta_fit = fit_beta_last_period_country_sector(country, sector)
        results.append(
            {"country": country.cou, "sector": sector, "beta_last_period": beta_fit}
        )

results_df = pd.DataFrame(results)
print(results_df)

# Save results to Excel
results_df.to_excel("../output/data/beta_last_period_results.xlsx", index=False)
"""
