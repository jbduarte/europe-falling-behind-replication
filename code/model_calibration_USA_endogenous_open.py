"""
=======================================================================================
Project: Structural Transformation and Productivity in Europe (with Buiatti and Duarte)
Filename: model_calibration_USA_open.py
Description: This program calibrates the BDS open economy modeland provides the test of
        the theory.

Author: Joao B. Duarte
Last Modified: Feb 2026
=======================================================================================
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, root, fsolve
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict
import numpy as np

rc("text", usetex=True)
rc("font", family="serif")

from model_calibration_USA import (
    sigma,
    eps_agr,
    eps_trd,
    eps_bss,
    eps_fin,
    eps_nps,
    eps_ser,
)

"""
-----------
	Data
-----------
"""

"KLEMS"
data = pd.read_csv("../data/euklems_2023.csv", index_col=[0, 1])
data.rename(index={"US": "USA"}, inplace=True)
data.rename(columns={"sector": "sec"}, inplace=True)

"OECD"
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

"Labor Productivity"
data["y_l"] = (data["VA_Q"] / data["H"]) * 100

"""
---------------
	US Data
---------------
"""
data = data.loc["USA"]
data = data[data.index >= 1995]

GDP_ph = GDP_ph.loc["USA"]
GDP_ph = GDP_ph[GDP_ph.index >= 1995]

data_NX_sec = data_NX_sec.loc["USA"]
data_NX_sec = data_NX_sec[data_NX_sec.index >= 1995]

data_NX_agg = data_NX_agg.loc["USA"]
data_NX_agg = data_NX_agg[data_NX_agg.index >= 1995]


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
c_GDP_ph, GDP_ph = sm.tsa.filters.hpfilter(GDP_ph["Value"], 100)

"GDP Growth"
g_GDP_ph = np.array(GDP_ph / GDP_ph.shift(1) - 1).flatten()  # GDP Growth from OECD

"Employment hours"
h_agr_c, h_agr = sm.tsa.filters.hpfilter(data_agr["H"], 100)
h_man_c, h_man = sm.tsa.filters.hpfilter(data_man["H"], 100)
h_trd_c, h_trd = sm.tsa.filters.hpfilter(data_trd["H"], 100)
h_bss_c, h_bss = sm.tsa.filters.hpfilter(data_bss["H"], 100)
h_fin_c, h_fin = sm.tsa.filters.hpfilter(data_fin["H"], 100)
h_nps_c, h_nps = sm.tsa.filters.hpfilter(data_nps["H"], 100)
h_ser_c, h_ser = sm.tsa.filters.hpfilter(data_ser["H"], 100)
h_tot_c, h_tot = sm.tsa.filters.hpfilter(data_tot["H"], 100)

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
g_y_l_agr = np.array(y_l_agr / y_l_agr.shift(1) - 1)
g_y_l_man = np.array(y_l_man / y_l_man.shift(1) - 1)
g_y_l_trd = np.array(y_l_trd / y_l_trd.shift(1) - 1)
g_y_l_bss = np.array(y_l_bss / y_l_bss.shift(1) - 1)
g_y_l_fin = np.array(y_l_fin / y_l_fin.shift(1) - 1)
g_y_l_nps = np.array(y_l_nps / y_l_nps.shift(1) - 1)
g_y_l_ser = np.array(y_l_ser / y_l_ser.shift(1) - 1)
g_y_l_tot = np.array(y_l_tot / y_l_tot.shift(1) - 1)

"Prices"
p_agr_c, p_agr = sm.tsa.filters.hpfilter(data_agr["VA"] / data_agr["VA_Q"], 100)
p_man_c, p_man = sm.tsa.filters.hpfilter(data_man["VA"] / data_man["VA_Q"], 100)
p_trd_c, p_trd = sm.tsa.filters.hpfilter(data_trd["VA"] / data_trd["VA_Q"], 100)
p_bss_c, p_bss = sm.tsa.filters.hpfilter(data_bss["VA"] / data_bss["VA_Q"], 100)
p_fin_c, p_fin = sm.tsa.filters.hpfilter(data_fin["VA"] / data_fin["VA_Q"], 100)
p_nps_c, p_nps = sm.tsa.filters.hpfilter(data_nps["VA"] / data_nps["VA_Q"], 100)
p_ser_c, p_ser = sm.tsa.filters.hpfilter(data_ser["VA"] / data_ser["VA_Q"], 100)
p_tot_c, p_tot = sm.tsa.filters.hpfilter(data_tot["VA"] / data_tot["VA_Q"], 100)

"Employment Shares"
share_c_agr, share_agr = sm.tsa.filters.hpfilter(
    (h_agr / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)
share_c_man, share_man = sm.tsa.filters.hpfilter(
    (h_man / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)
share_c_trd, share_trd = sm.tsa.filters.hpfilter(
    (h_trd / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)
share_c_bss, share_bss = sm.tsa.filters.hpfilter(
    (h_bss / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)
share_c_fin, share_fin = sm.tsa.filters.hpfilter(
    (h_fin / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)
share_c_nps, share_nps = sm.tsa.filters.hpfilter(
    (h_nps / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)
share_c_ser, share_ser = sm.tsa.filters.hpfilter(
    (h_ser / (h_agr + h_man + h_trd + h_bss + h_fin + h_nps)), 100
)

"Adjusted Value Added (Net of Exports)"
VA_adj_c_tot, VA_adj_tot = sm.tsa.filters.hpfilter(
    data_tot["VA"] - (data_tot["nx"]), 100
)

"Net Exports as a Share of Credited Expenditures"
nx_c_agr_E, nx_agr_E = sm.tsa.filters.hpfilter(
    data_agr["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)
nx_c_man_E, nx_man_E = sm.tsa.filters.hpfilter(
    data_man["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)
nx_c_trd_E, nx_trd_E = sm.tsa.filters.hpfilter(
    data_trd["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)
nx_c_bss_E, nx_bss_E = sm.tsa.filters.hpfilter(
    data_bss["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)
nx_c_fin_E, nx_fin_E = sm.tsa.filters.hpfilter(
    data_fin["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)
nx_c_nps_E, nx_nps_E = sm.tsa.filters.hpfilter(
    data_nps["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)
nx_ser_E = nx_trd_E + nx_bss_E + nx_fin_E + nx_nps_E
nx_c_tot_E, nx_tot_E = sm.tsa.filters.hpfilter(
    data_tot["nx"] / (data_tot["VA"] - data_tot["nx"]), 100
)

"Exports as a Share of Credited Expenditures"
x_c_agr_E, x_agr_E = sm.tsa.filters.hpfilter(
    data_agr["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
x_c_man_E, x_man_E = sm.tsa.filters.hpfilter(
    data_man["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
x_c_trd_E, x_trd_E = sm.tsa.filters.hpfilter(
    data_trd["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
x_c_bss_E, x_bss_E = sm.tsa.filters.hpfilter(
    data_bss["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
x_c_fin_E, x_fin_E = sm.tsa.filters.hpfilter(
    data_fin["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
x_c_nps_E, x_nps_E = sm.tsa.filters.hpfilter(
    data_nps["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
x_ser_E = x_trd_E + x_bss_E + x_fin_E + x_nps_E
x_c_tot_E, x_tot_E = sm.tsa.filters.hpfilter(
    data_tot["expo"] / (data_tot["VA"] - data_tot["nx"]), 100
)

"Imports as a Share of Credited Expenditures"
m_c_agr_E, m_agr_E = sm.tsa.filters.hpfilter(
    data_agr["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
m_c_man_E, m_man_E = sm.tsa.filters.hpfilter(
    data_man["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
m_c_trd_E, m_trd_E = sm.tsa.filters.hpfilter(
    data_trd["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
m_c_bss_E, m_bss_E = sm.tsa.filters.hpfilter(
    data_bss["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
m_c_fin_E, m_fin_E = sm.tsa.filters.hpfilter(
    data_fin["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
m_c_nps_E, m_nps_E = sm.tsa.filters.hpfilter(
    data_nps["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)
m_ser_E = m_trd_E + m_bss_E + m_fin_E + m_nps_E
m_c_tot_E, m_tot_E = sm.tsa.filters.hpfilter(
    data_tot["impo"] / (data_tot["VA"] - data_tot["nx"]), 100
)

"Net Exports as a Share of GDP"
nx_c_agr_Y, nx_agr_Y = sm.tsa.filters.hpfilter(data_agr["nx"] / (data_tot["gdp"]), 100)
nx_c_man_Y, nx_man_Y = sm.tsa.filters.hpfilter(data_man["nx"] / (data_tot["gdp"]), 100)
nx_c_trd_Y, nx_trd_Y = sm.tsa.filters.hpfilter(data_trd["nx"] / (data_tot["gdp"]), 100)
nx_c_bss_Y, nx_bss_Y = sm.tsa.filters.hpfilter(data_bss["nx"] / (data_tot["gdp"]), 100)
nx_c_fin_Y, nx_fin_Y = sm.tsa.filters.hpfilter(data_fin["nx"] / (data_tot["gdp"]), 100)
nx_c_nps_Y, nx_nps_Y = sm.tsa.filters.hpfilter(data_nps["nx"] / (data_tot["gdp"]), 100)
nx_ser_Y = nx_trd_Y + nx_bss_Y + nx_fin_Y + nx_nps_Y
nx_c_tot_Y, nx_tot_Y = sm.tsa.filters.hpfilter(data_tot["nx"] / (data_tot["gdp"]), 100)

"Exports as a Share of GDP"
x_c_agr_Y, x_agr_Y = sm.tsa.filters.hpfilter(data_agr["expo"] / (data_tot["gdp"]), 100)
x_c_man_Y, x_man_Y = sm.tsa.filters.hpfilter(data_man["expo"] / (data_tot["gdp"]), 100)
x_c_trd_Y, x_trd_Y = sm.tsa.filters.hpfilter(data_trd["expo"] / (data_tot["gdp"]), 100)
x_c_bss_Y, x_bss_Y = sm.tsa.filters.hpfilter(data_bss["expo"] / (data_tot["gdp"]), 100)
x_c_fin_Y, x_fin_Y = sm.tsa.filters.hpfilter(data_fin["expo"] / (data_tot["gdp"]), 100)
x_c_nps_Y, x_nps_Y = sm.tsa.filters.hpfilter(data_nps["expo"] / (data_tot["gdp"]), 100)
x_ser_Y = x_trd_Y + x_bss_Y + x_fin_Y + x_nps_Y
x_c_tot_Y, x_tot_Y = sm.tsa.filters.hpfilter(data_tot["expo"] / (data_tot["gdp"]), 100)

"Imports as a Share of GDP"
m_c_agr_Y, m_agr_Y = sm.tsa.filters.hpfilter(data_agr["impo"] / (data_tot["gdp"]), 100)
m_c_man_Y, m_man_Y = sm.tsa.filters.hpfilter(data_man["impo"] / (data_tot["gdp"]), 100)
m_c_trd_Y, m_trd_Y = sm.tsa.filters.hpfilter(data_trd["impo"] / (data_tot["gdp"]), 100)
m_c_bss_Y, m_bss_Y = sm.tsa.filters.hpfilter(data_bss["impo"] / (data_tot["gdp"]), 100)
m_c_fin_Y, m_fin_Y = sm.tsa.filters.hpfilter(data_fin["impo"] / (data_tot["gdp"]), 100)
m_c_nps_Y, m_nps_Y = sm.tsa.filters.hpfilter(data_nps["impo"] / (data_tot["gdp"]), 100)
m_ser_Y = m_trd_Y + m_bss_Y + m_fin_Y + m_nps_Y
m_c_tot_Y, m_tot_Y = sm.tsa.filters.hpfilter(data_tot["impo"] / (data_tot["gdp"]), 100)


# Compute imports as a share of GDP index for all sectors, starting at exports as a share of GDP in initial period,
# then propagate using observed growth rates of imports as share of GDP

sector_list = ["agr", "man", "trd", "bss", "fin", "nps", "ser", "tot"]

# Prepare observed series for each sector
x_Y_series = {sec: np.array(eval(f"x_{sec}_Y")) for sec in sector_list}
m_Y_series = {sec: np.array(eval(f"m_{sec}_Y")) for sec in sector_list}
m_Y_index = eval("m_man_Y").index  # All have same index

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
nx_Y_series = {sec: x_Y_series[sec] - im_share_index_all[sec] for sec in sector_list}

# Store each sector's net export series (adjusted) independently
nx_agr_Y_adj = nx_Y_series["agr"]
nx_man_Y_adj = nx_Y_series["man"]
nx_trd_Y_adj = nx_Y_series["trd"]
nx_bss_Y_adj = nx_Y_series["bss"]
nx_fin_Y_adj = nx_Y_series["fin"]
nx_nps_Y_adj = nx_Y_series["nps"]
nx_ser_Y_adj = nx_Y_series["ser"]
nx_tot_Y_adj = nx_Y_series["tot"]

"Real Net Exports"
nx_c_agr_q, nx_agr_q = sm.tsa.filters.hpfilter(
    data_agr["nx"] / (data_agr["VA"] / data_agr["VA_Q"]), 100
)
nx_c_man_q, nx_man_q = sm.tsa.filters.hpfilter(
    data_man["nx"] / (data_man["VA"] / data_man["VA_Q"]), 100
)
nx_c_trd_q, nx_trd_q = sm.tsa.filters.hpfilter(
    data_trd["nx"] / (data_trd["VA"] / data_trd["VA_Q"]), 100
)
nx_c_bss_q, nx_bss_q = sm.tsa.filters.hpfilter(
    data_bss["nx"] / (data_bss["VA"] / data_bss["VA_Q"]), 100
)
nx_c_fin_q, nx_fin_q = sm.tsa.filters.hpfilter(
    data_fin["nx"] / (data_fin["VA"] / data_fin["VA_Q"]), 100
)
nx_c_nps_q, nx_nps_q = sm.tsa.filters.hpfilter(
    data_nps["nx"] / (data_nps["VA"] / data_nps["VA_Q"]), 100
)
nx_ser_q = nx_trd_q + nx_bss_q + nx_fin_q + nx_nps_q
nx_c_tot_q, nx_tot_q = sm.tsa.filters.hpfilter(
    data_tot["nx"] / (data_tot["VA"] / data_tot["VA_Q"]), 100
)

"Real Exports"
x_c_agr_q, x_agr_q = sm.tsa.filters.hpfilter(
    data_agr["expo"] / (data_agr["VA"] / data_agr["VA_Q"]), 100
)
x_c_man_q, x_man_q = sm.tsa.filters.hpfilter(
    data_man["expo"] / (data_man["VA"] / data_man["VA_Q"]), 100
)
x_c_trd_q, x_trd_q = sm.tsa.filters.hpfilter(
    data_trd["expo"] / (data_trd["VA"] / data_trd["VA_Q"]), 100
)
x_c_bss_q, x_bss_q = sm.tsa.filters.hpfilter(
    data_bss["expo"] / (data_bss["VA"] / data_bss["VA_Q"]), 100
)
x_c_fin_q, x_fin_q = sm.tsa.filters.hpfilter(
    data_fin["expo"] / (data_fin["VA"] / data_fin["VA_Q"]), 100
)
x_c_nps_q, x_nps_q = sm.tsa.filters.hpfilter(
    data_nps["expo"] / (data_nps["VA"] / data_nps["VA_Q"]), 100
)
x_ser_q = x_trd_q + x_bss_q + x_fin_q + x_nps_q
x_c_tot_q, x_tot_q = sm.tsa.filters.hpfilter(
    data_tot["expo"] / (data_tot["VA"] / data_tot["VA_Q"]), 100
)

"Real Imports"
m_c_agr_q, m_agr_q = sm.tsa.filters.hpfilter(
    (-1) * data_agr["impo"] / (data_agr["VA"] / data_agr["VA_Q"]), 100
)
m_c_man_q, m_man_q = sm.tsa.filters.hpfilter(
    (-1) * data_man["impo"] / (data_man["VA"] / data_man["VA_Q"]), 100
)
m_c_trd_q, m_trd_q = sm.tsa.filters.hpfilter(
    (-1) * data_trd["impo"] / (data_trd["VA"] / data_trd["VA_Q"]), 100
)
m_c_bss_q, m_bss_q = sm.tsa.filters.hpfilter(
    (-1) * data_bss["impo"] / (data_bss["VA"] / data_bss["VA_Q"]), 100
)
m_c_fin_q, m_fin_q = sm.tsa.filters.hpfilter(
    (-1) * data_fin["impo"] / (data_fin["VA"] / data_fin["VA_Q"]), 100
)
m_c_nps_q, m_nps_q = sm.tsa.filters.hpfilter(
    (-1) * data_nps["impo"] / (data_nps["VA"] / data_nps["VA_Q"]), 100
)
m_ser_q = (-1) * (m_trd_q + m_bss_q + m_fin_q + m_nps_q)
m_c_tot_q, m_tot_q = sm.tsa.filters.hpfilter(
    (-1) * data_tot["impo"] / (data_tot["VA"] / data_tot["VA_Q"]), 100
)


# Compute real exports index for all sectors using growth rates from x_{sec}_q  starting at initial period of x_{sec}_E

sectors = ["agr", "man", "trd", "bss", "fin", "nps", "ser"]

# Prepare initial values for exports and imports for each sector
x_0 = {sec: np.array(eval(f"x_{sec}_Y"))[0] for sec in sectors}

# Prepare observed real exports and imports series for each sector
x_q_series = {sec: np.array(eval(f"x_{sec}_q")) for sec in sectors}

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

x_agr_index = x_index["agr"]
x_man_index = x_index["man"]
x_trd_index = x_index["trd"]
x_bss_index = x_index["bss"]
x_fin_index = x_index["fin"]
x_nps_index = x_index["nps"]
x_ser_index = x_index["ser"]

sector_list = ["agr", "man", "trd", "bss", "fin", "nps", "ser"]
M_E = {}

for sec in sector_list:
    x_index_sec = eval(f"x_{sec}_index")
    m_Y_sec = eval(f"m_{sec}_Y")

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
M_agr_E = M_E["agr"]
M_man_E = M_E["man"]
M_trd_E = M_E["trd"]
M_bss_E = M_E["bss"]
M_fin_E = M_E["fin"]
M_nps_E = M_E["nps"]
M_ser_E = M_E["ser"]

"""
---------------------------------------------
		Time Series (Inputs of the Model)
---------------------------------------------
"""

t_0 = np.array(data.index)[0]
ts_length = np.array(data.index)[-1] - np.array(data.index)[0]

"First period. Normalization."
A_agr = [1]
A_man = [1]
A_bss = [1]
A_trd = [1]
A_fin = [1]
A_nps = [1]
A_res = [1]
A_nps = [1]
A_ser = [1]
A_tot = [1]
year = [t_0]
"Productivity and Real Expenditure Growth"
for i in range(int(ts_length)):
    A_agr.append((1 + g_y_l_agr[i + 1]) * A_agr[i])
    A_man.append((1 + g_y_l_man[i + 1]) * A_man[i])
    A_bss.append((1 + g_y_l_bss[i + 1]) * A_bss[i])
    A_trd.append((1 + g_y_l_trd[i + 1]) * A_trd[i])
    A_fin.append((1 + g_y_l_fin[i + 1]) * A_fin[i])
    A_nps.append((1 + g_y_l_nps[i + 1]) * A_nps[i])
    A_ser.append((1 + g_y_l_ser[i + 1]) * A_ser[i])
    A_tot.append((1 + g_y_l_tot[i + 1]) * A_tot[i])
    year.append(t_0 + i + 1)

"Expenditures"
g_nom_exp = np.array(VA_adj_tot / VA_adj_tot.shift(1) - 1).flatten()
E = [1]
for i in range(int(ts_length)):
    E.append((1 + g_nom_exp[i + 1]) * E[i])

# Compute x_man_q_index from growth rates of x_man_q, starting at x_man_E[0]
sector_list = ["agr", "man", "trd", "bss", "fin", "nps", "ser", "tot"]
x_q_series = {sec: np.array(eval(f"x_{sec}_q")) for sec in sector_list}
x_E_series = {sec: np.array(eval(f"x_{sec}_E")) for sec in sector_list}
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
m_q_series = {sec: np.array(eval(f"m_{sec}_q")) for sec in sector_list}
m_q_index = {}
for sec in sector_list:
    m_q_arr = m_q_series[sec]
    x_E0 = x_E_series[sec][0]
    idx = [x_E0]
    for i in range(1, len(m_q_arr)):
        g = m_q_arr[i] / m_q_arr[i - 1] if m_q_arr[i - 1] != 0 else 1
        idx.append(idx[-1] * g)
    m_q_index[sec] = np.array(idx)

x_agr_q_index = np.array(x_q_index["agr"])
x_man_q_index = np.array(x_q_index["man"])
x_trd_q_index = np.array(x_q_index["trd"])
x_bss_q_index = np.array(x_q_index["bss"])
x_fin_q_index = np.array(x_q_index["fin"])
x_nps_q_index = np.array(x_q_index["nps"])
x_ser_q_index = np.array(x_q_index["ser"])

m_agr_q_index = np.array(m_q_index["agr"])
m_man_q_index = np.array(m_q_index["man"])
m_trd_q_index = np.array(m_q_index["trd"])
m_bss_q_index = np.array(m_q_index["bss"])
m_fin_q_index = np.array(m_q_index["fin"])
m_nps_q_index = np.array(m_q_index["nps"])
m_ser_q_index = np.array(m_q_index["ser"])

nx_agr_q_index = x_agr_q_index - m_agr_q_index
nx_man_q_index = x_man_q_index - m_man_q_index
nx_trd_q_index = x_trd_q_index - m_trd_q_index
nx_bss_q_index = x_bss_q_index - m_bss_q_index
nx_fin_q_index = x_fin_q_index - m_fin_q_index
nx_nps_q_index = x_nps_q_index - m_nps_q_index
nx_ser_q_index = x_ser_q_index - m_ser_q_index


def xi_fit_sector(x_q_index, A):
    # Objective: minimize distance at last period
    def objective(xi):
        model_last = x_q_index[0] * A[-1] ** xi
        data_last = x_q_index[-1]
        return (model_last - data_last) ** 2  # squared error

    res = minimize_scalar(objective, method="bounded", bounds=(0, 10))
    return res.x  # returns xi


xi_agr = xi_fit_sector(x_agr_q_index, A_agr)
xi_man = xi_fit_sector(x_man_q_index, A_man)
xi_trd = xi_fit_sector(x_trd_q_index, A_trd)
xi_bss = xi_fit_sector(x_bss_q_index, A_bss)
xi_fin = xi_fit_sector(x_fin_q_index, A_fin)
xi_nps = xi_fit_sector(x_nps_q_index, A_nps)
xi_ser = xi_fit_sector(x_ser_q_index, A_ser)

print("Estimated xi parameters:")
print("xi_agr:", xi_agr)
print("xi_man:", xi_man)
print("xi_trd:", xi_trd)
print("xi_bss:", xi_bss)
print("xi_fin:", xi_fin)
print("xi_nps:", xi_nps)
print("xi_ser:", xi_ser)


om_agr = np.array(share_agr)[0]
om_man = np.array(share_man)[0]
om_trd = np.array(share_trd)[0]
om_bss = np.array(share_bss)[0]
om_fin = np.array(share_fin)[0]
om_nps = np.array(share_nps)[0]
om_ser = np.array(share_ser)[0]

"Non-homothetic CES index"


def C_index(om_i, li, xn_i, lm, xn_m, pi_pm, sigma, epsilon_i):
    C_level = (
        (om_man / om_i) * ((li - xn_i) / (lm - xn_m)) * (pi_pm ** (sigma - 1))
    ) ** (1 / (epsilon_i - 1))
    g_C = np.array(C_level / C_level.shift(1) - 1)
    C = [1]
    for i in range(len(g_C) - 1):
        C.append((1 + g_C[i + 1]) * C[i])
    return C


C = C_index(
    (1 - om_agr - om_man),
    h_ser,
    nx_ser_q_index,
    h_man,
    nx_man_q_index,
    p_ser / p_man,
    sigma,
    eps_ser,
)

C_closed = C_index(
    (1 - om_agr - om_man),
    h_ser,
    0,
    h_man,
    0,
    p_ser / p_man,
    sigma,
    eps_ser,
)

"""
-------------------------
		The Models
-------------------------
"""


class model_ams:
    "Structural Transformation with Agriculture, Manufacturing and Services"

    def __init__(
        self,
        sigma=sigma,
        eps_agr=eps_agr,
        eps_man=1,
        eps_ser=eps_ser,
        om_agr=om_agr,
        om_man=om_man,
        om_ser=om_ser,
        xi_agr=xi_agr,
        xi_man=xi_man,
        xi_ser=xi_ser,
    ):
        "Initialize the Parameters"
        (
            self.sigma,
            self.eps_agr,
            self.eps_man,
            self.eps_ser,
            self.om_agr,
            self.om_man,
            self.om_ser,
            self.xi_agr,
            self.xi_man,
            self.xi_ser,
        ) = (
            sigma,
            eps_agr,
            1,
            eps_ser,
            om_agr,
            om_man,
            om_ser,
            xi_agr,
            xi_man,
            xi_ser,
        )

    def labor_agr(self, C, E, A_agr, M_agr_E):
        "Labor Demand in Agriculture"
        return self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_agr_q_index[0] * A_agr ** (self.xi_agr - 1)) / E - M_agr_E)

    def labor_man(self, C, E, A_man, M_man_E):
        "Labor Demand in Manufacturing"
        return self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_man_q_index[0] * A_man ** (self.xi_man - 1)) / E - M_man_E)

    def labor_ser(self, C, E, A_ser, M_ser_E):
        "Labor Demand in Services"
        return self.om_ser * (C**self.eps_ser) * (A_ser ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_ser_q_index[0] * A_ser ** (self.xi_ser - 1)) / E - M_ser_E)

    def agg_labor_demand(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Aggregate Labor Demand"
        return (
            self.labor_agr(C, E, A_agr, M_agr_E)
            + self.labor_man(C, E, A_man, M_man_E)
            + self.labor_ser(C, E, A_ser, M_ser_E)
        )

    def share_agr(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Employment Share in Agriculture"
        return self.labor_agr(C, E, A_agr, M_agr_E) / self.agg_labor_demand(
            C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
        )

    def share_man(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Employment Share in Manufacturing"
        return self.labor_man(C, E, A_man, M_man_E) / self.agg_labor_demand(
            C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
        )

    def share_ser(self, C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E):
        "Employment Share in Services"
        return self.labor_ser(C, E, A_ser, M_ser_E) / self.agg_labor_demand(
            C, E, A_agr, A_man, A_ser, M_agr_E, M_man_E, M_ser_E
        )


class model_nps:
    "Structural Transformation with Progressive and Non-Progressive Services"

    def __init__(
        self,
        sigma=sigma,
        eps_agr=eps_agr,
        eps_man=1,
        eps_trd=eps_trd,
        eps_bss=eps_bss,
        eps_fin=eps_fin,
        eps_nps=eps_nps,
        om_agr=om_agr,
        om_man=om_man,
        om_trd=om_trd,
        om_bss=om_bss,
        om_fin=om_fin,
        om_nps=om_nps,
        xi_agr=xi_agr,
        xi_man=xi_man,
        xi_trd=xi_trd,
        xi_bss=xi_bss,
        xi_fin=xi_fin,
        xi_nps=xi_nps,
    ):
        "Initialize the Parameters"
        (
            self.sigma,
            self.eps_agr,
            self.eps_man,
            self.eps_trd,
            self.eps_bss,
            self.eps_fin,
            self.eps_nps,
            self.om_agr,
            self.om_man,
            self.om_trd,
            self.om_bss,
            self.om_fin,
            self.om_nps,
            self.xi_agr,
            self.xi_man,
            self.xi_trd,
            self.xi_bss,
            self.xi_fin,
            self.xi_nps,
        ) = (
            sigma,
            eps_agr,
            1,
            eps_trd,
            eps_bss,
            eps_fin,
            eps_nps,
            om_agr,
            om_man,
            om_trd,
            om_bss,
            om_fin,
            om_nps,
            xi_agr,
            xi_man,
            xi_trd,
            xi_bss,
            xi_fin,
            xi_nps,
        )

    def labor_agr(self, C, E, A_agr, M_agr_E):
        "Labor Demand in Agriculture"
        return self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_agr_index[0] * A_agr ** (self.xi_agr - 1)) / E - M_agr_E)

    def labor_man(self, C, E, A_man, M_man_E):
        "Labor Demand in Manufacturing"
        return self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_man_index[0] * A_man ** (self.xi_man - 1)) / E - M_man_E)

    def labor_trd(self, C, E, A_trd, M_trd_E):
        "Labor Demand in Trade Services"
        return self.om_trd * (C**self.eps_trd) * (A_trd ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_trd_index[0] * A_trd ** (self.xi_trd - 1)) / E - M_trd_E)

    def labor_bss(self, C, E, A_bss, M_bss_E):
        "Labor Demand in Business Services"
        return self.om_bss * (C**self.eps_bss) * (A_bss ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_bss_index[0] * A_bss ** (self.xi_bss - 1)) / E - M_bss_E)

    def labor_fin(self, C, E, A_fin, M_fin_E):
        "Labor Demand in Financial Services"
        return self.om_fin * (C**self.eps_fin) * (A_fin ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_fin_index[0] * A_fin ** (self.xi_fin - 1)) / E - M_fin_E)

    def labor_nps(self, C, E, A_nps, M_nps_E):
        "Labor Demand in Non-Progressive Services"
        return self.om_nps * (C**self.eps_nps) * (A_nps ** (self.sigma - 1)) + (
            E ** (1 - self.sigma)
        ) * ((x_nps_index[0] * A_nps ** (self.xi_nps - 1)) / E - M_nps_E)

    def agg_labor_demand(
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

    def share_agr(
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
        return self.labor_agr(C, E, A_agr, M_agr_E) / self.agg_labor_demand(
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

    def share_man(
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
        return self.labor_man(C, E, A_man, M_man_E) / self.agg_labor_demand(
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

    def share_trd(
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
        return self.labor_trd(C, E, A_trd, M_trd_E) / self.agg_labor_demand(
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

    def share_bss(
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
        return self.labor_bss(C, E, A_bss, M_bss_E) / self.agg_labor_demand(
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

    def share_fin(
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
        return self.labor_fin(C, E, A_fin, M_fin_E) / self.agg_labor_demand(
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

    def share_nps(
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
        return self.labor_nps(C, E, A_nps, M_nps_E) / self.agg_labor_demand(
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


class model_ams_closed:
    "Structural Transformation with Agriculture, Manufacturing and Services (Closed Economy)"

    def __init__(
        self,
        sigma=sigma,
        eps_agr=eps_agr,
        eps_man=1,
        eps_ser=eps_ser,
        om_agr=om_agr,
        om_man=om_man,
        om_ser=om_ser,
    ):
        "Initialize the Parameters"
        (
            self.sigma,
            self.eps_agr,
            self.eps_man,
            self.eps_ser,
            self.om_agr,
            self.om_man,
            self.om_ser,
        ) = (
            sigma,
            eps_agr,
            1,
            eps_ser,
            om_agr,
            om_man,
            om_ser,
        )

    def labor_agr(self, C, A_agr):
        "Labor Demand in Agriculture"
        return self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))

    def labor_man(self, C, A_man):
        "Labor Demand in Manufacturing"
        return self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1))

    def labor_ser(self, C, A_ser):
        "Labor Demand in Services"
        return self.om_ser * (C**self.eps_ser) * (A_ser ** (self.sigma - 1))

    def agg_labor_demand(self, C, A_agr, A_man, A_ser):
        "Aggregate Labor Demand"
        return (
            self.labor_agr(C, A_agr)
            + self.labor_man(C, A_man)
            + self.labor_ser(C, A_ser)
        )

    def share_agr(self, C, A_agr, A_man, A_ser):
        "Employment Share in Agriculture"
        return self.labor_agr(C, A_agr) / self.agg_labor_demand(C, A_agr, A_man, A_ser)

    def share_man(self, C, A_agr, A_man, A_ser):
        "Employment Share in Manufacturing"
        return self.labor_man(C, A_man) / self.agg_labor_demand(C, A_agr, A_man, A_ser)

    def share_ser(self, C, A_agr, A_man, A_ser):
        "Employment Share in Services"
        return self.labor_ser(C, A_ser) / self.agg_labor_demand(C, A_agr, A_man, A_ser)


class model_nps_closed:
    "Structural Transformation with Progressive and Non-Progressive Services (Closed Economy)"

    def __init__(
        self,
        sigma=sigma,
        eps_agr=eps_agr,
        eps_man=1,
        eps_trd=eps_trd,
        eps_bss=eps_bss,
        eps_fin=eps_fin,
        eps_nps=eps_nps,
        om_agr=om_agr,
        om_man=om_man,
        om_trd=om_trd,
        om_bss=om_bss,
        om_fin=om_fin,
        om_nps=om_nps,
    ):
        "Initialize the Parameters"
        (
            self.sigma,
            self.eps_agr,
            self.eps_man,
            self.eps_trd,
            self.eps_bss,
            self.eps_fin,
            self.eps_nps,
            self.om_agr,
            self.om_man,
            self.om_trd,
            self.om_bss,
            self.om_fin,
            self.om_nps,
        ) = (
            sigma,
            eps_agr,
            1,
            eps_trd,
            eps_bss,
            eps_fin,
            eps_nps,
            om_agr,
            om_man,
            om_trd,
            om_bss,
            om_fin,
            om_nps,
        )

    def labor_agr(self, C, A_agr):
        "Labor Demand in Agriculture"
        return self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))

    def labor_man(self, C, A_man):
        "Labor Demand in Manufacturing"
        return self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1))

    def labor_trd(self, C, A_trd):
        "Labor Demand in Trade Services"
        return self.om_trd * (C**self.eps_trd) * (A_trd ** (self.sigma - 1))

    def labor_bss(self, C, A_bss):
        "Labor Demand in Business Services"
        return self.om_bss * (C**self.eps_bss) * (A_bss ** (self.sigma - 1))

    def labor_fin(self, C, A_fin):
        "Labor Demand in Financial Services"
        return self.om_fin * (C**self.eps_fin) * (A_fin ** (self.sigma - 1))

    def labor_nps(self, C, A_nps):
        "Labor Demand in Non-Progressive Services"
        return self.om_nps * (C**self.eps_nps) * (A_nps ** (self.sigma - 1))

    def agg_labor_demand(
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
            self.labor_agr(C, A_agr)
            + self.labor_man(C, A_man)
            + self.labor_trd(C, A_trd)
            + self.labor_bss(C, A_bss)
            + self.labor_fin(C, A_fin)
            + self.labor_nps(C, A_nps)
        )

    def share_agr(
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
        return self.labor_agr(C, A_agr) / self.agg_labor_demand(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_man(
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
        return self.labor_man(C, A_man) / self.agg_labor_demand(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_trd(
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
        return self.labor_trd(C, A_trd) / self.agg_labor_demand(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_bss(
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
        return self.labor_bss(C, A_bss) / self.agg_labor_demand(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_fin(
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
        return self.labor_fin(C, A_fin) / self.agg_labor_demand(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )

    def share_nps(
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
        return self.labor_nps(C, A_nps) / self.agg_labor_demand(
            C,
            A_agr,
            A_man,
            A_trd,
            A_bss,
            A_fin,
            A_nps,
        )


"Model projections"

ams = model_ams()
share_agr_ams = [
    ams.share_agr(
        C[0], E[0], A_agr[0], A_man[0], A_ser[0], M_agr_E[0], M_man_E[0], M_ser_E[0]
    )
]
share_man_ams = [
    ams.share_man(
        C[0], E[0], A_agr[0], A_man[0], A_ser[0], M_agr_E[0], M_man_E[0], M_ser_E[0]
    )
]
share_ser_ams = [
    ams.share_ser(
        C[0], E[0], A_agr[0], A_man[0], A_ser[0], M_agr_E[0], M_man_E[0], M_ser_E[0]
    )
]

for i in range(int(ts_length)):
    share_agr_ams.append(
        ams.share_agr(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_ser[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_ser_E[i + 1],
        )
    )
    share_man_ams.append(
        ams.share_man(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_ser[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_ser_E[i + 1],
        )
    )
    share_ser_ams.append(
        ams.share_ser(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_ser[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_ser_E[i + 1],
        )
    )


ams_closed = model_ams_closed()
share_agr_ams_closed = [ams_closed.share_agr(C_closed[0], A_agr[0], A_man[0], A_ser[0])]
share_man_ams_closed = [ams_closed.share_man(C_closed[0], A_agr[0], A_man[0], A_ser[0])]
share_ser_ams_closed = [ams_closed.share_ser(C_closed[0], A_agr[0], A_man[0], A_ser[0])]

for i in range(int(ts_length)):
    share_agr_ams_closed.append(
        ams_closed.share_agr(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_ser[i + 1],
        )
    )
    share_man_ams_closed.append(
        ams_closed.share_man(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_ser[i + 1],
        )
    )
    share_ser_ams_closed.append(
        ams_closed.share_ser(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_ser[i + 1],
        )
    )


# Model projections for NPS model
nps = model_nps()
share_agr_nps = [
    nps.share_agr(
        C[0],
        E[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
        M_agr_E[0],
        M_man_E[0],
        M_trd_E[0],
        M_bss_E[0],
        M_fin_E[0],
        M_nps_E[0],
    )
]
share_man_nps = [
    nps.share_man(
        C[0],
        E[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
        M_agr_E[0],
        M_man_E[0],
        M_trd_E[0],
        M_bss_E[0],
        M_fin_E[0],
        M_nps_E[0],
    )
]
share_trd_nps = [
    nps.share_trd(
        C[0],
        E[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
        M_agr_E[0],
        M_man_E[0],
        M_trd_E[0],
        M_bss_E[0],
        M_fin_E[0],
        M_nps_E[0],
    )
]
share_bss_nps = [
    nps.share_bss(
        C[0],
        E[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
        M_agr_E[0],
        M_man_E[0],
        M_trd_E[0],
        M_bss_E[0],
        M_fin_E[0],
        M_nps_E[0],
    )
]
share_fin_nps = [
    nps.share_fin(
        C[0],
        E[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
        M_agr_E[0],
        M_man_E[0],
        M_trd_E[0],
        M_bss_E[0],
        M_fin_E[0],
        M_nps_E[0],
    )
]
share_nps_nps = [
    nps.share_nps(
        C[0],
        E[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
        M_agr_E[0],
        M_man_E[0],
        M_trd_E[0],
        M_bss_E[0],
        M_fin_E[0],
        M_nps_E[0],
    )
]

for i in range(int(ts_length)):
    share_agr_nps.append(
        nps.share_agr(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_trd_E[i + 1],
            M_bss_E[i + 1],
            M_fin_E[i + 1],
            M_nps_E[i + 1],
        )
    )
    share_man_nps.append(
        nps.share_man(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_trd_E[i + 1],
            M_bss_E[i + 1],
            M_fin_E[i + 1],
            M_nps_E[i + 1],
        )
    )
    share_trd_nps.append(
        nps.share_trd(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_trd_E[i + 1],
            M_bss_E[i + 1],
            M_fin_E[i + 1],
            M_nps_E[i + 1],
        )
    )
    share_bss_nps.append(
        nps.share_bss(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_trd_E[i + 1],
            M_bss_E[i + 1],
            M_fin_E[i + 1],
            M_nps_E[i + 1],
        )
    )
    share_fin_nps.append(
        nps.share_fin(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_trd_E[i + 1],
            M_bss_E[i + 1],
            M_fin_E[i + 1],
            M_nps_E[i + 1],
        )
    )
    share_nps_nps.append(
        nps.share_nps(
            C[i + 1],
            E[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
            M_agr_E[i + 1],
            M_man_E[i + 1],
            M_trd_E[i + 1],
            M_bss_E[i + 1],
            M_fin_E[i + 1],
            M_nps_E[i + 1],
        )
    )


# Model projections for NPS model closed economys
nps_closed = model_nps_closed()
share_agr_nps_closed = [
    nps_closed.share_agr(
        C_closed[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
    )
]
share_man_nps_closed = [
    nps_closed.share_man(
        C_closed[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
    )
]
share_trd_nps_closed = [
    nps_closed.share_trd(
        C_closed[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
    )
]
share_bss_nps_closed = [
    nps_closed.share_bss(
        C_closed[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
    )
]
share_fin_nps_closed = [
    nps_closed.share_fin(
        C_closed[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
    )
]
share_nps_nps_closed = [
    nps_closed.share_nps(
        C_closed[0],
        A_agr[0],
        A_man[0],
        A_trd[0],
        A_bss[0],
        A_fin[0],
        A_nps[0],
    )
]

for i in range(int(ts_length)):
    share_agr_nps_closed.append(
        nps_closed.share_agr(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_man_nps_closed.append(
        nps_closed.share_man(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_trd_nps_closed.append(
        nps_closed.share_trd(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_bss_nps_closed.append(
        nps_closed.share_bss(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_fin_nps_closed.append(
        nps_closed.share_fin(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_nps_nps_closed.append(
        nps_closed.share_nps(
            C_closed[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )


"Aggregate Productivity"
weighted_ams_A_agr = [a * b for a, b in zip(share_agr_ams, A_agr)]
weighted_ams_A_man = [a * b for a, b in zip(share_man_ams, A_man)]
weighted_ams_A_ser = [a * b for a, b in zip(share_ser_ams, A_ser)]

A_tot_ams = [
    sum(x) for x in zip(weighted_ams_A_agr, weighted_ams_A_man, weighted_ams_A_ser)
]
A_tot_ams_weighted = share_agr * A_agr + share_man * A_man + share_ser * A_ser

"Aggregate Productivity"
weighted_ams_A_agr_closed = [a * b for a, b in zip(share_agr_ams_closed, A_agr)]
weighted_ams_A_man_closed = [a * b for a, b in zip(share_man_ams_closed, A_man)]
weighted_ams_A_ser_closed = [a * b for a, b in zip(share_ser_ams_closed, A_ser)]

A_tot_ams_closed = [
    sum(x)
    for x in zip(
        weighted_ams_A_agr_closed, weighted_ams_A_man_closed, weighted_ams_A_ser_closed
    )
]


weighted_agr_nps_A_agr = [a * b for a, b in zip(share_agr_nps, A_agr)]
weighted_man_nps_A_man = [a * b for a, b in zip(share_man_nps, A_man)]
weighted_trd_nps_A_trd = [a * b for a, b in zip(share_trd_nps, A_trd)]
weighted_bss_nps_A_bss = [a * b for a, b in zip(share_bss_nps, A_bss)]
weighted_fin_nps_A_bss = [a * b for a, b in zip(share_fin_nps, A_fin)]
weighted_nps_nps_A_nps = [a * b for a, b in zip(share_nps_nps, A_nps)]

A_tot_nps = [
    sum(x)
    for x in zip(
        weighted_agr_nps_A_agr,
        weighted_man_nps_A_man,
        weighted_trd_nps_A_trd,
        weighted_bss_nps_A_bss,
        weighted_fin_nps_A_bss,
        weighted_nps_nps_A_nps,
    )
]
A_tot_nps_weighted = (
    share_agr * A_agr
    + share_man * A_man
    + share_trd * A_trd
    + share_bss * A_bss
    + share_fin * A_fin
    + share_nps * A_nps
)


weighted_agr_nps_A_agr_closed = [a * b for a, b in zip(share_agr_nps_closed, A_agr)]
weighted_man_nps_A_man_closed = [a * b for a, b in zip(share_man_nps_closed, A_man)]
weighted_trd_nps_A_trd_closed = [a * b for a, b in zip(share_trd_nps_closed, A_trd)]
weighted_bss_nps_A_bss_closed = [a * b for a, b in zip(share_bss_nps_closed, A_bss)]
weighted_fin_nps_A_bss_closed = [a * b for a, b in zip(share_fin_nps_closed, A_fin)]
weighted_nps_nps_A_nps_closed = [a * b for a, b in zip(share_nps_nps_closed, A_nps)]

A_tot_nps_closed = [
    sum(x)
    for x in zip(
        weighted_agr_nps_A_agr_closed,
        weighted_man_nps_A_man_closed,
        weighted_trd_nps_A_trd_closed,
        weighted_bss_nps_A_bss_closed,
        weighted_fin_nps_A_bss_closed,
        weighted_nps_nps_A_nps_closed,
    )
]
