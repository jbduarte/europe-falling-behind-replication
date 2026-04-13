"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        model_calibration_USA.py
Purpose:     Calibrate the baseline closed-economy non-homothetic CES model on US
             data: recover the price elasticity sigma, the sectoral non-homotheticity
             parameters (eps_agr, eps_trd, eps_bss, eps_fin, eps_nps, eps_ser), and
             the sectoral productivity levels (A_agr, A_man, ..., A_tot) that
             rationalize observed US labor reallocation.
Pipeline:    Step 1/19 — Baseline closed-economy calibration on US.
Inputs:      ../data/euklems_2023.csv (EUKLEMS 2023, VA_Q and H by sector)
             ../data/raw/OECD_GDP_ph.xlsx (OECD GDP per hour, constant PPP USD)
Outputs:     Module-level variables imported downstream: sigma, eps_*, A_*, E,
             GDP, share_* (sectoral employment shares), plus the 6-sector
             aggregates A_tot_ams / A_tot_nps used for the "more aggregated" and
             "no-personal-services" robustness specifications.
Dependencies: None (this is the root of the closed-economy branch).
"""

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex=True)
rc("font", family="serif")


"""
-----------
	Data
-----------
"""

"KLEMS"
data = pd.read_csv("../data/euklems_2023.csv", index_col=[0, 1])
data.rename(index={"US": "USA"}, inplace=True)

"OECD"
GDP_ph = pd.read_excel(
    "../data/raw/OECD_GDP_ph.xlsx", index_col=[0, 5], engine="openpyxl"
)  # Measured in USD (constant prices 2010 and PPPs).
GDP_ph = GDP_ph[GDP_ph["MEASURE"] == "USD"]
GDP_ph.index.rename(["country", "year"])

"Labor Productivity"
data["y_l"] = (data["VA_Q"] / data["H"]) * 100

"US data"
data = data.loc["USA"]
GDP_ph = GDP_ph.loc["USA"]

"Sectoral Data"
data_agr = data[data["sector"] == "agr"]
data_man = data[data["sector"] == "man"]
data_trd = data[data["sector"] == "trd"]
data_bss = data[data["sector"] == "bss"]
data_fin = data[data["sector"] == "fin"]
data_nps = data[data["sector"] == "nps"]
data_ser = data[data["sector"] == "ser"]
data_tot = data[data["sector"] == "tot"]


"""
-------------------------------
	Time Series (Filtering)
-------------------------------
"""

"GDP"
c_GDP, GDP = sm.tsa.filters.hpfilter(GDP_ph["Value"], 100)

"GDP Growth"
g_GDP = np.array(GDP / GDP.shift(1) - 1).flatten()  # GDP Growth from OECD

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

"Employment Shares Without Manufacturing (Weights of C)"
share_c_agr_no_man, share_agr_no_man = sm.tsa.filters.hpfilter(
    h_agr / (h_agr + h_trd + h_bss + h_fin + h_nps), 100
)
share_c_trd_no_man, share_trd_no_man = sm.tsa.filters.hpfilter(
    h_trd / (h_agr + h_trd + h_bss + h_fin + h_nps), 100
)
share_c_bss_no_man, share_bss_no_man = sm.tsa.filters.hpfilter(
    h_bss / (h_agr + h_trd + h_bss + h_fin + h_nps), 100
)
share_c_fin_no_man, share_fin_no_man = sm.tsa.filters.hpfilter(
    h_fin / (h_agr + h_trd + h_bss + h_fin + h_nps), 100
)
share_c_nps_no_man, share_nps_no_man = sm.tsa.filters.hpfilter(
    h_nps / (h_agr + h_trd + h_bss + h_fin + h_nps), 100
)
share_c_ser_no_man, share_ser_no_man = sm.tsa.filters.hpfilter(
    h_ser / (h_agr + h_trd + h_bss + h_fin + h_nps), 100
)

"Employment Shares Without Agriculture and Manufacturing (Weights of C)"
share_c_trd_no_agm, share_trd_no_agm = sm.tsa.filters.hpfilter(
    h_trd / (h_trd + h_bss + h_fin + h_nps), 100
)
share_c_bss_no_agm, share_bss_no_agm = sm.tsa.filters.hpfilter(
    h_bss / (h_trd + h_bss + h_fin + h_nps), 100
)
share_c_fin_no_agm, share_fin_no_agm = sm.tsa.filters.hpfilter(
    h_fin / (h_trd + h_bss + h_fin + h_nps), 100
)
share_c_nps_no_agm, share_nps_no_agm = sm.tsa.filters.hpfilter(
    h_nps / (h_trd + h_bss + h_fin + h_nps), 100
)
share_c_ser_no_agm, share_ser_no_agm = sm.tsa.filters.hpfilter(
    h_ser / (h_trd + h_bss + h_fin + h_nps), 100
)


""" 
------------------------
	Parameterization 
------------------------
"""

"Initial employment shares"
om_agr = np.array(share_agr)[0]
om_man = np.array(share_man)[0]
om_trd = np.array(share_trd)[0]
om_bss = np.array(share_bss)[0]
om_fin = np.array(share_fin)[0]
om_nps = np.array(share_nps)[0]
om_ser = np.array(share_ser)[0]

"Relative labor demand"
l_agr_l_man = h_agr / h_man
l_trd_l_man = h_trd / h_man
l_bss_l_man = h_bss / h_man
l_fin_l_man = h_fin / h_man
l_nps_l_man = h_nps / h_man
l_ser_l_man = h_ser / h_man

l_agr_l_man_last = np.array(l_agr_l_man)[-1]
l_trd_l_man_last = np.array(l_trd_l_man)[-1]
l_bss_l_man_last = np.array(l_bss_l_man)[-1]
l_fin_l_man_last = np.array(l_fin_l_man)[-1]
l_nps_l_man_last = np.array(l_nps_l_man)[-1]
l_ser_l_man_last = np.array(l_ser_l_man)[-1]

"Relative prices"
p_agr_p_man = p_agr / p_man
p_trd_p_man = p_trd / p_man
p_bss_p_man = p_bss / p_man
p_fin_p_man = p_fin / p_man
p_nps_p_man = p_nps / p_man
p_ser_p_man = p_ser / p_man

p_agr_p_man_last = np.array(p_agr_p_man)[-1]
p_trd_p_man_last = np.array(p_trd_p_man)[-1]
p_bss_p_man_last = np.array(p_bss_p_man)[-1]
p_fin_p_man_last = np.array(p_fin_p_man)[-1]
p_nps_p_man_last = np.array(p_nps_p_man)[-1]
p_ser_p_man_last = np.array(p_ser_p_man)[-1]

"Expenditure relative to manufacturing prices"
E_pm_c, E_pm = sm.tsa.filters.hpfilter((data_tot["VA"] * 100) / p_man, 100)
E_pm_last = np.array(E_pm)[-1]

"Manufacturing Value Added Share"
man_va_share_c, man_va_share = sm.tsa.filters.hpfilter(
    data_man["VA"] / data_tot["VA"], 100
)
man_va_share_last = np.array(man_va_share)[-1]


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

E = [1]
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
    E.append((1 + g_GDP[i + 1]) * E[i])
    year.append(t_0 + i + 1)


"""
-------------------
	Calibration
-------------------
"""
# Identification strategy (Comin, Lashkari, Mestieri 2021 ECMA, adapted):
# with linear-in-labor production y_i = A_i ell_i, the first-order conditions
# of the non-homothetic CES deliver a log-linear relative-labor equation for
# every sector i paired with the numeraire (manufacturing). We exploit two
# moments of that equation in the terminal year:
#   (i) Fix eps_ser at the Comin et al. (2021) Table 9 value (eps_ser = 1.2)
#       and invert the services-vs-manufacturing equation for sigma.
#   (ii) Given sigma, invert the i-vs-manufacturing equation for each remaining
#        non-homotheticity eps_i (i in {agr, trd, bss, fin, nps}).
# (The commented-out grid-search below was the previous strategy that treated
#  eps_agr as free and minimized the residual between the induced CES
#  consumption index C and measured real expenditure E; we keep it for
#  reference but the published calibration uses the closed-form inversion.)

"""
-----------------
	Functions
-----------------
"""

"Price elasticity: closed-form inversion of the services-vs-manufacturing"
"log relative-labor equation for sigma, given eps_s and terminal-year data."


def sigma_ft(eps_s):
    nom = (
        np.log(l_ser_l_man_last)
        - np.log(om_ser / om_man)
        - (eps_s - 1) * np.log(man_va_share_last / om_man)
    )
    den = np.log(p_ser_p_man_last) + (eps_s - 1) * np.log(E_pm_last)
    return 1 - nom / den


"Sectoral non-homotheticity (income elasticity): closed-form inversion of"
"the i-vs-manufacturing log relative-labor equation for eps_i, given sigma."
"The name 'Engel curve' here refers to eps_i in the Comin et al. (2021)"
"notation (i.e., the income-elasticity parameter, not a full Engel curve)."


def eps_i_ft(sigma, rel_l, rel_om, rel_p):
    nom = np.log(rel_l) - np.log(rel_om) - (1 - sigma) * np.log(rel_p)
    den = np.log(man_va_share_last / om_man) + (1 - sigma) * np.log(E_pm_last)
    return 1 + nom / den


"Non-homothetic CES consumption index recovered from one relative-labor"
"equation. Under linear-in-labor technology the sectoral FOC implies"
"C^(eps_i-1) = (om_m/om_i) * (l_i/l_m) * (p_i/p_m)^(sigma-1); inverting"
"this for C at every t yields an observable time series for the index."


def C_index(om_i, li_lm, pi_pm, sigma, epsilon_i):
    C_level = ((om_man / om_i) * li_lm * (pi_pm ** (sigma - 1))) ** (
        1 / (epsilon_i - 1)
    )
    g_C = np.array(C_level / C_level.shift(1) - 1)
    C = [1]
    for i in range(len(g_C) - 1):
        C.append((1 + g_C[i + 1]) * C[i])
    return C


"""
'Loop over Engel curve in agriculture'
tot_sq_err = [] #Initialize total sum of sectoral squared errors
x = np.linspace(0.01,0.99,199)
for i in range(len(x)):
	eps_agr_loop = x[i]
	sigma_loop = sigma_ft(eps_agr_loop)	
	eps_ser_loop = eps_i_ft(sigma_loop, l_ser_l_man_last, (om_ser/om_man), p_ser_p_man_last)

	C_ams_agr_loop = C_index(om_agr, l_agr_l_man, p_agr_p_man, sigma_loop, eps_agr_loop)
	C_ams_ser_loop = C_index(om_ser, l_ser_l_man, p_ser_p_man, sigma_loop, eps_ser_loop)
#	C_ams_loop = np.array(share_agr_no_man*C_ams_agr_loop + share_ser_no_man*C_ams_ser_loop)
	C_ams_loop = C_ams_agr_loop
	tot_sq_err.append((E[-1] - C_ams_loop[-1])**2)

loc_min = np.where(np.array(tot_sq_err) == np.array(tot_sq_err).min())#Find least squared error	
"""
"Elasticities: fix services non-homotheticity, invert for sigma, then"
"invert for the remaining eps_i (agr, trd, bss, fin, nps)."
esp_ser = 1.2  # Comin et al. 2021, Table 9.
sigma = sigma_ft(esp_ser)
eps_agr = eps_i_ft(sigma, l_agr_l_man_last, (om_agr / om_man), p_agr_p_man_last)
eps_trd = eps_i_ft(sigma, l_trd_l_man_last, (om_trd / om_man), p_trd_p_man_last)
eps_bss = eps_i_ft(sigma, l_bss_l_man_last, (om_bss / om_man), p_bss_p_man_last)
eps_fin = eps_i_ft(sigma, l_fin_l_man_last, (om_fin / om_man), p_fin_p_man_last)
eps_nps = eps_i_ft(sigma, l_nps_l_man_last, (om_nps / om_man), p_nps_p_man_last)
eps_ser = eps_i_ft(sigma, l_ser_l_man_last, (om_ser / om_man), p_ser_p_man_last)

print("------------------------------------------")
print("sigma is " + str(sigma))
print("The Engel curve for agr is " + str(eps_agr))
print("The Engel curve for trd is " + str(eps_trd))
print("The Engel curve for bss is " + str(eps_bss))
print("The Engel curve for fin is " + str(eps_fin))
print("The Engel curve for nps is " + str(eps_nps))
print("The Engel curve for ser is " + str(eps_ser))
print(
    "Annualized Growth Rate in the US (1970-2019): "
    + str(((E[-1] / E[0]) ** (1 / ts_length) - 1) * 100)
)
print(
    "Annualized Growth Rate in the US (1995-2019): "
    + str(((E[-1] / E[25]) ** (1 / (ts_length - 25)) - 1) * 100)
)


"""
-------------------------
		The Models
-------------------------
"""


class model_ams:
    """Three-sector specification (agr, man, ser). With linear-in-labor
    technology y_i = A_i ell_i, sectoral employment share equals expenditure
    share: ell_i/L = omega_i = Omega_i * C^eps_i * A_i^(sigma-1) / sum_j(...),
    consistent with Comin, Lashkari, Mestieri (2021). Manufacturing serves as
    the numeraire sector with eps_man = 1."""

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
        ) = sigma, eps_agr, 1, eps_ser, om_agr, om_man, om_ser

    def labor_demand(self, C, A_agr, A_man, A_ser):
        "Denominator (sum of un-normalized CES weights) that converts each sectoral weight into an employment (= expenditure) share."
        L = (
            self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))
            + self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1))
            + self.om_ser * (C**self.eps_ser) * (A_ser ** (self.sigma - 1))
        )
        return L

    def share_agr(self, C, A_agr, A_man, A_ser):
        "Employment Share in Agriculture"
        l_agr = (self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_ser)
        )
        return l_agr

    def share_man(self, C, A_agr, A_man, A_ser):
        "Employment Share in Manufacturing"
        l_man = (self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_ser)
        )
        return l_man

    def share_ser(self, C, A_agr, A_man, A_ser):
        "Employment Share in Services"
        l_ser = (self.om_ser * (C**self.eps_ser) * (A_ser ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_ser)
        )
        return l_ser


class model_nps:
    """Six-sector specification (agr, man, trd, bss, fin, nps). Same
    non-homothetic CES + linear-in-labor structure as model_ams, with
    services disaggregated into wholesale/retail trade, business services,
    financial services, and non-progressive (personal) services."""

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
            eps_man,
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

    def labor_demand(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Sum of un-normalized CES weights (six-sector denominator)."
        L = (
            self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))
            + self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1))
            + self.om_trd * (C**self.eps_trd) * (A_trd ** (self.sigma - 1))
            + self.om_bss * (C**self.eps_bss) * (A_bss ** (self.sigma - 1))
            + self.om_fin * (C**self.eps_fin) * (A_fin ** (self.sigma - 1))
            + self.om_nps * (C**self.eps_nps) * (A_nps ** (self.sigma - 1))
        )
        return L

    def share_agr(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Employment Share in Agriculture"
        l_agr = (self.om_agr * (C**self.eps_agr) * (A_agr ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
        )
        return l_agr

    def share_man(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Employment Share in Manufacturing"
        l_man = (self.om_man * (C**self.eps_man) * (A_man ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
        )
        return l_man

    def share_trd(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Employment Share in Whole Sale and Retail Trade"
        l_trd = (self.om_trd * (C**self.eps_trd) * (A_trd ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
        )
        return l_trd

    def share_bss(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Employment Share in Business Services"
        l_bss = (self.om_bss * (C**self.eps_bss) * (A_bss ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
        )
        return l_bss

    def share_fin(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Employment Share in Financial Services"
        l_fin = (self.om_fin * (C**self.eps_fin) * (A_fin ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
        )
        return l_fin

    def share_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        "Employment Share in the Rest of Services"
        l_nps = (self.om_nps * (C**self.eps_nps) * (A_nps ** (self.sigma - 1))) / (
            self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
        )
        return l_nps


"Model Economy for Agriculture, Manufacturing and Services "
econ = model_ams()

"Computation of Non-Homothetic CES index"
C_ams_agr = C_index(om_agr, l_agr_l_man, p_agr_p_man, sigma, eps_agr)
C_ams_ser = C_index(om_ser, l_ser_l_man, p_ser_p_man, sigma, eps_ser)
C_ams = np.array(share_agr_no_man * C_ams_agr + share_ser_no_man * C_ams_ser)

"Alternative aggregators"
C_ams_simple_av = (np.array(C_ams_agr) + np.array(C_ams_ser)) / 2
C_ams_man_level = (man_va_share / om_man) * (E_pm ** (1 - sigma))
g_C_ams_man = np.array(C_ams_man_level / C_ams_man_level.shift(1) - 1)
C_ams_man = [1]
for i in range(len(g_C_ams_man) - 1):
    C_ams_man.append((1 + g_C_ams_man[i + 1]) * C_ams_man[i])
C_ams_alt = np.array(
    share_agr * C_ams_agr + share_man * C_ams_man + share_ser * C_ams_ser
)

"Alternative C series: recover C_t at each period by imposing the aggregate"
"resource constraint sum_i ell_i = L_t and solving for the level of the CES"
"index that clears it (consistent with the implicit-utility identity in"
"Comin et al. 2021). Used as a cross-check against the C_index series."


def C_exp_ams(C):
    return L_t ** (1 - sigma) - (
        econ.om_agr * (C**eps_agr) * (A_agr_t ** (sigma - 1))
        + econ.om_man * C * (A_man_t ** (sigma - 1))
        + econ.om_ser * (C**eps_ser) * (A_ser_t ** (sigma - 1))
    )


C_lev_E_ams = []
for i in range(len(E)):
    L_t = np.array(h_tot)[i]
    A_agr_t = np.array(y_l_agr)[i]
    A_man_t = np.array(y_l_man)[i]
    A_ser_t = np.array(y_l_ser)[i]
    C_lev_E_ams.append(fsolve(C_exp_ams, L_t).item())

C_level_E_ams = pd.DataFrame(C_lev_E_ams)

g_C_E_ams = np.array(C_level_E_ams / C_level_E_ams.shift(1) - 1).flatten()
C_E_ams = [1]
for i in range(len(g_C_E_ams) - 1):
    C_E_ams.append((1 + g_C_E_ams[i + 1]) * C_E_ams[i])

"Aggregator as input of the model"
C = C_ams_ser

"Initial Values (Match perfectly by construction)"
share_agr_ams = [econ.share_agr(C[0], A_agr[0], A_man[0], A_ser[0])]
share_man_ams = [econ.share_man(C[0], A_agr[0], A_man[0], A_ser[0])]
share_ser_ams = [econ.share_ser(C[0], A_agr[0], A_man[0], A_ser[0])]

"Subsequent Time Series Feeding Observed Growth Rates of Income and Productivity (Test of the Theory)"
for i in range(int(ts_length)):
    share_agr_ams.append(
        econ.share_agr(C[i + 1], A_agr[i + 1], A_man[i + 1], A_ser[i + 1])
    )
    share_man_ams.append(
        econ.share_man(C[i + 1], A_agr[i + 1], A_man[i + 1], A_ser[i + 1])
    )
    share_ser_ams.append(
        econ.share_ser(C[i + 1], A_agr[i + 1], A_man[i + 1], A_ser[i + 1])
    )

"Aggregate Productivity: A = sum_i omega_i * A_i where omega_i is the"
"model-implied employment share. Under linear-in-labor technology this is"
"the labor-productivity analog of Hulten's theorem (Domar-weighted sectoral"
"productivities). A_tot_ams uses model-implied shares; A_tot_ams_weighted"
"below uses observed (HP-filtered) data shares as a reference benchmark."
weighted_ams_A_agr = [a * b for a, b in zip(share_agr_ams, A_agr)]
weighted_ams_A_man = [a * b for a, b in zip(share_man_ams, A_man)]
weighted_ams_A_ser = [a * b for a, b in zip(share_ser_ams, A_ser)]

A_tot_ams = [
    sum(x) for x in zip(weighted_ams_A_agr, weighted_ams_A_man, weighted_ams_A_ser)
]
A_tot_ams_weighted = share_agr * A_agr + share_man * A_man + share_ser * A_ser


"Model Economy for Agriculture, Manufacturing, Whole Sale and Retail Trade, Business Services, Financial Services and Non-Progressive Services"
econ = model_nps()

"Computation of Non-Homothetic CES index"
C_nps_agr = C_index(om_agr, l_agr_l_man, p_agr_p_man, sigma, eps_agr)
C_nps_trd = C_index(om_trd, l_trd_l_man, p_trd_p_man, sigma, eps_trd)
C_nps_bss = C_index(om_bss, l_bss_l_man, p_bss_p_man, sigma, eps_bss)
C_nps_fin = C_index(om_fin, l_fin_l_man, p_fin_p_man, sigma, eps_fin)
C_nps_nps = C_index(om_nps, l_nps_l_man, p_nps_p_man, sigma, eps_nps)
# C_nps = np.array(share_agr_no_man*C_nps_agr + share_trd_no_man*C_nps_trd + share_bss_no_man*C_nps_bss + share_fin_no_man*C_nps_fin + share_nps_no_man*C_nps_nps)
C_nps = np.array(
    share_trd_no_agm * C_nps_trd
    + share_bss_no_agm * C_nps_bss
    + share_fin_no_agm * C_nps_fin
    + share_nps_no_agm * C_nps_nps
)

"Alternative aggregators"
C_nps_simple_av = (
    np.array(C_nps_agr)
    + np.array(C_nps_trd)
    + np.array(C_nps_bss)
    + np.array(C_nps_fin)
    + np.array(C_nps_nps)
) / 5
C_nps_man_level = (man_va_share / om_man) * (E_pm ** (1 - sigma))
g_C_nps_man = np.array(C_nps_man_level / C_nps_man_level.shift(1) - 1)
C_nps_man = [1]
for i in range(len(g_C_nps_man) - 1):
    C_nps_man.append((1 + g_C_nps_man[i + 1]) * C_nps_man[i])
C_nps_alt = np.array(
    share_agr * C_nps_agr
    + share_man * C_nps_man
    + share_trd * C_nps_trd
    + share_bss * C_nps_bss
    + share_fin * C_nps_fin
    + share_nps * C_nps_nps
)

"Alternative C series (six-sector): same logic as C_exp_ams above, using the"
"six-sector aggregate resource constraint."


def C_exp_nps(C):
    return L_t ** (1 - sigma) - (
        econ.om_agr * (C**eps_agr) * (A_agr_t ** (sigma - 1))
        + econ.om_man * C * (A_man_t ** (sigma - 1))
        + econ.om_trd * (C**eps_trd) * (A_trd_t ** (sigma - 1))
        + econ.om_bss * (C**eps_bss) * (A_bss_t ** (sigma - 1))
        + econ.om_fin * (C**eps_fin) * (A_fin_t ** (sigma - 1))
        + econ.om_nps * (C**eps_nps) * (A_nps_t ** (sigma - 1))
    )


C_lev_E_nps = []
for i in range(len(E)):
    L_t = np.array(h_tot)[i]
    A_trd_t = np.array(y_l_trd)[i]
    A_bss_t = np.array(y_l_bss)[i]
    A_fin_t = np.array(y_l_fin)[i]
    A_nps_t = np.array(y_l_nps)[i]
    C_lev_E_nps.append(fsolve(C_exp_nps, L_t).item())
C_level_E_nps = pd.DataFrame(C_lev_E_nps)
g_C_E_nps = np.array(C_level_E_nps / C_level_E_nps.shift(1) - 1).flatten()
C_E_nps = [1]
for i in range(len(g_C_E_nps) - 1):
    C_E_nps.append((1 + g_C_E_nps[i + 1]) * C_E_nps[i])

"Aggregator as input of the model"
# C = C_nps
C = C_ams_ser

"Initial Values (Match perfectly by construction)"
share_agr_nps = [
    econ.share_agr(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])
]
share_man_nps = [
    econ.share_man(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])
]
share_trd_nps = [
    econ.share_trd(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])
]
share_bss_nps = [
    econ.share_bss(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])
]
share_fin_nps = [
    econ.share_fin(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])
]
share_nps_nps = [
    econ.share_nps(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])
]

"Subsequent Time Series Feeding Observed Growth Rates of Income and Productivity (Test of the Theory)"
for i in range(int(ts_length)):
    share_agr_nps.append(
        econ.share_agr(
            C[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_man_nps.append(
        econ.share_man(
            C[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_trd_nps.append(
        econ.share_trd(
            C[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_bss_nps.append(
        econ.share_bss(
            C[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_fin_nps.append(
        econ.share_fin(
            C[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )
    share_nps_nps.append(
        econ.share_nps(
            C[i + 1],
            A_agr[i + 1],
            A_man[i + 1],
            A_trd[i + 1],
            A_bss[i + 1],
            A_fin[i + 1],
            A_nps[i + 1],
        )
    )

"Aggregate Productivity"
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

"""
------------
	Plots
------------
"""

"Non-Homothetic C"
"ams"
plt.figure(1)
plt.plot(
    year,
    C_ams_agr,
    "--v",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{agr},t}/l_{\texttt{man},t}$",
)
plt.plot(
    year,
    C_ams_ser,
    "--^",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{ser},t}/l_{\texttt{man},t}$",
)
plt.plot(year, E, "-s", lw=2, alpha=0.75, label="Data: GDP ph")
plt.plot(year, C_ams_simple_av, "-D", lw=2, alpha=0.75, label=r"$C_t$: Simple Average")
plt.plot(year, C_ams, "k-o", lw=2, alpha=0.75, label=r"$C_t$: Weighted Average")
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.ylabel("", fontsize=12)
# plt.savefig('../output/figures/fig_USA_C_ams_appendix.pdf', bbox_inches='tight')
plt.close()

"nps"
fig = plt.figure(2)
plt.plot(
    year,
    C_nps_agr,
    "--v",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{agr},t}/l_{\texttt{man},t}$",
)
plt.plot(
    year,
    C_nps_trd,
    "--^",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{trd},t}/l_{\texttt{man},t}$",
)
plt.plot(
    year,
    C_nps_bss,
    "--<",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{bss},t}/l_{\texttt{man},t}$",
)
plt.plot(
    year,
    C_nps_fin,
    "-->",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{fin},t}/l_{\texttt{man},t}$",
)
plt.plot(
    year,
    C_nps_nps,
    "--8",
    lw=1,
    markersize=4,
    alpha=0.35,
    label=r"Eq: $l_{\texttt{nps},t}/l_{\texttt{man},t}$",
)
plt.plot(year, E, "-s", lw=2, alpha=0.75, label="Data: GDP ph")
plt.plot(year, C_nps_simple_av, "-D", lw=2, alpha=0.75, label=r"$C_t$: Simple Average")
plt.plot(year, C_nps, "k-o", lw=2, alpha=0.75, label=r"$C_t$: Weighted Average")
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.ylabel("", fontsize=12)
# plt.savefig('../output/figures/fig_USA_C_nps_appendix.pdf', bbox_inches='tight')
plt.close()

"Ai"
plt.figure(3)
plt.plot(year, A_agr, "-o", lw=1, alpha=0.5, label=r"$A_{\texttt{agr},t}$")
plt.plot(year, A_man, "->", lw=1, alpha=0.5, label=r"$A_{\texttt{man},t}$")
plt.plot(year, A_bss, "-s", lw=1, alpha=0.5, label=r"$A_{\texttt{bss},t}$")
plt.plot(year, A_trd, "-v", lw=1, alpha=0.5, label=r"$A_{\texttt{trd},t}$")
plt.plot(year, A_fin, "-8", lw=1, alpha=0.5, label=r"$A_{\texttt{fin},t}$")
plt.plot(year, A_nps, "-p", lw=1, alpha=0.5, label=r"$A_{\texttt{per},t}$")
plt.plot(year, A_ser, "-s", lw=2, alpha=0.75, label=r"$A_{\texttt{ser},t}$")
plt.plot(year, A_nps, "-D", lw=2, alpha=0.75, label=r"$A_{\texttt{nps},t}$")
plt.plot(year, A_tot, "k-o", lw=2, alpha=0.75, label=r"$A_{\texttt{tot},t}$")
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.ylabel("", fontsize=12)
plt.savefig("../output/figures/fig_USA_Ai_appendix.pdf", bbox_inches="tight")
plt.close()


"""
-----------------------------------------------------
	Predictions for the Structural Transformation
-----------------------------------------------------
"""

"agr"
plt.figure(4)
plt.plot(year, share_agr, "b-", lw=2, alpha=0.95, label=r"Data: \texttt{agr}")
plt.plot(
    year,
    share_agr_ams,
    "D-",
    markersize=5,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    alpha=0.95,
    label=r"Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}",
)
plt.plot(
    year,
    share_agr_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
    label=r"Model (2): \texttt{nps}, progressive services",
)
plt.ylabel("Employment Share", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.tight_layout()
# plt.savefig('../output/figures/fig_USA_calib_agr.pdf', bbox_inches='tight')
plt.close()

"man"
plt.figure(5)
plt.plot(year, share_man, "b-", lw=2, alpha=0.95, label=r"Data: \texttt{man}")
plt.plot(
    year,
    share_man_ams,
    "D-",
    markersize=5,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    alpha=0.95,
)
plt.plot(
    year,
    share_man_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
)
plt.ylabel("Employment Share", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.savefig('../output/figures/fig_USA_calib_man.pdf', bbox_inches='tight')
plt.close()

"ser and nps"
plt.figure(6)
plt.plot(year, share_ser, "b-", lw=2, alpha=0.95, label=r"Data: \texttt{ser}")
plt.plot(
    year,
    share_ser_ams,
    "D-",
    markersize=5,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    alpha=0.95,
)
plt.plot(
    year, share_nps, "c--", lw=2, markersize=5, alpha=0.95, label=r"Data: \texttt{nps}"
)
plt.plot(
    year,
    share_nps_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
)
plt.legend(fontsize=16)
plt.ylabel("Employment Share", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.savefig('../output/figures/fig_USA_calib_ser_nps.pdf', bbox_inches='tight')
plt.close()

"trd, bss and fin"
plt.figure(7)
plt.plot(year, share_trd, "b-", lw=2, alpha=0.95, label=r"Data: \texttt{trd}")
plt.plot(
    year,
    share_trd_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
)
plt.plot(year, share_bss, "c--", lw=2, alpha=0.95, label=r"Data: \texttt{bss}")
plt.plot(
    year,
    share_bss_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
)
plt.plot(year, share_fin, "y-.", lw=2, alpha=0.95, label=r"Data: \texttt{fin}")
plt.plot(
    year,
    share_fin_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.95,
)
plt.legend(fontsize=16)
plt.ylabel("Employment Share", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.savefig('../output/figures/fig_USA_calib_trd_bss_fin.pdf', bbox_inches='tight')
plt.close()

fig = plt.figure(8)
fig.set_figheight(6)
fig.set_figwidth(6)

plt.plot(
    year,
    share_agr_ams,
    "D-",
    markersize=6,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    alpha=0.5,
    label=r"Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}",
)
plt.plot(
    year,
    share_agr_nps,
    "H-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.5,
    label=r"Model (2): \texttt{nps}, progressive services",
)
plt.plot(
    year,
    share_man_ams,
    "D-",
    markersize=6,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    alpha=0.5,
)
plt.plot(
    year,
    share_man_nps,
    "H-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.5,
)
plt.plot(
    year,
    share_bss_nps,
    "H-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.5,
)
plt.plot(
    year,
    share_trd_nps,
    "H-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.5,
)
plt.plot(
    year,
    share_fin_nps,
    "H-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.5,
)
plt.plot(
    year,
    share_nps_nps,
    "H-",
    markersize=6,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.5,
)
plt.plot(
    year, share_agr, "-", color="blue", lw=2, alpha=0.95, label=r"Data: \texttt{agr}"
)
plt.plot(
    year, share_man, "-", color="red", lw=2, alpha=0.95, label=r"Data: \texttt{man}"
)
plt.plot(
    year, share_trd, "-", color="gold", lw=2, alpha=0.95, label=r"Data: \texttt{trd}"
)
plt.plot(
    year, share_bss, "-", color="purple", lw=2, alpha=0.95, label=r"Data: \texttt{bss}"
)
plt.plot(
    year, share_fin, "-", color="grey", lw=2, alpha=0.95, label=r"Data: \texttt{fin}"
)
plt.plot(
    year,
    share_nps,
    "-",
    color="darkcyan",
    lw=2,
    markersize=5,
    alpha=0.95,
    label=r"Data: \texttt{nps}",
)
plt.ylabel("Employment Share", fontsize=14)
plt.xlabel("Year", fontsize=14)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
# plt.savefig('../output/figures/fig_USA_calib_all.pdf', bbox_inches='tight')
plt.close()

"""
---------------------------------------------------
	Prediction for Aggregate Labor Productivity
---------------------------------------------------
"""

plt.figure(8)
plt.plot(year, A_tot, "r--", lw=2, alpha=0.95, label=r"$A_t$: Data")
plt.plot(year, E, "b-", lw=2, alpha=0.95, label=r"$\frac{Y_t}{L_t}$: Data")
# plt.plot(year, A_tot_ams_weighted, 'r--', lw=2, alpha = 0.95, label = r'$\frac{Y_t}{L_t}$. Weighted Average: \texttt{agr}, \texttt{man} and \texttt{ser}')
# plt.plot(year, A_tot_nps_weighted, 'm-.', lw=2, alpha = 0.95, label = r'$\frac{Y_t}{L_t}$ Weighted Average: \texttt{nps} and progressive services')
plt.plot(
    year,
    A_tot_ams,
    "D-",
    markersize=5,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    alpha=0.65,
    label=r"Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}",
)
plt.plot(
    year,
    A_tot_nps,
    "H-",
    markersize=5,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    alpha=0.65,
    label=r"Model (2): \texttt{nps} and progressive services",
)
plt.yticks(fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.legend(fontsize=12)
# plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=14)
plt.tight_layout()
# plt.savefig('../output/figures/fig_USA_A_data_model.pdf', bbox_inches='tight')
plt.close()


"""
---------------------------
	Plots for the paper
---------------------------
"""

fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(5)
fig.set_figwidth(10)

"Agriculture, Manufacturing and Services"
ax = plt.subplot(1, 2, 1)

ax.plot(
    year,
    share_agr,
    "D-",
    markersize=4,
    color="darkgreen",
    markerfacecolor="lime",
    markeredgecolor="darkgreen",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{agr}$: Data",
)
ax.plot(
    year,
    share_agr_nps,
    "D--",
    markersize=4,
    color="darkgreen",
    markeredgecolor="darkgreen",
    mfc="none",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{agr}$: Model",
)
ax.plot(
    year,
    share_man,
    "o-",
    markersize=4,
    color="darkblue",
    markerfacecolor="lightskyblue",
    markeredgecolor="darkblue",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{man}$: Data",
)
ax.plot(
    year,
    share_man_nps,
    "o--",
    markersize=4,
    color="darkblue",
    markeredgecolor="darkblue",
    mfc="none",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{man}$: Model",
)
ax.plot(
    year,
    share_trd,
    "s-",
    markersize=4,
    color="darkred",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{trd}$: Data",
)
ax.plot(
    year,
    share_trd_nps,
    "s--",
    markersize=4,
    color="darkred",
    markeredgecolor="darkred",
    mfc="none",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{trd}$: Model",
)
ax.plot(
    year,
    share_bss,
    "^-",
    markersize=4,
    color="darkmagenta",
    markerfacecolor="violet",
    markeredgecolor="darkmagenta",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{bss}$: Data",
)
ax.plot(
    year,
    share_bss_nps,
    "^--",
    markersize=4,
    color="darkmagenta",
    markeredgecolor="darkmagenta",
    mfc="none",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{bss}$: Model",
)
ax.plot(
    year,
    share_fin,
    "p-",
    markersize=4,
    color="darkcyan",
    markerfacecolor="cyan",
    markeredgecolor="darkcyan",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{fin}$: Data",
)
ax.plot(
    year,
    share_fin_nps,
    "p--",
    markersize=4,
    color="darkcyan",
    markeredgecolor="darkcyan",
    mfc="none",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{fin}$: Model",
)
ax.plot(
    year,
    share_nps,
    "v-",
    markersize=4,
    color="saddlebrown",
    markerfacecolor="sandybrown",
    markeredgecolor="saddlebrown",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{nps}$: Data",
)
ax.plot(
    year,
    share_nps_nps,
    "v--",
    markersize=4,
    color="saddlebrown",
    markeredgecolor="saddlebrown",
    mfc="none",
    markevery=7,
    alpha=0.95,
    label=r"$\texttt{nps}$: Model",
)
plt.ylabel("Employment Share", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.axis([1968, 2021, -0.025, 0.525])
plt.grid()

plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

"Aggregate Productivity"
ax = plt.subplot(1, 2, 2)
ax.plot(
    year,
    E,
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
    year,
    A_tot,
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
    year,
    A_tot_nps,
    "s-",
    markersize=6,
    color="darkred",
    markerfacecolor="lightcoral",
    markeredgecolor="darkred",
    markevery=7,
    alpha=0.95,
    label="Model",
)
plt.ylabel("Labor Productivity", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], fontsize=12)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=12)
plt.legend(fontsize=12)
plt.grid()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=12)

plt.tight_layout()
plt.savefig("../output/figures/fig_calibration_USA.pdf", bbox_inches="tight")
plt.close()

print(
    "Annualized Growth Rate in the USA (OECD) (1970-2019): "
    + str(((E[-1] / E[0]) ** (1 / 49) - 1) * 100)
)
print(
    "Annualized Growth Rate in the USA (KLEMS) (1970-2019): "
    + str(((A_tot[-1] / A_tot[0]) ** (1 / 49) - 1) * 100)
)
print(
    "Annualized Growth Rate in the USA (KLEMS. Weighted Average) (1970-2019): "
    + str(
        (
            (np.array(A_tot_nps_weighted)[-1] / np.array(A_tot_nps_weighted)[0])
            ** (1 / 49)
            - 1
        )
        * 100
    )
)
print(
    "Annualized Growth Rate in the USA (Model) (1970-2019): "
    + str(((A_tot_nps[-1] / A_tot_nps[0]) ** (1 / 49) - 1) * 100)
)


E_ams = (
    om_agr * ((1 / np.array(A_agr)) ** (1 - sigma)) * (np.array(C) ** eps_agr)
    + om_man * ((1 / np.array(A_man)) ** (1 - sigma)) * (np.array(C))
    + om_ser * ((1 / np.array(A_ser)) ** (1 - sigma)) * (np.array(C) ** eps_ser)
) ** (1 / (1 - sigma))


P_ams = (
    om_agr
    * ((1 / np.array(A_agr)) ** (1 - sigma))
    * (np.array(C) ** (eps_agr - (1 - sigma)))
    + om_man
    * ((1 / np.array(A_man)) ** (1 - sigma))
    * (np.array(C) ** (1 - (1 - sigma)))
    + om_ser
    * ((1 / np.array(A_ser)) ** (1 - sigma))
    * (np.array(C) ** (eps_ser - (1 - sigma)))
) ** (1 / (1 - sigma))
