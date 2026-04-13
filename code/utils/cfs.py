"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        cfs.py
Purpose:     Build counterfactual contribution tables comparing endogenous (model)
             vs. exogenous (shift-share) employment shares across regions and sectors.
Pipeline:    Step 15/19 — Generates Tables 2, A.4, A.8 (counterfactual tables).
Inputs:      ../output/data/predictions_vs_data.obj (model predictions vs. data),
             ../output/figures/Counterfactual_*.xlsx (CF growth gaps by region/sector),
             EU4 shift-share inputs imported from utils.table_1_ss_eu4.
Outputs:     ../output/data/table_cf1_<region>.tex,
             ../output/data/table_cf2_<region>.tex,
             ../output/data/table_cf1_ams_<region>.tex,
             ../output/data/table_cf2_ams_<region>.tex
             for region in {EU4, EU15, EUCORE, EUPERI, GBR} (AMS variants: EU4, EU15).
Dependencies: utils.table_1_ss_eu4 (Step 14) — provides shift_share and base LP/LS arrays.
              The Counterfactual_*.xlsx workbooks must already exist (produced upstream
              in the model pipeline).
"""

import pandas as pd
import pickle
from matplotlib import rc
import numpy as np
from .table_1_ss_eu4 import shift_share, annualized, lp_1970_eu_nps, lp_2019_eu_nps, l_1970_eu_nps, l_2019_eu_nps
rc('text', usetex=True)
rc('font', family='serif')

# Convert a cumulative growth factor over `year` years to an annualized
# percentage rate. Default 49 = 1970-2019 horizon used in the main text.
# This shadows the `annualized` import from utils.table_1_ss_eu4 deliberately
# so that the local default lines up with the panels built below.
def annualized(x, year=49):
    return ((x) ** (1 / year) - 1) * 100

# data_predictions is a long-format DataFrame with model employment shares
# (LS_m) by (country, year, sector). Built upstream by the model_test_europe*
# scripts and pickled to disk so cfs.py can be run without re-solving the model.
filehandler = open('../output/data/predictions_vs_data.obj', 'rb')
data_load = pickle.load(filehandler)
filehandler.close()

data_pred = data_load["data_predictions"]

# Five regions enter the six-sector counterfactual tables. EU4 is the main
# text aggregate (DE+FR+GB+IT); the other four feed Tables A.4 / A.8 robustness.
regions = ['EU4', 'EU15', 'EUCORE', 'EUPERI', 'GBR']

for region in regions:
    l_1970_eu_model = data_pred.loc[(data_pred.country == region) & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")
    l_2019_eu_model = data_pred.loc[(data_pred.country == region) & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")
    
    ss_eu_data = shift_share(np.ones(6),
                1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values,
                l_1970_eu_nps['LS'].values,
                l_2019_eu_nps['LS'].values)
    
    ss_eu_model = shift_share(np.ones(6),
                1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values,
                l_1970_eu_model['LS_m'].values,
                l_2019_eu_model['LS_m'].values)
    
    A_tot_model = 1 + ss_eu_model["LP_growth"].sum()
    A_tot_data = 1 + ss_eu_data["LP_growth"].sum()
    
    # CF1 — feed each EU sector the U.S. growth rate of sectoral productivity
    # (one sector at a time). Reported with both the structural model (cf1)
    # and the matched shift-share decomposition (cf1_ss); the difference
    # between the two columns isolates the contribution of endogenous
    # reallocation triggered by the productivity shock.
    cf1 = pd.read_excel("../output/figures/Counterfactual_1_nps.xlsx", index_col=0)[region]
    cf1_ss = pd.read_excel("../output/figures/Counterfactual_1_nps_ss.xlsx", index_col=0)[region]

    # CF2 — give the EU the U.S. *level* of sectoral productivity in 2019
    # (catch-up). Same model vs. shift-share contrast as CF1.
    cf2 = pd.read_excel("../output/figures/Counterfactual_2_catch_nps.xlsx", index_col=0)[region]
    cf2_ss = pd.read_excel("../output/figures/Counterfactual_2_catch_nps_ss.xlsx", index_col=0)[region]
    # CF3 — labour reallocation gap zeroed after 1990 (post-divergence period)
    cf3 = pd.read_excel("../output/figures/Counterfactual_3.xlsx")[region]
    # CF4 — sectoral productivity gap zeroed after 1990 (model)
    cf4 = pd.read_excel("../output/figures/Counterfactual_2_nps.xlsx")[region]
    # CF5 — sectoral productivity gap zeroed after 1990 (shift-share counterpart)
    cf5 = pd.read_excel("../output/figures/Counterfactual_2_nps_ss.xlsx")[region]
    
    # Aggregate productivity factor (1 + cumulative LP growth) under model
    # and data weights, used as the base on which each cfX shock scales.
    A_tot_model = 1 + ss_eu_model["LP_growth"].sum()
    A_tot_data = 1 + ss_eu_data["LP_growth"].sum()
    # Seven entries: six structural sectors plus the personal-services (prs)
    # split that appears only in CF1's appendix variant.
    A_tot_model_cf1 = np.zeros(7)
    for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps', 'prs']):
        A_tot_model_cf1[i] = A_tot_model*(1+cf1.loc[sec]/100)
    A_tot_model_cf1_ss = np.zeros(7)
    for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps', 'prs']):
        A_tot_model_cf1_ss[i] = A_tot_data*(1+cf1_ss.loc[sec]/100)
    
    A_tot_model_cf2 = np.zeros(6)
    for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps']):
        A_tot_model_cf2[i] = A_tot_model*(1+cf2.loc[sec]/100)
    A_tot_model_cf2_ss = np.zeros(6)
    for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps']):
        A_tot_model_cf2_ss[i] = A_tot_data*(1+cf2_ss.loc[sec]/100)

    A_tot_model_cf3 = A_tot_model * (1 + cf3.values / 100)
    A_tot_model_cf4 = A_tot_model * (1 + cf4.values / 100)
    A_tot_model_cf5 = A_tot_model * (1 + cf5.values / 100)


    annualized(A_tot_model)
    A_tot_model_cf1_g = annualized(A_tot_model_cf1) - annualized(A_tot_model)
    A_tot_model_cf1_ss_g = annualized(A_tot_model_cf1_ss) - annualized(A_tot_data)
    A_tot_model_cf1_rel_ss_g = (A_tot_model_cf1_g - A_tot_model_cf1_ss_g)
    
    A_tot_model_cf2_g = annualized(A_tot_model_cf2) - annualized(A_tot_model)
    A_tot_model_cf2_ss_g = annualized(A_tot_model_cf2_ss) - annualized(A_tot_data)
    A_tot_model_cf2_rel_ss_g = (A_tot_model_cf2_g - A_tot_model_cf2_ss_g)

    A_tot_model_cf3 = annualized(A_tot_model_cf3) - annualized(A_tot_model)
    A_tot_model_cf4 = annualized(A_tot_model_cf4) - annualized(A_tot_model)
    A_tot_model_cf5 = annualized(A_tot_model_cf5) - annualized(A_tot_model)

    print(A_tot_model_cf3, A_tot_model_cf4, A_tot_model_cf5)

    
    cf1_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps', 'bss, fin, trd'],
                             columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
    cf1_table['End. emp. shares (model)'] = A_tot_model_cf1_g.round(2)
    cf1_table['Exo. emp. shares (data)'] = A_tot_model_cf1_ss_g.round(2)
    cf1_table['Difference'] = A_tot_model_cf1_rel_ss_g.round(2)
    
    cf1_table.style.format("{:.2f}").to_latex('../output/data/table_cf1_' + region +'.tex')


    cf2_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps'],
                             columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
    cf2_table['End. emp. shares (model)'] = A_tot_model_cf2_g.round(2)
    cf2_table['Exo. emp. shares (data)'] = A_tot_model_cf2_ss_g.round(2)
    cf2_table['Difference'] = A_tot_model_cf2_rel_ss_g.round(2)
    
    cf2_table.style.format("{:.2f}").to_latex('../output/data/table_cf2_' + region + '.tex')

# ────────────────────────────────────────────────────────────────────────────
# Appendix variant — collapse the six base sectors to the classical three
# (agr / man / ser) used in Acemoglu-style structural-change papers (Tables
# A.4 and A.8 with the "_ams" suffix). Re-load the predictions object so the
# block is self-contained when the script is rerun cell-by-cell.
# ────────────────────────────────────────────────────────────────────────────
filehandler = open('../output/data/predictions_vs_data.obj', 'rb')
data_load = pickle.load(filehandler)
filehandler.close()

data_pred = data_load["data_predictions"]

regions = ['EU4', 'EU15']

for region in regions:
    cf1 = pd.read_excel("../output/figures/Counterfactual_1_ams.xlsx", index_col=0)[region]
    cf1_ss = pd.read_excel("../output/figures/Counterfactual_1_ams_ss.xlsx", index_col=0)[region]

    cf2 = pd.read_excel("../output/figures/Counterfactual_2_catch_ams.xlsx", index_col=0)[region]
    cf2_ss = pd.read_excel("../output/figures/Counterfactual_2_catch_ams_ss.xlsx", index_col=0)[region]


    A_tot_model = 1 + ss_eu_model["LP_growth"].sum()
    A_tot_data = 1 + ss_eu_data["LP_growth"].sum()
    A_tot_model_cf1 = np.zeros(3)
    for i, sec in enumerate(['agr', 'man', 'ser']):
        A_tot_model_cf1[i] = A_tot_model * (1 + cf1.loc[sec] / 100)
    A_tot_model_cf1_ss = np.zeros(3)
    for i, sec in enumerate(['agr', 'man', 'ser']):
        A_tot_model_cf1_ss[i] = A_tot_data * (1 + cf1_ss.loc[sec] / 100)

    A_tot_model_cf2 = np.zeros(3)
    for i, sec in enumerate(['agr', 'man', 'ser']):
        A_tot_model_cf2[i] = A_tot_model * (1 + cf2.loc[sec] / 100)
    A_tot_model_cf2_ss = np.zeros(3)
    for i, sec in enumerate(['agr', 'man', 'ser']):
        A_tot_model_cf2_ss[i] = A_tot_data * (1 + cf2_ss.loc[sec] / 100)


    annualized(A_tot_model)
    A_tot_model_cf1_g = annualized(A_tot_model_cf1) - annualized(A_tot_model)
    A_tot_model_cf1_ss_g = annualized(A_tot_model_cf1_ss) - annualized(A_tot_data)
    A_tot_model_cf1_rel_ss_g = (A_tot_model_cf1_g - A_tot_model_cf1_ss_g)

    A_tot_model_cf2_g = annualized(A_tot_model_cf2) - annualized(A_tot_model)
    A_tot_model_cf2_ss_g = annualized(A_tot_model_cf2_ss) - annualized(A_tot_data)
    A_tot_model_cf2_rel_ss_g = (A_tot_model_cf2_g - A_tot_model_cf2_ss_g)

    cf1_table = pd.DataFrame(index=['agr', 'man', 'ser'],
                             columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
    cf1_table['End. emp. shares (model)'] = A_tot_model_cf1_g.round(2)
    cf1_table['Exo. emp. shares (data)'] = A_tot_model_cf1_ss_g.round(2)
    cf1_table['Difference'] = A_tot_model_cf1_rel_ss_g.round(2)

    cf1_table.style.format("{:.2f}").to_latex('../output/data/table_cf1_ams_' + region + '.tex')

    cf2_table = pd.DataFrame(index=['agr', 'man', 'ser'],
                             columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
    cf2_table['End. emp. shares (model)'] = A_tot_model_cf2_g.round(2)
    cf2_table['Exo. emp. shares (data)'] = A_tot_model_cf2_ss_g.round(2)
    cf2_table['Difference'] = A_tot_model_cf2_rel_ss_g.round(2)

    cf2_table.style.format("{:.2f}").to_latex('../output/data/table_cf2_ams_' + region + '.tex')

