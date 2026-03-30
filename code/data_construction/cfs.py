import pandas as pd
import pickle
import statsmodels.api as sm
from matplotlib import rc
import numpy as np
from Dofiles.table_1_ss_eu4 import shift_share, annualized, lp_1970_us_nps, lp_1995_us_nps, lp_2019_us_nps, lp_1970_eu_nps, lp_1995_eu_nps, lp_2019_eu_nps, l_1970_us_nps, l_1995_us_nps, l_2019_us_nps, l_1970_eu_nps, l_1995_eu_nps, l_2019_eu_nps
rc('text', usetex=True)
rc('font', family='serif')

## New table CFs

def annualized(x, year=49):
    return ((x) ** (1 / year) - 1) * 100

filehandler = open('../Outputs/Data/predictions_vs_data.obj', 'rb')
data_load = pickle.load(filehandler)
filehandler.close()

data_pred = data_load["data_predictions"]

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
    
    cf1 = pd.read_excel("../Figures/Counterfactual_1_nps.xlsx", index_col=0)[region]
    cf1_ss = pd.read_excel("../Figures/Counterfactual_1_nps_ss.xlsx", index_col=0)[region]
    
    cf2 = pd.read_excel("../Figures/Counterfactual_2_catch_nps.xlsx", index_col=0)[region]
    cf2_ss = pd.read_excel("../Figures/Counterfactual_2_catch_nps_ss.xlsx", index_col=0)[region]
    # reallocation gap is zero after 1990
    cf3 = pd.read_excel("../Figures/Counterfactual_3.xlsx")[region]
    # sectoral productivity gap is zero after 1990
    cf4 = pd.read_excel("../Figures/Counterfactual_2_nps.xlsx")[region]
    # sectoral productivity gap is zero after 1990 in shift share
    cf5 = pd.read_excel("../Figures/Counterfactual_2_nps_ss.xlsx")[region]
    
    A_tot_model = 1 + ss_eu_model["LP_growth"].sum()
    A_tot_data = 1 + ss_eu_data["LP_growth"].sum()
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
    
    cf1_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_cf1_' + region +'.tex')


    cf2_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps'],
                             columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
    cf2_table['End. emp. shares (model)'] = A_tot_model_cf2_g.round(2)
    cf2_table['Exo. emp. shares (data)'] = A_tot_model_cf2_ss_g.round(2)
    cf2_table['Difference'] = A_tot_model_cf2_rel_ss_g.round(2)
    
    cf2_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_cf2_' + region + '.tex')

filehandler = open('../Outputs/Data/predictions_vs_data.obj', 'rb')
data_load = pickle.load(filehandler)
filehandler.close()

data_pred = data_load["data_predictions"]

regions = ['EU4', 'EU15']

for region in regions:
    cf1 = pd.read_excel("../Figures/Counterfactual_1_ams.xlsx", index_col=0)[region]
    cf1_ss = pd.read_excel("../Figures/Counterfactual_1_ams_ss.xlsx", index_col=0)[region]

    cf2 = pd.read_excel("../Figures/Counterfactual_2_catch_ams.xlsx", index_col=0)[region]
    cf2_ss = pd.read_excel("../Figures/Counterfactual_2_catch_ams_ss.xlsx", index_col=0)[region]


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

    cf1_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_cf1_ams_' + region + '.tex')

    cf2_table = pd.DataFrame(index=['agr', 'man', 'ser'],
                             columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
    cf2_table['End. emp. shares (model)'] = A_tot_model_cf2_g.round(2)
    cf2_table['Exo. emp. shares (data)'] = A_tot_model_cf2_ss_g.round(2)
    cf2_table['Difference'] = A_tot_model_cf2_rel_ss_g.round(2)

    cf2_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_cf2_ams_' + region + '.tex')

#####################################################################################################################
########## Get table C.4
#
# for country in ['EU15', 'GBR']:
#     cf1 = pd.read_excel("../Figures/Counterfactual_1_nps.xlsx", index_col=0)[country]
#     cf1_ss = pd.read_excel("../Figures/Counterfactual_1_nps_ss.xlsx", index_col=0)[country]
#
#     cf2 = pd.read_excel("../Figures/Counterfactual_2_catch_nps.xlsx", index_col=0)[country]
#     cf2_ss = pd.read_excel("../Figures/Counterfactual_2_catch_nps_ss.xlsx", index_col=0)[country]
#
#     A_tot_model = 1 + ss_eu_model["LP_growth"].sum()
#     A_tot_data = 1 + ss_eu_data["LP_growth"].sum()
#     A_tot_model_cf1 = np.zeros(7)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps', 'prs']):
#         A_tot_model_cf1[i] = A_tot_model*(1+cf1.loc[sec]/100)
#     A_tot_model_cf1_ss = np.zeros(7)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps', 'prs']):
#         A_tot_model_cf1_ss[i] = A_tot_data*(1+cf1_ss.loc[sec]/100)
#
#     A_tot_model_cf2 = np.zeros(6)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps']):
#         A_tot_model_cf2[i] = A_tot_model*(1+cf2.loc[sec]/100)
#     A_tot_model_cf2_ss = np.zeros(6)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps']):
#         A_tot_model_cf2_ss[i] = A_tot_data*(1+cf2_ss.loc[sec]/100)
#
#     annualized(A_tot_model)
#     A_tot_model_cf1_g = annualized(A_tot_model_cf1) - annualized(A_tot_model)
#     A_tot_model_cf1_ss_g = annualized(A_tot_model_cf1_ss) - annualized(A_tot_data)
#     A_tot_model_cf1_rel_ss_g = (A_tot_model_cf1_g - A_tot_model_cf1_ss_g)
#
#     A_tot_model_cf2_g = annualized(A_tot_model_cf2) - annualized(A_tot_model)
#     A_tot_model_cf2_ss_g = annualized(A_tot_model_cf2_ss) - annualized(A_tot_data)
#     A_tot_model_cf2_rel_ss_g = (A_tot_model_cf2_g - A_tot_model_cf2_ss_g)
#
#     cf1_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps', 'bss, fin, trd'],
#                              columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
#     cf1_table['End. emp. shares (model)'] = A_tot_model_cf1_g.round(2)
#     cf1_table['Exo. emp. shares (data)'] = A_tot_model_cf1_ss_g.round(2)
#     cf1_table['Difference'] = A_tot_model_cf1_rel_ss_g.round(2)
#
#     cf1_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_c4_' + country + '.tex')
#
#
#     cf2_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps'],
#                              columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Rel. difference (%)'])
#     cf2_table['End. emp. shares (model)'] = A_tot_model_cf2_g.round(2)
#     cf2_table['Exo. emp. shares (data)'] = A_tot_model_cf2_ss_g.round(2)
#     cf2_table['Difference'] = A_tot_model_cf2_rel_ss_g.round(2)
#
#     cf2_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_c4_cf2_' + country + '.tex')
#
#
# ########## Get table C.3
#
# for country in ['EUCORE', 'EUPERI']:
#     cf1 = pd.read_excel("../Figures/Counterfactual_1_nps.xlsx", index_col=0)[country]
#     cf1_ss = pd.read_excel("../Figures/Counterfactual_1_nps_ss.xlsx", index_col=0)[country]
#
#     cf2 = pd.read_excel("../Figures/Counterfactual_2_catch_nps.xlsx", index_col=0)[country]
#     cf2_ss = pd.read_excel("../Figures/Counterfactual_2_catch_nps_ss.xlsx", index_col=0)[country]
#
#     A_tot_model = 1 + ss_eu_model["LP_growth"].sum()
#     A_tot_data = 1 + ss_eu_data["LP_growth"].sum()
#     A_tot_model_cf1 = np.zeros(7)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps', 'prs']):
#         A_tot_model_cf1[i] = A_tot_model*(1+cf1.loc[sec]/100)
#     A_tot_model_cf1_ss = np.zeros(7)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps', 'prs']):
#         A_tot_model_cf1_ss[i] = A_tot_data*(1+cf1_ss.loc[sec]/100)
#
#     A_tot_model_cf2 = np.zeros(6)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps']):
#         A_tot_model_cf2[i] = A_tot_model*(1+cf2.loc[sec]/100)
#     A_tot_model_cf2_ss = np.zeros(6)
#     for i,sec in enumerate(['agr','man', 'bss', 'fin', 'trd', 'nps']):
#         A_tot_model_cf2_ss[i] = A_tot_data*(1+cf2_ss.loc[sec]/100)
#
#     annualized(A_tot_model)
#     A_tot_model_cf1_g = annualized(A_tot_model_cf1) - annualized(A_tot_model)
#     A_tot_model_cf1_ss_g = annualized(A_tot_model_cf1_ss) - annualized(A_tot_data)
#     A_tot_model_cf1_rel_ss_g = (A_tot_model_cf1_g - A_tot_model_cf1_ss_g)
#
#     A_tot_model_cf2_g = annualized(A_tot_model_cf2) - annualized(A_tot_model)
#     A_tot_model_cf2_ss_g = annualized(A_tot_model_cf2_ss) - annualized(A_tot_data)
#     A_tot_model_cf2_rel_ss_g = (A_tot_model_cf2_g - A_tot_model_cf2_ss_g)
#
#     cf1_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps', 'bss, fin, trd'],
#                              columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Difference'])
#     cf1_table['End. emp. shares (model)'] = A_tot_model_cf1_g.round(2)
#     cf1_table['Exo. emp. shares (data)'] = A_tot_model_cf1_ss_g.round(2)
#     cf1_table['Difference'] = A_tot_model_cf1_rel_ss_g.round(2)
#
#     cf1_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_c3_core_peri' + country + '.tex')
#
#
#     cf2_table = pd.DataFrame(index=['agr', 'man', 'bss', 'fin', 'trd', 'nps'],
#                              columns=['End. emp. shares (model)', 'Exo. emp. shares (data)', 'Difference'])
#     cf2_table['End. emp. shares (model)'] = A_tot_model_cf2_g.round(2)
#     cf2_table['Exo. emp. shares (data)'] = A_tot_model_cf2_ss_g.round(2)
#     cf2_table['Difference'] = A_tot_model_cf2_rel_ss_g.round(2)
#
#     cf2_table.style.format("{:.2f}").to_latex('../Outputs/Data/table_c3_cf2_core_peri' + country + '.tex')
#
#
#
#
# #############################################################################################
# ### OTHER CHECKS
# #############################################################################################
# #############################################################################################
# print("\n\n 1970-2019 period")
#
#
# filehandler = open('../Outputs/Data/predictions_vs_data.obj', 'rb')
# data_load = pickle.load(filehandler)
# filehandler.close()
#
# data_pred = data_load["data_predictions"]
#
# l_1970_eu_data_felipe = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS"]].sort_values("sector")
# l_1995_eu_data_felipe = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1995), ["sector", "LS"]].sort_values("sector")
# l_2019_eu_data_felipe = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS"]].sort_values("sector")
#
# l_1970_eu_model = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")
# l_1995_eu_model = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1995), ["sector", "LS_m"]].sort_values("sector")
# l_2019_eu_model = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")
#
# # CFs
# lp_cf_agr =  1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
# lp_cf_agr[0] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[0]
# lp_cf_bss =  1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
# lp_cf_bss[1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[1]
# lp_cf_fin =  1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
# lp_cf_fin[2] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[2]
# lp_cf_man =  1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
# lp_cf_man[3] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[3]
# lp_cf_nps =  1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
# lp_cf_nps[4] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[4]
# lp_cf_trd =  1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
# lp_cf_trd[5] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[5]
#
# l_1970_eu_model_cf_agr = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_agr"]].sort_values("sector")
# l_1995_eu_model_cf_agr = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1995), ["sector", "LS_m_cf1_agr"]].sort_values("sector")
# l_2019_eu_model_cf_agr = data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_agr"]].sort_values("sector")
#
#
# ss_eu_data = shift_share(np.ones(6),
#             1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values,
#             l_1970_eu_nps['LS'].values,
#             l_2019_eu_nps['LS'].values)
#
# ss_eu_data_felipe = shift_share(np.ones(6),
#             1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values,
#             l_1970_eu_data_felipe['LS'].values,
#             l_2019_eu_data_felipe['LS'].values)
#
# ss_eu_model = shift_share(np.ones(6),
#             1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values,
#             l_1970_eu_model['LS_m'].values,
#             l_2019_eu_model['LS_m'].values)
#
# ss_eu_model_cf_agr = shift_share(np.ones(6),
#             lp_cf_agr,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_agr"]].sort_values("sector")["LS_m_cf1_agr"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_agr"]].sort_values("sector")["LS_m_cf1_agr"].values)
#
# ss_eu_model_cf_man = shift_share(np.ones(6),
#             lp_cf_man,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_man"]].sort_values("sector")["LS_m_cf1_man"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_man"]].sort_values("sector")["LS_m_cf1_man"].values)
#
# ss_eu_model_cf_bss = shift_share(np.ones(6),
#             lp_cf_bss,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_bss"]].sort_values("sector")["LS_m_cf1_bss"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_bss"]].sort_values("sector")["LS_m_cf1_bss"].values)
#
# ss_eu_model_cf_fin = shift_share(np.ones(6),
#             lp_cf_fin,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_fin"]].sort_values("sector")["LS_m_cf1_fin"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_fin"]].sort_values("sector")["LS_m_cf1_fin"].values)
#
# ss_eu_model_cf_trd = shift_share(np.ones(6),
#             lp_cf_trd,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_trd"]].sort_values("sector")["LS_m_cf1_trd"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_trd"]].sort_values("sector")["LS_m_cf1_trd"].values)
#
# ss_eu_model_cf_nps = shift_share(np.ones(6),
#             lp_cf_nps,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m_cf1_nps"]].sort_values("sector")["LS_m_cf1_nps"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m_cf1_nps"]].sort_values("sector")["LS_m_cf1_nps"].values)
#
#
# ss_eu_model_cf_agr_ss = shift_share(np.ones(6),
#             lp_cf_agr,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values)
#
# ss_eu_model_cf_man_ss = shift_share(np.ones(6),
#             lp_cf_man,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values)
#
# ss_eu_model_cf_bss_ss = shift_share(np.ones(6),
#             lp_cf_bss,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values)
#
# ss_eu_model_cf_fin_ss = shift_share(np.ones(6),
#             lp_cf_fin,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values)
#
# ss_eu_model_cf_trd_ss = shift_share(np.ones(6),
#             lp_cf_trd,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values)
#
# ss_eu_model_cf_nps_ss = shift_share(np.ones(6),
#             lp_cf_nps,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 1970), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values,
#             data_pred.loc[(data_pred.country == "EU4") & (data_pred.year == 2019), ["sector", "LS_m"]].sort_values("sector")["LS_m"].values)
#
#
# def annualized(x):
#     return ((x) ** (1 / 49) - 1) * 100
#
# LP_eu_data = annualized(1 + ss_eu_data["LP_growth"].sum())
# LP_eu_data_felipe = annualized(1 + ss_eu_data_felipe["LP_growth"].sum())
# LP_eu_model = annualized(1 + ss_eu_model["LP_growth"].sum())
#
# print("LP DATA", round(LP_eu_data, 2))
# print("LP DATA FELIPE", round(LP_eu_data_felipe, 2))  # to double-check
# print("LP MODEL", round(LP_eu_model, 2))
#
#
# print("\n\n COUNTERFACTUAL")
# LP_eu_model_cf_agr = annualized(1 + ss_eu_model_cf_agr["LP_growth"].sum())
# LP_eu_model_cf_man = annualized(1 + ss_eu_model_cf_man["LP_growth"].sum())
# LP_eu_model_cf_bss = annualized(1 + ss_eu_model_cf_bss["LP_growth"].sum())
# LP_eu_model_cf_fin = annualized(1 + ss_eu_model_cf_fin["LP_growth"].sum())
# LP_eu_model_cf_trd = annualized(1 + ss_eu_model_cf_trd["LP_growth"].sum())
# LP_eu_model_cf_nps = annualized(1 + ss_eu_model_cf_nps["LP_growth"].sum())
#
# LP_eu_model_cf_agr_ss = annualized(1 + ss_eu_model_cf_agr_ss["LP_growth"].sum())
# LP_eu_model_cf_man_ss = annualized(1 + ss_eu_model_cf_man_ss["LP_growth"].sum())
# LP_eu_model_cf_bss_ss = annualized(1 + ss_eu_model_cf_bss_ss["LP_growth"].sum())
# LP_eu_model_cf_fin_ss = annualized(1 + ss_eu_model_cf_fin_ss["LP_growth"].sum())
# LP_eu_model_cf_trd_ss = annualized(1 + ss_eu_model_cf_trd_ss["LP_growth"].sum())
# LP_eu_model_cf_nps_ss = annualized(1 + ss_eu_model_cf_nps_ss["LP_growth"].sum())
#
# print("\nLP MODEL", round(LP_eu_model, 2))
# print("LP MODEL CF agr", round(LP_eu_model_cf_agr, 2))
# print("LP MODEL CF man", round(LP_eu_model_cf_man, 2))
# print("LP MODEL CF bss", round(LP_eu_model_cf_bss, 6))
# print("LP MODEL CF fin", round(LP_eu_model_cf_fin, 2))
# print("LP MODEL CF trd", round(LP_eu_model_cf_trd, 2))
# print("LP MODEL CF nps", round(LP_eu_model_cf_nps, 2))
#
#
# print("\nLP MODEL", round(LP_eu_model, 2))
# print("LP MODEL CF agr FIXED WEIGHTS", round(LP_eu_model_cf_agr_ss, 2))
# print("LP MODEL CF man FIXED WEIGHTS", round(LP_eu_model_cf_man_ss, 2))
# print("LP MODEL CF bss FIXED WEIGHTS", round(LP_eu_model_cf_bss_ss, 6))
# print("LP MODEL CF fin FIXED WEIGHTS", round(LP_eu_model_cf_fin_ss, 2))
# print("LP MODEL CF trd FIXED WEIGHTS", round(LP_eu_model_cf_trd_ss, 2))
# print("LP MODEL CF nps FIXED WEIGHTS", round(LP_eu_model_cf_nps_ss, 2))
#
# # Shit-share decomposition
# sectors = lp_2019_eu_nps["sector"].values
#
# # Shit-share decomposition
# print("\n Shift share for aggregate LP")
# print("EU Data within effect", round(ss_eu_data["within_effect"].sum() / ss_eu_data["LP_growth"].sum() * LP_eu_data, 2))
# print("EU data shift effect", round(ss_eu_data["shift_effect"].sum() / ss_eu_data["LP_growth"].sum() * LP_eu_data, 2))
#
# print("EU Model within effect", round(ss_eu_model["within_effect"].sum() / ss_eu_model["LP_growth"].sum() * LP_eu_model, 2))
# print("EU Model shift effect", round(ss_eu_model["shift_effect"].sum() / ss_eu_model["LP_growth"].sum() * LP_eu_model, 2))
#
# # Shit-share decomposition by sector
# print("\n Shift share for each sector")
# for i in range(6):
#     print("EU data within effect", sectors[i], round(ss_eu_data["within_effect"][i] / ss_eu_data["LP_growth"][i] * round(ss_eu_data["LP_growth"][i] / ss_eu_data["LP_growth"].sum() * LP_eu_data, 2), 2))
#     print("EU data shift effect", sectors[i], round(ss_eu_data["shift_effect"][i] / ss_eu_data["LP_growth"][i] * round(ss_eu_data["LP_growth"][i] / ss_eu_data["LP_growth"].sum() * LP_eu_data, 2), 2))
#
# print("")
# for i in range(6):
#     print("EU model within effect", sectors[i], round(ss_eu_model["within_effect"][i] / ss_eu_model["LP_growth"][i] * round(ss_eu_model["LP_growth"][i] / ss_eu_model["LP_growth"].sum() * LP_eu_model, 2), 2))
#     print("EU model shift effect", sectors[i], round(ss_eu_model["shift_effect"][i] / ss_eu_model["LP_growth"][i] * round(ss_eu_model["LP_growth"][i] / ss_eu_model["LP_growth"].sum() * LP_eu_model, 2), 2))
#
#
