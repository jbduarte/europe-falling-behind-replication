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

import pandas as pd 
import numpy as np 
from scipy.optimize import minimize_scalar, root, fsolve
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

'''
-----------
	Data
-----------
'''

'KLEMS'
data = pd.read_csv('../data/euklems_2023.csv', index_col=[0,1])
data.rename(index={'US':'USA'},inplace=True)
data.rename(columns={'sector': 'sec'}, inplace=True)

'OECD'
GDP_ph = pd.read_excel('../data/raw/OECD_GDP_ph.xlsx', index_col=[0,5], engine = 'openpyxl') #Measured in USD (constant prices 2010 and PPPs).
GDP_ph = GDP_ph[GDP_ph['MEASURE'] == 'USD']
GDP_ph.index.rename(['country', 'year'], inplace=True)

'OECD-IO'
data_NX_sec = pd.read_excel('../data/io_panel.xlsx', index_col=[0,1], engine = 'openpyxl')
data_NX_sec['nx'] = data_NX_sec['expo']+data_NX_sec['impo']
#data_NX_sec['nx'] = 0
data_NX_agg = pd.read_excel('../data/exp_imp_aggregate_panel.xlsx', index_col=[0,1], engine = 'openpyxl')

'Labor Productivity'
data['y_l'] = (data['VA_Q'] / data['H'])*100

'''
---------------
	US Data
---------------
'''
data = data.loc['USA']
data = data[data.index >= 1995]

GDP_ph = GDP_ph.loc['USA']
GDP_ph = GDP_ph[GDP_ph.index >= 1995]

data_NX_sec = data_NX_sec.loc['USA']
data_NX_sec = data_NX_sec[data_NX_sec.index >= 1995]

data_NX_agg = data_NX_agg.loc['USA']
data_NX_agg = data_NX_agg[data_NX_agg.index >= 1995]


'''
--------------------------
   Trade Stylized Fact
--------------------------
'''


'Sectoral Data'
# Ensure indexes are aligned for merging
data_NX_sec = data_NX_sec.reset_index()
data = data.reset_index()

# Merge sectoral data
data = pd.merge(data, data_NX_sec, on=['year', 'sec'], how='left')
data.set_index('year', inplace=True)

data_agr = data[data['sec']=='agr']
data_man = data[data['sec']=='man']
data_trd = data[data['sec']=='trd']
data_bss = data[data['sec']=='bss']
data_fin = data[data['sec']=='fin']
data_nps = data[data['sec']=='nps']
data_ser = data[data['sec']=='ser']
data_tot = data[data['sec']=='tot']

# Adding trade in services
data_ser.loc[:, 'expo'] = data_trd.loc[:,'expo']+data_bss.loc[:,'expo']+data_fin.loc[:,'expo']+data_nps.loc[:,'expo']
data_ser.loc[:, 'impo'] = data_trd.loc[:,'impo']+data_bss.loc[:,'impo']+data_fin.loc[:,'impo']+data_nps.loc[:,'impo']
data_ser.loc[:, 'gdp'] = data_trd.loc[:,'gdp']+data_bss.loc[:,'gdp']+data_fin.loc[:,'gdp']+data_nps.loc[:,'gdp']
data_ser.loc[:, 'nx'] = data_trd.loc[:,'nx']+data_bss.loc[:,'nx']+data_fin.loc[:,'nx']+data_nps.loc[:,'nx']


'''
-------------------------------
	Time Series (Filtering)
-------------------------------
'''

'GDP'
c_GDP_ph, GDP_ph = sm.tsa.filters.hpfilter(GDP_ph['Value'],100)

'GDP Growth'
g_GDP_ph = np.array(GDP_ph/GDP_ph.shift(1) - 1).flatten() #GDP Growth from OECD

'Employment hours'
h_agr_c, h_agr = sm.tsa.filters.hpfilter(data_agr['H'],100)
h_man_c, h_man = sm.tsa.filters.hpfilter(data_man['H'],100)
h_trd_c, h_trd = sm.tsa.filters.hpfilter(data_trd['H'],100)
h_bss_c, h_bss = sm.tsa.filters.hpfilter(data_bss['H'],100)
h_fin_c, h_fin = sm.tsa.filters.hpfilter(data_fin['H'],100)
h_nps_c, h_nps = sm.tsa.filters.hpfilter(data_nps['H'],100)
h_ser_c, h_ser = sm.tsa.filters.hpfilter(data_ser['H'],100)
h_tot_c, h_tot = sm.tsa.filters.hpfilter(data_tot['H'],100)

'Labor Productivity'
y_l_agr_c, y_l_agr = sm.tsa.filters.hpfilter(data_agr['y_l'],100)
y_l_man_c, y_l_man = sm.tsa.filters.hpfilter(data_man['y_l'],100)
y_l_trd_c, y_l_trd = sm.tsa.filters.hpfilter(data_trd['y_l'],100)
y_l_bss_c, y_l_bss = sm.tsa.filters.hpfilter(data_bss['y_l'],100)
y_l_fin_c, y_l_fin = sm.tsa.filters.hpfilter(data_fin['y_l'],100)
y_l_nps_c, y_l_nps = sm.tsa.filters.hpfilter(data_nps['y_l'],100)
y_l_ser_c, y_l_ser = sm.tsa.filters.hpfilter(data_ser['y_l'],100)
y_l_tot_c, y_l_tot = sm.tsa.filters.hpfilter(data_tot['y_l'],100)

'Labor Productivity Growth'
g_y_l_agr = np.array(y_l_agr/y_l_agr.shift(1)-1)
g_y_l_man = np.array(y_l_man/y_l_man.shift(1)-1)
g_y_l_trd = np.array(y_l_trd/y_l_trd.shift(1)-1)
g_y_l_bss = np.array(y_l_bss/y_l_bss.shift(1)-1)
g_y_l_fin = np.array(y_l_fin/y_l_fin.shift(1)-1)
g_y_l_nps = np.array(y_l_nps/y_l_nps.shift(1)-1)
g_y_l_ser = np.array(y_l_ser/y_l_ser.shift(1)-1)
g_y_l_tot = np.array(y_l_tot/y_l_tot.shift(1)-1)

'Prices'
p_agr_c, p_agr = sm.tsa.filters.hpfilter(data_agr['VA']/data_agr['VA_Q'],100)
p_man_c, p_man = sm.tsa.filters.hpfilter(data_man['VA']/data_man['VA_Q'],100)
p_trd_c, p_trd = sm.tsa.filters.hpfilter(data_trd['VA']/data_trd['VA_Q'],100)
p_bss_c, p_bss = sm.tsa.filters.hpfilter(data_bss['VA']/data_bss['VA_Q'],100)
p_fin_c, p_fin = sm.tsa.filters.hpfilter(data_fin['VA']/data_fin['VA_Q'],100)
p_nps_c, p_nps = sm.tsa.filters.hpfilter(data_nps['VA']/data_nps['VA_Q'],100)
p_ser_c, p_ser = sm.tsa.filters.hpfilter(data_ser['VA']/data_ser['VA_Q'],100)
p_tot_c, p_tot = sm.tsa.filters.hpfilter(data_tot['VA']/data_tot['VA_Q'],100)

'Employment Shares'
share_c_agr, share_agr = sm.tsa.filters.hpfilter((h_agr/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)
share_c_man, share_man = sm.tsa.filters.hpfilter((h_man/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)
share_c_trd, share_trd = sm.tsa.filters.hpfilter((h_trd/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)
share_c_bss, share_bss = sm.tsa.filters.hpfilter((h_bss/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)
share_c_fin, share_fin = sm.tsa.filters.hpfilter((h_fin/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)
share_c_nps, share_nps = sm.tsa.filters.hpfilter((h_nps/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)
share_c_ser, share_ser = sm.tsa.filters.hpfilter((h_ser/(h_agr+h_man+h_trd+h_bss+h_fin+h_nps)),100)

'Employment Shares Without Manufacturing (Weights of C)'
share_c_agr_no_man, share_agr_no_man =  sm.tsa.filters.hpfilter(h_agr/(h_agr+h_trd+h_bss+h_fin+h_nps), 100)
share_c_trd_no_man, share_trd_no_man =  sm.tsa.filters.hpfilter(h_trd/(h_agr+h_trd+h_bss+h_fin+h_nps), 100)
share_c_bss_no_man, share_bss_no_man =  sm.tsa.filters.hpfilter(h_bss/(h_agr+h_trd+h_bss+h_fin+h_nps), 100)
share_c_fin_no_man, share_fin_no_man =  sm.tsa.filters.hpfilter(h_fin/(h_agr+h_trd+h_bss+h_fin+h_nps), 100)
share_c_nps_no_man, share_nps_no_man =  sm.tsa.filters.hpfilter(h_nps/(h_agr+h_trd+h_bss+h_fin+h_nps), 100)
share_c_ser_no_man, share_ser_no_man =  sm.tsa.filters.hpfilter(h_ser/(h_agr+h_trd+h_bss+h_fin+h_nps), 100)

'Employment Shares Without Agriculture and Manufacturing (Weights of C)'
share_c_trd_no_agm, share_trd_no_agm =  sm.tsa.filters.hpfilter(h_trd/(h_trd+h_bss+h_fin+h_nps), 100)
share_c_bss_no_agm, share_bss_no_agm =  sm.tsa.filters.hpfilter(h_bss/(h_trd+h_bss+h_fin+h_nps), 100)
share_c_fin_no_agm, share_fin_no_agm =  sm.tsa.filters.hpfilter(h_fin/(h_trd+h_bss+h_fin+h_nps), 100)
share_c_nps_no_agm, share_nps_no_agm =  sm.tsa.filters.hpfilter(h_nps/(h_trd+h_bss+h_fin+h_nps), 100)
share_c_ser_no_agm, share_ser_no_agm =  sm.tsa.filters.hpfilter(h_ser/(h_trd+h_bss+h_fin+h_nps), 100)

#'Net Exports as a Share of Credited Expenditures'
#nx_c_agr_E, nx_agr_E = sm.tsa.filters.hpfilter(data_agr['nx']/(data_tot['VA']-data_tot['nx']),100)
#nx_c_man_E, nx_man_E = sm.tsa.filters.hpfilter(data_man['nx']/(data_tot['VA']-data_tot['nx']),100)
#nx_c_trd_E, nx_trd_E = sm.tsa.filters.hpfilter(data_trd['nx']/(data_tot['VA']-data_tot['nx']),100)
#nx_c_bss_E, nx_bss_E = sm.tsa.filters.hpfilter(data_bss['nx']/(data_tot['VA']-data_tot['nx']),100)
#nx_c_fin_E, nx_fin_E = sm.tsa.filters.hpfilter(data_fin['nx']/(data_tot['VA']-data_tot['nx']),100)
#nx_c_nps_E, nx_nps_E = sm.tsa.filters.hpfilter(data_nps['nx']/(data_tot['VA']-data_tot['nx']),100)
#nx_ser_E = nx_trd_E + nx_bss_E + nx_fin_E + nx_nps_E
#nx_c_tot_E, nx_tot_E = sm.tsa.filters.hpfilter(data_tot['nx']/(data_tot['VA']-data_tot['nx']),100)

'Real Net Exports'
nx_c_agr_q, nx_agr_q = sm.tsa.filters.hpfilter(data_agr['nx']/(data_agr['VA']/data_agr['VA_Q']),100)
nx_c_man_q, nx_man_q = sm.tsa.filters.hpfilter(data_man['nx']/(data_man['VA']/data_man['VA_Q']),100)
nx_c_trd_q, nx_trd_q = sm.tsa.filters.hpfilter(data_trd['nx']/(data_trd['VA']/data_trd['VA_Q']),100)
nx_c_bss_q, nx_bss_q = sm.tsa.filters.hpfilter(data_bss['nx']/(data_bss['VA']/data_bss['VA_Q']),100)
nx_c_fin_q, nx_fin_q = sm.tsa.filters.hpfilter(data_fin['nx']/(data_fin['VA']/data_fin['VA_Q']),100)
nx_c_nps_q, nx_nps_q = sm.tsa.filters.hpfilter(data_nps['nx']/(data_nps['VA']/data_nps['VA_Q']),100)
nx_ser_q = nx_trd_q + nx_bss_q + nx_fin_q + nx_nps_q
nx_c_tot_q, nx_tot_q = sm.tsa.filters.hpfilter(data_tot['nx']/(data_tot['VA']/data_tot['VA_Q']),100)

'Nominal Net Exports as a Share of Credited Expenditures'
nx_c_agr_E, nx_agr_E = sm.tsa.filters.hpfilter(((1/data_agr['y_l'])*(data_agr['nx']/(data_agr['VA']/data_agr['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)
nx_c_man_E, nx_man_E = sm.tsa.filters.hpfilter(((1/data_man['y_l'])*(data_man['nx']/(data_man['VA']/data_man['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)
nx_c_trd_E, nx_trd_E = sm.tsa.filters.hpfilter(((1/data_trd['y_l'])*(data_trd['nx']/(data_trd['VA']/data_trd['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)
nx_c_bss_E, nx_bss_E = sm.tsa.filters.hpfilter(((1/data_bss['y_l'])*(data_bss['nx']/(data_bss['VA']/data_bss['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)
nx_c_fin_E, nx_fin_E = sm.tsa.filters.hpfilter(((1/data_fin['y_l'])*(data_fin['nx']/(data_fin['VA']/data_fin['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)
nx_c_nps_E, nx_nps_E = sm.tsa.filters.hpfilter(((1/data_nps['y_l'])*(data_nps['nx']/(data_nps['VA']/data_nps['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)
nx_ser_E = nx_trd_E + nx_bss_E + nx_fin_E + nx_nps_E
nx_c_tot_E, nx_tot_E = sm.tsa.filters.hpfilter(((1/data_tot['y_l'])*(data_tot['nx']/(data_tot['VA']/data_tot['VA_Q'])))/(data_tot['VA']-data_tot['nx']),100)

'Net Exports as a Share of GDP'
nx_c_agr_Y, nx_agr_Y = sm.tsa.filters.hpfilter(data_agr['nx']/data_NX_agg['gdp'],100)
nx_c_man_Y, nx_man_Y = sm.tsa.filters.hpfilter(data_man['nx']/data_NX_agg['gdp'],100)
nx_c_trd_Y, nx_trd_Y = sm.tsa.filters.hpfilter(data_trd['nx']/data_NX_agg['gdp'],100)
nx_c_bss_Y, nx_bss_Y = sm.tsa.filters.hpfilter(data_bss['nx']/data_NX_agg['gdp'],100)
nx_c_fin_Y, nx_fin_Y = sm.tsa.filters.hpfilter(data_fin['nx']/data_NX_agg['gdp'],100)
nx_c_nps_Y, nx_nps_Y = sm.tsa.filters.hpfilter(data_nps['nx']/data_NX_agg['gdp'],100)
nx_ser_Y = nx_trd_Y + nx_bss_Y + nx_fin_Y + nx_nps_Y
nx_c_tot_Y, nx_tot_Y = sm.tsa.filters.hpfilter(data_tot['nx']/data_NX_agg['gdp'],100)

'Opennes (exp+imp)/GDP'
exp_imp_c_agr_Y, exp_imp_agr_Y = sm.tsa.filters.hpfilter((data_agr['expo'] + (-1)*data_agr['impo'])/data_NX_agg['gdp'],100)
exp_imp_c_man_Y, exp_imp_man_Y = sm.tsa.filters.hpfilter((data_man['expo'] + (-1)*data_man['impo'])/data_NX_agg['gdp'],100)
exp_imp_c_trd_Y, exp_imp_trd_Y = sm.tsa.filters.hpfilter((data_trd['expo'] + (-1)*data_trd['impo'])/data_NX_agg['gdp'],100)
exp_imp_c_bss_Y, exp_imp_bss_Y = sm.tsa.filters.hpfilter((data_bss['expo'] + (-1)*data_bss['impo'])/data_NX_agg['gdp'],100)
exp_imp_c_fin_Y, exp_imp_fin_Y = sm.tsa.filters.hpfilter((data_fin['expo'] + (-1)*data_fin['impo'])/data_NX_agg['gdp'],100)
exp_imp_c_nps_Y, exp_imp_nps_Y = sm.tsa.filters.hpfilter((data_nps['expo'] + (-1)*data_nps['impo'])/data_NX_agg['gdp'],100)
exp_imp_ser_Y = exp_imp_trd_Y + exp_imp_bss_Y + exp_imp_fin_Y + exp_imp_nps_Y
exp_imp_c_tot_Y, exp_imp_tot_Y = sm.tsa.filters.hpfilter((data_tot['expo'] + (-1)*data_tot['impo'])/data_NX_agg['gdp'],100)

'Nominal Exports'
expo_c_agr, expo_agr = sm.tsa.filters.hpfilter(data_agr['expo'],100)
expo_c_man, expo_man = sm.tsa.filters.hpfilter(data_man['expo'],100)
expo_c_trd, expo_trd = sm.tsa.filters.hpfilter(data_trd['expo'],100)
expo_c_bss, expo_bss = sm.tsa.filters.hpfilter(data_bss['expo'],100)
expo_c_fin, expo_fin = sm.tsa.filters.hpfilter(data_fin['expo'],100)
expo_c_nps, expo_nps = sm.tsa.filters.hpfilter(data_nps['expo'],100)
expo_ser= expo_trd + expo_bss + expo_fin + expo_nps
expo_c_tot, expo_tot = sm.tsa.filters.hpfilter(data_tot['expo'],100)

'Nominal Imports'
impo_c_agr, impo_agr = sm.tsa.filters.hpfilter((-1)*data_agr['impo'],100)
impo_c_man, impo_man = sm.tsa.filters.hpfilter((-1)*data_man['impo'],100)
impo_c_trd, impo_trd = sm.tsa.filters.hpfilter((-1)*data_trd['impo'],100)
impo_c_bss, impo_bss = sm.tsa.filters.hpfilter((-1)*data_bss['impo'],100)
impo_c_fin, impo_fin = sm.tsa.filters.hpfilter((-1)*data_fin['impo'],100)
impo_c_nps, impo_nps = sm.tsa.filters.hpfilter((-1)*data_nps['impo'],100)
impo_ser= impo_trd + impo_bss + impo_fin + impo_nps
impo_c_tot, impo_tot = sm.tsa.filters.hpfilter((-1)*data_tot['impo'],100)


''' 
------------------------
	Parameretization 
------------------------
'''
# The following string runs the calibration of the model on the US, and imports the parameter values that are used in this the baseline calibration.
from model_calibration_USA import sigma, eps_agr, eps_trd, eps_fin, eps_bss, eps_nps, eps_ser

'Relative labor demand'
l_agr_l_man = h_agr/h_man
l_trd_l_man = h_trd/h_man
l_bss_l_man = h_bss/h_man
l_fin_l_man = h_fin/h_man
l_nps_l_man = h_nps/h_man
l_ser_l_man = h_ser/h_man

'Relative Expenditures in Last Period'
rel_exp_agr_man_last = np.array((data_agr['VA']-data_agr['nx'])/(data_man['VA']-data_man['nx']))[-1]
rel_exp_trd_man_last = np.array((data_trd['VA']-data_trd['nx'])/(data_man['VA']-data_man['nx']))[-1]
rel_exp_bss_man_last = np.array((data_bss['VA']-data_bss['nx'])/(data_man['VA']-data_man['nx']))[-1]
rel_exp_fin_man_last = np.array((data_fin['VA']-data_fin['nx'])/(data_man['VA']-data_man['nx']))[-1]
rel_exp_nps_man_last = np.array((data_nps['VA']-data_nps['nx'])/(data_man['VA']-data_man['nx']))[-1]
rel_exp_ser_man_last = np.array((data_ser['VA']-data_ser['nx'])/(data_man['VA']-data_man['nx']))[-1]

'Relative Prices in Last Period'
rel_p_agr_man_last = np.array((data_agr['VA']/data_agr['VA_Q'])/(data_man['VA']/data_man['VA_Q']))[-1]
rel_p_trd_man_last = np.array((data_trd['VA']/data_trd['VA_Q'])/(data_man['VA']/data_man['VA_Q']))[-1]
rel_p_bss_man_last = np.array((data_bss['VA']/data_bss['VA_Q'])/(data_man['VA']/data_man['VA_Q']))[-1]
rel_p_fin_man_last = np.array((data_fin['VA']/data_fin['VA_Q'])/(data_man['VA']/data_man['VA_Q']))[-1]
rel_p_nps_man_last = np.array((data_nps['VA']/data_nps['VA_Q'])/(data_man['VA']/data_man['VA_Q']))[-1]
rel_p_ser_man_last = np.array((data_ser['VA']/data_ser['VA_Q'])/(data_man['VA']/data_man['VA_Q']))[-1]

'Expenditure Relative to Manufacturing Prices in Last Period'
E_pm_last = np.array(((data_tot['VA']-data_tot['nx'])*100)/(data_man['VA']/data_man['VA_Q']))[-1] 

'Manufacturing Value Added Share'
share_va_man_c, share_va_man = sm.tsa.filters.hpfilter((data_man['VA']-data_man['nx'])/(data_tot['VA']-data_tot['nx']),100) 
share_va_man_last = np.array((data_man['VA']-data_man['nx'])/(data_tot['VA']-data_tot['nx']))[-1] 


'''
---------------------------------------------
		Time Series (Inputs of the Model)
---------------------------------------------
'''
nom_exp_c, nom_exp = sm.tsa.filters.hpfilter((data_tot['VA']-data_tot['nx']), 100)
g_nom_exp = np.array(nom_exp/nom_exp.shift(1) - 1).flatten()

t_0 = np.array(data.index)[0]
ts_length = np.array(data.index)[-1] - np.array(data.index)[0]

'First period. Normalization.'
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
GDP_per_h = [1]
E = [1]
year = [t_0]

'Productivity and Real Expenditure Growth'
for i in range(int(ts_length)):
	A_agr.append((1 + g_y_l_agr[i+1])*A_agr[i])
	A_man.append((1 + g_y_l_man[i+1])*A_man[i])
	A_bss.append((1 + g_y_l_bss[i+1])*A_bss[i])
	A_trd.append((1 + g_y_l_trd[i+1])*A_trd[i])
	A_fin.append((1 + g_y_l_fin[i+1])*A_fin[i])
	A_nps.append((1 + g_y_l_nps[i+1])*A_nps[i])
	A_ser.append((1 + g_y_l_ser[i+1])*A_ser[i])
	A_tot.append((1 + g_y_l_tot[i+1])*A_tot[i])
	E.append((1 + g_nom_exp[i+1])*E[i])
	GDP_per_h.append((1 + g_GDP_ph[i+1])*GDP_per_h[i])
	year.append(t_0 + i + 1)

'Expenditures Relative to Manufacturing'
rel_exp_agr_man_c, rel_exp_agr_man = sm.tsa.filters.hpfilter((data_agr['VA']-data_agr['nx'])/(data_man['VA']-data_man['nx']),100)  
rel_exp_trd_man_c, rel_exp_trd_man = sm.tsa.filters.hpfilter((data_trd['VA']-data_trd['nx'])/(data_man['VA']-data_man['nx']),100)  
rel_exp_bss_man_c, rel_exp_bss_man = sm.tsa.filters.hpfilter((data_bss['VA']-data_bss['nx'])/(data_man['VA']-data_man['nx']),100)  
rel_exp_fin_man_c, rel_exp_fin_man = sm.tsa.filters.hpfilter((data_fin['VA']-data_fin['nx'])/(data_man['VA']-data_man['nx']),100)  
rel_exp_nps_man_c, rel_exp_nps_man = sm.tsa.filters.hpfilter((data_nps['VA']-data_nps['nx'])/(data_man['VA']-data_man['nx']),100)  
rel_exp_ser_man_c, rel_exp_ser_man = sm.tsa.filters.hpfilter((data_ser['VA']-data_ser['nx'])/(data_man['VA']-data_man['nx']),100)  

'Prices Relative to Manufacturing'
rel_p_agr_man_c, rel_p_agr_man = sm.tsa.filters.hpfilter((data_agr['VA']/data_agr['VA_Q'])/(data_man['VA']/data_man['VA_Q']),100)
rel_p_trd_man_c, rel_p_trd_man = sm.tsa.filters.hpfilter((data_trd['VA']/data_trd['VA_Q'])/(data_man['VA']/data_man['VA_Q']),100)
rel_p_bss_man_c, rel_p_bss_man = sm.tsa.filters.hpfilter((data_bss['VA']/data_bss['VA_Q'])/(data_man['VA']/data_man['VA_Q']),100)
rel_p_fin_man_c, rel_p_fin_man = sm.tsa.filters.hpfilter((data_fin['VA']/data_fin['VA_Q'])/(data_man['VA']/data_man['VA_Q']),100)
rel_p_nps_man_c, rel_p_nps_man = sm.tsa.filters.hpfilter((data_nps['VA']/data_nps['VA_Q'])/(data_man['VA']/data_man['VA_Q']),100)
rel_p_ser_man_c, rel_p_ser_man = sm.tsa.filters.hpfilter((data_ser['VA']/data_ser['VA_Q'])/(data_man['VA']/data_man['VA_Q']),100)

'Initial employment shares'
om_agr = np.array(share_agr)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_agr)[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0]
om_man = np.array(share_man)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_man)[0])/np.array(p_man)[0])*np.array(nx_man_E)[0]
om_trd = np.array(share_trd)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_trd)[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0]
om_bss = np.array(share_bss)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_bss)[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0]
om_fin = np.array(share_fin)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_fin)[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0]
om_nps = np.array(share_nps)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_nps)[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]
om_ser = np.array(share_ser)[0]*(1+((1/np.array(A_tot)[0])/np.array(p_tot)[0])*np.array(nx_tot_E)[0])-((1/np.array(A_ser)[0])/np.array(p_ser)[0])*np.array(nx_ser_E)[0]

om_agr_closed = np.array(share_agr)[0]
om_man_closed = np.array(share_man)[0]
om_trd_closed = np.array(share_trd)[0]
om_bss_closed = np.array(share_bss)[0]
om_fin_closed = np.array(share_fin)[0]
om_nps_closed = np.array(share_nps)[0]
om_ser_closed = np.array(share_ser)[0]

'Non-homothetic CES index'
def C_index(om_m, om_i, exp_i_m, pi_pm, sigma, epsilon_i):
	C_level = ((om_m/om_i)*exp_i_m*(pi_pm**(sigma-1)))**(1/(epsilon_i-1))
	g_C = np.array(C_level/C_level.shift(1) - 1)
	C = [1]
	for i in range(len(g_C) - 1):
		C.append((1+g_C[i+1])*C[i])
	return C

'''
-------------------------
		The Models
-------------------------
'''

class model_ams:
	"Structural Transformation with Agriculture, Manufacturing and Services"

	def __init__(self, sigma=sigma, eps_agr=eps_agr, eps_man=1, eps_ser=eps_ser, om_agr=om_agr, om_man=om_man, om_ser=om_ser):
		'Initialize the Parameters'
		self.sigma, self.eps_agr, self.eps_man, self.eps_ser, self.om_agr, self.om_man, self.om_ser, self.om_agr_closed, self.om_man_closed, self.om_ser_closed = sigma, eps_agr, 1, eps_ser, om_agr, om_man, om_ser, om_agr_closed, om_man_closed, om_ser_closed

	def E(self, C, A_agr, A_man, A_ser):
		'Expenditure'
		weight_agr = self.om_agr*(A_agr**(self.sigma-1))*(C**self.eps_agr) 
		weight_man = self.om_man*(A_man**(self.sigma-1))*(C**self.eps_man) 
		weight_ser = self.om_ser*(A_ser**(self.sigma-1))*(C**self.eps_ser)
		return (weight_agr + weight_man + weight_ser)**(1/(1-self.sigma))

	def labor_demand(self, C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E):
		'total Labor Demand (domestic)'
		L = (self.om_agr*(C**self.eps_agr)*(A_agr**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_ser)**(1-self.sigma))*nx_agr_E + 
			 self.om_man*(C**self.eps_man)*(A_man**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_ser)**(1-self.sigma))*nx_man_E + 
			 self.om_ser*(C**self.eps_ser)*(A_ser**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_ser)**(1-self.sigma))*nx_ser_E)
		return L

	def share_agr(self, C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E):
		'Employment Share in Agriculture'
		l_agr = (self.om_agr*(C**self.eps_agr)*(A_agr**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_ser)**(1-self.sigma))*nx_agr_E)/(self.labor_demand(C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E))
		return l_agr

	def share_man(self, C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E):
		'Employment Share in Manufacturing'
		l_man = (self.om_man*(C**self.eps_man)*(A_man**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_ser)**(1-self.sigma))*nx_man_E)/(self.labor_demand(C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E))
		return l_man

	def share_ser(self, C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E):
		'Employment Share in Services'
		l_ser = (self.om_ser*(C**self.eps_ser)*(A_ser**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_ser)**(1-self.sigma))*nx_ser_E)/(self.labor_demand(C, A_agr, A_man, A_ser, nx_agr_E, nx_man_E, nx_ser_E))
		return l_ser

	def labor_demand_closed(self, C, A_agr, A_man, A_ser):
		'total Labor Demand (domestic)'
		L = (self.om_agr_closed*(C**self.eps_agr)*(A_agr**(self.sigma - 1)) + 
			 self.om_man_closed*(C**self.eps_man)*(A_man**(self.sigma - 1)) + 
			 self.om_ser_closed*(C**self.eps_ser)*(A_ser**(self.sigma - 1)))
		return L

	def share_agr_closed(self, C, A_agr, A_man, A_ser):
		'Employment Share in Agriculture'
		l_agr = (self.om_agr_closed*(C**self.eps_agr)*(A_agr**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_ser)
		return l_agr

	def share_man_closed(self, C, A_agr, A_man, A_ser):
		'Employment Share in Manufacturing'
		l_man = (self.om_man_closed*(C**self.eps_man)*(A_man**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_ser)
		return l_man

	def share_ser_closed(self, C, A_agr, A_man, A_ser):
		'Employment Share in Services'
		l_ser = (self.om_ser_closed*(C**self.eps_ser)*(A_ser**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_ser)
		return l_ser

class model_nps:
	"Agriculture, Manufacturing, Whole Sale and Retail Trade, Business Services, and the Rest of Services"
	def __init__(self, sigma=sigma, eps_agr=eps_agr, eps_man=1, eps_trd=eps_trd, eps_bss=eps_bss, eps_fin=eps_fin, eps_nps=eps_nps, om_agr=om_agr, om_man=om_man, om_trd=om_trd, om_bss=om_bss, om_fin=om_fin, om_nps=om_nps):
		'Initialize the Parameters'
		self.sigma, self.eps_agr, self.eps_man, self.eps_trd, self.eps_bss, self.eps_fin, self.eps_nps, self.om_agr, self.om_man, self.om_trd, self.om_bss, self.om_fin, self.om_nps, self.om_agr_closed, self.om_man_closed, self.om_trd_closed, self.om_bss_closed, self.om_fin_closed, self.om_nps_closed = sigma, eps_agr, eps_man, eps_trd, eps_bss, eps_fin, eps_nps, om_agr, om_man, om_trd, om_bss, om_fin, om_nps, om_agr_closed, om_man_closed, om_trd_closed, om_bss_closed, om_fin_closed, om_nps_closed 

	def E(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Expenditure'
		weight_agr = self.om_agr*(A_agr**(self.sigma-1))*(C**self.eps_agr) 
		weight_man = self.om_man*(A_man**(self.sigma-1))*(C**self.eps_man) 
		weight_trd = self.om_trd*(A_trd**(self.sigma-1))*(C**self.eps_trd)
		weight_bss = self.om_bss*(A_bss**(self.sigma-1))*(C**self.eps_bss)
		weight_fin = self.om_fin*(A_fin**(self.sigma-1))*(C**self.eps_fin)
		weight_nps = self.om_nps*(A_nps**(self.sigma-1))*(C**self.eps_nps)
		return (weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps)**(1/(1-self.sigma))

	def labor_demand(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'total Labor Demand'
		L = (self.om_agr*(C**self.eps_agr)*(A_agr**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_agr_E + 
			 self.om_man*(C**self.eps_man)*(A_man**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_man_E + 
			 self.om_trd*(C**self.eps_trd)*(A_trd**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_trd_E + 
			 self.om_bss*(C**self.eps_bss)*(A_bss**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_bss_E + 
			 self.om_fin*(C**self.eps_fin)*(A_fin**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_fin_E + 
			 self.om_nps*(C**self.eps_nps)*(A_nps**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_nps_E)
		return L

	def share_agr(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'Employment Share in Agriculture'
		l_agr = (self.om_agr*(C**self.eps_agr)*(A_agr**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_agr_E)/(self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E))
		return l_agr

	def share_man(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'Employment Share in Manufacturing'
		l_man = (self.om_man*(C**self.eps_man)*(A_man**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_man_E)/(self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E))
		return l_man

	def share_trd(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'Employment Share in Whole Sale and Retail Trade'
		l_trd = (self.om_trd*(C**self.eps_trd)*(A_trd**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_trd_E)/(self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E))
		return l_trd

	def share_bss(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'Employment Share in Business Services'
		l_bss = (self.om_bss*(C**self.eps_bss)*(A_bss**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_bss_E)/(self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E))
		return l_bss

	def share_fin(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'Employment Share in Financial Services'
		l_fin = (self.om_fin*(C**self.eps_fin)*(A_fin**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_fin_E)/(self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E))
		return l_fin

	def share_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E):
		'Employment Share in the Rest of Services'
		l_nps = (self.om_nps*(C**self.eps_nps)*(A_nps**(self.sigma - 1)) + (self.E(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)**(1-self.sigma))*nx_nps_E)/(self.labor_demand(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, nx_agr_E, nx_man_E, nx_trd_E, nx_bss_E, nx_fin_E, nx_nps_E))
		return l_nps

	def labor_demand_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'total Labor Demand'
		L = (self.om_agr_closed*(C**self.eps_agr)*(A_agr**(self.sigma - 1)) + 
			 self.om_man_closed*(C**self.eps_man)*(A_man**(self.sigma - 1)) + 
			 self.om_trd_closed*(C**self.eps_trd)*(A_trd**(self.sigma - 1)) + 
			 self.om_bss_closed*(C**self.eps_bss)*(A_bss**(self.sigma - 1)) + 
			 self.om_fin_closed*(C**self.eps_fin)*(A_fin**(self.sigma - 1)) + 
			 self.om_nps_closed*(C**self.eps_nps)*(A_nps**(self.sigma - 1)))
		return L

	def share_agr_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Employment Share in Agriculture'
		l_agr = (self.om_agr_closed*(C**self.eps_agr)*(A_agr**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
		return l_agr

	def share_man_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Employment Share in Manufacturing'
		l_man = (self.om_man_closed*(C**self.eps_man)*(A_man**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
		return l_man

	def share_trd_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Employment Share in Whole Sale and Retail Trade'
		l_trd = (self.om_trd_closed*(C**self.eps_trd)*(A_trd**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
		return l_trd

	def share_bss_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Employment Share in Business Services'
		l_bss = (self.om_bss_closed*(C**self.eps_bss)*(A_bss**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
		return l_bss

	def share_fin_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Employment Share in Financial Services'
		l_fin = (self.om_fin_closed*(C**self.eps_fin)*(A_fin**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
		return l_fin

	def share_nps_closed(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
		'Employment Share in the Rest of Services'
		l_nps = (self.om_nps_closed*(C**self.eps_nps)*(A_nps**(self.sigma - 1)))/self.labor_demand_closed(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps)
		return l_nps

'Model Economy for Agriculture, Manufacturing and Services '
econ = model_ams()

'Computation of Non-Homothetic CES index'
C_agr = C_index(econ.om_agr, econ.om_man, rel_exp_agr_man, rel_p_agr_man, sigma, eps_agr)
C_ser = C_index((1-econ.om_agr-econ.om_man), econ.om_man, rel_exp_ser_man, rel_p_ser_man, sigma, eps_ser)
C_agr_closed = C_index(econ.om_agr_closed, econ.om_man_closed, l_agr_l_man, rel_p_agr_man, sigma, eps_agr)
C_ser_closed = C_index((1-econ.om_agr_closed-econ.om_man_closed), econ.om_man_closed, l_ser_l_man, rel_p_ser_man, sigma, eps_ser)
C_ams = np.array(share_agr_no_man*C_agr + share_ser_no_man*C_ser)

'Alternative Non-Homothetic CES indexes'

#Simple Average Excluding Manufacturing
C_ams_simple_av = (np.array(C_agr) + np.array(C_ser))/2
E_pm_c, E_pm = sm.tsa.filters.hpfilter(((data_tot['VA']-data_tot['nx'])*100)/(data_man['VA']/data_man['VA_Q']),100)

#C implied from manufacturing
C_man_level = (share_va_man/om_man)*(E_pm**(1-sigma))
g_C_man = np.array(C_man_level/C_man_level.shift(1) - 1)
C_man = [1]
for i in range(len(g_C_man) - 1):
	C_man.append((1+g_C_man[i+1])*C_man[i])

#Simple Average Including Manufacturing
C_ams_simple_av_alt = np.array(share_agr*C_agr + share_man*C_man + share_ser*C_ser)

'Aggregator as input of the model'
#C=C_ams
C=C_ser
C_closed=C_ser_closed

'Initial Values (Match perfectly by construction)'
share_agr_ams = [econ.share_agr(C[0], A_agr[0], A_man[0], A_ser[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_ser[0])/np.array(p_ser)[0])*np.array(nx_ser_E)[0])]
share_man_ams = [econ.share_man(C[0], A_agr[0], A_man[0], A_ser[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_ser[0])/np.array(p_ser)[0])*np.array(nx_ser_E)[0])]
share_ser_ams = [econ.share_ser(C[0], A_agr[0], A_man[0], A_ser[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_ser[0])/np.array(p_ser)[0])*np.array(nx_ser_E)[0])]
share_agr_ams_closed = [econ.share_agr_closed(C_closed[0], A_agr[0], A_man[0], A_ser[0])]
share_man_ams_closed = [econ.share_man_closed(C_closed[0], A_agr[0], A_man[0], A_ser[0])]
share_ser_ams_closed = [econ.share_ser_closed(C_closed[0], A_agr[0], A_man[0], A_ser[0])]

'Subsequent Time Series Feeding Observed Growth Rates of Income and Productivity (Test of the Theory)'
for i in range(int(ts_length)):
	share_agr_ams.append(econ.share_agr(C[i+1], A_agr[i+1], A_man[i+1], A_ser[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_ser[i+1])/np.array(p_ser)[0])*np.array(nx_ser_E)[i+1]))
	share_man_ams.append(econ.share_man(C[i+1], A_agr[i+1], A_man[i+1], A_ser[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_ser[i+1])/np.array(p_ser)[0])*np.array(nx_ser_E)[i+1]))
	share_ser_ams.append(econ.share_ser(C[i+1], A_agr[i+1], A_man[i+1], A_ser[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_ser[i+1])/np.array(p_ser)[0])*np.array(nx_ser_E)[i+1]))
	share_agr_ams_closed.append(econ.share_agr_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_ser[i+1]))
	share_man_ams_closed.append(econ.share_man_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_ser[i+1]))
	share_ser_ams_closed.append(econ.share_ser_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_ser[i+1]))

'Aggregate Productivity'
weighted_ams_A_agr = [a*b for a,b in zip(share_agr_ams, A_agr)] 
weighted_ams_A_man = [a*b for a,b in zip(share_man_ams, A_man)] 
weighted_ams_A_ser = [a*b for a,b in zip(share_ser_ams, A_ser)] 

A_tot_ams = [sum(x) for x in zip(weighted_ams_A_agr, weighted_ams_A_man, weighted_ams_A_ser)]
A_tot_ams_weighted = share_agr*A_agr + share_man*A_man + share_ser*A_ser 

'Aggregate Productivity'
weighted_ams_A_agr_closed = [a*b for a,b in zip(share_agr_ams_closed, A_agr)] 
weighted_ams_A_man_closed = [a*b for a,b in zip(share_man_ams_closed, A_man)] 
weighted_ams_A_ser_closed = [a*b for a,b in zip(share_ser_ams_closed, A_ser)] 

A_tot_ams_closed = [sum(x) for x in zip(weighted_ams_A_agr_closed, weighted_ams_A_man_closed, weighted_ams_A_ser_closed)]

'Model Economy for Agriculture, Manufacturing, Whole Sale and Retail Trade, Business Services, Financial Services and Non-Progressive Services'
econ = model_nps()

'Computation of Non-Homothetic CES index'
C_trd = C_index(econ.om_trd, econ.om_man,  rel_exp_trd_man, rel_p_trd_man, sigma, eps_trd)
C_bss = C_index(econ.om_bss, econ.om_man,  rel_exp_bss_man, rel_p_bss_man, sigma, eps_bss)
C_fin = C_index(econ.om_fin, econ.om_man,  rel_exp_fin_man, rel_p_fin_man, sigma, eps_fin)
C_nps = C_index(econ.om_nps, econ.om_man,  rel_exp_nps_man, rel_p_nps_man, sigma, eps_nps)
C_nps = np.array(share_agr_no_man*C_agr + share_trd_no_man*C_trd + share_bss_no_man*C_bss + share_fin_no_man*C_fin + share_nps_no_man*C_nps)
C_nps_agm = np.array(share_trd_no_agm*C_trd + share_bss_no_agm*C_bss + share_fin_no_agm*C_fin + share_nps_no_agm*C_nps)

'Alternative aggregators'
C_nps_simple_av = (np.array(C_agr) + np.array(C_trd) + np.array(C_bss) + np.array(C_fin) + np.array(C_nps))/5
C_nps_simple_av_alt = (np.array(C_agr) + np.array(C_man) + np.array(C_trd) + np.array(C_bss) + np.array(C_fin) + np.array(C_nps))/6
C_nps_alt = np.array(share_agr*C_agr + share_man*C_man + share_trd*C_trd + share_bss*C_bss + share_fin*C_fin + share_nps*C_nps)


'Aggregator as input of the model'
#C=C_nps
C=C_ser
C_closed=C_ser_closed

'Initial Values (Match perfectly by construction)'
share_agr_nps = [econ.share_agr(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_trd[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0], ((1/A_bss[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0], ((1/A_fin[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0], ((1/A_nps[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0])]
share_man_nps = [econ.share_man(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_trd[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0], ((1/A_bss[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0], ((1/A_fin[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0], ((1/A_nps[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0])]
share_trd_nps = [econ.share_trd(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_trd[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0], ((1/A_bss[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0], ((1/A_fin[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0], ((1/A_nps[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0])]
share_bss_nps = [econ.share_bss(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_trd[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0], ((1/A_bss[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0], ((1/A_fin[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0], ((1/A_nps[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0])]
share_fin_nps = [econ.share_fin(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_trd[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0], ((1/A_bss[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0], ((1/A_fin[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0], ((1/A_nps[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0])]
share_nps_nps = [econ.share_nps(C[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0], ((1/A_agr[0])/np.array(p_agr)[0])*np.array(nx_agr_E)[0], ((1/A_man[0])/np.array(p_man)[0])*np.array(nx_man_E)[0], ((1/A_trd[0])/np.array(p_trd)[0])*np.array(nx_trd_E)[0], ((1/A_bss[0])/np.array(p_bss)[0])*np.array(nx_bss_E)[0], ((1/A_fin[0])/np.array(p_fin)[0])*np.array(nx_fin_E)[0], ((1/A_nps[0])/np.array(p_nps)[0])*np.array(nx_nps_E)[0])]
share_agr_nps_closed = [econ.share_agr_closed(C_closed[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])]
share_man_nps_closed = [econ.share_man_closed(C_closed[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])]
share_trd_nps_closed = [econ.share_trd_closed(C_closed[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])]
share_bss_nps_closed = [econ.share_bss_closed(C_closed[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])]
share_fin_nps_closed = [econ.share_fin_closed(C_closed[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])]
share_nps_nps_closed = [econ.share_nps_closed(C_closed[0], A_agr[0], A_man[0], A_trd[0], A_bss[0], A_fin[0], A_nps[0])]

'Subsequent Time Series Feeding Observed Growth Rates of Income and Productivity (Test of the Theory)'
for i in range(int(ts_length)):
	share_agr_nps.append(econ.share_agr(C[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_trd[i+1])/np.array(p_trd)[0])*np.array(nx_trd_E)[i+1], ((1/A_bss[i+1])/np.array(p_bss)[0])*np.array(nx_bss_E)[i+1], ((1/A_fin[i+1])/np.array(p_fin)[0])*np.array(nx_fin_E)[i+1], ((1/A_nps[i+1])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]))
	share_man_nps.append(econ.share_man(C[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_trd[i+1])/np.array(p_trd)[0])*np.array(nx_trd_E)[i+1], ((1/A_bss[i+1])/np.array(p_bss)[0])*np.array(nx_bss_E)[i+1], ((1/A_fin[i+1])/np.array(p_fin)[0])*np.array(nx_fin_E)[i+1], ((1/A_nps[i+1])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]))
	share_trd_nps.append(econ.share_trd(C[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_trd[i+1])/np.array(p_trd)[0])*np.array(nx_trd_E)[i+1], ((1/A_bss[i+1])/np.array(p_bss)[0])*np.array(nx_bss_E)[i+1], ((1/A_fin[i+1])/np.array(p_fin)[0])*np.array(nx_fin_E)[i+1], ((1/A_nps[i+1])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]))
	share_bss_nps.append(econ.share_bss(C[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_trd[i+1])/np.array(p_trd)[0])*np.array(nx_trd_E)[i+1], ((1/A_bss[i+1])/np.array(p_bss)[0])*np.array(nx_bss_E)[i+1], ((1/A_fin[i+1])/np.array(p_fin)[0])*np.array(nx_fin_E)[i+1], ((1/A_nps[i+1])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]))
	share_fin_nps.append(econ.share_fin(C[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_trd[i+1])/np.array(p_trd)[0])*np.array(nx_trd_E)[i+1], ((1/A_bss[i+1])/np.array(p_bss)[0])*np.array(nx_bss_E)[i+1], ((1/A_fin[i+1])/np.array(p_fin)[0])*np.array(nx_fin_E)[i+1], ((1/A_nps[i+1])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]))
	share_nps_nps.append(econ.share_nps(C[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1], ((1/A_agr[i+1])/np.array(p_agr)[0])*np.array(nx_agr_E)[i+1], ((1/A_man[i+1])/np.array(p_man)[0])*np.array(nx_man_E)[i+1], ((1/A_trd[i+1])/np.array(p_trd)[0])*np.array(nx_trd_E)[i+1], ((1/A_bss[i+1])/np.array(p_bss)[0])*np.array(nx_bss_E)[i+1], ((1/A_fin[i+1])/np.array(p_fin)[0])*np.array(nx_fin_E)[i+1], ((1/A_nps[i+1])/np.array(p_nps)[0])*np.array(nx_nps_E)[0]))
	share_agr_nps_closed.append(econ.share_agr_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1]))
	share_man_nps_closed.append(econ.share_man_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1]))
	share_trd_nps_closed.append(econ.share_trd_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1]))
	share_bss_nps_closed.append(econ.share_bss_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1]))
	share_fin_nps_closed.append(econ.share_fin_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1]))
	share_nps_nps_closed.append(econ.share_nps_closed(C_closed[i+1], A_agr[i+1], A_man[i+1], A_trd[i+1], A_bss[i+1], A_fin[i+1], A_nps[i+1]))

'Aggregate Productivity'
weighted_agr_nps_A_agr = [a*b for a,b in zip(share_agr_nps, A_agr)] 
weighted_man_nps_A_man = [a*b for a,b in zip(share_man_nps, A_man)] 
weighted_trd_nps_A_trd = [a*b for a,b in zip(share_trd_nps, A_trd)] 
weighted_bss_nps_A_bss = [a*b for a,b in zip(share_bss_nps, A_bss)] 
weighted_fin_nps_A_bss = [a*b for a,b in zip(share_fin_nps, A_fin)] 
weighted_nps_nps_A_nps = [a*b for a,b in zip(share_nps_nps, A_nps)] 

A_tot_nps = [sum(x) for x in zip(weighted_agr_nps_A_agr, weighted_man_nps_A_man, weighted_trd_nps_A_trd, weighted_bss_nps_A_bss, weighted_fin_nps_A_bss, weighted_nps_nps_A_nps)]
A_tot_nps_weighted = share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps

weighted_agr_nps_A_agr_closed = [a*b for a,b in zip(share_agr_nps_closed, A_agr)] 
weighted_man_nps_A_man_closed = [a*b for a,b in zip(share_man_nps_closed, A_man)] 
weighted_trd_nps_A_trd_closed = [a*b for a,b in zip(share_trd_nps_closed, A_trd)] 
weighted_bss_nps_A_bss_closed = [a*b for a,b in zip(share_bss_nps_closed, A_bss)] 
weighted_fin_nps_A_bss_closed = [a*b for a,b in zip(share_fin_nps_closed, A_fin)] 
weighted_nps_nps_A_nps_closed = [a*b for a,b in zip(share_nps_nps_closed, A_nps)] 

A_tot_nps_closed = [sum(x) for x in zip(weighted_agr_nps_A_agr_closed, weighted_man_nps_A_man_closed, weighted_trd_nps_A_trd_closed, weighted_bss_nps_A_bss_closed, weighted_fin_nps_A_bss_closed, weighted_nps_nps_A_nps_closed)]

'''
------------
	Plots
------------
'''

'Non-Homothetic C'
'ams'
plt.figure(1)
plt.plot(year, C_agr, '--v', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{agr},t}/l_{\texttt{man},t}$')
plt.plot(year, C_ser, '--^', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{ser},t}/l_{\texttt{man},t}$')
plt.plot(year, GDP_per_h, '-s', lw = 2, alpha = 0.75, label = 'Data: GDP ph')
plt.plot(year, C_ams_simple_av, '-D', lw = 2,  alpha = 0.75, label = r'$C_t$: Simple Average')
plt.plot(year, C_ams_simple_av_alt, '-x', lw = 2,  alpha = 0.75, label = r'$C_t$: Simple Average (alt)')
plt.plot(year, C_ams, 'k-o', lw = 2,  alpha = 0.75, label = r'$C_t$: Weighted Average')
plt.plot(year, C_man, '-o',  alpha = 0.75, label = r'$C_t$: Manufacturing')

plt.legend(ncol=2, fontsize=12)
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('', fontsize=12)
#plt.safefig('../output/figures/fig_USA_C_ams_appendix.pdf', bbox_inches='tight')
#plt.show()
plt.close()

'nps'
fig  = plt.figure(2)
plt.plot(year, C_agr, '--v', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{agr},t}/l_{\texttt{man},t}$')
plt.plot(year, C_trd, '--^', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{trd},t}/l_{\texttt{man},t}$')
plt.plot(year, C_bss, '--<', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{bss},t}/l_{\texttt{man},t}$')
plt.plot(year, C_fin, '-->', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{fin},t}/l_{\texttt{man},t}$')
plt.plot(year, C_nps, '--8', lw = 1, markersize = 4, alpha = 0.35, label = r'Eq: $l_{\texttt{nps},t}/l_{\texttt{man},t}$')
plt.plot(year, GDP_per_h, '-s', lw = 2, alpha = 0.75, label = 'Data: GDP ph')
plt.plot(year, C_nps_simple_av, '-D', lw = 2,  alpha = 0.75, label = r'$C_t$: Simple Average')
plt.plot(year, C_nps, 'k-o', lw = 2,  alpha = 0.75, label = r'$C_t$: Weighted Average')
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('', fontsize=12)
#plt.safefig('../output/figures/fig_USA_C_nps_appendix.pdf', bbox_inches='tight')
#plt.show()
plt.close()

'Ai'
plt.figure(3)
plt.plot(year, A_agr, '-o', lw = 1, alpha = 0.5, label = r'$A_{\texttt{agr},t}$')
plt.plot(year, A_man, '->', lw = 1, alpha = 0.5, label = r'$A_{\texttt{man},t}$')
plt.plot(year, A_bss, '-s', lw = 1, alpha = 0.5, label = r'$A_{\texttt{bss},t}$')
plt.plot(year, A_trd, '-v', lw = 1, alpha = 0.5, label = r'$A_{\texttt{trd},t}$')
plt.plot(year, A_fin, '-8', lw = 1, alpha = 0.5, label = r'$A_{\texttt{fin},t}$')
plt.plot(year, A_nps, '-p', lw = 1, alpha = 0.5, label = r'$A_{\texttt{per},t}$')
plt.plot(year, A_ser, '-s', lw = 2, alpha = 0.75, label = r'$A_{\texttt{ser},t}$')
plt.plot(year, A_nps, '-D', lw = 2, alpha = 0.75, label = r'$A_{\texttt{nps},t}$')
plt.plot(year, A_tot, 'k-o', lw = 2,  alpha = 0.75, label = r'$A_{\texttt{tot},t}$')
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('', fontsize=12)
#plt.safefig('../output/figures/fig_USA_Ai_appendix.pdf', bbox_inches='tight')
#plt.show()
plt.close()


'''
-----------------------------------------------------
	Predictions for the Structural Transformation
-----------------------------------------------------
'''

'agr'
plt.figure(4)
plt.plot(year, share_agr, 'b-', lw=2, alpha=0.95, label = r'Data: \texttt{agr}')
plt.plot(year, share_agr_ams, 'D-', markersize=5, color = 'saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha = 0.95, label = r'Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}')
#plt.plot(year, share_agr_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.95, label = r'Model (2): \texttt{nps}, progressive services')
plt.ylabel('Employment Share', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize = 14)
plt.tight_layout()
#plt.safefig('../output/figures/fig_USA_calib_agr.pdf', bbox_inches='tight')
#plt.show()
plt.close()

'man'
plt.figure(5)
plt.plot(year, share_man, 'b-', lw=2, alpha=0.95, label=r'Data: \texttt{man}')
plt.plot(year, share_man_ams, 'D-', markersize=5, color = 'saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha = 0.95)
#plt.plot(year, share_man_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.95)
plt.ylabel('Employment Share', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
#plt.safefig('../output/figures/fig_USA_calib_man.pdf', bbox_inches='tight')
#plt.show()
plt.close()

'ser and nps'
plt.figure(6)
plt.plot(year, share_ser, 'b-', lw=2, alpha=0.95, label = r'Data: \texttt{ser}')
plt.plot(year, share_ser_ams, 'D-', markersize=5, color = 'saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha = 0.95)
#plt.plot(year, share_nps, 'c--', lw=2, markersize=5, alpha = 0.95, label = r'Data: \texttt{nps}')
#plt.plot(year, share_nps_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.95)
plt.legend(fontsize = 16)
plt.ylabel('Employment Share', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
#plt.safefig('../output/figures/fig_USA_calib_ser_nps.pdf', bbox_inches='tight')
#plt.show()
plt.close()

plt.figure(1)
plt.plot(year, share_agr, 'g-', lw=2, alpha=0.95, label = r'Data: \texttt{agr}')
plt.plot(year, share_agr_ams, 'g--', lw=2, alpha = 0.95, label = r'Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}')
plt.plot(year, share_man, 'b-', lw=2, alpha=0.95, label=r'Data: \texttt{man}')
plt.plot(year, share_man_ams, 'b--', lw=2, alpha = 0.95)
plt.plot(year, share_ser, 'r-', lw=2, alpha=0.95, label = r'Data: \texttt{ser}')
plt.plot(year, share_ser_ams, 'r--', lw=2, alpha = 0.95)
#plt.show()
plt.close()

'trd, bss and fin'
plt.figure(7)
plt.plot(year, share_trd, 'b-', lw=2, alpha=0.95, label = r'Data: \texttt{trd}')
plt.plot(year, share_trd_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.95)
plt.plot(year, share_bss, 'c--', lw=2, alpha = 0.95, label = r'Data: \texttt{bss}')
plt.plot(year, share_bss_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.95)
plt.plot(year, share_fin, 'y-.', lw=2, alpha = 0.95, label = r'Data: \texttt{fin}')
plt.plot(year, share_fin_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.95)
plt.legend(fontsize = 16)
plt.ylabel('Employment Share', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
#plt.safefig('../output/figures/fig_USA_calib_trd_bss_fin.pdf', bbox_inches='tight')
#plt.show()
plt.close()

fig = plt.figure(8)
fig.set_figheight(6)
fig.set_figwidth(6)

plt.plot(year, share_agr_ams, 'D-', markersize=6, color = 'saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha = 0.5, label = r'Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}')
plt.plot(year, share_agr_nps, 'H-', markersize=6, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.5, label = r'Model (2): \texttt{nps}, progressive services')
plt.plot(year, share_man_ams, 'D-', markersize=6, color = 'saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha = 0.5)
plt.plot(year, share_man_nps, 'H-', markersize=6, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.5)
plt.plot(year, share_bss_nps, 'H-', markersize=6, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.5)
plt.plot(year, share_trd_nps, 'H-', markersize=6, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.5)
plt.plot(year, share_fin_nps, 'H-', markersize=6, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.5)
plt.plot(year, share_nps_nps, 'H-', markersize=6, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.5)
plt.plot(year, share_agr, '-', color='blue', lw=2, alpha=0.95, label = r'Data: \texttt{agr}')
plt.plot(year, share_man, '-', color='red', lw=2, alpha=0.95, label=r'Data: \texttt{man}')
plt.plot(year, share_trd, '-', color='gold', lw=2, alpha=0.95, label = r'Data: \texttt{trd}')
plt.plot(year, share_bss, '-', color='purple', lw=2, alpha = 0.95, label = r'Data: \texttt{bss}')
plt.plot(year, share_fin, '-', color='grey', lw=2, alpha = 0.95, label = r'Data: \texttt{fin}')
plt.plot(year, share_nps, '-', color='darkcyan', lw=2, markersize=5, alpha = 0.95, label = r'Data: \texttt{nps}')
plt.ylabel('Employment Share', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=14)
plt.xticks(fontsize = 12)
plt.tight_layout()
#plt.safefig('../output/figures/fig_USA_calib_all.pdf', bbox_inches='tight')
#plt.show()
plt.close()

'''
---------------------------------------------------
	Prediction for Aggregate Labor Productivity
---------------------------------------------------
'''

plt.figure(8)
plt.plot(year, A_tot, 'r--', lw=2, alpha  = 0.95, label = r'$A_t$: Data')
plt.plot(year, GDP_per_h, 'b-', lw=2, alpha = 0.95, label = r'$\frac{Y_t}{L_t}$: Data')
#plt.plot(year, A_tot_ams_weighted, 'r--', lw=2, alpha = 0.95, label = r'$\frac{Y_t}{L_t}$. Weighted Average: \texttt{agr}, \texttt{man} and \texttt{ser}')
#plt.plot(year, A_tot_nps_weighted, 'm-.', lw=2, alpha = 0.95, label = r'$\frac{Y_t}{L_t}$ Weighted Average: \texttt{nps} and progressive services')
plt.plot(year, A_tot_ams, 'D-', markersize=5, color = 'saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha = 0.65, label = r'Model (1): \texttt{agr}, \texttt{man} and \texttt{ser}')
plt.plot(year, A_tot_nps, 'H-', markersize=5, color = 'darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha = 0.65, label = r'Model (2): \texttt{nps} and progressive services')
plt.yticks(fontsize = 12)
plt.xlabel('Year', fontsize=12)
plt.legend(fontsize=12)
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=14)
plt.tight_layout()
#plt.safefig('../output/figures/fig_USA_A_data_model.pdf', bbox_inches='tight')
#plt.show()
plt.close()


'''
---------------------------
	Plots for the paper
---------------------------
'''

fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(5)
fig.set_figwidth(10)

'Agriculture, Manufacturing and Services'
ax = plt.subplot(1,2,1)

ax.plot(year, share_agr, 'D-', markersize=4, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Data')
ax.plot(year, share_agr_nps, 'D--', markersize=4, color='darkgreen', markeredgecolor='darkgreen', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{agr}$: Model')
ax.plot(year, share_man, 'o-', markersize=4, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Data')
ax.plot(year, share_man_nps, 'o--', markersize=4, color='darkblue', markeredgecolor='darkblue', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{man}$: Model')
ax.plot(year, share_trd, 's-', markersize=4, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Data')
ax.plot(year, share_trd_nps, 's--', markersize=4, color='darkred', markeredgecolor='darkred', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{trd}$: Model')
ax.plot(year, share_bss, '^-', markersize=4, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Data')
ax.plot(year, share_bss_nps, '^--', markersize=4, color='darkmagenta', markeredgecolor='darkmagenta', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{bss}$: Model')
ax.plot(year, share_fin, 'p-', markersize=4, color='darkcyan', markerfacecolor='cyan', markeredgecolor='darkcyan', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Data')
ax.plot(year, share_fin_nps, 'p--', markersize=4, color='darkcyan', markeredgecolor='darkcyan', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{fin}$: Model')
ax.plot(year, share_nps, 'v-', markersize=4, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Data')
ax.plot(year, share_nps_nps, 'v--', markersize=4, color='saddlebrown', markeredgecolor='saddlebrown', mfc='none', markevery=7, alpha=0.95, label=r'$\texttt{nps}$: Model')
plt.ylabel('Employment Share', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.axis([1993,2021, -0.025, 0.525])
plt.grid()

plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

'Aggregate Productivity'
ax = plt.subplot(1,2,2)
ax.plot(year, GDP_per_h, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha = 0.95, label = 'Data: OECD')
ax.plot(year, A_tot, 'D-', markersize=6, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha = 0.95, label = 'Data: KLEMS')
ax.plot(year, A_tot_nps, 's-', markersize=6, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label = 'Model')
plt.ylabel('Labor Productivity', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], fontsize = 12)
plt.xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020], fontsize = 12)
plt.legend(fontsize=12)
plt.grid()

#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=12)

plt.tight_layout()
#plt.safefig('../output/figures/fig_calibration_USA.pdf', bbox_inches="tight")
#plt.show()
plt.close()

#print('Annualized Growth Rate in the USA (OECD) (1970-2019): ' +str(((GDP_per_h[-1]/GDP_per_h[0])**(1/49)-1)*100))
#print('Annualized Growth Rate in the USA (KLEMS) (1970-2019): ' +str(((A_tot[-1]/A_tot[0])**(1/49)-1)*100))
#print('Annualized Growth Rate in the USA (KLEMS. Weighted Average) (1970-2019): ' +str(((np.array(A_tot_nps_weighted)[-1]/np.array(A_tot_nps_weighted)[0])**(1/49)-1)*100))
#print('Annualized Growth Rate in the USA (Model) (1970-2019): ' +str(((A_tot_nps[-1]/A_tot_nps[0])**(1/49)-1)*100))



