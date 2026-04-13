"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        model_test_europe.py
Purpose:     Apply the US-calibrated closed-economy model to European economies.
             Defines the model_country class that, given a country code, recovers
             sectoral productivity series from data and generates model-implied
             sectoral employment shares. Builds EU4, EU15, core, and periphery
             aggregates used throughout the paper.
Pipeline:    Step 2/19 — Closed-economy model test on Europe.
Inputs:      ../data/euklems_2023.csv (EUKLEMS 2023 VA_Q and H by country-sector)
             ../data/raw/OECD_GDP_ph.xlsx (OECD GDP per hour, PPP USD)
             Preference parameters (sigma, eps_*) and US aggregates (GDP, E,
             A_tot, share_*, A_tot_ams, A_tot_nps) from model_calibration_USA.py.
Outputs:     The model_country class and EU aggregates EUR4_*, EUR15_*, EURCORE_*,
             EURPERI_* (hours h_tot, productivities A_tot/A_tot_ams/A_tot_nps,
             employment shares share_*, share_*_ams_m, share_*_nps_m). These feed
             all downstream counterfactual and figure scripts.
Dependencies: model_calibration_USA.py (Step 1).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
import statsmodels.api as sm
from scipy.optimize import fsolve
import pandas as pd
rc('text', usetex=True)
rc('font', family='serif')

# Import calibrated preference parameters and US aggregates from Step 1.
# US GDP and productivity aggregates are needed to measure European sectoral
# productivity in comparable units.
from model_calibration_USA import sigma, eps_agr, eps_trd, eps_fin, eps_bss, eps_nps, eps_ser, \
    GDP, E, A_tot, \
    share_agr, share_man, share_trd, share_bss, share_fin, share_nps, share_ser, \
    share_agr_ams, share_man_ams, share_ser_ams, A_tot_ams, \
    share_agr_nps, share_man_nps, share_trd_nps, share_bss_nps, share_fin_nps, A_tot_nps

'''
----------------------------
       Model Economy
----------------------------
'''

class model_country:
    """Country-level non-homothetic CES model with linear-in-labor production.

    Preference parameters (sigma, eps_*) are held fixed at the US-calibrated
    values (Step 1) so Europe's fit is a genuine out-of-sample test. The only
    country-specific objects are (i) the CES weights Omega_i, backed out from
    each country's initial-period employment shares via fsolve on the CES
    share identity, and (ii) the sectoral productivity paths A_i,t, which
    inherit the observed growth rates of sectoral labor productivity. The
    initial-period level gap GDP^c_0 / GDP^US_0 is imposed on both C and on
    every A_i to place the country in comparable units to the US."""
    def __init__(self, country_code, sigma=sigma, eps_agr=eps_agr, eps_trd=eps_trd, eps_fin=eps_fin, eps_bss=eps_bss, eps_nps=eps_nps, eps_ser=eps_ser):

        'Initialize the Parameters'
        self.cou, self.sigma, self.eps_agr, self.eps_trd, self.eps_fin, self.eps_bss, self.eps_nps, self.eps_ser = country_code, sigma, eps_agr, eps_trd, eps_fin, eps_bss, eps_nps, eps_ser

        '''
        -----------
            Data
        -----------
        '''       

        'KLEMS'
        data = pd.read_csv('../data/euklems_2023.csv', index_col=[0,1])
        data.rename(index={'AT':'AUT'},inplace=True)
        data.rename(index={'BE':'BEL'},inplace=True)
        data.rename(index={'DE':'DEU'},inplace=True)
        data.rename(index={'DK':'DNK'},inplace=True)
        data.rename(index={'ES':'ESP'},inplace=True)
        data.rename(index={'FI':'FIN'},inplace=True)
        data.rename(index={'FR':'FRA'},inplace=True)
        data.rename(index={'GB':'GBR'},inplace=True)
        data.rename(index={'GR':'GRC'},inplace=True)
        data.rename(index={'IE':'IRL'},inplace=True)
        data.rename(index={'IT':'ITA'},inplace=True)
        data.rename(index={'LU':'LUX'},inplace=True)
        data.rename(index={'NL':'NLD'},inplace=True)
        data.rename(index={'PT':'PRT'},inplace=True)
        data.rename(index={'SE':'SWE'},inplace=True)
        data.rename(index={'US':'USA'},inplace=True)
        
        'OECD'
        if self.cou == "EU15":
            GDP_ph = pd.read_excel('../data/raw/OECD_GDP_ph_EU15.xlsx', index_col=[0, 5],
                                   engine='openpyxl')  # Measured in USD (constant prices 2010 and PPPs).
            GDP_ph = GDP_ph[GDP_ph['MEASURE'] == 'USD']
            GDP_ph.index.rename(['country', 'year'])
        else:
            GDP_ph = pd.read_excel('../data/raw/OECD_GDP_ph.xlsx', index_col=[0,5], engine = 'openpyxl') #Measured in USD (constant prices 2010 and PPPs).
            GDP_ph = GDP_ph[GDP_ph['MEASURE'] == 'USD']
            GDP_ph.index.rename(['country', 'year'])
        
        'Country data'
        data = data.loc[self.cou]
        GDP_ph = GDP_ph.loc[self.cou]

        'Labor Productivity'
        data['y_l'] = (data['VA_Q']/data['H'])*100

        'Sectoral Data'
        data_agr = data[data['sector']=='agr']
        data_man = data[data['sector']=='man']
        data_trd = data[data['sector']=='trd']
        data_bss = data[data['sector']=='bss']
        data_fin = data[data['sector']=='fin']
        data_nps = data[data['sector']=='nps']
        data_ser = data[data['sector']=='ser']
        data_tot = data[data['sector']=='tot']

        'GDP'
        c_GDP, self.GDP = sm.tsa.filters.hpfilter(GDP_ph['Value'],100)

        'GDP Growth'
        self.g_GDP = np.array(self.GDP/self.GDP.shift(1) - 1).flatten() #GDP Growth from OECD
        
        'Employment hours'
        h_agr_c, self.h_agr = sm.tsa.filters.hpfilter(data_agr['H'],100)
        h_man_c, self.h_man = sm.tsa.filters.hpfilter(data_man['H'],100)
        h_trd_c, self.h_trd = sm.tsa.filters.hpfilter(data_trd['H'],100)
        h_bss_c, self.h_bss = sm.tsa.filters.hpfilter(data_bss['H'],100)
        h_fin_c, self.h_fin = sm.tsa.filters.hpfilter(data_fin['H'],100)
        h_nps_c, self.h_nps = sm.tsa.filters.hpfilter(data_nps['H'],100)
        h_ser_c, self.h_ser = sm.tsa.filters.hpfilter(data_ser['H'],100)
        h_tot_c, self.h_tot = sm.tsa.filters.hpfilter(data_tot['H'],100)

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
        self.g_y_l_agr = np.array(y_l_agr/y_l_agr.shift(1) - 1)
        self.g_y_l_man = np.array(y_l_man/y_l_man.shift(1) - 1)
        self.g_y_l_trd = np.array(y_l_trd/y_l_trd.shift(1) - 1)
        self.g_y_l_bss = np.array(y_l_bss/y_l_bss.shift(1) - 1)
        self.g_y_l_fin = np.array(y_l_fin/y_l_fin.shift(1) - 1)
        self.g_y_l_nps = np.array(y_l_nps/y_l_nps.shift(1) - 1)
        self.g_y_l_ser = np.array(y_l_ser/y_l_ser.shift(1) - 1)
        self.g_y_l_tot = np.array(y_l_tot/y_l_tot.shift(1) - 1)
        
        'Prices'
        p_agr_c, p_agr = sm.tsa.filters.hpfilter(data_agr['VA'] / data_agr['VA_Q'], 100)
        p_man_c, p_man = sm.tsa.filters.hpfilter(data_man['VA'] / data_man['VA_Q'], 100)
        p_trd_c, p_trd = sm.tsa.filters.hpfilter(data_trd['VA'] / data_trd['VA_Q'], 100)
        p_bss_c, p_bss = sm.tsa.filters.hpfilter(data_bss['VA'] / data_bss['VA_Q'], 100)
        p_fin_c, p_fin = sm.tsa.filters.hpfilter(data_fin['VA'] / data_fin['VA_Q'], 100)
        p_nps_c, p_nps = sm.tsa.filters.hpfilter(data_nps['VA'] / data_nps['VA_Q'], 100)
        p_ser_c, p_ser = sm.tsa.filters.hpfilter(data_ser['VA'] / data_ser['VA_Q'], 100)
        p_tot_c, p_tot = sm.tsa.filters.hpfilter(data_tot['VA'] / data_tot['VA_Q'], 100)
        
        'Employment Shares'
        share_c_agr, self.share_agr = sm.tsa.filters.hpfilter((self.h_agr/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        share_c_man, self.share_man = sm.tsa.filters.hpfilter((self.h_man/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        share_c_trd, self.share_trd = sm.tsa.filters.hpfilter((self.h_trd/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        share_c_bss, self.share_bss = sm.tsa.filters.hpfilter((self.h_bss/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        share_c_fin, self.share_fin = sm.tsa.filters.hpfilter((self.h_fin/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        share_c_nps, self.share_nps = sm.tsa.filters.hpfilter((self.h_nps/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        share_c_ser, self.share_ser = sm.tsa.filters.hpfilter((self.h_ser/(self.h_agr+self.h_man+self.h_trd+self.h_bss+self.h_fin+self.h_nps)),100)
        
        'Employment Shares Without Manufacturing (Weights of C)'
        share_c_agr_no_man, self.share_agr_no_man =  sm.tsa.filters.hpfilter(self.h_agr/(self.h_agr+self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_trd_no_man, self.share_trd_no_man =  sm.tsa.filters.hpfilter(self.h_trd/(self.h_agr+self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_bss_no_man, self.share_bss_no_man =  sm.tsa.filters.hpfilter(self.h_bss/(self.h_agr+self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_fin_no_man, self.share_fin_no_man =  sm.tsa.filters.hpfilter(self.h_fin/(self.h_agr+self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_nps_no_man, self.share_nps_no_man =  sm.tsa.filters.hpfilter(self.h_nps/(self.h_agr+self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_ser_no_man, self.share_ser_no_man =  sm.tsa.filters.hpfilter(self.h_ser/(self.h_agr+self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)

        share_c_trd_no_agm, self.share_trd_no_agm =  sm.tsa.filters.hpfilter(self.h_trd/(self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_bss_no_agm, self.share_bss_no_agm =  sm.tsa.filters.hpfilter(self.h_bss/(self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_fin_no_agm, self.share_fin_no_agm =  sm.tsa.filters.hpfilter(self.h_fin/(self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_nps_no_agm, self.share_nps_no_agm =  sm.tsa.filters.hpfilter(self.h_nps/(self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)
        share_c_ser_no_agm, self.share_ser_no_agm =  sm.tsa.filters.hpfilter(self.h_ser/(self.h_trd+self.h_bss+self.h_fin+self.h_nps), 100)

        'Relative labor demand'
        self.l_agr_l_man = self.h_agr/self.h_man
        self.l_trd_l_man = self.h_trd/self.h_man
        self.l_bss_l_man = self.h_bss/self.h_man
        self.l_fin_l_man = self.h_fin/self.h_man
        self.l_nps_l_man = self.h_nps/self.h_man
        self.l_ser_l_man = self.h_ser/self.h_man
        
        'Relative prices'
        self.p_agr_p_man = p_agr/p_man
        self.p_trd_p_man = p_trd/p_man
        self.p_bss_p_man = p_bss/p_man
        self.p_fin_p_man = p_fin/p_man
        self.p_nps_p_man = p_nps/p_man
        self.p_ser_p_man = p_ser/p_man
        
        '''
        ---------------------------------------------
                Time Series (Inputs of the Model)
        ---------------------------------------------
        '''

        # Initial-period normalization: set E_0 (and later A_i,0) equal to the
        # country's GDP-per-hour LEVEL relative to the US initial period,
        # GDP^c_0 / GDP^US_0. This anchors European productivity and income
        # in comparable PPP units so the model captures both the level gap
        # and the subsequent growth dynamics.
        t_0 = np.array(data.index)[0]
        self.E=[np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.year = [t_0]
        self.ts_length = np.array(data.index)[-1] - np.array(data.index)[0]
        for i in range(int(self.ts_length)):
            self.E.append((1 + self.g_GDP[i+1])*self.E[i])
            self.year.append(t_0 + i + 1)
                    
        'Back out country-specific CES weights Omega_i by requiring the'
        'model-implied employment shares in the INITIAL period to equal the'
        'observed initial-period data shares, given the GAP_init level of'
        'the CES consumption index. Two moments (agr and man shares) pin down'
        'the two free weights (Omega_ser = 1 - Omega_agr - Omega_man).'
        def ces_weights_ams(p):
            omega_agr, omega_man = p
            GAP_init = np.array(self.GDP)[0]/np.array(GDP)[0]
            l_agr = omega_agr*(GAP_init**eps_agr)*(GAP_init**(sigma - 1))
            l_man = omega_man*GAP_init*(GAP_init**(sigma - 1))
            l_ser = (1-omega_agr-omega_man)*(GAP_init**eps_ser)*(GAP_init**(sigma - 1))
            omegas = (l_agr/(l_agr+l_man+l_ser) - np.array(self.share_agr)[0],
                l_man/(l_agr+l_man+l_ser) - np.array(self.share_man)[0])
            return np.reshape(omegas, (2,))
        self.om_agr_ams, self.om_man_ams = fsolve(ces_weights_ams, (0.5,0.5))

        'Six-sector analogue: five initial-period share moments (agr, man,'
        'trd, bss, fin) pin down five CES weights; Omega_nps is the residual.'
        def ces_weights_nps(p):
            omega_agr, omega_man, omega_trd, omega_bss, omega_fin = p
            GAP_init = np.array(self.GDP)[0]/np.array(GDP)[0]
            l_agr = omega_agr*(GAP_init**eps_agr)*(GAP_init**(sigma - 1))
            l_man = omega_man*GAP_init*(GAP_init**(sigma - 1))
            l_trd = omega_trd*(GAP_init**eps_trd)*(GAP_init**(sigma - 1))
            l_bss = omega_bss*(GAP_init**eps_bss)*(GAP_init**(sigma - 1))
            l_fin = omega_fin*(GAP_init**eps_fin)*(GAP_init**(sigma - 1))
            l_nps = (1-omega_agr-omega_man-omega_trd-omega_bss-omega_fin)*(GAP_init**eps_nps)*(GAP_init**(sigma - 1))
            omegas = (l_agr/(l_agr+l_man+l_trd+l_bss+l_fin+l_nps) - np.array(self.share_agr)[0],
                l_man/(l_agr+l_man+l_trd+l_bss+l_fin+l_nps) - np.array(self.share_man)[0],
                l_trd/(l_agr+l_man+l_trd+l_bss+l_fin+l_nps) - np.array(self.share_trd)[0],
                l_bss/(l_agr+l_man+l_trd+l_bss+l_fin+l_nps) - np.array(self.share_bss)[0],
                l_fin/(l_agr+l_man+l_trd+l_bss+l_fin+l_nps) - np.array(self.share_fin)[0])
            return np.reshape(omegas, (5,))
        self.om_agr_nps, self.om_man_nps, self.om_trd_nps, self.om_bss_nps, self.om_fin_nps = fsolve(ces_weights_nps, (0.5, 0.5, 0.05,0.05,0.05))

    '''
    -------------------------
            The Models
    -------------------------
    '''

    'ams'
    "Agriculture, Manufacturing and Services"
    def labor_demand_ams(self, C, A_agr, A_man, A_ser):
        L = (self.om_agr_ams*(C**eps_agr)*(A_agr**(sigma - 1)) + 
            self.om_man_ams*C*(A_man**(sigma - 1)) + 
            (1-self.om_agr_ams-self.om_man_ams)*(C**eps_ser)*(A_ser**(sigma - 1)))
        return L

    def share_agr_ams(self, C, A_agr, A_man, A_ser):
        'Employment Share in Agriculture'
        return (self.om_agr_ams*(C**eps_agr)*(A_agr**(sigma - 1)))/(self.labor_demand_ams(C, A_agr, A_man, A_ser))

    def share_man_ams(self, C, A_agr, A_man, A_ser):
        'Employment Share in Manufacturing'
        return (self.om_man_ams*C*(A_man**(sigma - 1)))/(self.labor_demand_ams(C, A_agr, A_man, A_ser))

    def share_ser_ams(self, C, A_agr, A_man, A_ser):
        'Employment Share in Services'
        return ((1-self.om_agr_ams-self.om_man_ams)*(C**eps_ser)*(A_ser**(sigma - 1)))/(self.labor_demand_ams(C, A_agr, A_man, A_ser))

    'nps'
    "Agriculture, Manufacturing, Whole Sale and Retail Trade, Business Services, Financial Services and Non-Progressive Services"
    def labor_demand_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Total Labor Demand'
        L = (self.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) +
             self.om_man_nps*C*(A_man**(sigma - 1)) +
             self.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) +
             self.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) +
             self.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) +
             (1-self.om_agr_nps-self.om_man_nps-self.om_trd_nps-self.om_bss_nps-self.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)))
        return L

    def share_agr_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Employment Share in Agriculture'
        return (self.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)))/(self.labor_demand_nps(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps))
        
    def share_man_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Employment Share in Manufacturing'
        return (self.om_man_nps*C*(A_man**(sigma - 1)))/(self.labor_demand_nps(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps))
        

    def share_trd_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Employment Share in Whole Sale and Retail Trade'
        return (self.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)))/(self.labor_demand_nps(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps))
        

    def share_bss_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Employment Share in Business Services'
        return (self.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)))/(self.labor_demand_nps(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps))
        

    def share_fin_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Employment Share in Financial Services'
        return (self.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)))/(self.labor_demand_nps(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps))
        

    def share_nps_nps(self, C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps):
        'Employment Share in the Rest of Services'
        return ((1-self.om_agr_nps-self.om_man_nps-self.om_trd_nps-self.om_bss_nps-self.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)))/(self.labor_demand_nps(C, A_agr, A_man, A_trd, A_bss, A_fin, A_nps))
        
    'Country-level non-homothetic CES index recovered from the relative-'
    'labor equation, then initialized at the US-relative GDP level so C and'
    'the sectoral A_i paths live in the same units.'
    def C_index(self, om_i, om_m, li_lm, pi_pm, sigma, epsilon_i):
        C_level = ((om_m/om_i)*li_lm*(pi_pm**(sigma-1)))**(1/(epsilon_i-1))
        g_C = np.array(C_level/C_level.shift(1) - 1)
        C = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        for i in range(len(g_C) - 1):
            C.append((1+g_C[i+1])*C[i])
        return C

    '''
    ------------------------
        Model Predictions
    ------------------------
    '''

    'Productivity Time Indexes'  
    def productivity_series(self):
        'First period. Normalization' 
        self.A_agr = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_man = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_trd = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_bss = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_fin = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_nps = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_ser = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        self.A_tot = [np.array(self.GDP)[0]/np.array(GDP)[0]]
              
        'Productivity and Real Expenditure Growth'
        for i in range(int(self.ts_length)):
            self.A_agr.append((1 + self.g_y_l_agr[i+1])*self.A_agr[i])
            self.A_man.append((1 + self.g_y_l_man[i+1])*self.A_man[i])
            self.A_trd.append((1 + self.g_y_l_trd[i+1])*self.A_trd[i])
            self.A_bss.append((1 + self.g_y_l_bss[i+1])*self.A_bss[i])
            self.A_fin.append((1 + self.g_y_l_fin[i+1])*self.A_fin[i])
            self.A_nps.append((1 + self.g_y_l_nps[i+1])*self.A_nps[i])
            self.A_ser.append((1 + self.g_y_l_ser[i+1])*self.A_ser[i])
            self.A_tot.append((1 + self.g_y_l_tot[i+1])*self.A_tot[i])

        'Non-Homothetic C: ams'
        self.C_ams_agr = self.C_index(self.om_agr_ams, self.om_man_ams, self.l_agr_l_man, self.p_agr_p_man, sigma, eps_agr)
        self.C_ams_ser = self.C_index((1-self.om_agr_ams-self.om_man_ams), self.om_man_ams, self.l_ser_l_man, self.p_ser_p_man, sigma, eps_ser)
#        self.C_ams_simple_av = (np.array(C_ams_agr) + np.array(C_ams_ser))/2
#        self.C_ams = np.array(self.share_agr_no_man*self.C_ams_agr + self.share_ser_no_man*self.C_ams_ser)
        self.C_ams = self.C_ams_ser
#        self.C_ams = self.E

        'C from aggregate labor: ams'
        def C_exp_ams(C):
            return L_t**(1-sigma) - (self.om_agr_ams*(C**eps_agr)*(A_agr_t**(sigma-1)) + self.om_man_ams*C*(A_man_t**(sigma-1)) + (1-self.om_agr_ams-self.om_man_ams)*(C**eps_ser)*(A_ser_t**(sigma-1)))
        C_lev_E_ams=[]
        for i in range(len(E)):
            L_t=np.array(self.h_tot)[i]
            A_agr_t=np.array(self.A_agr)[i]
            A_man_t=np.array(self.A_man)[i]
            A_ser_t=np.array(self.A_ser)[i]
            C_lev_E_ams.append(fsolve(C_exp_ams, L_t).item())
        C_level_E_ams = pd.DataFrame(C_lev_E_ams)
        g_C_E_ams = np.array(C_level_E_ams/C_level_E_ams.shift(1) - 1).flatten()
        self.C_E_ams = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        for i in range(len(g_C_E_ams) - 1):
            self.C_E_ams.append((1+g_C_E_ams[i+1])*self.C_E_ams[i])

        'Non-Homothetic C: nps'
        self.C_nps_agr = self.C_index(self.om_agr_nps, self.om_man_nps, self.l_agr_l_man, self.p_agr_p_man, sigma, eps_agr)
        self.C_nps_trd = self.C_index(self.om_trd_nps, self.om_man_nps, self.l_trd_l_man, self.p_trd_p_man, sigma, eps_trd)
        self.C_nps_bss = self.C_index(self.om_bss_nps, self.om_man_nps, self.l_bss_l_man, self.p_bss_p_man, sigma, eps_bss)
        self.C_nps_fin = self.C_index(self.om_fin_nps, self.om_man_nps, self.l_fin_l_man, self.p_fin_p_man, sigma, eps_fin)
        self.C_nps_nps = self.C_index((1-self.om_agr_nps-self.om_man_nps-self.om_trd_nps-self.om_bss_nps-self.om_fin_nps), self.om_man_nps, self.l_nps_l_man, self.p_nps_p_man, sigma, eps_nps)
#        self.C_nps_simple_av = (np.array(C_nps_agr) + np.array(C_nps_trd) + np.array(C_nps_bss) + np.array(C_nps_fin) + np.array(C_nps_nps))/5
#        self.C_nps = np.array(self.share_agr_no_man*self.C_nps_agr + self.share_trd_no_man*self.C_nps_trd + self.share_bss_no_man*self.C_nps_bss + self.share_fin_no_man*self.C_nps_fin + self.share_nps_no_man*self.C_nps_nps)
#        self.C_nps = np.array(self.share_trd_no_agm*self.C_nps_trd + self.share_bss_no_agm*self.C_nps_bss + self.share_fin_no_agm*self.C_nps_fin + self.share_nps_no_agm*self.C_nps_nps)
#        self.C_nps = self.E
        self.C_nps = self.C_ams_ser

        'C from aggregate labor: nps'
        def C_exp_nps(C):
            return L_t**(1-sigma) - (self.om_agr_nps*(C**eps_agr)*(A_agr_t**(sigma-1)) + self.om_man_nps*C*(A_man_t**(sigma-1)) + self.om_trd_nps*(C**eps_trd)*(A_trd_t**(sigma-1)) + self.om_bss_nps*(C**eps_bss)*(A_bss_t**(sigma-1)) + self.om_fin_nps*(C**eps_fin)*(A_fin_t**(sigma-1)) + (1-self.om_agr_nps-self.om_man_nps-self.om_trd_nps-self.om_bss_nps-self.om_fin_nps)*(C**eps_nps)*(A_nps_t**(sigma-1)))
        C_lev_E_nps=[]
        for i in range(len(E)):
            L_t=np.array(self.h_tot)[i]
            A_trd_t=np.array(self.A_trd)[i]
            A_bss_t=np.array(self.A_bss)[i]
            A_fin_t=np.array(self.A_fin)[i]
            A_nps_t=np.array(self.A_nps)[i]
            C_lev_E_nps.append(fsolve(C_exp_nps, L_t).item())
        C_level_E_nps = pd.DataFrame(C_lev_E_nps)
        g_C_E_nps = np.array(C_level_E_nps/C_level_E_nps.shift(1) - 1).flatten()
        self.C_E_nps = [np.array(self.GDP)[0]/np.array(GDP)[0]]
        for i in range(len(g_C_E_nps) - 1):
            self.C_E_nps.append((1+g_C_E_nps[i+1])*self.C_E_nps[i])

    'ams'
    def predictions_ams(self):
        'Model Implied Employment Shares'
        self.share_agr_ams_m = [self.share_agr_ams(self.C_ams[0], self.A_agr[0], self.A_man[0], self.A_ser[0])]
        self.share_man_ams_m = [self.share_man_ams(self.C_ams[0], self.A_agr[0], self.A_man[0], self.A_ser[0])]
        self.share_ser_ams_m = [self.share_ser_ams(self.C_ams[0], self.A_agr[0], self.A_man[0], self.A_ser[0])]
        for i in range(int(self.ts_length)):
            self.share_agr_ams_m.append(self.share_agr_ams(self.C_ams[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_ser[i+1]))
            self.share_man_ams_m.append(self.share_man_ams(self.C_ams[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_ser[i+1]))
            self.share_ser_ams_m.append(self.share_ser_ams(self.C_ams[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_ser[i+1]))

        'Aggregate Productivity'
        weighted_ams_A_agr = [a*b for a,b in zip(self.share_agr_ams_m, self.A_agr)]
        weighted_ams_A_man = [a*b for a,b in zip(self.share_man_ams_m, self.A_man)]
        weighted_ams_A_ser = [a*b for a,b in zip(self.share_ser_ams_m, self.A_ser)]
        self.A_tot_ams = [sum(x) for x in zip(weighted_ams_A_agr, weighted_ams_A_man, weighted_ams_A_ser)]

    'nps'
    def predictions_nps(self):
        'Model Implied Emloyment Shares'
        self.share_agr_nps_m = [self.share_agr_nps(self.C_nps[0], self.A_agr[0], self.A_man[0], self.A_trd[0], self.A_bss[0], self.A_fin[0], self.A_nps[0])]
        self.share_man_nps_m = [self.share_man_nps(self.C_nps[0], self.A_agr[0], self.A_man[0], self.A_trd[0], self.A_bss[0], self.A_fin[0], self.A_nps[0])]
        self.share_trd_nps_m = [self.share_trd_nps(self.C_nps[0], self.A_agr[0], self.A_man[0], self.A_trd[0], self.A_bss[0], self.A_fin[0], self.A_nps[0])]
        self.share_bss_nps_m = [self.share_bss_nps(self.C_nps[0], self.A_agr[0], self.A_man[0], self.A_trd[0], self.A_bss[0], self.A_fin[0], self.A_nps[0])]
        self.share_fin_nps_m = [self.share_fin_nps(self.C_nps[0], self.A_agr[0], self.A_man[0], self.A_trd[0], self.A_bss[0], self.A_fin[0], self.A_nps[0])]
        self.share_nps_nps_m = [self.share_nps_nps(self.C_nps[0], self.A_agr[0], self.A_man[0], self.A_trd[0], self.A_bss[0], self.A_fin[0], self.A_nps[0])]
        for i in range(int(self.ts_length)):
            self.share_agr_nps_m.append(self.share_agr_nps(self.C_nps[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_trd[i+1], self.A_bss[i+1], self.A_fin[i+1], self.A_nps[i+1]))
            self.share_man_nps_m.append(self.share_man_nps(self.C_nps[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_trd[i+1], self.A_bss[i+1], self.A_fin[i+1], self.A_nps[i+1]))
            self.share_trd_nps_m.append(self.share_trd_nps(self.C_nps[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_trd[i+1], self.A_bss[i+1], self.A_fin[i+1], self.A_nps[i+1]))
            self.share_bss_nps_m.append(self.share_bss_nps(self.C_nps[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_trd[i+1], self.A_bss[i+1], self.A_fin[i+1], self.A_nps[i+1]))
            self.share_fin_nps_m.append(self.share_fin_nps(self.C_nps[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_trd[i+1], self.A_bss[i+1], self.A_fin[i+1], self.A_nps[i+1]))
            self.share_nps_nps_m.append(self.share_nps_nps(self.C_nps[i+1], self.A_agr[i+1], self.A_man[i+1], self.A_trd[i+1], self.A_bss[i+1], self.A_fin[i+1], self.A_nps[i+1]))

        'Aggregate Productivity'
        weighted_nps_A_agr = [a*b for a,b in zip(self.share_agr_nps_m, self.A_agr)]
        weighted_nps_A_man = [a*b for a,b in zip(self.share_man_nps_m, self.A_man)]
        weighted_nps_A_trd = [a*b for a,b in zip(self.share_trd_nps_m, self.A_trd)]
        weighted_nps_A_bss = [a*b for a,b in zip(self.share_bss_nps_m, self.A_bss)]
        weighted_nps_A_fin = [a*b for a,b in zip(self.share_fin_nps_m, self.A_fin)]
        weighted_nps_A_nps = [a*b for a,b in zip(self.share_nps_nps_m, self.A_nps)]
        self.A_tot_nps = [sum(x) for x in zip(weighted_nps_A_agr, weighted_nps_A_man, weighted_nps_A_trd, weighted_nps_A_bss, weighted_nps_A_fin, weighted_nps_A_nps)]


'''
-----------------------------------------
    Predictions for European Countries
-----------------------------------------
'''

# EUR15 = model_country('EU15')
# EUR15.productivity_series()
# EUR15.predictions_ams()
# EUR15.predictions_nps()

AUT = model_country('AUT')
AUT.productivity_series()
AUT.predictions_ams()
AUT.predictions_nps()

BEL = model_country('BEL')
BEL.productivity_series()
BEL.predictions_ams()
BEL.predictions_nps()

DEU = model_country('DEU')
DEU.productivity_series()
DEU.predictions_ams()
DEU.predictions_nps()

DNK = model_country('DNK')
DNK.productivity_series()
DNK.predictions_ams()
DNK.predictions_nps()

ESP = model_country('ESP')
ESP.productivity_series()
ESP.predictions_ams()
ESP.predictions_nps()

FIN = model_country('FIN')
FIN.productivity_series()
FIN.predictions_ams()
FIN.predictions_nps()

FRA = model_country('FRA')
FRA.productivity_series()
FRA.predictions_ams()
FRA.predictions_nps()

GBR = model_country('GBR')
GBR.productivity_series()
GBR.predictions_ams()
GBR.predictions_nps()

GRC = model_country('GRC')
GRC.productivity_series()
GRC.predictions_ams()
GRC.predictions_nps()

IRL = model_country('IRL')
IRL.productivity_series()
IRL.predictions_ams()
IRL.predictions_nps()

ITA = model_country('ITA')
ITA.productivity_series()
ITA.predictions_ams()
ITA.predictions_nps()

LUX = model_country('LUX')
LUX.productivity_series()
LUX.predictions_ams()
LUX.predictions_nps()

NLD = model_country('NLD')
NLD.productivity_series()
NLD.predictions_ams()
NLD.predictions_nps()

PRT = model_country('PRT')
PRT.productivity_series()
PRT.predictions_ams()
PRT.predictions_nps()

SWE = model_country('SWE')
SWE.productivity_series()
SWE.predictions_ams()
SWE.predictions_nps()

'Export EUKLEMS labor productivity growth rates data'
db_export = pd.DataFrame({'year': np.arange(1970, 2020)})
db_export['country'] = 'AUT'
sectors = ['agr', 'bss', 'fin', 'man', 'nps', 'ser', 'tot', 'trd']

for sec in sectors:
    db_export['g_y_l_' + sec] = getattr(AUT, 'g_y_l_' + sec)

countries = ['BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE']

for country in countries:
    db_export_temp = pd.DataFrame({'year': np.arange(1970, 2020)})
    db_export_temp['country'] = country
    for sec in sectors:
        db_export_temp['g_y_l_' + sec] = getattr(locals().get(country), 'g_y_l_' + sec)
    db_export = pd.concat((db_export, db_export_temp), axis=0)

db_export.to_excel('../output/data/lp_KLEMS_data.xlsx', index=False)

# Bayoumi and Eichengreen (1993) core and periphery classification
# Core:Germany, France, Belgium, Netherlands, and Denmark
# PERI: Greece, Ireland, Portugal, Spain, Italy, and the UK

'Productivity and GDP in Europe'
EUR4_h_tot = np.array(DEU.h_tot) + np.array(FRA.h_tot) + np.array(GBR.h_tot) + np.array(ITA.h_tot)
EURCORE_h_tot = np.array(DEU.h_tot) + np.array(FRA.h_tot) + np.array(BEL.h_tot) + np.array(NLD.h_tot) + np.array(DNK.h_tot)
EURPERI_h_tot = np.array(GRC.h_tot) + np.array(IRL.h_tot) + np.array(PRT.h_tot) + np.array(ESP.h_tot) + np.array(ITA.h_tot) + np.array(GBR.h_tot)
EUR15_h_tot = np.array(AUT.h_tot) + np.array(BEL.h_tot) + np.array(DEU.h_tot) + np.array(DNK.h_tot) + np.array(ESP.h_tot) + np.array(FIN.h_tot) + np.array(FRA.h_tot) + np.array(GBR.h_tot) + np.array(GRC.h_tot) + np.array(IRL.h_tot) + np.array(ITA.h_tot) + np.array(LUX.h_tot) + np.array(NLD.h_tot) + np.array(PRT.h_tot) + np.array(SWE.h_tot)

EUR4_A_tot = (np.array(DEU.h_tot)*np.array(DEU.A_tot) + np.array(FRA.h_tot)*np.array(FRA.A_tot) + np.array(GBR.h_tot)*np.array(GBR.A_tot) + np.array(ITA.h_tot)*np.array(ITA.A_tot))/EUR4_h_tot
EURCORE_A_tot = (np.array(DEU.h_tot)*np.array(DEU.A_tot) + np.array(FRA.h_tot)*np.array(FRA.A_tot) + np.array(BEL.h_tot)*np.array(BEL.A_tot) + np.array(NLD.h_tot)*np.array(NLD.A_tot) + np.array(DNK.h_tot)*np.array(DNK.A_tot))/EURCORE_h_tot
EURPERI_A_tot = (np.array(GRC.h_tot)*np.array(GRC.A_tot) + np.array(IRL.h_tot)*np.array(IRL.A_tot) + np.array(PRT.h_tot)*np.array(PRT.A_tot) + np.array(ESP.h_tot)*np.array(ESP.A_tot) + np.array(ITA.h_tot)*np.array(ITA.A_tot) + np.array(GBR.h_tot)*np.array(GBR.A_tot))/EURPERI_h_tot
EUR15_A_tot = (np.array(AUT.h_tot)*np.array(AUT.A_tot) + np.array(BEL.h_tot)*np.array(BEL.A_tot) + np.array(DEU.h_tot)*np.array(DEU.A_tot) + np.array(DNK.h_tot)*np.array(DNK.A_tot) + np.array(ESP.h_tot)*np.array(ESP.A_tot) + np.array(FIN.h_tot)*np.array(FIN.A_tot) + np.array(FRA.h_tot)*np.array(FRA.A_tot) + np.array(GBR.h_tot)*np.array(GBR.A_tot) + np.array(GRC.h_tot)*np.array(GRC.A_tot) + np.array(IRL.h_tot)*np.array(IRL.A_tot) + np.array(ITA.h_tot)*np.array(ITA.A_tot) + np.array(LUX.h_tot)*np.array(LUX.A_tot) + np.array(NLD.h_tot)*np.array(NLD.A_tot) + np.array(PRT.h_tot)*np.array(PRT.A_tot) + np.array(SWE.h_tot)*np.array(SWE.A_tot))/EUR15_h_tot

EUR4_rel_A_tot = EUR4_A_tot/A_tot
EURCORE_rel_A_tot = EURCORE_A_tot/A_tot
EURPERI_rel_A_tot = EURPERI_A_tot/A_tot
EUR15_rel_A_tot = EUR15_A_tot/A_tot

EUR4_E = (np.array(DEU.E)*np.array(DEU.h_tot) + np.array(FRA.E)*np.array(FRA.h_tot) + np.array(GBR.E)*np.array(GBR.h_tot) + np.array(ITA.E)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_E = (np.array(DEU.E)*np.array(DEU.h_tot) + np.array(FRA.E)*np.array(FRA.h_tot) + np.array(BEL.E)*np.array(BEL.h_tot) + np.array(NLD.E)*np.array(NLD.h_tot) + np.array(DNK.E)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_E = (np.array(GRC.E)*np.array(GRC.h_tot) + np.array(IRL.E)*np.array(IRL.h_tot) + np.array(PRT.E)*np.array(PRT.h_tot) + np.array(ESP.E)*np.array(ESP.h_tot) + np.array(ITA.E)*np.array(ITA.h_tot) + np.array(GBR.E)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_E = (np.array(AUT.E)*np.array(AUT.h_tot) + np.array(BEL.E)*np.array(BEL.h_tot) + np.array(DEU.E)*np.array(DEU.h_tot) + np.array(DNK.E)*np.array(DNK.h_tot) + np.array(ESP.E)*np.array(ESP.h_tot) + np.array(FIN.E)*np.array(FIN.h_tot) + np.array(FRA.E)*np.array(FRA.h_tot) + np.array(GBR.E)*np.array(GBR.h_tot) + np.array(GRC.E)*np.array(GRC.h_tot) + np.array(IRL.E)*np.array(IRL.h_tot) + np.array(ITA.E)*np.array(ITA.h_tot) + np.array(LUX.E)*np.array(LUX.h_tot) + np.array(NLD.E)*np.array(NLD.h_tot) + np.array(PRT.E)*np.array(PRT.h_tot) + np.array(SWE.E)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_rel_E = ((np.array(DEU.E)/np.array(E))*np.array(DEU.h_tot) + (np.array(FRA.E)/np.array(E))*np.array(FRA.h_tot) + (np.array(GBR.E)/np.array(E))*np.array(GBR.h_tot) + (np.array(ITA.E)/np.array(E))*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_rel_E = ((np.array(DEU.E)/np.array(E))*np.array(DEU.h_tot) + (np.array(FRA.E)/np.array(E))*np.array(FRA.h_tot) + (np.array(BEL.E)/np.array(E))*np.array(BEL.h_tot) + (np.array(NLD.E)/np.array(E))*np.array(NLD.h_tot) + (np.array(DNK.E)/np.array(E))*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_rel_E = ((np.array(GRC.E)/np.array(E))*np.array(GRC.h_tot) + (np.array(IRL.E)/np.array(E))*np.array(IRL.h_tot) + (np.array(PRT.E)/np.array(E))*np.array(PRT.h_tot) + (np.array(ESP.E)/np.array(E))*np.array(ESP.h_tot) + (np.array(ITA.E)/np.array(E))*np.array(ITA.h_tot) + (np.array(GBR.E)/np.array(E))*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_rel_E = ((np.array(AUT.E)/np.array(E))*np.array(AUT.h_tot) + (np.array(BEL.E)/np.array(E))*np.array(BEL.h_tot) + (np.array(DEU.E)/np.array(E))*np.array(DEU.h_tot) + (np.array(DNK.E)/np.array(E))*np.array(DNK.h_tot) + (np.array(ESP.E)/np.array(E))*np.array(ESP.h_tot) + (np.array(FIN.E)/np.array(E))*np.array(FIN.h_tot) + (np.array(FRA.E)/np.array(E))*np.array(FRA.h_tot) + (np.array(GBR.E)/np.array(E))*np.array(GBR.h_tot) + (np.array(GRC.E)/np.array(E))*np.array(GRC.h_tot) + (np.array(IRL.E)/np.array(E))*np.array(IRL.h_tot) + (np.array(ITA.E)/np.array(E))*np.array(ITA.h_tot) + (np.array(LUX.E)/np.array(E))*np.array(LUX.h_tot) + (np.array(NLD.E)/np.array(E))*np.array(NLD.h_tot) + (np.array(PRT.E)/np.array(E))*np.array(PRT.h_tot) + (np.array(SWE.E)/np.array(E))*np.array(SWE.h_tot))/EUR15_h_tot

'Predictions for Aggregate Productivity'
'ams'
AUT_A_tot_ams_h_tot = np.array(AUT.A_tot_ams).flatten()*np.array(AUT.h_tot).flatten()
BEL_A_tot_ams_h_tot = np.array(BEL.A_tot_ams).flatten()*np.array(BEL.h_tot).flatten()
DEU_A_tot_ams_h_tot = np.array(DEU.A_tot_ams).flatten()*np.array(DEU.h_tot).flatten()
DNK_A_tot_ams_h_tot = np.array(DNK.A_tot_ams).flatten()*np.array(DNK.h_tot).flatten()
ESP_A_tot_ams_h_tot = np.array(ESP.A_tot_ams).flatten()*np.array(ESP.h_tot).flatten()
FIN_A_tot_ams_h_tot = np.array(FIN.A_tot_ams).flatten()*np.array(FIN.h_tot).flatten()
FRA_A_tot_ams_h_tot = np.array(FRA.A_tot_ams).flatten()*np.array(FRA.h_tot).flatten()
GBR_A_tot_ams_h_tot = np.array(GBR.A_tot_ams).flatten()*np.array(GBR.h_tot).flatten()
GRC_A_tot_ams_h_tot = np.array(GRC.A_tot_ams).flatten()*np.array(GRC.h_tot).flatten()
IRL_A_tot_ams_h_tot = np.array(IRL.A_tot_ams).flatten()*np.array(IRL.h_tot).flatten()
ITA_A_tot_ams_h_tot = np.array(ITA.A_tot_ams).flatten()*np.array(ITA.h_tot).flatten()
LUX_A_tot_ams_h_tot = np.array(LUX.A_tot_ams).flatten()*np.array(LUX.h_tot).flatten()
NLD_A_tot_ams_h_tot = np.array(NLD.A_tot_ams).flatten()*np.array(NLD.h_tot).flatten()
PRT_A_tot_ams_h_tot = np.array(PRT.A_tot_ams).flatten()*np.array(PRT.h_tot).flatten()
SWE_A_tot_ams_h_tot = np.array(SWE.A_tot_ams).flatten()*np.array(SWE.h_tot).flatten()

EUR4_A_tot_ams = (DEU_A_tot_ams_h_tot + FRA_A_tot_ams_h_tot + GBR_A_tot_ams_h_tot + ITA_A_tot_ams_h_tot)/EUR4_h_tot
EURCORE_A_tot_ams = (DEU_A_tot_ams_h_tot + FRA_A_tot_ams_h_tot + BEL_A_tot_ams_h_tot + NLD_A_tot_ams_h_tot + DNK_A_tot_ams_h_tot)/EURCORE_h_tot
EURPERI_A_tot_ams = (GRC_A_tot_ams_h_tot + IRL_A_tot_ams_h_tot + PRT_A_tot_ams_h_tot + ESP_A_tot_ams_h_tot + ITA_A_tot_ams_h_tot + GBR_A_tot_ams_h_tot)/EURPERI_h_tot
EUR15_A_tot_ams = (AUT_A_tot_ams_h_tot + BEL_A_tot_ams_h_tot + DEU_A_tot_ams_h_tot + DNK_A_tot_ams_h_tot + ESP_A_tot_ams_h_tot + FIN_A_tot_ams_h_tot + FRA_A_tot_ams_h_tot + GBR_A_tot_ams_h_tot + GRC_A_tot_ams_h_tot + IRL_A_tot_ams_h_tot + ITA_A_tot_ams_h_tot + LUX_A_tot_ams_h_tot + NLD_A_tot_ams_h_tot + PRT_A_tot_ams_h_tot + SWE_A_tot_ams_h_tot)/EUR15_h_tot

'nps'
AUT_A_tot_nps_h_tot = np.array(AUT.A_tot_nps).flatten()*np.array(AUT.h_tot).flatten()
BEL_A_tot_nps_h_tot = np.array(BEL.A_tot_nps).flatten()*np.array(BEL.h_tot).flatten()
DEU_A_tot_nps_h_tot = np.array(DEU.A_tot_nps).flatten()*np.array(DEU.h_tot).flatten()
DNK_A_tot_nps_h_tot = np.array(DNK.A_tot_nps).flatten()*np.array(DNK.h_tot).flatten()
ESP_A_tot_nps_h_tot = np.array(ESP.A_tot_nps).flatten()*np.array(ESP.h_tot).flatten()
FIN_A_tot_nps_h_tot = np.array(FIN.A_tot_nps).flatten()*np.array(FIN.h_tot).flatten()
FRA_A_tot_nps_h_tot = np.array(FRA.A_tot_nps).flatten()*np.array(FRA.h_tot).flatten()
GBR_A_tot_nps_h_tot = np.array(GBR.A_tot_nps).flatten()*np.array(GBR.h_tot).flatten()
GRC_A_tot_nps_h_tot = np.array(GRC.A_tot_nps).flatten()*np.array(GRC.h_tot).flatten()
IRL_A_tot_nps_h_tot = np.array(IRL.A_tot_nps).flatten()*np.array(IRL.h_tot).flatten()
ITA_A_tot_nps_h_tot = np.array(ITA.A_tot_nps).flatten()*np.array(ITA.h_tot).flatten()
LUX_A_tot_nps_h_tot = np.array(LUX.A_tot_nps).flatten()*np.array(LUX.h_tot).flatten()
NLD_A_tot_nps_h_tot = np.array(NLD.A_tot_nps).flatten()*np.array(NLD.h_tot).flatten()
PRT_A_tot_nps_h_tot = np.array(PRT.A_tot_nps).flatten()*np.array(PRT.h_tot).flatten()
SWE_A_tot_nps_h_tot = np.array(SWE.A_tot_nps).flatten()*np.array(SWE.h_tot).flatten()

EUR4_A_tot_nps = (DEU_A_tot_nps_h_tot + FRA_A_tot_nps_h_tot + GBR_A_tot_nps_h_tot + ITA_A_tot_nps_h_tot)/EUR4_h_tot
EURCORE_A_tot_nps = (DEU_A_tot_nps_h_tot + FRA_A_tot_nps_h_tot + BEL_A_tot_nps_h_tot + NLD_A_tot_nps_h_tot + DNK_A_tot_nps_h_tot)/EURCORE_h_tot
EURPERI_A_tot_nps = (GRC_A_tot_nps_h_tot + IRL_A_tot_nps_h_tot + PRT_A_tot_nps_h_tot + ESP_A_tot_nps_h_tot + ITA_A_tot_nps_h_tot + GBR_A_tot_nps_h_tot)/EURPERI_h_tot
EUR15_A_tot_nps = (AUT_A_tot_nps_h_tot + BEL_A_tot_nps_h_tot + DEU_A_tot_nps_h_tot + DNK_A_tot_nps_h_tot + ESP_A_tot_nps_h_tot + FIN_A_tot_nps_h_tot + FRA_A_tot_nps_h_tot + GBR_A_tot_nps_h_tot + GRC_A_tot_nps_h_tot + IRL_A_tot_nps_h_tot + ITA_A_tot_nps_h_tot + LUX_A_tot_nps_h_tot + NLD_A_tot_nps_h_tot + PRT_A_tot_nps_h_tot + SWE_A_tot_nps_h_tot)/EUR15_h_tot

'Euroshares (data)'
EUR4_share_agr = (np.array(DEU.share_agr)*np.array(DEU.h_tot) + np.array(FRA.share_agr)*np.array(FRA.h_tot) + np.array(GBR.share_agr)*np.array(GBR.h_tot) + np.array(ITA.share_agr)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_agr = (np.array(DEU.share_agr)*np.array(DEU.h_tot) + np.array(FRA.share_agr)*np.array(FRA.h_tot) + np.array(BEL.share_agr)*np.array(BEL.h_tot) + np.array(NLD.share_agr)*np.array(NLD.h_tot) + np.array(DNK.share_agr)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_agr = (np.array(GRC.share_agr)*np.array(GRC.h_tot) + np.array(IRL.share_agr)*np.array(IRL.h_tot) + np.array(PRT.share_agr)*np.array(PRT.h_tot) + np.array(ESP.share_agr)*np.array(ESP.h_tot) + np.array(ITA.share_agr)*np.array(ITA.h_tot) + np.array(GBR.share_agr)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_agr = (np.array(AUT.share_agr)*np.array(AUT.h_tot) + np.array(BEL.share_agr)*np.array(BEL.h_tot) + np.array(DEU.share_agr)*np.array(DEU.h_tot) + np.array(DNK.share_agr)*np.array(DNK.h_tot) + np.array(ESP.share_agr)*np.array(ESP.h_tot) + np.array(FIN.share_agr)*np.array(FIN.h_tot) + np.array(FRA.share_agr)*np.array(FRA.h_tot) + np.array(GBR.share_agr)*np.array(GBR.h_tot) + np.array(GRC.share_agr)*np.array(GRC.h_tot) + np.array(IRL.share_agr)*np.array(IRL.h_tot) + np.array(ITA.share_agr)*np.array(ITA.h_tot) + np.array(LUX.share_agr)*np.array(LUX.h_tot) + np.array(NLD.share_agr)*np.array(NLD.h_tot) + np.array(PRT.share_agr)*np.array(PRT.h_tot) + np.array(SWE.share_agr)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_man=(np.array(DEU.share_man)*np.array(DEU.h_tot) + np.array(FRA.share_man)*np.array(FRA.h_tot) + np.array(GBR.share_man)*np.array(GBR.h_tot) + np.array(ITA.share_man)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_man = (np.array(DEU.share_man)*np.array(DEU.h_tot) + np.array(FRA.share_man)*np.array(FRA.h_tot) + np.array(BEL.share_man)*np.array(BEL.h_tot) + np.array(NLD.share_man)*np.array(NLD.h_tot) + np.array(DNK.share_man)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_man = (np.array(GRC.share_man)*np.array(GRC.h_tot) + np.array(IRL.share_man)*np.array(IRL.h_tot) + np.array(PRT.share_man)*np.array(PRT.h_tot) + np.array(ESP.share_man)*np.array(ESP.h_tot) + np.array(ITA.share_man)*np.array(ITA.h_tot) + np.array(GBR.share_man)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_man=(np.array(AUT.share_man)*np.array(AUT.h_tot) + np.array(BEL.share_man)*np.array(BEL.h_tot) + np.array(DEU.share_man)*np.array(DEU.h_tot) + np.array(DNK.share_man)*np.array(DNK.h_tot) + np.array(ESP.share_man)*np.array(ESP.h_tot) + np.array(FIN.share_man)*np.array(FIN.h_tot) + np.array(FRA.share_man)*np.array(FRA.h_tot) + np.array(GBR.share_man)*np.array(GBR.h_tot) + np.array(GRC.share_man)*np.array(GRC.h_tot) + np.array(IRL.share_man)*np.array(IRL.h_tot) + np.array(ITA.share_man)*np.array(ITA.h_tot) + np.array(LUX.share_man)*np.array(LUX.h_tot) + np.array(NLD.share_man)*np.array(NLD.h_tot) + np.array(PRT.share_man)*np.array(PRT.h_tot) + np.array(SWE.share_man)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_ser=(np.array(DEU.share_ser)*np.array(DEU.h_tot) + np.array(FRA.share_ser)*np.array(FRA.h_tot) + np.array(GBR.share_ser)*np.array(GBR.h_tot)+ np.array(ITA.share_ser)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_ser = (np.array(DEU.share_ser)*np.array(DEU.h_tot) + np.array(FRA.share_ser)*np.array(FRA.h_tot) + np.array(BEL.share_ser)*np.array(BEL.h_tot) + np.array(NLD.share_ser)*np.array(NLD.h_tot) + np.array(DNK.share_ser)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_ser = (np.array(GRC.share_ser)*np.array(GRC.h_tot) + np.array(IRL.share_ser)*np.array(IRL.h_tot) + np.array(PRT.share_ser)*np.array(PRT.h_tot) + np.array(ESP.share_ser)*np.array(ESP.h_tot) + np.array(ITA.share_ser)*np.array(ITA.h_tot) + np.array(GBR.share_ser)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_ser=(np.array(AUT.share_ser)*np.array(AUT.h_tot) + np.array(BEL.share_ser)*np.array(BEL.h_tot) + np.array(DEU.share_ser)*np.array(DEU.h_tot) + np.array(DNK.share_ser)*np.array(DNK.h_tot) + np.array(ESP.share_ser)*np.array(ESP.h_tot) + np.array(FIN.share_ser)*np.array(FIN.h_tot) + np.array(FRA.share_ser)*np.array(FRA.h_tot) + np.array(GBR.share_ser)*np.array(GBR.h_tot) + np.array(GRC.share_ser)*np.array(GRC.h_tot) + np.array(IRL.share_ser)*np.array(IRL.h_tot) + np.array(ITA.share_ser)*np.array(ITA.h_tot) + np.array(LUX.share_ser)*np.array(LUX.h_tot) + np.array(NLD.share_ser)*np.array(NLD.h_tot) + np.array(PRT.share_ser)*np.array(PRT.h_tot) + np.array(SWE.share_ser)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_trd=(np.array(DEU.share_trd)*np.array(DEU.h_tot) + np.array(FRA.share_trd)*np.array(FRA.h_tot) + np.array(GBR.share_trd)*np.array(GBR.h_tot) + np.array(ITA.share_trd)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_trd = (np.array(DEU.share_trd)*np.array(DEU.h_tot) + np.array(FRA.share_trd)*np.array(FRA.h_tot) + np.array(BEL.share_trd)*np.array(BEL.h_tot) + np.array(NLD.share_trd)*np.array(NLD.h_tot) + np.array(DNK.share_trd)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_trd = (np.array(GRC.share_trd)*np.array(GRC.h_tot) + np.array(IRL.share_trd)*np.array(IRL.h_tot) + np.array(PRT.share_trd)*np.array(PRT.h_tot) + np.array(ESP.share_trd)*np.array(ESP.h_tot) + np.array(ITA.share_trd)*np.array(ITA.h_tot) + np.array(GBR.share_trd)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_trd=(np.array(AUT.share_trd)*np.array(AUT.h_tot) + np.array(BEL.share_trd)*np.array(BEL.h_tot) + np.array(DEU.share_trd)*np.array(DEU.h_tot) + np.array(DNK.share_trd)*np.array(DNK.h_tot) + np.array(ESP.share_trd)*np.array(ESP.h_tot) + np.array(FIN.share_trd)*np.array(FIN.h_tot) + np.array(FRA.share_trd)*np.array(FRA.h_tot) + np.array(GBR.share_trd)*np.array(GBR.h_tot) + np.array(GRC.share_trd)*np.array(GRC.h_tot) + np.array(IRL.share_trd)*np.array(IRL.h_tot) + np.array(ITA.share_trd)*np.array(ITA.h_tot) + np.array(LUX.share_trd)*np.array(LUX.h_tot) + np.array(NLD.share_trd)*np.array(NLD.h_tot) + np.array(PRT.share_trd)*np.array(PRT.h_tot) + np.array(SWE.share_trd)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_bss=(np.array(DEU.share_bss)*np.array(DEU.h_tot) + np.array(FRA.share_bss)*np.array(FRA.h_tot) + np.array(GBR.share_bss)*np.array(GBR.h_tot) + np.array(ITA.share_bss)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_bss = (np.array(DEU.share_bss)*np.array(DEU.h_tot) + np.array(FRA.share_bss)*np.array(FRA.h_tot) + np.array(BEL.share_bss)*np.array(BEL.h_tot) + np.array(NLD.share_bss)*np.array(NLD.h_tot) + np.array(DNK.share_bss)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_bss = (np.array(GRC.share_bss)*np.array(GRC.h_tot) + np.array(IRL.share_bss)*np.array(IRL.h_tot) + np.array(PRT.share_bss)*np.array(PRT.h_tot) + np.array(ESP.share_bss)*np.array(ESP.h_tot) + np.array(ITA.share_bss)*np.array(ITA.h_tot) + np.array(GBR.share_bss)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_bss=(np.array(AUT.share_bss)*np.array(AUT.h_tot) + np.array(BEL.share_bss)*np.array(BEL.h_tot) + np.array(DEU.share_bss)*np.array(DEU.h_tot) + np.array(DNK.share_bss)*np.array(DNK.h_tot) + np.array(ESP.share_bss)*np.array(ESP.h_tot) + np.array(FIN.share_bss)*np.array(FIN.h_tot) + np.array(FRA.share_bss)*np.array(FRA.h_tot) + np.array(GBR.share_bss)*np.array(GBR.h_tot) + np.array(GRC.share_bss)*np.array(GRC.h_tot) + np.array(IRL.share_bss)*np.array(IRL.h_tot) + np.array(ITA.share_bss)*np.array(ITA.h_tot) + np.array(LUX.share_bss)*np.array(LUX.h_tot) + np.array(NLD.share_bss)*np.array(NLD.h_tot) + np.array(PRT.share_bss)*np.array(PRT.h_tot) + np.array(SWE.share_bss)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_fin=(np.array(DEU.share_fin)*np.array(DEU.h_tot) + np.array(FRA.share_fin)*np.array(FRA.h_tot) + np.array(GBR.share_fin)*np.array(GBR.h_tot) + np.array(ITA.share_fin)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_fin = (np.array(DEU.share_fin)*np.array(DEU.h_tot) + np.array(FRA.share_fin)*np.array(FRA.h_tot) + np.array(BEL.share_fin)*np.array(BEL.h_tot) + np.array(NLD.share_fin)*np.array(NLD.h_tot) + np.array(DNK.share_fin)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_fin = (np.array(GRC.share_fin)*np.array(GRC.h_tot) + np.array(IRL.share_fin)*np.array(IRL.h_tot) + np.array(PRT.share_fin)*np.array(PRT.h_tot) + np.array(ESP.share_fin)*np.array(ESP.h_tot) + np.array(ITA.share_fin)*np.array(ITA.h_tot) + np.array(GBR.share_fin)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_fin=(np.array(AUT.share_fin)*np.array(AUT.h_tot) + np.array(BEL.share_fin)*np.array(BEL.h_tot) + np.array(DEU.share_fin)*np.array(DEU.h_tot) + np.array(DNK.share_fin)*np.array(DNK.h_tot) + np.array(ESP.share_fin)*np.array(ESP.h_tot) + np.array(FIN.share_fin)*np.array(FIN.h_tot) + np.array(FRA.share_fin)*np.array(FRA.h_tot) + np.array(GBR.share_fin)*np.array(GBR.h_tot) + np.array(GRC.share_fin)*np.array(GRC.h_tot) + np.array(IRL.share_fin)*np.array(IRL.h_tot) + np.array(ITA.share_fin)*np.array(ITA.h_tot) + np.array(LUX.share_fin)*np.array(LUX.h_tot) + np.array(NLD.share_fin)*np.array(NLD.h_tot) + np.array(PRT.share_fin)*np.array(PRT.h_tot) + np.array(SWE.share_fin)*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_nps=(np.array(DEU.share_nps)*np.array(DEU.h_tot) + np.array(FRA.share_nps)*np.array(FRA.h_tot) + np.array(GBR.share_nps)*np.array(GBR.h_tot) + np.array(ITA.share_nps)*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_nps = (np.array(DEU.share_nps)*np.array(DEU.h_tot) + np.array(FRA.share_nps)*np.array(FRA.h_tot) + np.array(BEL.share_nps)*np.array(BEL.h_tot) + np.array(NLD.share_nps)*np.array(NLD.h_tot) + np.array(DNK.share_nps)*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_nps = (np.array(GRC.share_nps)*np.array(GRC.h_tot) + np.array(IRL.share_nps)*np.array(IRL.h_tot) + np.array(PRT.share_nps)*np.array(PRT.h_tot) + np.array(ESP.share_nps)*np.array(ESP.h_tot) + np.array(ITA.share_nps)*np.array(ITA.h_tot) + np.array(GBR.share_nps)*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_nps=(np.array(AUT.share_nps)*np.array(AUT.h_tot) + np.array(BEL.share_nps)*np.array(BEL.h_tot) + np.array(DEU.share_nps)*np.array(DEU.h_tot) + np.array(DNK.share_nps)*np.array(DNK.h_tot) + np.array(ESP.share_nps)*np.array(ESP.h_tot) + np.array(FIN.share_nps)*np.array(FIN.h_tot) + np.array(FRA.share_nps)*np.array(FRA.h_tot) + np.array(GBR.share_nps)*np.array(GBR.h_tot) + np.array(GRC.share_nps)*np.array(GRC.h_tot) + np.array(IRL.share_nps)*np.array(IRL.h_tot) + np.array(ITA.share_nps)*np.array(ITA.h_tot) + np.array(LUX.share_nps)*np.array(LUX.h_tot) + np.array(NLD.share_nps)*np.array(NLD.h_tot) + np.array(PRT.share_nps)*np.array(PRT.h_tot) + np.array(SWE.share_nps)*np.array(SWE.h_tot))/EUR15_h_tot

'Euroshares (model)'
EUR4_share_agr_ams_m=(np.array(DEU.share_agr_ams_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_agr_ams_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_agr_ams_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_agr_ams_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_agr_ams_m=(np.array(DEU.share_agr_ams_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_agr_ams_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_agr_ams_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_agr_ams_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_agr_ams_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_agr_ams_m=(np.array(GRC.share_agr_ams_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_agr_ams_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_agr_ams_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_agr_ams_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_agr_ams_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_agr_ams_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_agr_ams_m=(np.array(AUT.share_agr_ams_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_agr_ams_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_agr_ams_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_agr_ams_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_agr_ams_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_agr_ams_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_agr_ams_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_agr_ams_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_agr_ams_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_agr_ams_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_agr_ams_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_agr_ams_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_agr_ams_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_agr_ams_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_agr_ams_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_agr_nps_m=(np.array(DEU.share_agr_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_agr_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_agr_nps_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_agr_nps_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_agr_nps_m=(np.array(DEU.share_agr_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_agr_nps_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_agr_nps_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_agr_nps_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_agr_nps_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_agr_nps_m=(np.array(GRC.share_agr_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_agr_nps_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_agr_nps_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_agr_nps_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_agr_nps_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_agr_nps_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_agr_nps_m=(np.array(AUT.share_agr_nps_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_agr_nps_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_agr_nps_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_agr_nps_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_agr_nps_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_agr_nps_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_agr_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_agr_nps_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_agr_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_agr_nps_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_agr_nps_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_agr_nps_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_agr_nps_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_agr_nps_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_agr_nps_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_man_ams_m=(np.array(DEU.share_man_ams_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_man_ams_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_man_ams_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_man_ams_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_man_ams_m=(np.array(DEU.share_man_ams_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_man_ams_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_man_ams_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_man_ams_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_man_ams_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_man_ams_m=(np.array(GRC.share_man_ams_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_man_ams_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_man_ams_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_man_ams_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_man_ams_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_man_ams_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_man_ams_m=(np.array(AUT.share_man_ams_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_man_ams_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_man_ams_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_man_ams_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_man_ams_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_man_ams_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_man_ams_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_man_ams_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_man_ams_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_man_ams_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_man_ams_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_man_ams_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_man_ams_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_man_ams_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_man_ams_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_man_nps_m=(np.array(DEU.share_man_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_man_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_man_nps_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_man_nps_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_man_nps_m=(np.array(DEU.share_man_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_man_nps_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_man_nps_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_man_nps_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_man_nps_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_man_nps_m=(np.array(GRC.share_man_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_man_nps_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_man_nps_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_man_nps_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_man_nps_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_man_nps_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_man_nps_m=(np.array(AUT.share_man_nps_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_man_nps_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_man_nps_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_man_nps_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_man_nps_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_man_nps_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_man_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_man_nps_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_man_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_man_nps_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_man_nps_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_man_nps_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_man_nps_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_man_nps_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_man_nps_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_ser_ams_m=(np.array(DEU.share_ser_ams_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_ser_ams_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_ser_ams_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_ser_ams_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_ser_ams_m=(np.array(DEU.share_ser_ams_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_ser_ams_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_ser_ams_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_ser_ams_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_ser_ams_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_ser_ams_m=(np.array(GRC.share_ser_ams_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_ser_ams_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_ser_ams_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_ser_ams_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_ser_ams_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_ser_ams_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_ser_ams_m=(np.array(AUT.share_ser_ams_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_ser_ams_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_ser_ams_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_ser_ams_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_ser_ams_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_ser_ams_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_ser_ams_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_ser_ams_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_ser_ams_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_ser_ams_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_ser_ams_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_ser_ams_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_ser_ams_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_ser_ams_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_ser_ams_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_trd_nps_m=(np.array(DEU.share_trd_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_trd_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_trd_nps_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_trd_nps_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_trd_nps_m=(np.array(DEU.share_trd_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_trd_nps_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_trd_nps_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_trd_nps_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_trd_nps_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_trd_nps_m=(np.array(GRC.share_trd_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_trd_nps_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_trd_nps_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_trd_nps_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_trd_nps_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_trd_nps_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_trd_nps_m=(np.array(AUT.share_trd_nps_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_trd_nps_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_trd_nps_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_trd_nps_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_trd_nps_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_trd_nps_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_trd_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_trd_nps_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_trd_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_trd_nps_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_trd_nps_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_trd_nps_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_trd_nps_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_trd_nps_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_trd_nps_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_bss_nps_m=(np.array(DEU.share_bss_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_bss_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_bss_nps_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_bss_nps_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_bss_nps_m=(np.array(DEU.share_bss_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_bss_nps_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_bss_nps_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_bss_nps_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_bss_nps_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_bss_nps_m=(np.array(GRC.share_bss_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_bss_nps_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_bss_nps_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_bss_nps_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_bss_nps_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_bss_nps_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_bss_nps_m=(np.array(AUT.share_bss_nps_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_bss_nps_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_bss_nps_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_bss_nps_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_bss_nps_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_bss_nps_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_bss_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_bss_nps_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_bss_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_bss_nps_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_bss_nps_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_bss_nps_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_bss_nps_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_bss_nps_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_bss_nps_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_fin_nps_m=(np.array(DEU.share_fin_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_fin_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_fin_nps_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_fin_nps_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_fin_nps_m=(np.array(DEU.share_fin_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_fin_nps_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_fin_nps_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_fin_nps_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_fin_nps_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_fin_nps_m=(np.array(GRC.share_fin_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_fin_nps_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_fin_nps_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_fin_nps_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_fin_nps_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_fin_nps_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_fin_nps_m=(np.array(AUT.share_fin_nps_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_fin_nps_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_fin_nps_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_fin_nps_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_fin_nps_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_fin_nps_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_fin_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_fin_nps_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_fin_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_fin_nps_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_fin_nps_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_fin_nps_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_fin_nps_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_fin_nps_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_fin_nps_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot

EUR4_share_nps_nps_m=(np.array(DEU.share_nps_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_nps_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_nps_nps_m).flatten()*np.array(GBR.h_tot) + np.array(ITA.share_nps_nps_m).flatten()*np.array(ITA.h_tot))/EUR4_h_tot
EURCORE_share_nps_nps_m=(np.array(DEU.share_nps_nps_m).flatten()*np.array(DEU.h_tot) + np.array(FRA.share_nps_nps_m).flatten()*np.array(FRA.h_tot) + np.array(BEL.share_nps_nps_m).flatten()*np.array(BEL.h_tot) + np.array(NLD.share_nps_nps_m).flatten()*np.array(NLD.h_tot) + np.array(DNK.share_nps_nps_m).flatten()*np.array(DNK.h_tot))/EURCORE_h_tot
EURPERI_share_nps_nps_m=(np.array(GRC.share_nps_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_nps_nps_m).flatten()*np.array(IRL.h_tot) + np.array(PRT.share_nps_nps_m).flatten()*np.array(PRT.h_tot) + np.array(ESP.share_nps_nps_m).flatten()*np.array(ESP.h_tot) + np.array(ITA.share_nps_nps_m).flatten()*np.array(ITA.h_tot) + np.array(GBR.share_nps_nps_m).flatten()*np.array(GBR.h_tot))/EURPERI_h_tot
EUR15_share_nps_nps_m=(np.array(AUT.share_nps_nps_m).flatten()*np.array(AUT.h_tot) + np.array(BEL.share_nps_nps_m).flatten()*np.array(BEL.h_tot) + np.array(DEU.share_nps_nps_m).flatten()*np.array(DEU.h_tot) + np.array(DNK.share_nps_nps_m).flatten()*np.array(DNK.h_tot) + np.array(ESP.share_nps_nps_m).flatten()*np.array(ESP.h_tot) + np.array(FIN.share_nps_nps_m).flatten()*np.array(FIN.h_tot) + np.array(FRA.share_nps_nps_m).flatten()*np.array(FRA.h_tot) + np.array(GBR.share_nps_nps_m).flatten()*np.array(GBR.h_tot) + np.array(GRC.share_nps_nps_m).flatten()*np.array(GRC.h_tot) + np.array(IRL.share_nps_nps_m).flatten()*np.array(IRL.h_tot) + np.array(ITA.share_nps_nps_m).flatten()*np.array(ITA.h_tot) + np.array(LUX.share_nps_nps_m).flatten()*np.array(LUX.h_tot) + np.array(NLD.share_nps_nps_m).flatten()*np.array(NLD.h_tot) + np.array(PRT.share_nps_nps_m).flatten()*np.array(PRT.h_tot) + np.array(SWE.share_nps_nps_m).flatten()*np.array(SWE.h_tot))/EUR15_h_tot


'''
---------------------------
    Tests of the Theory
---------------------------
'''
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(8)
fig.set_figwidth(8)

'Agriculture'
ax = plt.subplot(3,3,1)
ax.plot(np.array(AUT.share_agr)[-1], AUT.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(BEL.share_agr)[-1], BEL.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_agr)[-1], DEU.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_agr)[-1], DNK.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_agr)[-1], ESP.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_agr)[-1], FIN.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_agr)[-1], FRA.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_agr)[-1], GBR.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_agr)[-1], GRC.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_agr)[-1], IRL.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_agr)[-1], ITA.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_agr)[-1], LUX.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_agr)[-1], NLD.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_agr)[-1], PRT.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_agr)[-1], SWE.share_agr_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=10, label = 'Model (1)')
ax.plot(EUR15_share_agr_ams_m[-1], EUR15_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=10)

#ax.annotate('AUT', (np.array(AUT.share_agr)[-1], AUT.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('BEL', (np.array(BEL.share_agr)[-1], BEL.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('DEU', (np.array(DEU.share_agr)[-1], DEU.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('DNK', (np.array(DNK.share_agr)[-1], DNK.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('ESP', (np.array(ESP.share_agr)[-1], ESP.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('FIN', (np.array(FIN.share_agr)[-1], FIN.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('FRA', (np.array(FRA.share_agr)[-1], FRA.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('GBR', (np.array(GBR.share_agr)[-1], GBR.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('GRC', (np.array(GRC.share_agr)[-1], GRC.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('IRL', (np.array(IRL.share_agr)[-1], IRL.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('ITA', (np.array(ITA.share_agr)[-1], ITA.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('LUX', (np.array(LUX.share_agr)[-1], LUX.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('NLD', (np.array(NLD.share_agr)[-1], NLD.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('PRT', (np.array(PRT.share_agr)[-1], PRT.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('SWE', (np.array(SWE.share_agr)[-1], SWE.share_agr_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('EU4', (EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1]), fontsize=20)
#ax.annotate('EU15', (EUR15_share_agr_ams_m[-1], EUR15_share_agr[-1]), fontsize=20)


ax.plot(np.array(AUT.share_agr)[-1], AUT.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(BEL.share_agr)[-1], BEL.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_agr)[-1], DEU.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_agr)[-1], DNK.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_agr)[-1], ESP.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_agr)[-1], FIN.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_agr)[-1], FRA.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_agr)[-1], GBR.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_agr)[-1], GRC.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_agr)[-1], IRL.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_agr)[-1], ITA.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_agr)[-1], LUX.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_agr)[-1], NLD.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_agr)[-1], PRT.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_agr)[-1], SWE.share_agr_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10, label = 'Model (2)')
ax.plot(EUR15_share_agr_nps_m[-1], EUR15_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_agr)[-1], AUT.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_agr)[-1], BEL.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_agr)[-1], DEU.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_agr)[-1], DNK.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_agr)[-1], ESP.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_agr)[-1], FIN.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_agr)[-1], FRA.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_agr)[-1], GBR.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_agr)[-1], GRC.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_agr)[-1], IRL.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_agr)[-1], ITA.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_agr)[-1], LUX.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_agr)[-1], NLD.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_agr)[-1], PRT.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_agr)[-1], SWE.share_agr_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_agr_nps_m[-1], EUR15_share_agr[-1]), fontsize=12)

#ax.legend(fontsize=20)
plt.title('Agriculture', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.01, 0.04, 0.07, 0.10, 0.13],fontsize=10)
#plt.yticks([0.01, 0.04, 0.07, 0.10, 0.13],fontsize=10)

handles, labels = ax.get_legend_handles_labels()

'Manufacturing'
ax = plt.subplot(3,3,2)
ax.plot(np.array(AUT.share_man)[-1], AUT.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label='Model (1)')
ax.plot(np.array(BEL.share_man)[-1], BEL.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_man)[-1], DEU.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_man)[-1], DNK.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_man)[-1], ESP.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_man)[-1], FIN.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_man)[-1], FRA.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_man)[-1], GBR.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_man)[-1], GRC.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_man)[-1], IRL.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_man)[-1], ITA.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_man)[-1], LUX.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_man)[-1], NLD.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_man)[-1], PRT.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_man)[-1], SWE.share_man_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EUR4_share_man_ams_m[-1], EUR4_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=10)
ax.plot(EUR15_share_man_ams_m[-1], EUR15_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=10)

#ax.annotate('AUT', (np.array(AUT.share_man)[-1], AUT.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('BEL', (np.array(BEL.share_man)[-1], BEL.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('DEU', (np.array(DEU.share_man)[-1], DEU.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('DNK', (np.array(DNK.share_man)[-1], DNK.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('ESP', (np.array(ESP.share_man)[-1], ESP.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('FIN', (np.array(FIN.share_man)[-1], FIN.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('FRA', (np.array(FRA.share_man)[-1], FRA.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('GBR', (np.array(GBR.share_man)[-1], GBR.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('GRC', (np.array(GRC.share_man)[-1], GRC.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('IRL', (np.array(IRL.share_man)[-1], IRL.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('ITA', (np.array(ITA.share_man)[-1], ITA.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
##ax.annotate('LUX', (np.array(LUX.share_man)[-1], LUX.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('NLD', (np.array(NLD.share_man)[-1], NLD.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('PRT', (np.array(PRT.share_man)[-1], PRT.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('SWE', (np.array(SWE.share_man)[-1], SWE.share_man_ams_m[-1]),alpha=0.6,fontsize=10)
#ax.annotate('EU4', (EUR4_share_man_ams_m[-1], EUR4_share_man[-1]), fontsize=20)
#ax.annotate('EU15', (EUR15_share_man_ams_m[-1], EUR15_share_man[-1]), fontsize=20)

ax.plot(np.array(AUT.share_man)[-1], AUT.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label='Model (2)')
ax.plot(np.array(BEL.share_man)[-1], BEL.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_man)[-1], DEU.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_man)[-1], DNK.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_man)[-1], ESP.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_man)[-1], FIN.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_man)[-1], FRA.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_man)[-1], GBR.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_man)[-1], GRC.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_man)[-1], IRL.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_man)[-1], ITA.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_man)[-1], LUX.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_man)[-1], NLD.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_man)[-1], PRT.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_man)[-1], SWE.share_man_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(EUR4_share_man_nps_m[-1], EUR4_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)
ax.plot(EUR15_share_man_nps_m[-1], EUR15_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_man)[-1], AUT.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_man)[-1], BEL.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_man)[-1], DEU.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_man)[-1], DNK.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_man)[-1], ESP.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_man)[-1], FIN.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_man)[-1], FRA.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_man)[-1], GBR.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_man)[-1], GRC.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_man)[-1], IRL.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_man)[-1], ITA.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_man)[-1], LUX.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_man)[-1], NLD.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_man)[-1], PRT.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_man)[-1], SWE.share_man_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_man_nps_m[-1], EUR4_share_man[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_man_nps_m[-1], EUR15_share_man[-1]), fontsize=12)

plt.title('Manufacturing', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.17,0.20,0.23,0.26],fontsize=10)
#plt.yticks([0.17,0.20,0.23,0.26],fontsize=10)

'Services'
ax = plt.subplot(3,3,3)
ax.plot(np.array(AUT.share_ser)[-1], AUT.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label='Model (1)')
ax.plot(np.array(BEL.share_ser)[-1], BEL.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_ser)[-1], DEU.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_ser)[-1], DNK.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_ser)[-1], ESP.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_ser)[-1], FIN.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_ser)[-1], FRA.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_ser)[-1], GBR.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_ser)[-1], GRC.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_ser)[-1], IRL.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_ser)[-1], ITA.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_ser)[-1], LUX.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_ser)[-1], NLD.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_ser)[-1], PRT.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_ser)[-1], SWE.share_ser_ams_m[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=10)
ax.plot(EUR15_share_ser_ams_m[-1], EUR15_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_ser)[-1], AUT.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_ser)[-1], BEL.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_ser)[-1], DEU.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_ser)[-1], DNK.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_ser)[-1], ESP.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_ser)[-1], FIN.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_ser)[-1], FRA.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_ser)[-1], GBR.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_ser)[-1], GRC.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_ser)[-1], IRL.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_ser)[-1], ITA.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_ser)[-1], LUX.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_ser)[-1], NLD.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_ser)[-1], PRT.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_ser)[-1], SWE.share_ser_ams_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_ser_ams_m[-1], EUR15_share_ser[-1]), fontsize=12)

plt.title('Services', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.60,0.70,0.80],fontsize=10)
#plt.yticks([0.60,0.70,0.80],fontsize=10)

'Wholesale and Retail Trade'
ax = plt.subplot(3,3,4)
ax.plot(np.array(AUT.share_trd)[-1], AUT.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label='Model (2)')
ax.plot(np.array(BEL.share_trd)[-1], BEL.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_trd)[-1], DEU.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_trd)[-1], DNK.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_trd)[-1], ESP.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_trd)[-1], FIN.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_trd)[-1], FRA.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_trd)[-1], GBR.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_trd)[-1], GRC.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_trd)[-1], IRL.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_trd)[-1], ITA.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_trd)[-1], LUX.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_trd)[-1], NLD.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_trd)[-1], PRT.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_trd)[-1], SWE.share_trd_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)
ax.plot(EUR15_share_trd_nps_m[-1], EUR15_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_trd)[-1], AUT.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_trd)[-1], BEL.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_trd)[-1], DEU.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_trd)[-1], DNK.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_trd)[-1], ESP.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_trd)[-1], FIN.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_trd)[-1], FRA.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_trd)[-1], GBR.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_trd)[-1], GRC.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_trd)[-1], IRL.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_trd)[-1], ITA.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_trd)[-1], LUX.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_trd)[-1], NLD.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_trd)[-1], PRT.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_trd)[-1], SWE.share_trd_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_trd_nps_m[-1], EUR15_share_trd[-1]), fontsize=12)

plt.title('Trade', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.10, 0.14, 0.18],fontsize=10)
#plt.yticks([0.10, 0.14, 0.18],fontsize=10)

'Business Services'
ax = plt.subplot(3,3,5)
ax.plot(np.array(AUT.share_bss)[-1], AUT.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label='Model (2)')
ax.plot(np.array(BEL.share_bss)[-1], BEL.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_bss)[-1], DEU.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_bss)[-1], DNK.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_bss)[-1], ESP.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_bss)[-1], FIN.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_bss)[-1], FRA.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_bss)[-1], GBR.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_bss)[-1], GRC.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_bss)[-1], IRL.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_bss)[-1], ITA.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_bss)[-1], LUX.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_bss)[-1], NLD.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_bss)[-1], PRT.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_bss)[-1], SWE.share_bss_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)
ax.plot(EUR15_share_bss_nps_m[-1], EUR15_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_bss)[-1], AUT.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_bss)[-1], BEL.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_bss)[-1], DEU.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_bss)[-1], DNK.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_bss)[-1], ESP.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_bss)[-1], FIN.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_bss)[-1], FRA.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_bss)[-1], GBR.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_bss)[-1], GRC.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_bss)[-1], IRL.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_bss)[-1], ITA.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_bss)[-1], LUX.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_bss)[-1], NLD.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_bss)[-1], PRT.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_bss)[-1], SWE.share_bss_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_bss_nps_m[-1], EUR15_share_bss[-1]), fontsize=12)

plt.title('Business', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.10, 0.15, 0.20, 0.25],fontsize=10)
#plt.yticks([0.10, 0.15, 0.20, 0.25],fontsize=10)

'Finance'
ax = plt.subplot(3,3,6)
ax.plot(np.array(AUT.share_fin)[-1], AUT.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label='Model (2)')
ax.plot(np.array(BEL.share_fin)[-1], BEL.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_fin)[-1], DEU.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_fin)[-1], DNK.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_fin)[-1], ESP.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_fin)[-1], FIN.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_fin)[-1], FRA.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_fin)[-1], GBR.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_fin)[-1], GRC.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_fin)[-1], IRL.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_fin)[-1], ITA.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_fin)[-1], LUX.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_fin)[-1], NLD.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_fin)[-1], PRT.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_fin)[-1], SWE.share_fin_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)
ax.plot(EUR15_share_fin_nps_m[-1], EUR15_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_fin)[-1], AUT.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_fin)[-1], BEL.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_fin)[-1], DEU.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_fin)[-1], DNK.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_fin)[-1], ESP.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_fin)[-1], FIN.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_fin)[-1], FRA.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_fin)[-1], GBR.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_fin)[-1], GRC.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_fin)[-1], IRL.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_fin)[-1], ITA.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_fin)[-1], LUX.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_fin)[-1], NLD.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_fin)[-1], PRT.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_fin)[-1], SWE.share_fin_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_fin_nps_m[-1], EUR15_share_fin[-1]), fontsize=12)

plt.title('Finance', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.01, 0.02, 0.03, 0.04],fontsize=10)
#plt.yticks([0.01, 0.02, 0.03, 0.04],fontsize=10)

'Non-Progressive Services'
ax = plt.subplot(3,3,7)
ax.plot(np.array(AUT.share_nps)[-1], AUT.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label='Model (2)')
ax.plot(np.array(BEL.share_nps)[-1], BEL.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DEU.share_nps)[-1], DEU.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(DNK.share_nps)[-1], DNK.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ESP.share_nps)[-1], ESP.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FIN.share_nps)[-1], FIN.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(FRA.share_nps)[-1], FRA.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GBR.share_nps)[-1], GBR.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(GRC.share_nps)[-1], GRC.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(IRL.share_nps)[-1], IRL.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(ITA.share_nps)[-1], ITA.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(LUX.share_nps)[-1], LUX.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(NLD.share_nps)[-1], NLD.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(PRT.share_nps)[-1], PRT.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(np.array(SWE.share_nps)[-1], SWE.share_nps_nps_m[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.plot(EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)
ax.plot(EUR15_share_nps_nps_m[-1], EUR15_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=10)

ax.annotate('AUT', (np.array(AUT.share_nps)[-1], AUT.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('BEL', (np.array(BEL.share_nps)[-1], BEL.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DEU', (np.array(DEU.share_nps)[-1], DEU.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('DNK', (np.array(DNK.share_nps)[-1], DNK.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ESP', (np.array(ESP.share_nps)[-1], ESP.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FIN', (np.array(FIN.share_nps)[-1], FIN.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('FRA', (np.array(FRA.share_nps)[-1], FRA.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GBR', (np.array(GBR.share_nps)[-1], GBR.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('GRC', (np.array(GRC.share_nps)[-1], GRC.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('IRL', (np.array(IRL.share_nps)[-1], IRL.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('ITA', (np.array(ITA.share_nps)[-1], ITA.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('LUX', (np.array(LUX.share_nps)[-1], LUX.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('NLD', (np.array(NLD.share_nps)[-1], NLD.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('PRT', (np.array(PRT.share_nps)[-1], PRT.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('SWE', (np.array(SWE.share_nps)[-1], SWE.share_nps_nps_m[-1]),alpha=0.6,fontsize=10)
ax.annotate('EU4', (EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1]), fontsize=12)
ax.annotate('EU15', (EUR15_share_nps_nps_m[-1], EUR15_share_nps[-1]), fontsize=12)

plt.title('Non-Progressive', fontsize=16)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.xlabel('Data', fontsize=10)
plt.ylabel('Model', fontsize=10)
#plt.xticks([0.30, 0.40, 0.50],fontsize=10)
#plt.yticks([0.30, 0.40, 0.50],fontsize=10)


fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
plt.tight_layout()
#plt.savefig('../output/figures/fig_test.pdf', bbox_inches="tight")
plt.close()


'Test of the Theory as Time Series'
rc('axes', linewidth=0.4)

fig = plt.figure()
ax = plt.subplot(4,4,1)
plt.plot(AUT.year, np.array(AUT.E)/E, 'b-', alpha=0.75) 
plt.plot(AUT.year, np.array(AUT.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(AUT.year, np.array(AUT.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Austria', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,2)
plt.plot(BEL.year, np.array(BEL.E)/E, 'b-', alpha=0.75) 
plt.plot(BEL.year, np.array(BEL.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(BEL.year, np.array(BEL.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Belgium', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,3)
plt.plot(DEU.year, np.array(DEU.E)/E, 'b-', alpha=0.75) 
plt.plot(DEU.year, np.array(DEU.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(DEU.year, np.array(DEU.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Germany', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,4)
plt.plot(DNK.year, np.array(DNK.E)/E, 'b-', alpha=0.75) 
plt.plot(DNK.year, np.array(DNK.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(DNK.year, np.array(DNK.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Denmark', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,5)
plt.plot(ESP.year, np.array(ESP.E)/E, 'b-', alpha=0.75) 
plt.plot(ESP.year, np.array(ESP.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(ESP.year, np.array(ESP.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Spain', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,6)
plt.plot(FIN.year, np.array(FIN.E)/E, 'b-', alpha=0.75) 
plt.plot(FIN.year, np.array(FIN.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(FIN.year, np.array(FIN.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Finland', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,7)
plt.plot(FRA.year, np.array(FRA.E)/E, 'b-', alpha=0.75) 
plt.plot(FRA.year, np.array(FRA.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(FRA.year, np.array(FRA.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'France', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,8)
plt.plot(GBR.year, np.array(GBR.E)/E, 'b-', alpha=0.75) 
plt.plot(GBR.year, np.array(GBR.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(GBR.year, np.array(GBR.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Great Britan', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,9)
plt.plot(GRC.year, np.array(GRC.E)/E, 'b-', alpha=0.75) 
plt.plot(GRC.year, np.array(GRC.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(GRC.year, np.array(GRC.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Greece', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,10)
plt.plot(IRL.year, np.array(IRL.E)/E, 'b-', alpha=0.75) 
plt.plot(IRL.year, np.array(IRL.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(IRL.year, np.array(IRL.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Ireland', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,11)
plt.plot(ITA.year, np.array(ITA.E)/E, 'b-', alpha=0.75) 
plt.plot(ITA.year, np.array(ITA.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(ITA.year, np.array(ITA.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Italy', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

ax = plt.subplot(4,4,12)
plt.plot(LUX.year, np.array(LUX.E)/E, 'b-', alpha=0.75) 
plt.plot(LUX.year, np.array(LUX.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(LUX.year, np.array(LUX.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Luxembourg', fontsize = 12, y = -0.05)
plt.xticks([])
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize = 10)

ax = plt.subplot(4,4,13)
plt.plot(NLD.year, np.array(NLD.E)/E, 'b-', alpha=0.75) 
plt.plot(NLD.year, np.array(NLD.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(NLD.year, np.array(NLD.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'The Netherlands', fontsize = 12, y = -0.05)
plt.xticks([1970, 1995, 2020], fontsize=10)
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

ax = plt.subplot(4,4,14)
plt.plot(PRT.year, np.array(PRT.E)/E, 'b-', alpha=0.75) 
plt.plot(PRT.year, np.array(PRT.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(PRT.year, np.array(PRT.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Portugal', fontsize = 12, y = -0.05)
plt.xticks([1970, 1995, 2020], fontsize=10)
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

ax = plt.subplot(4,4,15)
plt.plot(SWE.year, np.array(SWE.E)/E, 'b-', alpha=0.75) 
plt.plot(SWE.year, np.array(SWE.A_tot_ams)/A_tot_ams, '--', color = 'saddlebrown', alpha=0.75) 
plt.plot(SWE.year, np.array(SWE.A_tot_nps)/A_tot_nps, '-.', color = 'darkgreen', alpha=0.75) 
#plt.axis([1968, 2021, 0.35, 1.25])
plt.grid(axis = 'y',alpha=0.35)
plt.title(r'Sweeden', fontsize = 12, y = -0.05)
plt.xticks([1970, 1995, 2020], fontsize=10)
#plt.yticks([0.4,0.6,0.8,1,1.2], fontsize=10)

plt.tight_layout()
#plt.savefig('../output/figures/fig_test_A_EUR_countries.pdf', bbox_inches="tight")
plt.close()


fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
plt.plot(AUT.year, EUR4_rel_E, 'b-', lw=2, alpha=0.95,label='EU4 (OECD)')
plt.plot(AUT.year, EUR4_rel_A_tot, 'r--', lw=2, alpha=0.95,label='EU4 (KLEMS)')
plt.plot(AUT.year, EUR4_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=5, alpha=0.95, label='EU4. Model (1)')
plt.plot(AUT.year, EUR4_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=5, alpha=0.95, label='EU4. Model (2)')
plt.axis([1968, 2021, 0.65, 1.05])
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
plt.legend(fontsize=14)
plt.grid()

ax = plt.subplot(1, 2, 2)
plt.plot(AUT.year, EUR15_rel_E, 'b-', lw=2, alpha=0.75, label='EU15 (OECD)')
plt.plot(AUT.year, EUR15_rel_A_tot, 'r--', lw=2, alpha=0.75, label='EU15 (KLEMS)')
plt.plot(AUT.year, EUR15_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=5, alpha=0.95, label='EU15. Model (1)')
plt.plot(AUT.year, EUR15_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=5, alpha=0.95, label='EU15. Model (2)')
plt.axis([1968, 2021, 0.65, 1.05])
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
plt.legend(fontsize=14)
plt.grid()

plt.tight_layout()
#plt.savefig('../output/figures/fig_test_A_EUR.pdf', bbox_inches="tight")
plt.close()

'''
---------------------------
    Plots for the paper
---------------------------
'''


#FIGURE 1
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
#plt.plot(DEU.year, DEU.E/np.array(E), 'o-', markersize=4, color='darkgoldenrod', markerfacecolor='gold', markeredgecolor='darkgoldenrod', alpha=0.5, label='Germany')
#plt.plot(DEU.year, FRA.E/np.array(E), 'D-', markersize=4, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.5, label='France')
#plt.plot(DEU.year, GBR.E/np.array(E), 's-', markersize=4, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', alpha=0.5, label='Great Britain')
#plt.plot(DEU.year, ITA.E/np.array(E), '^-', markersize=4, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', alpha=0.5, label='Italy')
ax.plot(DEU.year, EUR4_rel_E, 'b-', lw=2, alpha=0.95, label='Europe')
plt.title('Labor Productivity Gap (Relative to the U.S.)', fontsize=16)
ax.axis([1968, 2021, 0.68, 1.04])
plt.xticks(fontsize=14)
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], fontsize=14)
plt.legend(loc='upper left', fontsize=14)
plt.grid()

ax = plt.subplot(1, 2, 2)
ax.plot(DEU.year, share_nps, '^-', markersize=6, color='darkmagenta', mfc='none', markevery=2, alpha=0.8, label='USA: Non-Progressive')
ax.plot(DEU.year, EUR4_share_nps, '^-', markersize=6, color='darkmagenta', markerfacecolor='violet', markeredgecolor='darkmagenta', markevery=2, alpha=0.8, label='Europe: Non-Progressive')
ax.plot(DEU.year, share_trd, 'o-', markersize=6, color='darkblue', mfc='none', markevery=2, alpha=0.8, label='USA: Trade')
ax.plot(DEU.year, EUR4_share_trd, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=2, alpha=0.8, label='Europe: Trade')
ax.plot(DEU.year, share_bss, 'D-', markersize=6, color='darkgreen', mfc='none', markevery=2, alpha=0.8, label='USA: Business')
ax.plot(DEU.year, EUR4_share_bss, 'D-', markersize=6, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=2, alpha=0.8, label='Europe: Business')
ax.plot(DEU.year, share_fin, 's-', markersize=6, color='darkred', mfc='none', markevery=2, alpha=0.8, label='USA: Financial')
ax.plot(DEU.year, EUR4_share_fin, 's-', markersize=6, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=2, alpha=0.8, label='Europe: Financial')
plt.title('Employment Shares Within Services', fontsize=16)
ax.axis([1968, 2021, 0, 0.48])
plt.xticks(fontsize=14)
plt.yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], fontsize=14)
plt.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=14)

plt.tight_layout()
plt.savefig("../output/figures/fig_1.pdf", bbox_inches="tight")
plt.close()


#FIGURE 2
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
ax.plot(np.array(share_agr)[-1], share_agr_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75, label = 'United States')
ax.plot(EUR4_share_agr[-1], EUR4_share_agr_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75, label = 'Europe')
ax.annotate(r'$\texttt{agr}$', (np.array(share_agr)[-1], share_agr_nps[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr[-1], EUR4_share_agr_nps_m[-1]), alpha=0.75, fontsize=16)
ax.plot(np.array(share_man)[-1], share_man_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75)
ax.plot(EUR4_share_man[-1], EUR4_share_man_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{man}$', (np.array(share_man)[-1], share_man_nps[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{man}$', (EUR4_share_man[-1], EUR4_share_man_nps_m[-1]), alpha=0.75, fontsize=16)
ax.plot(np.array(share_trd)[-1], share_trd_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75)
ax.plot(EUR4_share_trd[-1], EUR4_share_trd_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (np.array(share_trd)[-1], share_trd_nps[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{trd}$', (EUR4_share_trd[-1], EUR4_share_trd_nps_m[-1]), alpha=0.75, fontsize=16)
ax.plot(np.array(share_bss)[-1], share_bss_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75)
ax.plot(EUR4_share_bss[-1], EUR4_share_bss_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (np.array(share_bss)[-1], share_bss_nps[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{bss}$', (EUR4_share_bss[-1], EUR4_share_bss_nps_m[-1]), alpha=0.75, fontsize=16)
ax.plot(np.array(share_fin)[-1], share_fin_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75)
ax.plot(EUR4_share_fin[-1], EUR4_share_fin_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (np.array(share_fin)[-1], share_fin_nps[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{fin}$', (EUR4_share_fin[-1], EUR4_share_fin_nps_m[-1]), alpha=0.75, fontsize=16)
#ax.plot(np.array(share_nps)[-1], share_nps_nps[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6,alpha=0.75)
#ax.plot(EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{nps}$', (np.array(share_nps)[-1], share_nps_nps[-1]), alpha=0.75, fontsize=16)
#ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps[-1], EUR4_share_nps_nps_m[-1]), alpha=0.75, fontsize=16)

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
#plt.axis([0, 0.52, 0, 0.52])
plt.title('Employment Shares in 2019', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid()

#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)

ax = plt.subplot(1,2,2)
ax.plot(DEU.year, EUR4_rel_E, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha = 0.95, label = 'Data: OECD')
ax.plot(DEU.year, EUR4_rel_A_tot, 'D-', markersize=6, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha = 0.95, label = 'Data: KLEMS')
ax.plot(DEU.year, EUR4_A_tot_nps/A_tot_nps, 's-', markersize=6, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label = 'Model')
plt.title('Labor Productivity (Relative to U.S.)', fontsize=16)
ax.axis([1968, 2021, 0.68, 1.04])
plt.xticks(fontsize=14)
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid()

plt.tight_layout()
plt.savefig('../output/figures/fig_2.pdf', bbox_inches="tight")
plt.close()#

# NOTE: Figure 3 (counterfactual decomposition) moved to generate_paper_outputs.py
# because it requires Counterfactual_ts.xlsx produced by counterfactuals.py (Step 3).

'''
---------------------------
    Plots for the paper
---------------------------
'''

#FIG TEST EUROPE APPENDIX
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(6)
fig.set_figwidth(6)

ax = plt.subplot(3, 2, 1)
plt.plot(AUT.year, EUR4_rel_E, 'b-', alpha=0.95,label='Data')
#plt.plot(AUT.year, EUR4_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(AUT.year, EUR4_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', markersize=5, markevery=2, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.95, label='Model')
plt.axis([1968, 2021, 0.68, 1.02])
plt.title('EU4', fontsize=12)
#plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel('Year', fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 2)
#ax.plot(EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label = 'Model')
#ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1]), alpha=0.75, fontsize=14)
#ax.plot(EUR4_share_man_ams_m[-1], EUR4_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EUR4_share_man_nps_m[-1], EUR4_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{man}$', (EUR4_share_man_ams_m[-1], EUR4_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{man}$', (EUR4_share_man_nps_m[-1], EUR4_share_man[-1]), alpha=0.75, fontsize=14)
#ax.plot(EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{ser}$', (EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1]), alpha=0.75, fontsize=14)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0,0.5,0,0.5])
plt.title('EU4', fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 3)
plt.plot(AUT.year, GBR.E/np.array(E), 'b-', alpha=0.95,label=r'$\frac{Y_t}{L_t}$: Data')
#plt.plot(AUT.year, EUR15_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(AUT.year, GBR.A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', markersize=5, markevery=2, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.95, label='Model')
plt.axis([1968, 2021, 0.68, 1.02])
plt.title('GBR', fontsize=12)
#plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel('Year', fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 4)
ax.plot(GBR.share_agr_nps_m[-1], GBR.share_agr.values[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label = 'Model')
ax.annotate(r'$\texttt{agr}$', (GBR.share_agr_nps_m[-1], GBR.share_agr.values[-1]), alpha=0.75, fontsize=14)
ax.plot(GBR.share_man_nps_m[-1], GBR.share_man.values[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{man}$', (GBR.share_man_nps_m[-1], GBR.share_man.values[-1]), fontsize=14)
ax.plot(GBR.share_trd_nps_m[-1], GBR.share_trd.values[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (GBR.share_trd_nps_m[-1], GBR.share_trd.values[-1]), alpha=0.75, fontsize=14)
ax.plot(GBR.share_bss_nps_m[-1], GBR.share_bss.values[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (GBR.share_bss_nps_m[-1], GBR.share_bss.values[-1]), alpha=0.75, fontsize=14)
ax.plot(GBR.share_fin_nps_m[-1], GBR.share_fin.values[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (GBR.share_fin_nps_m[-1], GBR.share_fin.values[-1]), alpha=0.75, fontsize=14)
ax.plot(GBR.share_nps_nps_m[-1], GBR.share_nps.values[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{nps}$', (GBR.share_nps_nps_m[-1], GBR.share_nps.values[-1]), alpha=0.75, fontsize=14)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0,0.5,0,0.5])
plt.title('GBR', fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 5)
plt.plot(AUT.year, EUR15_rel_E, 'b-', alpha=0.95,label=r'$\frac{Y_t}{L_t}$: Data')
#plt.plot(AUT.year, EUR15_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(AUT.year, EUR15_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', markersize=5, markevery=2, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.95, label='Model')
plt.axis([1968, 2021, 0.68, 1.02])
plt.title('EU15', fontsize=12)
#plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel('Year', fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=10)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 6)
#ax.plot(EUR15_share_agr_ams_m[-1], EUR15_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(EUR15_share_agr_nps_m[-1], EUR15_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label = 'Model')
#ax.annotate(r'$\texttt{agr}$', (EUR15_share_agr_ams_m[-1], EUR15_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{agr}$', (EUR15_share_agr_nps_m[-1], EUR15_share_agr[-1]), alpha=0.75, fontsize=14)
#ax.plot(EUR15_share_man_ams_m[-1], EUR15_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EUR15_share_man_nps_m[-1], EUR15_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{man}$', (EUR15_share_man_ams_m[-1], EUR15_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{man}$', (EUR15_share_man_nps_m[-1], EUR15_share_man[-1]), fontsize=14)
#ax.plot(EUR15_share_ser_ams_m[-1], EUR15_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{ser}$', (EUR15_share_ser_ams_m[-1], EUR15_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR15_share_trd_nps_m[-1], EUR15_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (EUR15_share_trd_nps_m[-1], EUR15_share_trd[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR15_share_bss_nps_m[-1], EUR15_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (EUR15_share_bss_nps_m[-1], EUR15_share_bss[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR15_share_fin_nps_m[-1], EUR15_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (EUR15_share_fin_nps_m[-1], EUR15_share_fin[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR15_share_nps_nps_m[-1], EUR15_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{nps}$', (EUR15_share_nps_nps_m[-1], EUR15_share_nps[-1]), alpha=0.75, fontsize=14)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0,0.5,0,0.5])
plt.title('EU15', fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

plt.tight_layout()
plt.savefig("../output/figures/fig_test_EUR_appendix.pdf", bbox_inches="tight")
plt.close()

#FIG TEST EUROPE APPENDIX
fig = plt.figure()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
fig.set_figheight(6)
fig.set_figwidth(6)

ax = plt.subplot(3, 2, 1)
plt.plot(AUT.year, EUR4_rel_E, 'b-', alpha=0.95,label='Data')
#plt.plot(AUT.year, EUR4_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(AUT.year, EUR4_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', markersize=5, markevery=2, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.95, label='Model')
plt.axis([1968, 2021, 0.68, 1.02])
plt.title('EU4', fontsize=12)
#plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel('Year', fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 2)
#ax.plot(EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label = 'Model')
#ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr_ams_m[-1], EUR4_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr_nps_m[-1], EUR4_share_agr[-1]), alpha=0.75, fontsize=14)
#ax.plot(EUR4_share_man_ams_m[-1], EUR4_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EUR4_share_man_nps_m[-1], EUR4_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{man}$', (EUR4_share_man_ams_m[-1], EUR4_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{man}$', (EUR4_share_man_nps_m[-1], EUR4_share_man[-1]), alpha=0.75, fontsize=14)
#ax.plot(EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{ser}$', (EUR4_share_ser_ams_m[-1], EUR4_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (EUR4_share_trd_nps_m[-1], EUR4_share_trd[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (EUR4_share_bss_nps_m[-1], EUR4_share_bss[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (EUR4_share_fin_nps_m[-1], EUR4_share_fin[-1]), alpha=0.75, fontsize=14)
ax.plot(EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{nps}$', (EUR4_share_nps_nps_m[-1], EUR4_share_nps[-1]), alpha=0.75, fontsize=14)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0,0.5,0,0.5])
plt.title('EU4', fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 3)
plt.plot(AUT.year, EURCORE_E/np.array(E), 'b-', alpha=0.95,label=r'$\frac{Y_t}{L_t}$: Data')
#plt.plot(AUT.year, EUR15_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(AUT.year, EURCORE_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', markersize=5, markevery=2, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.95, label='Model')
plt.axis([1968, 2021, 0.68, 1.02])
plt.title('EU Core', fontsize=12)
#plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel('Year', fontsize=11)
plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2], fontsize=10)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 4)
ax.plot(EURCORE_share_agr_nps_m[-1], EURCORE_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label = 'Model')
ax.annotate(r'$\texttt{agr}$', (EURCORE_share_agr_nps_m[-1], EURCORE_share_agr[-1]), alpha=0.75, fontsize=14)
ax.plot(EURCORE_share_man_nps_m[-1], EURCORE_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{man}$', (EURCORE_share_man_nps_m[-1], EURCORE_share_man[-1]), fontsize=14)
ax.plot(EURCORE_share_trd_nps_m[-1], EURCORE_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (EURCORE_share_trd_nps_m[-1], EURCORE_share_trd[-1]), alpha=0.75, fontsize=14)
ax.plot(EURCORE_share_bss_nps_m[-1], EURCORE_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (EURCORE_share_bss_nps_m[-1], EURCORE_share_bss[-1]), alpha=0.75, fontsize=14)
ax.plot(EURCORE_share_fin_nps_m[-1], EURCORE_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (EURCORE_share_fin_nps_m[-1], EURCORE_share_fin[-1]), alpha=0.75, fontsize=14)
ax.plot(EURCORE_share_nps_nps_m[-1], EURCORE_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{nps}$', (EURCORE_share_nps_nps_m[-1], EURCORE_share_nps[-1]), alpha=0.75, fontsize=14)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0,0.5,0,0.5])
plt.title('EU Core', fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 5)
plt.plot(AUT.year, EURPERI_rel_E, 'b-', alpha=0.95,label=r'$\frac{Y_t}{L_t}$: Data')
#plt.plot(AUT.year, EURPERI_A_tot_ams/np.array(A_tot_ams).flatten(), 'D-', markersize=5, markevery=2, color='saddlebrown', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', alpha=0.95, label='Model (1)')
plt.plot(AUT.year, EURPERI_A_tot_nps/np.array(A_tot_nps).flatten(), 'H-', markersize=5, markevery=2, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.95, label='Model')
plt.axis([1968, 2021, 0.68, 1.02])
plt.title('EU Periphery', fontsize=12)
#plt.ylabel(r'Labor Productivity (rel. to U.S.)', fontsize=11)
plt.xlabel('Year', fontsize=11)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1], fontsize=10)
plt.xticks([1970, 1980, 1990, 2000, 2010, 2020], fontsize=10)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=10)
plt.grid()

ax = plt.subplot(3, 2, 6)
#ax.plot(EURPERI_share_agr_ams_m[-1], EURPERI_share_agr[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75, label = 'Model (1)')
ax.plot(EURPERI_share_agr_nps_m[-1], EURPERI_share_agr[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75, label = 'Model')
#ax.annotate(r'$\texttt{agr}$', (EURPERI_share_agr_ams_m[-1], EURPERI_share_agr[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{agr}$', (EURPERI_share_agr_nps_m[-1], EURPERI_share_agr[-1]), alpha=0.75, fontsize=14)
#ax.plot(EURPERI_share_man_ams_m[-1], EURPERI_share_man[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
ax.plot(EURPERI_share_man_nps_m[-1], EURPERI_share_man[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{man}$', (EURPERI_share_man_ams_m[-1], EURPERI_share_man[-1]), alpha=0.75, fontsize=14)
ax.annotate(r'$\texttt{man}$', (EURPERI_share_man_nps_m[-1], EURPERI_share_man[-1]), fontsize=14)
#ax.plot(EURPERI_share_ser_ams_m[-1], EURPERI_share_ser[-1], 'D', markerfacecolor='sandybrown', markeredgecolor='saddlebrown', markersize=6, alpha=0.75)
#ax.annotate(r'$\texttt{ser}$', (EURPERI_share_ser_ams_m[-1], EURPERI_share_ser[-1]), alpha=0.75, fontsize=14)
ax.plot(EURPERI_share_trd_nps_m[-1], EURPERI_share_trd[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{trd}$', (EURPERI_share_trd_nps_m[-1], EURPERI_share_trd[-1]), alpha=0.75, fontsize=14)
ax.plot(EURPERI_share_bss_nps_m[-1], EURPERI_share_bss[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{bss}$', (EURPERI_share_bss_nps_m[-1], EURPERI_share_bss[-1]), alpha=0.75, fontsize=14)
ax.plot(EURPERI_share_fin_nps_m[-1], EURPERI_share_fin[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{fin}$', (EURPERI_share_fin_nps_m[-1], EURPERI_share_fin[-1]), alpha=0.75, fontsize=14)
ax.plot(EURPERI_share_nps_nps_m[-1], EURPERI_share_nps[-1], 'H', markerfacecolor='lime', markeredgecolor='darkgreen', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{nps}$', (EURPERI_share_nps_nps_m[-1], EURPERI_share_nps[-1]), alpha=0.75, fontsize=14)
plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0,0.5,0,0.5])
plt.title('EU Periphery', fontsize=12)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=10)
ax.set_ylabel("Data")
ax.set_xlabel("Model")
#plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.grid()

plt.tight_layout()
plt.savefig("../output/figures/fig_test_EUR_2_appendix.pdf", bbox_inches="tight")
plt.close()

print ('------------------------------------------')
print('Annualized Growth Rate in the EU4 (1970-1995): ' +str((((EUR4_rel_E[25]*E[25])/EUR4_rel_E[0])**(1/25)-1)*100))
print('Annualized Growth Rate in the EU15 (1970-1995): ' +str((((EUR15_rel_E[25]*E[25])/EUR15_rel_E[0])**(1/25)-1)*100))
print('Annualized Growth Rate in the EU4 (1995-2019): ' +str((((EUR4_rel_E[-1]*E[-1])/(EUR4_rel_E[25]*E[25]))**(1/(49-25))-1)*100))
print('Annualized Growth Rate in the EU15 (1995-2019): ' +str((((EUR15_rel_E[-1]*E[-1])/(EUR15_rel_E[25]*E[25]))**(1/(49-25))-1)*100))
print('Percentage of convergence explain by Model (2): ' +str((np.array(EUR4_A_tot_nps/np.array(A_tot_nps).flatten())[25]-np.array(EUR4_A_tot_nps/np.array(A_tot_nps).flatten())[0])/(EUR15_rel_E[25]-EUR15_rel_E[0])))
print('Percentage of divergence explain by Model (2): ' +str((np.array(EUR4_A_tot_nps/np.array(A_tot_nps).flatten())[-1]-np.array(EUR4_A_tot_nps/np.array(A_tot_nps).flatten())[25])/(EUR15_rel_E[-1]-EUR15_rel_E[25])))


'''
PLOTS WITH AMS MODEL
'''

# FIGURE 2 with AMS
fig = plt.figure()
plt.subplots_adjust(wspace=0.05)
fig.set_figheight(5)
fig.set_figwidth(10)

ax = plt.subplot(1, 2, 1)
ax.plot(np.array(share_agr)[-1], share_agr_ams[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75, label = 'United States')
ax.plot(EUR4_share_agr[-1], EUR4_share_agr_ams_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75, label = 'Europe')
ax.annotate(r'$\texttt{agr}$', (np.array(share_agr)[-1], share_agr_ams[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{agr}$', (EUR4_share_agr[-1], EUR4_share_agr_ams_m[-1]), alpha=0.75, fontsize=16)
ax.plot(np.array(share_man)[-1], share_man_ams[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75)
ax.plot(EUR4_share_man[-1], EUR4_share_man_ams_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{man}$', (np.array(share_man)[-1], share_man_ams[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{man}$', (EUR4_share_man[-1], EUR4_share_man_ams_m[-1]), alpha=0.75, fontsize=16)
ax.plot(np.array(share_ser)[-1], share_ser_ams[-1], 'D', markerfacecolor='lightcoral', markeredgecolor='darkred', markersize=6, alpha=0.75)
ax.plot(EUR4_share_ser[-1], EUR4_share_ser_ams_m[-1], 'o', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markersize=6, alpha=0.75)
ax.annotate(r'$\texttt{ser}$', (np.array(share_ser)[-1], share_ser_ams[-1]), alpha=0.75, fontsize=16)
ax.annotate(r'$\texttt{ser}$', (EUR4_share_ser[-1], EUR4_share_ser_ams_m[-1]), alpha=0.75, fontsize=16)

plt.plot(ax.get_xlim(), ax.get_xlim(), ls="-", c=".3")
plt.axis([0, 0.85, 0, 0.85])
plt.title('Employment Shares in 2019', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid()

#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc=9, bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)

ax = plt.subplot(1,2,2)
ax.plot(DEU.year, EUR4_rel_E, 'o-', markersize=6, color='darkblue', markerfacecolor='lightskyblue', markeredgecolor='darkblue', markevery=7, alpha = 0.95, label = 'Data: OECD')
ax.plot(DEU.year, EUR4_rel_A_tot, 'D-', markersize=6, color='darkgreen', markerfacecolor='lime', markeredgecolor='darkgreen', markevery=7, alpha = 0.95, label = 'Data: KLEMS')
ax.plot(DEU.year, EUR4_A_tot_nps/A_tot_nps, 's-', markersize=6, color='darkred', markerfacecolor='lightcoral', markeredgecolor='darkred', markevery=7, alpha=0.95, label = 'Baseline model')
ax.plot(DEU.year, EUR4_A_tot_ams/A_tot_ams, '^-', markersize=6, color='black', markerfacecolor='black', markeredgecolor='black', markevery=7, alpha=0.95, label = 'Three-sector model')
plt.title('Labor Productivity (Relative to U.S.)', fontsize=16)
ax.axis([1968, 2021, 0.68, 1.04])
plt.xticks(fontsize=14)
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid()

plt.tight_layout()
plt.savefig("../output/figures/fig_2_ams.pdf", bbox_inches="tight")
plt.close()


