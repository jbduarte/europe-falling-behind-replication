"""
=======================================================================================
Project: Structural Transformation and Productivity in Europe (with Duarte and Saenz)
Filename: trade_counterfactual.py
Description: This program uses the BDS model with calibrated parameters to establish
	counterfactual experiments. 

Author: Joao B. Duarte
Last Modified: Feb 2026
=======================================================================================
"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import minimize_scalar, root, fsolve
rc('text', usetex=True)
rc('font', family='serif')

# The following string runs the calibration of the model on the US, and imports the parameter values that are used in this program. Also, it imports US GDP, needed in the European sectoral productivity measurement.
from model_calibration_USA import sigma, eps_agr, eps_trd, eps_fin, eps_bss, eps_nps, eps_ser
from model_calibration_USA_open import GDP_ph, E, A_tot, \
    share_agr, share_man, share_trd, share_bss, share_fin, share_nps, share_ser, \
    share_agr_ams, share_man_ams, share_ser_ams, A_tot_ams, \
    share_agr_ams_closed, share_man_ams_closed, share_ser_ams_closed, A_tot_ams_closed, \
    share_agr_nps, share_man_nps, share_trd_nps, share_bss_nps, share_fin_nps, share_nps_nps, A_tot_nps, \
    share_agr_nps_closed, share_man_nps_closed, share_trd_nps_closed, share_bss_nps_closed, share_fin_nps_closed, share_nps_nps_closed, A_tot_nps_closed

# Rename US Aggregates
GDP_ph_USA, E_USA = GDP_ph, E

#The following string runs the measurement of productivity in Europe recovering the initial levels with our model.
from model_test_europe_open import model_country, EUR4_h_tot, EURCORE_h_tot, EURPERI_h_tot,  EUR13_h_tot, EUR4_A_tot, EURCORE_A_tot, EURPERI_A_tot, EUR13_A_tot, EUR4_rel_A_tot, EUR13_rel_A_tot, EUR4_E, EUR13_E, EUR4_rel_E, EUR13_rel_E, \
	EUR4_share_agr, EUR13_share_agr, EUR4_share_man, EUR13_share_man, EUR4_share_ser, EUR13_share_ser, EUR4_share_trd, EUR13_share_trd, EUR4_share_bss, EUR13_share_bss, EUR4_share_fin, EUR13_share_fin, EUR4_share_nps, EUR13_share_nps, \
	EUR4_share_agr_ams_m, EUR13_share_agr_ams_m, EUR4_share_agr_nps_m, EUR13_share_agr_nps_m, EUR4_share_man_ams_m, EUR13_share_man_ams_m, EUR4_share_man_nps_m, EUR13_share_man_nps_m, EUR4_share_ser_ams_m, EUR13_share_ser_ams_m, EUR4_share_trd_nps_m, EUR13_share_trd_nps_m, EUR4_share_bss_nps_m, EUR13_share_bss_nps_m, EUR4_share_fin_nps_m, EUR13_share_fin_nps_m, EUR4_share_nps_nps_m, EUR13_share_nps_nps_m, \
	EUR4_A_tot_ams, EUR13_A_tot_ams, EUR4_A_tot_nps, EURCORE_A_tot_nps, EURPERI_A_tot_nps, EUR13_A_tot_nps


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

#IRL = model_country('IRL')
#IRL.productivity_series()
#IRL.predictions_ams()
#IRL.predictions_nps()

ITA = model_country('ITA')
ITA.productivity_series()
ITA.predictions_ams()
ITA.predictions_nps()

#LUX = model_country('LUX')
#LUX.productivity_series()
#LUX.predictions_ams()
#LUX.predictions_nps()

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

# UNITED STATES
USA = model_country('USA')
USA.productivity_series()
USA.predictions_ams()
USA.predictions_nps()


class counterfactual:
	'Counterfactuals'
	def __init__(self, country_code):
		self.country_code = country_code
		self.cou = model_country(self.country_code)
		self.cou.productivity_series()

	'Baseline'
	def baseline(self):

		'Shift-share baseline'
		self.ss_A_base_ams_init  = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_ser)[0]*self.cou.A_ser[-1]
		self.ss_A_base_ams  = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_ser*self.cou.A_ser

		self.ss_A_base_nps_init = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_trd)[0]*self.cou.A_trd[-1] + np.array(self.cou.share_bss)[0]*self.cou.A_bss[-1] + np.array(self.cou.share_fin)[0]*self.cou.A_fin[-1] + np.array(self.cou.share_nps)[0]*self.cou.A_nps[-1]
		self.ss_A_base_nps = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_trd*self.cou.A_trd + self.cou.share_bss*self.cou.A_bss + self.cou.share_fin*self.cou.A_fin + self.cou.share_nps*self.cou.A_nps

		'Baseline computation of C'
		C_lev_E_ams=[]
		for i in range(len(self.cou.h_tot)):
			L_t=np.array(self.cou.h_tot)[i]
			A_agr_t=np.array(self.cou.A_agr)[i]
			A_man_t=np.array(self.cou.A_man)[i]
			A_ser_t=np.array(self.cou.A_ser)[i]			
			def C_exp_ams(C):
	   			return L_t**(1-sigma) - (self.cou.om_agr_ams*(C**eps_agr)*(A_agr_t**(sigma-1)) + self.cou.om_man_ams*C*(A_man_t**(sigma-1)) + (1-self.cou.om_agr_ams-self.cou.om_man_ams)*(C**eps_ser)*(A_ser_t**(sigma-1)))
			C_lev_E_ams.append(fsolve(C_exp_ams, L_t).item()) 
		C_level_E_ams = pd.DataFrame(C_lev_E_ams)       
		g_C_E_ams = np.array(C_level_E_ams/C_level_E_ams.shift(1) - 1).flatten()
		self.C_E_ams_baseline = [np.array(self.cou.GDP_ph)[0]/np.array(GDP_ph_USA)[0]]
		for i in range(len(g_C_E_ams) - 1):
			self.C_E_ams_baseline.append((1+g_C_E_ams[i+1])*self.C_E_ams_baseline[i])

		C_lev_E_nps=[]
		for i in range(len(self.cou.h_tot)):
			L_t=np.array(self.cou.h_tot)[i]
			A_agr_t=np.array(self.cou.A_agr)[i]
			A_man_t=np.array(self.cou.A_man)[i]
			A_trd_t=np.array(self.cou.A_trd)[i]
			A_bss_t=np.array(self.cou.A_bss)[i]
			A_fin_t=np.array(self.cou.A_fin)[i]
			A_nps_t=np.array(self.cou.A_nps)[i]
			def C_exp_nps(C):
				return L_t**(1-sigma) - (self.cou.om_agr_nps*(C**eps_agr)*(A_agr_t**(sigma-1)) + self.cou.om_man_nps*C*(A_man_t**(sigma-1)) + self.cou.om_trd_nps*(C**eps_trd)*(A_trd_t**(sigma-1)) + self.cou.om_bss_nps*(C**eps_bss)*(A_bss_t**(sigma-1)) + self.cou.om_fin_nps*(C**eps_fin)*(A_fin_t**(sigma-1)) + (1-self.cou.om_agr_nps-self.cou.om_man_nps-self.cou.om_trd_nps-self.cou.om_bss_nps-self.cou.om_fin_nps)*(C**eps_nps)*(A_nps_t**(sigma-1)))
			C_lev_E_nps.append(fsolve(C_exp_nps, L_t).item())
		C_level_E_nps = pd.DataFrame(C_lev_E_nps)
		g_C_E_nps = np.array(C_level_E_nps/C_level_E_nps.shift(1) - 1).flatten()
		self.C_E_nps_baseline = [np.array(self.cou.GDP_ph)[0]/np.array(GDP_ph_USA)[0]]
		for i in range(len(g_C_E_nps) - 1):
			self.C_E_nps_baseline.append((1+g_C_E_nps[i+1])*self.C_E_nps_baseline[i])
  
	def feed_US_productivity_growth(self, init_year, sec):
		'Baseline'
		self.baseline()

		'Feeding US productivity growth into sectors'
		if sec == 'agr':
			self.cou.g_y_l_agr[init_year:] = USA.g_y_l_agr[init_year:]			
		if sec == 'man':
			self.cou.g_y_l_man[init_year:] = USA.g_y_l_man[init_year:]
		if sec == 'trd':
			self.cou.g_y_l_trd[init_year:] = USA.g_y_l_trd[init_year:]
		if sec == 'bss':
			self.cou.g_y_l_bss[init_year:] = USA.g_y_l_bss[init_year:]
		if sec == 'fin':
			self.cou.g_y_l_fin[init_year:] = USA.g_y_l_fin[init_year:]
		if sec == 'nps':
			self.cou.g_y_l_nps[init_year:] = USA.g_y_l_nps[init_year:]
		if sec == 'ser':
			self.cou.g_y_l_ser[init_year:] = USA.g_y_l_ser[init_year:]
		if sec == 'prs':
			self.cou.g_y_l_trd[init_year:] = USA.g_y_l_trd[init_year:]
			self.cou.g_y_l_bss[init_year:] = USA.g_y_l_bss[init_year:]
			self.cou.g_y_l_fin[init_year:] = USA.g_y_l_fin[init_year:]
			self.cou.g_y_l_ser[init_year:] = USA.g_y_l_ser[init_year:]
		if sec == 'all':
			self.cou.g_y_l_agr[init_year:] = USA.g_y_l_agr[init_year:]
			self.cou.g_y_l_man[init_year:] = USA.g_y_l_man[init_year:]
			self.cou.g_y_l_trd[init_year:] = USA.g_y_l_trd[init_year:]
			self.cou.g_y_l_bss[init_year:] = USA.g_y_l_bss[init_year:]
			self.cou.g_y_l_fin[init_year:] = USA.g_y_l_fin[init_year:]
			self.cou.g_y_l_nps[init_year:] = USA.g_y_l_nps[init_year:]
			self.cou.g_y_l_ser[init_year:] = USA.g_y_l_ser[init_year:]

		'Generate counterfactual series'
		self.cou.productivity_series()

		'ams'
		self.cou.p_agr_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_agr)
		self.cou.p_ser_p_man = np.array(self.cou.A_ser)/np.array(self.cou.A_agr)
		self.cou.C_ams = self.cou.C_ams_ser*(np.array(self.cou.C_E_ams)/np.array(self.C_E_ams_baseline))
		self.C_ams = self.cou.C_ams
		self.cou.predictions_ams()
		
		'nps'
		self.cou.p_trd_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_trd)
		self.cou.p_bss_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_bss)
		self.cou.p_fin_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_fin)
		self.cou.p_nps_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_nps)
		self.cou.C_nps = self.cou.C_ams_ser*(np.array(self.cou.C_E_nps)/np.array(self.C_E_nps_baseline))
		self.C_nps = self.cou.C_nps
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
		'Baseline'
		self.baseline()

		if sec == 'agr':
			A_agr_catch = E_USA[-1]/np.array(self.cou.share_agr)[-1] - (np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1] + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1] + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1] + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_agr)[-1]
			g_y_l_agr_catch = (A_agr_catch/np.array(self.cou.A_agr)[0])**(1/self.cou.ts_length)-1
			g_catch = np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_agr_catch)
			self.cou.g_y_l_agr[init_year:] = g_catch[init_year:]
			self.A_agr_catch = A_agr_catch
			self.cou.A_agr=A_agr_catch
			self.g_y_l_agr_catch = g_y_l_agr_catch

		if sec == 'man':
			A_man_catch=E_USA[-1]/np.array(self.cou.share_man)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_man)[-1]
			g_y_l_man_catch=(A_man_catch/np.array(self.cou.A_man)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_man_catch)
			self.cou.g_y_l_man[init_year:] = g_catch[init_year:]
			self.A_man_catch=A_man_catch
			self.cou.A_man=A_man_catch 
			self.g_y_l_man_catch=g_y_l_man_catch			

		if sec == 'trd':
			A_trd_catch=E_USA[-1]/np.array(self.cou.share_trd)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_trd)[-1]
			g_y_l_trd_catch=(A_trd_catch/np.array(self.cou.A_trd)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_trd_catch)
			self.cou.g_y_l_trd[init_year:] = g_catch[init_year:]
			self.A_trd_catch=A_trd_catch
			self.cou.A_trd=A_trd_catch 
			self.g_y_l_trd_catch=g_y_l_trd_catch			

		if sec == 'bss':
			A_bss_catch=E_USA[-1]/np.array(self.cou.share_bss)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_bss)[-1]
			g_y_l_bss_catch=(A_bss_catch/np.array(self.cou.A_bss)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_bss_catch)
			self.cou.g_y_l_bss[init_year:] = g_catch[init_year:]
			self.A_bss_catch=A_bss_catch
			self.cou.A_bss=A_bss_catch
			self.g_y_l_bss_catch=g_y_l_bss_catch			

		if sec == 'fin':
			A_fin_catch=E_USA[-1]/np.array(self.cou.share_fin)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_fin)[-1]
			g_y_l_fin_catch=(A_fin_catch/np.array(self.cou.A_fin)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_fin_catch)
			self.cou.g_y_l_fin[init_year:] = g_catch[init_year:]
			self.A_fin_catch=A_fin_catch
			self.cou.A_fin=A_fin_catch
			self.g_y_l_fin_catch=g_y_l_fin_catch			

		if sec == 'nps':
			A_nps_catch=E_USA[-1]/np.array(self.cou.share_nps)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1])/np.array(self.cou.h_nps)[-1]
			g_y_l_nps_catch=(A_nps_catch/np.array(self.cou.A_nps)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_nps_catch)
			self.cou.g_y_l_nps[init_year:] = g_catch[init_year:]
			self.A_nps_catch=A_nps_catch
			self.cou.A_nps=A_nps_catch
			self.g_y_l_nps_catch=g_y_l_nps_catch			


		'Generate counterfactual series'
		self.cou.productivity_series()

#		'ams'
#		self.cou.p_ser_p_man = np.array(self.cou.A_man) / np.array(self.cou.A_ser)
#		self.cou.C_ams = self.cou.C_ams_ser * (np.array(self.cou.C_E_ams) / np.array(self.C_E_ams_baseline))
#		self.C_ams = self.cou.C_ams
#		self.cou.predictions_ams()
#
#		self.A_agr = self.cou.A_agr
#		self.A_man = self.cou.A_man
#		self.A_ser = self.cou.A_ser
#
#		self.share_agr_ams_m = self.cou.share_agr_ams_m
#		self.share_man_ams_m = self.cou.share_man_ams_m
#		self.share_ser_ams_m = self.cou.share_ser_ams_m
#	
#		self.A_tot_ams = self.cou.A_tot_ams

		'nps'
		self.cou.p_trd_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_trd)
		self.cou.p_bss_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_bss)
		self.cou.p_fin_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_fin)
		self.cou.p_nps_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_nps)
		self.cou.C_nps = self.cou.C_ams_ser*(np.array(self.cou.C_E_nps)/np.array(self.C_E_nps_baseline))
		self.C_nps = self.cou.C_nps

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
		'ams'
		self.ss_A_ams_init  = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_ser)[0]*self.cou.A_ser[-1]
		self.ss_A_ams  = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_ser*self.cou.A_ser

		'nps'
		self.ss_A_nps_init = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_trd)[0]*self.cou.A_trd[-1] + np.array(self.cou.share_bss)[0]*self.cou.A_bss[-1] + np.array(self.cou.share_fin)[0]*self.cou.A_fin[-1] + np.array(self.cou.share_nps)[0]*self.cou.A_nps[-1]
		self.ss_A_nps = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_trd*self.cou.A_trd + self.cou.share_bss*self.cou.A_bss + self.cou.share_fin*self.cou.A_fin + self.cou.share_nps*self.cou.A_nps


'''
----------------------------------------------------------------------------------------
	Counterfactual 1: Each sector keeping the pace with the US for the entire period
----------------------------------------------------------------------------------------
'''

'ams'
cf_1_ams_ss_init = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', 'EU13']]
cf_1_ams_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', 'EU13']]
cf_1_ams = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', 'EU13']]

for sec in ['agr', 'man', 'ser']:		
	AUT_cf=counterfactual('AUT')
	BEL_cf=counterfactual('BEL')
	DEU_cf=counterfactual('DEU')
	DNK_cf=counterfactual('DNK')
	ESP_cf=counterfactual('ESP')
	FIN_cf=counterfactual('FIN')
	FRA_cf=counterfactual('FRA')
	GBR_cf=counterfactual('GBR')
	GRC_cf=counterfactual('GRC')
	ITA_cf=counterfactual('ITA')
	NLD_cf=counterfactual('NLD')
	PRT_cf=counterfactual('PRT')
	SWE_cf=counterfactual('SWE')

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

	cf_1_sec_ams_init_ss = 	[sec,
				(AUT_cf.ss_A_ams_init/AUT_cf.ss_A_base_ams_init-1)*100,
				(BEL_cf.ss_A_ams_init/BEL_cf.ss_A_base_ams_init-1)*100,
				(DEU_cf.ss_A_ams_init/DEU_cf.ss_A_base_ams_init-1)*100,
				(DNK_cf.ss_A_ams_init/DNK_cf.ss_A_base_ams_init-1)*100,
				(ESP_cf.ss_A_ams_init/ESP_cf.ss_A_base_ams_init-1)*100,
				(FIN_cf.ss_A_ams_init/FIN_cf.ss_A_base_ams_init-1)*100,
				(FRA_cf.ss_A_ams_init/FRA_cf.ss_A_base_ams_init-1)*100,
				(GBR_cf.ss_A_ams_init/GBR_cf.ss_A_base_ams_init-1)*100,
				(GRC_cf.ss_A_ams_init/GRC_cf.ss_A_base_ams_init-1)*100,
				(ITA_cf.ss_A_ams_init/ITA_cf.ss_A_base_ams_init-1)*100,
				(NLD_cf.ss_A_ams_init/NLD_cf.ss_A_base_ams_init-1)*100,
				(PRT_cf.ss_A_ams_init/PRT_cf.ss_A_base_ams_init-1)*100,
				(SWE_cf.ss_A_ams_init/SWE_cf.ss_A_base_ams_init-1)*100,
				np.array([(np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_ams_init/DEU_cf.ss_A_base_ams_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_ams_init/FRA_cf.ss_A_base_ams_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_ams_init/ITA_cf.ss_A_base_ams_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_ams_init/GBR_cf.ss_A_base_ams_init-1)*100)/EUR4_h_tot[0]]),
				np.array([(np.array(AUT.h_tot)[0]*(AUT_cf.ss_A_ams_init/AUT_cf.ss_A_base_ams_init-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf.ss_A_ams_init/BEL_cf.ss_A_base_ams_init-1)*100 + np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_ams_init/DEU_cf.ss_A_base_ams_init-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf.ss_A_ams_init/DNK_cf.ss_A_base_ams_init-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf.ss_A_ams_init/ESP_cf.ss_A_base_ams_init-1)*100 + np.array(FIN.h_tot)[0]*(FIN_cf.ss_A_ams_init/FIN_cf.ss_A_base_ams_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_ams_init/FRA_cf.ss_A_base_ams_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_ams_init/GBR_cf.ss_A_base_ams_init-1)*100 + np.array(GRC.h_tot)[0]*(GRC_cf.ss_A_ams_init/GRC_cf.ss_A_base_ams_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_ams_init/ITA_cf.ss_A_base_ams_init-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf.ss_A_ams_init/NLD_cf.ss_A_base_ams_init-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf.ss_A_ams_init/PRT_cf.ss_A_base_ams_init-1)*100 + np.array(SWE.h_tot)[0]*(SWE_cf.ss_A_ams_init/SWE_cf.ss_A_base_ams_init-1)*100)/EUR13_h_tot[0]])]

	cf_1_sec_ams_init_ss[1:] = [ '%.1f' % elem for elem in cf_1_sec_ams_init_ss[1:] ]
	cf_1_ams_ss_init.append(cf_1_sec_ams_init_ss)

	cf_1_sec_ams_ss = 	[sec,
				(np.array(AUT_cf.ss_A_ams)[-1]/np.array(AUT_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(BEL_cf.ss_A_ams)[-1]/np.array(BEL_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(DEU_cf.ss_A_ams)[-1]/np.array(DEU_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(DNK_cf.ss_A_ams)[-1]/np.array(DNK_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(ESP_cf.ss_A_ams)[-1]/np.array(ESP_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(FIN_cf.ss_A_ams)[-1]/np.array(FIN_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(FRA_cf.ss_A_ams)[-1]/np.array(FRA_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(GBR_cf.ss_A_ams)[-1]/np.array(GBR_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(GRC_cf.ss_A_ams)[-1]/np.array(GRC_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(ITA_cf.ss_A_ams)[-1]/np.array(ITA_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(NLD_cf.ss_A_ams)[-1]/np.array(NLD_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(PRT_cf.ss_A_ams)[-1]/np.array(PRT_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(SWE_cf.ss_A_ams)[-1]/np.array(SWE_cf.ss_A_base_ams)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_ams)[-1]/np.array(DEU_cf.ss_A_base_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_ams)[-1]/np.array(FRA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_ams)[-1]/np.array(ITA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_ams)[-1]/np.array(GBR_cf.ss_A_base_ams)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.ss_A_ams)[-1]/np.array(AUT_cf.ss_A_base_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.ss_A_ams)[-1]/np.array(BEL_cf.ss_A_base_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_ams)[-1]/np.array(DEU_cf.ss_A_base_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.ss_A_ams)[-1]/np.array(DNK_cf.ss_A_base_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.ss_A_ams)[-1]/np.array(ESP_cf.ss_A_base_ams)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.ss_A_ams)[-1]/np.array(FIN_cf.ss_A_base_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_ams)[-1]/np.array(FRA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_ams)[-1]/np.array(GBR_cf.ss_A_base_ams)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.ss_A_ams)[-1]/np.array(GRC_cf.ss_A_base_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_ams)[-1]/np.array(ITA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.ss_A_ams)[-1]/np.array(NLD_cf.ss_A_base_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.ss_A_ams)[-1]/np.array(PRT_cf.ss_A_base_ams)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.ss_A_ams)[-1]/np.array(SWE_cf.ss_A_base_ams)[-1]-1)*100)/EUR13_h_tot[-1]])]
	
	cf_1_sec_ams_ss[1:] = [ '%.1f' % elem for elem in cf_1_sec_ams_ss[1:] ]
	cf_1_ams_ss.append(cf_1_sec_ams_ss)

	cf_1_sec_ams = 	[sec,
				(np.array(AUT_cf.A_tot_ams)[-1]/np.array(AUT.A_tot_ams)[-1]-1)*100,
				(np.array(BEL_cf.A_tot_ams)[-1]/np.array(BEL.A_tot_ams)[-1]-1)*100,
				(np.array(DEU_cf.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100,
				(np.array(DNK_cf.A_tot_ams)[-1]/np.array(DNK.A_tot_ams)[-1]-1)*100,
				(np.array(ESP_cf.A_tot_ams)[-1]/np.array(ESP.A_tot_ams)[-1]-1)*100,
				(np.array(FIN_cf.A_tot_ams)[-1]/np.array(FIN.A_tot_ams)[-1]-1)*100,
				(np.array(FRA_cf.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100,
				(np.array(GBR_cf.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100,
				(np.array(GRC_cf.A_tot_ams)[-1]/np.array(GRC.A_tot_ams)[-1]-1)*100,
				(np.array(ITA_cf.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100,
				(np.array(NLD_cf.A_tot_ams)[-1]/np.array(NLD.A_tot_ams)[-1]-1)*100,
				(np.array(PRT_cf.A_tot_ams)[-1]/np.array(PRT.A_tot_ams)[-1]-1)*100,
				(np.array(SWE_cf.A_tot_ams)[-1]/np.array(SWE.A_tot_ams)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.A_tot_ams)[-1]/np.array(AUT.A_tot_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.A_tot_ams)[-1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.A_tot_ams)[-1]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.A_tot_ams)[-1]/np.array(ESP.A_tot_ams)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.A_tot_ams)[-1]/np.array(FIN.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.A_tot_ams)[-1]/np.array(GRC.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.A_tot_ams)[-1]/np.array(NLD.A_tot_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.A_tot_ams)[-1]/np.array(PRT.A_tot_ams)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.A_tot_ams)[-1]/np.array(SWE.A_tot_ams)[-1]-1)*100)/EUR13_h_tot[-1]])]
	cf_1_sec_ams[1:] = [ '%.1f' % elem for elem in cf_1_sec_ams[1:] ]
	cf_1_ams.append(cf_1_sec_ams)

pd.DataFrame(cf_1_ams_ss_init).to_excel('../output/figures/Counterfactual_1_ams_ss_init_trade.xlsx', index = False, header = False)
pd.DataFrame(cf_1_ams_ss).to_excel('../output/figures/Counterfactual_1_ams_ss_trade.xlsx', index = False, header = False)
pd.DataFrame(cf_1_ams).to_excel('../output/figures/Counterfactual_1_ams_trade.xlsx', index = False, header = False)


'nps'
cf_1_nps_ss_init = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI','EU13']]
cf_1_nps_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI', 'EU13']]
cf_1_nps = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI', 'EU13']]

for sec in ['agr', 'man', 'trd', 'fin', 'bss', 'prs', 'nps']:
	AUT_cf=counterfactual('AUT')
	BEL_cf=counterfactual('BEL')
	DEU_cf=counterfactual('DEU')
	DNK_cf=counterfactual('DNK')
	ESP_cf=counterfactual('ESP')
	FIN_cf=counterfactual('FIN')
	FRA_cf=counterfactual('FRA')
	GBR_cf=counterfactual('GBR')
	GRC_cf=counterfactual('GRC')
	ITA_cf=counterfactual('ITA')
	NLD_cf=counterfactual('NLD')
	PRT_cf=counterfactual('PRT')
	SWE_cf=counterfactual('SWE')

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

	cf_1_sec_nps_init_ss = 	[sec,
				(AUT_cf.ss_A_nps_init/AUT_cf.ss_A_base_nps_init-1)*100,
				(BEL_cf.ss_A_nps_init/BEL_cf.ss_A_base_nps_init-1)*100,
				(DEU_cf.ss_A_nps_init/DEU_cf.ss_A_base_nps_init-1)*100,
				(DNK_cf.ss_A_nps_init/DNK_cf.ss_A_base_nps_init-1)*100,
				(ESP_cf.ss_A_nps_init/ESP_cf.ss_A_base_nps_init-1)*100,
				(FIN_cf.ss_A_nps_init/FIN_cf.ss_A_base_nps_init-1)*100,
				(FRA_cf.ss_A_nps_init/FRA_cf.ss_A_base_nps_init-1)*100,
				(GBR_cf.ss_A_nps_init/GBR_cf.ss_A_base_nps_init-1)*100,
				(GRC_cf.ss_A_nps_init/GRC_cf.ss_A_base_nps_init-1)*100,
				(ITA_cf.ss_A_nps_init/ITA_cf.ss_A_base_nps_init-1)*100,
				(NLD_cf.ss_A_nps_init/NLD_cf.ss_A_base_nps_init-1)*100,
				(PRT_cf.ss_A_nps_init/PRT_cf.ss_A_base_nps_init-1)*100,
				(SWE_cf.ss_A_nps_init/SWE_cf.ss_A_base_nps_init-1)*100,
				np.array([(np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_nps_init/DEU_cf.ss_A_base_nps_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_nps_init/FRA_cf.ss_A_base_nps_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_nps_init/ITA_cf.ss_A_base_nps_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_nps_init/GBR_cf.ss_A_base_nps_init-1)*100)/EUR4_h_tot[0]]),
			    np.array([(np.array(DEU.h_tot)[0] * (DEU_cf.ss_A_nps_init / DEU_cf.ss_A_base_nps_init - 1) * 100 + np.array(FRA.h_tot)[0] * (FRA_cf.ss_A_nps_init / FRA_cf.ss_A_base_nps_init - 1) * 100 + np.array(BEL.h_tot)[0] * (BEL_cf.ss_A_nps_init / BEL_cf.ss_A_base_nps_init - 1) * 100 + np.array(NLD.h_tot)[0] * (NLD_cf.ss_A_nps_init / NLD_cf.ss_A_base_nps_init - 1) * 100 + np.array(DNK.h_tot)[0] * (DNK_cf.ss_A_nps_init / DNK_cf.ss_A_base_nps_init - 1) * 100) / EURCORE_h_tot[0]]),
			    np.array([(np.array(GRC.h_tot)[0] * (GRC_cf.ss_A_nps_init / GRC_cf.ss_A_base_nps_init - 1) * 100 + np.array(PRT.h_tot)[0] * (PRT_cf.ss_A_nps_init / PRT_cf.ss_A_base_nps_init - 1) * 100 + np.array(ESP.h_tot)[0] * (ESP_cf.ss_A_nps_init / ESP_cf.ss_A_base_nps_init - 1) * 100 + np.array(ITA.h_tot)[0] * (ITA_cf.ss_A_nps_init / ITA_cf.ss_A_base_nps_init - 1) * 100 + np.array(GBR.h_tot)[0] * (GBR_cf.ss_A_nps_init / GBR_cf.ss_A_base_nps_init - 1) * 100) / EURPERI_h_tot[0]]),
				np.array([(np.array(AUT.h_tot)[0]*(AUT_cf.ss_A_nps_init/AUT_cf.ss_A_base_nps_init-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf.ss_A_nps_init/BEL_cf.ss_A_base_nps_init-1)*100 + np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_nps_init/DEU_cf.ss_A_base_nps_init-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf.ss_A_nps_init/DNK_cf.ss_A_base_nps_init-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf.ss_A_nps_init/ESP_cf.ss_A_base_nps_init-1)*100 + np.array(FIN.h_tot)[0]*(FIN_cf.ss_A_nps_init/FIN_cf.ss_A_base_nps_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_nps_init/FRA_cf.ss_A_base_nps_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_nps_init/GBR_cf.ss_A_base_nps_init-1)*100 + np.array(GRC.h_tot)[0]*(GRC_cf.ss_A_nps_init/GRC_cf.ss_A_base_nps_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_nps_init/ITA_cf.ss_A_base_nps_init-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf.ss_A_nps_init/NLD_cf.ss_A_base_nps_init-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf.ss_A_nps_init/PRT_cf.ss_A_base_nps_init-1)*100 + np.array(SWE.h_tot)[0]*(SWE_cf.ss_A_nps_init/SWE_cf.ss_A_base_nps_init-1)*100)/EUR13_h_tot[0]])]

	cf_1_sec_nps_init_ss[1:] = [ '%.1f' % elem for elem in cf_1_sec_nps_init_ss[1:] ]
	cf_1_nps_ss_init.append(cf_1_sec_nps_init_ss)

	cf_1_sec_nps_ss = 	[sec,
				(np.array(AUT_cf.ss_A_nps)[-1]/np.array(AUT_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(BEL_cf.ss_A_nps)[-1]/np.array(BEL_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(DNK_cf.ss_A_nps)[-1]/np.array(DNK_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(ESP_cf.ss_A_nps)[-1]/np.array(ESP_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(FIN_cf.ss_A_nps)[-1]/np.array(FIN_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(GRC_cf.ss_A_nps)[-1]/np.array(GRC_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(NLD_cf.ss_A_nps)[-1]/np.array(NLD_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(PRT_cf.ss_A_nps)[-1]/np.array(PRT_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(SWE_cf.ss_A_nps)[-1]/np.array(SWE_cf.ss_A_base_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.ss_A_nps)[-1]/np.array(BEL_cf.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.ss_A_nps)[-1]/np.array(NLD_cf.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.ss_A_nps)[-1]/np.array(DNK_cf.ss_A_base_nps)[-1]-1)*100)/EURCORE_h_tot[-1]]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.ss_A_nps)[-1]/np.array(GRC_cf.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.ss_A_nps)[-1]/np.array(PRT_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.ss_A_nps)[-1]/np.array(ESP_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.ss_A_nps)[-1]/np.array(AUT_cf.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.ss_A_nps)[-1]/np.array(BEL_cf.ss_A_base_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.ss_A_nps)[-1]/np.array(DNK_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.ss_A_nps)[-1]/np.array(ESP_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.ss_A_nps)[-1]/np.array(FIN_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.ss_A_nps)[-1]/np.array(GRC_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.ss_A_nps)[-1]/np.array(NLD_cf.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.ss_A_nps)[-1]/np.array(PRT_cf.ss_A_base_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.ss_A_nps)[-1]/np.array(SWE_cf.ss_A_base_nps)[-1]-1)*100)/EUR13_h_tot[-1]])]

	cf_1_sec_nps_ss[1:] = [ '%.1f' % elem for elem in cf_1_sec_nps_ss[1:] ]
	cf_1_nps_ss.append(cf_1_sec_nps_ss)

	cf_1_sec_nps = 	[sec,
				(np.array(AUT_cf.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100,
				(np.array(BEL_cf.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100,
				(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100,
				(np.array(DNK_cf.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100,
				(np.array(ESP_cf.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100,
				(np.array(FIN_cf.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100,
				(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100,
				(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100,
				(np.array(GRC_cf.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100,
				(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100,
				(np.array(NLD_cf.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100,
				(np.array(PRT_cf.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100,
				(np.array(SWE_cf.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100)/EURCORE_h_tot[-1]]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100)/EUR13_h_tot[-1]])]
	cf_1_sec_nps[1:] = [ '%.1f' % elem for elem in cf_1_sec_nps[1:] ]
	cf_1_nps.append(cf_1_sec_nps)

pd.DataFrame(cf_1_nps_ss_init).to_excel('../output/figures/Counterfactual_1_nps_ss_init_trade.xlsx', index = False, header = False)
pd.DataFrame(cf_1_nps_ss).to_excel('../output/figures/Counterfactual_1_nps_ss_trade.xlsx', index = False, header = False)
pd.DataFrame(cf_1_nps).to_excel('../output/figures/Counterfactual_1_nps_trade.xlsx', index = False, header = False)


'''
------------------------------------------------------------
	Counterfactual 2: Catch up Productivity with the US
------------------------------------------------------------
'''

'ams'
cf_2_catch_ams_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU13']]
cf_2_catch_ams = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU13']]
for sec in ['agr', 'man', 'ser']:
	AUT_cf2_catch=counterfactual('AUT')
	BEL_cf2_catch=counterfactual('BEL')
	DEU_cf2_catch=counterfactual('DEU')
	DNK_cf2_catch=counterfactual('DNK')
	ESP_cf2_catch=counterfactual('ESP')
	FIN_cf2_catch=counterfactual('FIN')
	FRA_cf2_catch=counterfactual('FRA')
	GBR_cf2_catch=counterfactual('GBR')
	GRC_cf2_catch=counterfactual('GRC')
	ITA_cf2_catch=counterfactual('ITA')
	NLD_cf2_catch=counterfactual('NLD')
	PRT_cf2_catch=counterfactual('PRT')
	SWE_cf2_catch=counterfactual('SWE')

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

	cf_2_catch_sec_ams_ss = 	[sec,
				(E_USA[-1]/np.array(AUT_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(BEL_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(FIN_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(SWE_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(E_USA[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(E_USA[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100)/(EURCORE_h_tot[-1] -np.array(BEL.h_tot)[-1])]),
				np.array([(np.array(GRC.h_tot)[-1]*(E_USA[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(E_USA[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(E_USA[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(E_USA[-1]/np.array(AUT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(E_USA[-1]/np.array(BEL_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(E_USA[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(E_USA[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(E_USA[-1]/np.array(FIN_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(E_USA[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(E_USA[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(E_USA[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(E_USA[-1]/np.array(SWE_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR13_h_tot[-1]])]

	cf_2_catch_sec_ams_ss[1:] = [ '%.1f' % elem for elem in cf_2_catch_sec_ams_ss[1:] ]
	cf_2_catch_ams_ss.append(cf_2_catch_sec_ams_ss)

	cf_2_catch_sec_ams = [sec,
				(np.array(AUT_cf2_catch.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100,
				(np.array(BEL_cf2_catch.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100,
				(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100,
				(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100,
				(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100,
				(np.array(FIN_cf2_catch.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100,
				(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100,
				(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100,
				(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100,
				(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100,
				(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100,
				(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100,
				(np.array(SWE_cf2_catch.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100)/(EURCORE_h_tot[-1] -np.array(BEL.h_tot)[-1])]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2_catch.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2_catch.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2_catch.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100)/(EUR13_h_tot[-1]-np.array(BEL.h_tot)[-1])])]

	cf_2_catch_sec_ams[1:] = [ '%.1f' % elem for elem in cf_2_catch_sec_ams[1:] ]
	cf_2_catch_ams.append(cf_2_catch_sec_ams)

pd.DataFrame(cf_2_catch_ams_ss).to_excel('../output/figures/Counterfactual_2_catch_ams_ss_trade.xlsx', index=False, header=False)
pd.DataFrame(cf_2_catch_ams).to_excel('../output/figures/Counterfactual_2_catch_ams_trade.xlsx', index=False, header=False)


'nps'
cf_2_catch_nps_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU13']]
cf_2_catch_nps = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'ITA', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU13']]
for sec in ['agr', 'man', 'trd', 'fin', 'bss', 'nps']:
	AUT_cf2_catch=counterfactual('AUT')
	BEL_cf2_catch=counterfactual('BEL')
	DEU_cf2_catch=counterfactual('DEU')
	DNK_cf2_catch=counterfactual('DNK')
	ESP_cf2_catch=counterfactual('ESP')
	FIN_cf2_catch=counterfactual('FIN')
	FRA_cf2_catch=counterfactual('FRA')
	GBR_cf2_catch=counterfactual('GBR')
	GRC_cf2_catch=counterfactual('GRC')
	ITA_cf2_catch=counterfactual('ITA')
	NLD_cf2_catch=counterfactual('NLD')
	PRT_cf2_catch=counterfactual('PRT')
	SWE_cf2_catch=counterfactual('SWE')

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

	cf_2_catch_sec_nps_ss = 	[sec,
				(E_USA[-1]/np.array(AUT_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(BEL_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(FIN_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E_USA[-1]/np.array(SWE_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(E_USA[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(E_USA[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100)/(EURCORE_h_tot[-1] -np.array(BEL.h_tot)[-1])]),
				np.array([(np.array(GRC.h_tot)[-1]*(E_USA[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(E_USA[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(E_USA[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(E_USA[-1]/np.array(AUT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(E_USA[-1]/np.array(BEL_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(E_USA[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(E_USA[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(E_USA[-1]/np.array(FIN_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(E_USA[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(E_USA[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(E_USA[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(E_USA[-1]/np.array(SWE_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR13_h_tot[-1]])]

	cf_2_catch_sec_nps_ss[1:] = [ '%.1f' % elem for elem in cf_2_catch_sec_nps_ss[1:] ]
	cf_2_catch_nps_ss.append(cf_2_catch_sec_nps_ss)

	cf_2_catch_sec_nps = [sec,
				(np.array(AUT_cf2_catch.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100,
				(np.array(BEL_cf2_catch.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100,
				(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100,
				(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100,
				(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100,
				(np.array(FIN_cf2_catch.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100,
				(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100,
				(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100,
				(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100,
				(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100,
				(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100,
				(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100,
				(np.array(SWE_cf2_catch.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100)/(EURCORE_h_tot[-1] -np.array(BEL.h_tot)[-1])]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2_catch.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2_catch.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2_catch.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100)/(EUR13_h_tot[-1]-np.array(BEL.h_tot)[-1])])]

	cf_2_catch_sec_nps[1:] = [ '%.1f' % elem for elem in cf_2_catch_sec_nps[1:] ]
	cf_2_catch_nps.append(cf_2_catch_sec_nps)


pd.DataFrame(cf_2_catch_nps_ss).to_excel('../output/figures/Counterfactual_2_catch_nps_ss_trade.xlsx', index=False, header=False)
pd.DataFrame(cf_2_catch_nps).to_excel('../output/figures/Counterfactual_2_catch_nps_trade.xlsx', index=False, header=False)


'Germany'
for sec in ['agr', 'man', 'trd', 'fin', 'bss', 'nps']:
	DEU_cf2_catch_trade=counterfactual('DEU')
	DEU_cf2_catch_trade.baseline()
	DEU_cf2_catch_trade.feed_catch_up_growth(0, sec)

	A_agr = np.array(DEU_cf2_catch_trade.A_agr)[-1]
	A_man = np.array(DEU_cf2_catch_trade.A_man)[-1]
	A_trd = np.array(DEU_cf2_catch_trade.A_trd)[-1]
	A_fin = np.array(DEU_cf2_catch_trade.A_fin)[-1]
	A_bss = np.array(DEU_cf2_catch_trade.A_bss)[-1]
	A_nps = np.array(DEU_cf2_catch_trade.A_nps)[-1]
	
	nx_agr_E = np.array(DEU.nx_agr_E)[-1]
	nx_man_E = np.array(DEU.nx_man_E)[-1]
	nx_trd_E = np.array(DEU.nx_trd_E)[-1]
	nx_fin_E = np.array(DEU.nx_fin_E)[-1]
	nx_bss_E = np.array(DEU.nx_bss_E)[-1]
	nx_nps_E = np.array(DEU.nx_nps_E)[-1]
	
	C=DEU_cf2_catch_trade.C_nps[-1]

	weight_agr = DEU.om_agr_nps*(A_agr**(sigma-1))*(C**eps_agr) 
	weight_man = DEU.om_man_nps*(A_man**(sigma-1))*C 
	weight_trd = DEU.om_trd_nps*(A_trd**(sigma-1))*(C**eps_trd)
	weight_bss = DEU.om_bss_nps*(A_bss**(sigma-1))*(C**eps_bss)
	weight_fin = DEU.om_fin_nps*(A_fin**(sigma-1))*(C**eps_fin)
	weight_nps = (1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(A_nps**(sigma-1))*(C**eps_nps)
	
	E = (weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps)**(1/(1-sigma))

	if sec == 'agr':
		def find_nx(NX_CF):
			L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, 0)
		print('Counterfactual trade in agr is ' + str(nx_cf) + ' while current net trade is ' +str(nx_agr_E))
		print('Counterfactual trade in agr is ' + str(nx_cf/nx_agr_E) + ' fold')
		print('Counterfactual productivity in agr is ' + str(A_agr) + ' fold')
		print('Current productivity in agr is ' + str(DEU.A_agr[-1]) + ' fold')
		DEU.nx_cf_agr = nx_cf

	if sec == 'man':
		def find_nx(NX_CF):
			L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_man_E)
		print('Counterfactual trade in man is ' +str(nx_cf) + ' while current net trade is ' +str(nx_man_E))
		print('Counterfactual trade in man is ' +str(nx_cf/nx_man_E) + ' fold')
		print('Counterfactual productivity in man is ' + str(A_man) + ' fold')
		print('Current productivity in man is ' + str(DEU.A_man[-1]) + ' fold')
		DEU.nx_cf_man = nx_cf

	if sec == 'trd':
		def find_nx(NX_CF):
			L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_trd_E)
		print('Counterfactual trade in trd is ' +str(nx_cf) + ' while current net trade is ' +str(nx_trd_E))
		print('Counterfactual trade in trd is ' +str(nx_cf/nx_trd_E) + ' fold')
		print('Counterfactual productivity in trd is ' + str(A_trd) + ' fold')
		print('Current productivity in trd is ' + str(DEU.A_trd[-1]) + ' fold')
		DEU.nx_cf_trd = nx_cf

	if sec == 'bss':
		def find_nx(NX_CF):
			L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_bss_E)
		print('Counterfactual trade in bss is ' +str(nx_cf) + ' while current net trade is ' +str(nx_bss_E))
		print('Counterfactual trade in bss is ' +str(nx_cf/nx_bss_E) + ' fold')
		print('Counterfactual productivity in bss is ' + str(A_bss) + ' fold')
		print('Current productivity in bss is ' + str(DEU.A_bss[-1]) + ' fold')
		DEU.nx_cf_bss = nx_cf

	if sec == 'fin':
		def find_nx(NX_CF):
			L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_fin_E)
		print('Counterfactual trade in fin is ' +str(nx_cf) + ' while current net trade is ' +str(nx_fin_E))
		print('Counterfactual trade in fin is ' +str(nx_cf/nx_fin_E) + ' fold')
		print('Counterfactual productivity in fin is ' + str(A_fin) + ' fold')
		print('Current productivity in fin is ' + str(DEU.A_fin[-1]) + ' fold')
		DEU.nx_cf_fin = nx_cf

	if sec == 'nps':
		def find_nx(NX_CF):
			L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*NX_CF)
		
			share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_nps_E)
		print('Counterfactual trade in nps is ' +str(nx_cf) + ' while current net trade is ' +str(nx_nps_E))
		print('Counterfactual trade in nps is ' +str(nx_cf/nx_nps_E) + ' fold')
		print('Counterfactual productivity in nps is ' + str(A_nps) + ' fold')
		print('Current productivity in nps is ' + str(DEU.A_nps[-1]) + ' fold')
		DEU.nx_cf_nps = nx_cf

'France'
for sec in ['agr', 'man', 'trd', 'fin', 'bss', 'nps']:
	FRA_cf2_catch_trade=counterfactual('FRA')
	FRA_cf2_catch_trade.baseline()
	FRA_cf2_catch_trade.feed_catch_up_growth(0, sec)

	A_agr = np.array(FRA_cf2_catch_trade.A_agr)[-1]
	A_man = np.array(FRA_cf2_catch_trade.A_man)[-1]
	A_trd = np.array(FRA_cf2_catch_trade.A_trd)[-1]
	A_fin = np.array(FRA_cf2_catch_trade.A_fin)[-1]
	A_bss = np.array(FRA_cf2_catch_trade.A_bss)[-1]
	A_nps = np.array(FRA_cf2_catch_trade.A_nps)[-1]
	
	nx_agr_E = np.array(FRA.nx_agr_E)[-1]
	nx_man_E = np.array(FRA.nx_man_E)[-1]
	nx_trd_E = np.array(FRA.nx_trd_E)[-1]
	nx_fin_E = np.array(FRA.nx_fin_E)[-1]
	nx_bss_E = np.array(FRA.nx_bss_E)[-1]
	nx_nps_E = np.array(FRA.nx_nps_E)[-1]
	
	C=FRA_cf2_catch_trade.C_nps[-1]

	weight_agr = FRA.om_agr_nps*(A_agr**(sigma-1))*(C**eps_agr) 
	weight_man = FRA.om_man_nps*(A_man**(sigma-1))*C 
	weight_trd = FRA.om_trd_nps*(A_trd**(sigma-1))*(C**eps_trd)
	weight_bss = FRA.om_bss_nps*(A_bss**(sigma-1))*(C**eps_bss)
	weight_fin = FRA.om_fin_nps*(A_fin**(sigma-1))*(C**eps_fin)
	weight_nps = (1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(A_nps**(sigma-1))*(C**eps_nps)
	
	E = (weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps)**(1/(1-sigma))

	if sec == 'agr':
		def find_nx(NX_CF):
			L = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 FRA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_man = (FRA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, 0)
		print('Counterfactual trade in agr is ' + str(nx_cf) + ' while current net trade is ' +str(nx_agr_E))
		print('Counterfactual trade in agr is ' + str(nx_cf/nx_agr_E) + ' fold')
		print('Counterfactual productivity in agr is ' + str(A_agr) + ' fold')
		print('Current productivity in agr is ' + str(FRA.A_agr[-1]) + ' fold')
		FRA.nx_cf_agr = nx_cf

	if sec == 'man':
		def find_nx(NX_CF):
			L = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 FRA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (FRA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_trd = (FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_man_E)
		print('Counterfactual trade in man is ' +str(nx_cf) + ' while current net trade is ' +str(nx_man_E))
		print('Counterfactual trade in man is ' +str(nx_cf/nx_man_E) + ' fold')
		print('Counterfactual productivity in man is ' + str(A_man) + ' fold')
		print('Current productivity in man is ' + str(FRA.A_man[-1]) + ' fold')
		FRA.nx_cf_man = nx_cf

	if sec == 'trd':
		def find_nx(NX_CF):
			L = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 FRA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (FRA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_bss = (FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_trd_E)
		print('Counterfactual trade in trd is ' +str(nx_cf) + ' while current net trade is ' +str(nx_trd_E))
		print('Counterfactual trade in trd is ' +str(nx_cf/nx_trd_E) + ' fold')
		print('Counterfactual productivity in trd is ' + str(A_trd) + ' fold')
		print('Current productivity in trd is ' + str(FRA.A_trd[-1]) + ' fold')
		FRA.nx_cf_trd = nx_cf

	if sec == 'bss':
		def find_nx(NX_CF):
			L = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 FRA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (FRA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_fin = (FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_bss_E)
		print('Counterfactual trade in bss is ' +str(nx_cf) + ' while current net trade is ' +str(nx_bss_E))
		print('Counterfactual trade in bss is ' +str(nx_cf/nx_bss_E) + ' fold')
		print('Counterfactual productivity in bss is ' + str(A_bss) + ' fold')
		print('Current productivity in bss is ' + str(FRA.A_bss[-1]) + ' fold')
		FRA.nx_cf_bss = nx_cf

	if sec == 'fin':
		def find_nx(NX_CF):
			L = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 FRA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			(1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (FRA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_nps = ((1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_fin_E)
		print('Counterfactual trade in fin is ' +str(nx_cf) + ' while current net trade is ' +str(nx_fin_E))
		print('Counterfactual trade in fin is ' +str(nx_cf/nx_fin_E) + ' fold')
		print('Counterfactual productivity in fin is ' + str(A_fin) + ' fold')
		print('Current productivity in fin is ' + str(FRA.A_fin[-1]) + ' fold')
		FRA.nx_cf_fin = nx_cf

	if sec == 'nps':
		def find_nx(NX_CF):
			L = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 FRA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*NX_CF)
		
			share_agr = (FRA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (FRA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (FRA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (FRA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (FRA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-FRA.om_agr_nps-FRA.om_man_nps-FRA.om_trd_nps-FRA.om_bss_nps-FRA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_nps_E)
		print('Counterfactual trade in nps is ' +str(nx_cf) + ' while current net trade is ' +str(nx_nps_E))
		print('Counterfactual trade in nps is ' +str(nx_cf/nx_nps_E) + ' fold')
		print('Counterfactual productivity in nps is ' + str(A_nps) + ' fold')
		print('Current productivity in nps is ' + str(FRA.A_nps[-1]) + ' fold')
		FRA.nx_cf_nps = nx_cf
		
'Great Britain'
for sec in ['agr', 'man', 'trd', 'fin', 'bss', 'nps']:
	GBR_cf2_catch_trade=counterfactual('GBR')
	GBR_cf2_catch_trade.baseline()
	GBR_cf2_catch_trade.feed_catch_up_growth(0, sec)

	A_agr = np.array(GBR_cf2_catch_trade.A_agr)[-1]
	A_man = np.array(GBR_cf2_catch_trade.A_man)[-1]
	A_trd = np.array(GBR_cf2_catch_trade.A_trd)[-1]
	A_fin = np.array(GBR_cf2_catch_trade.A_fin)[-1]
	A_bss = np.array(GBR_cf2_catch_trade.A_bss)[-1]
	A_nps = np.array(GBR_cf2_catch_trade.A_nps)[-1]
	
	nx_agr_E = np.array(GBR.nx_agr_E)[-1]
	nx_man_E = np.array(GBR.nx_man_E)[-1]
	nx_trd_E = np.array(GBR.nx_trd_E)[-1]
	nx_fin_E = np.array(GBR.nx_fin_E)[-1]
	nx_bss_E = np.array(GBR.nx_bss_E)[-1]
	nx_nps_E = np.array(GBR.nx_nps_E)[-1]
	
	C=GBR_cf2_catch_trade.C_nps[-1]

	weight_agr = GBR.om_agr_nps*(A_agr**(sigma-1))*(C**eps_agr) 
	weight_man = GBR.om_man_nps*(A_man**(sigma-1))*C 
	weight_trd = GBR.om_trd_nps*(A_trd**(sigma-1))*(C**eps_trd)
	weight_bss = GBR.om_bss_nps*(A_bss**(sigma-1))*(C**eps_bss)
	weight_fin = GBR.om_fin_nps*(A_fin**(sigma-1))*(C**eps_fin)
	weight_nps = (1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(A_nps**(sigma-1))*(C**eps_nps)
	
	E = (weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps)**(1/(1-sigma))

	if sec == 'agr':
		def find_nx(NX_CF):
			L = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 GBR.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_man = (GBR.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, 0)
		print('Counterfactual trade in agr is ' + str(nx_cf) + ' while current net trade is ' +str(nx_agr_E))
		print('Counterfactual trade in agr is ' + str(nx_cf/nx_agr_E) + ' fold')
		print('Counterfactual productivity in agr is ' + str(A_agr) + ' fold')
		print('Current productivity in agr is ' + str(GBR.A_agr[-1]) + ' fold')
		GBR.nx_cf_agr = nx_cf

	if sec == 'man':
		def find_nx(NX_CF):
			L = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 GBR.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (GBR.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_trd = (GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_man_E)
		print('Counterfactual trade in man is ' +str(nx_cf) + ' while current net trade is ' +str(nx_man_E))
		print('Counterfactual trade in man is ' +str(nx_cf/nx_man_E) + ' fold')
		print('Counterfactual productivity in man is ' + str(A_man) + ' fold')
		print('Current productivity in man is ' + str(GBR.A_man[-1]) + ' fold')
		GBR.nx_cf_man = nx_cf

	if sec == 'trd':
		def find_nx(NX_CF):
			L = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 GBR.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (GBR.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_bss = (GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_trd_E)
		print('Counterfactual trade in trd is ' +str(nx_cf) + ' while current net trade is ' +str(nx_trd_E))
		print('Counterfactual trade in trd is ' +str(nx_cf/nx_trd_E) + ' fold')
		print('Counterfactual productivity in trd is ' + str(A_trd) + ' fold')
		print('Current productivity in trd is ' + str(GBR.A_trd[-1]) + ' fold')
		GBR.nx_cf_trd = nx_cf

	if sec == 'bss':
		def find_nx(NX_CF):
			L = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 GBR.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (GBR.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_fin = (GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_bss_E)
		print('Counterfactual trade in bss is ' +str(nx_cf) + ' while current net trade is ' +str(nx_bss_E))
		print('Counterfactual trade in bss is ' +str(nx_cf/nx_bss_E) + ' fold')
		print('Counterfactual productivity in bss is ' + str(A_bss) + ' fold')
		print('Current productivity in bss is ' + str(GBR.A_bss[-1]) + ' fold')
		GBR.nx_cf_bss = nx_cf

	if sec == 'fin':
		def find_nx(NX_CF):
			L = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 GBR.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			(1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (GBR.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_nps = ((1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_fin_E)
		print('Counterfactual trade in fin is ' +str(nx_cf) + ' while current net trade is ' +str(nx_fin_E))
		print('Counterfactual trade in fin is ' +str(nx_cf/nx_fin_E) + ' fold')
		print('Counterfactual productivity in fin is ' + str(A_fin) + ' fold')
		print('Current productivity in fin is ' + str(GBR.A_fin[-1]) + ' fold')
		GBR.nx_cf_fin = nx_cf

	if sec == 'nps':
		def find_nx(NX_CF):
			L = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 GBR.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*NX_CF)
		
			share_agr = (GBR.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (GBR.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (GBR.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (GBR.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (GBR.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-GBR.om_agr_nps-GBR.om_man_nps-GBR.om_trd_nps-GBR.om_bss_nps-GBR.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_nps_E)
		print('Counterfactual trade in nps is ' +str(nx_cf) + ' while current net trade is ' +str(nx_nps_E))
		print('Counterfactual trade in nps is ' +str(nx_cf/nx_nps_E) + ' fold')
		print('Counterfactual productivity in nps is ' + str(A_nps) + ' fold')
		print('Current productivity in nps is ' + str(GBR.A_nps[-1]) + ' fold')
		GBR.nx_cf_nps = nx_cf
'Italy'
for sec in ['agr', 'man', 'trd', 'fin', 'bss', 'nps']:
	ITA_cf2_catch_trade=counterfactual('ITA')
	ITA_cf2_catch_trade.baseline()
	ITA_cf2_catch_trade.feed_catch_up_growth(0, sec)

	A_agr = np.array(ITA_cf2_catch_trade.A_agr)[-1]
	A_man = np.array(ITA_cf2_catch_trade.A_man)[-1]
	A_trd = np.array(ITA_cf2_catch_trade.A_trd)[-1]
	A_fin = np.array(ITA_cf2_catch_trade.A_fin)[-1]
	A_bss = np.array(ITA_cf2_catch_trade.A_bss)[-1]
	A_nps = np.array(ITA_cf2_catch_trade.A_nps)[-1]
	
	nx_agr_E = np.array(ITA.nx_agr_E)[-1]
	nx_man_E = np.array(ITA.nx_man_E)[-1]
	nx_trd_E = np.array(ITA.nx_trd_E)[-1]
	nx_fin_E = np.array(ITA.nx_fin_E)[-1]
	nx_bss_E = np.array(ITA.nx_bss_E)[-1]
	nx_nps_E = np.array(ITA.nx_nps_E)[-1]
	
	C=ITA_cf2_catch_trade.C_nps[-1]

	weight_agr = ITA.om_agr_nps*(A_agr**(sigma-1))*(C**eps_agr) 
	weight_man = ITA.om_man_nps*(A_man**(sigma-1))*C 
	weight_trd = ITA.om_trd_nps*(A_trd**(sigma-1))*(C**eps_trd)
	weight_bss = ITA.om_bss_nps*(A_bss**(sigma-1))*(C**eps_bss)
	weight_fin = ITA.om_fin_nps*(A_fin**(sigma-1))*(C**eps_fin)
	weight_nps = (1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(A_nps**(sigma-1))*(C**eps_nps)
	
	E = (weight_agr + weight_man + weight_trd + weight_bss + weight_fin + weight_nps)**(1/(1-sigma))

	if sec == 'agr':
		def find_nx(NX_CF):
			L = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 ITA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_man = (ITA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, 0)
		print('Counterfactual trade in agr is ' + str(nx_cf) + ' while current net trade is ' +str(nx_agr_E))
		print('Counterfactual trade in agr is ' + str(nx_cf/nx_agr_E) + ' fold')
		print('Counterfactual productivity in agr is ' + str(A_agr) + ' fold')
		print('Current productivity in agr is ' + str(ITA.A_agr[-1]) + ' fold')
		ITA.nx_cf_agr = nx_cf

	if sec == 'man':
		def find_nx(NX_CF):
			L = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 ITA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (ITA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_trd = (ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_man_E)
		print('Counterfactual trade in man is ' +str(nx_cf) + ' while current net trade is ' +str(nx_man_E))
		print('Counterfactual trade in man is ' +str(nx_cf/nx_man_E) + ' fold')
		print('Counterfactual productivity in man is ' + str(A_man) + ' fold')
		print('Current productivity in man is ' + str(ITA.A_man[-1]) + ' fold')
		ITA.nx_cf_man = nx_cf

	if sec == 'trd':
		def find_nx(NX_CF):
			L = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 ITA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (ITA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_bss = (ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_trd_E)
		print('Counterfactual trade in trd is ' +str(nx_cf) + ' while current net trade is ' +str(nx_trd_E))
		print('Counterfactual trade in trd is ' +str(nx_cf/nx_trd_E) + ' fold')
		print('Counterfactual productivity in trd is ' + str(A_trd) + ' fold')
		print('Current productivity in trd is ' + str(ITA.A_trd[-1]) + ' fold')
		ITA.nx_cf_trd = nx_cf

	if sec == 'bss':
		def find_nx(NX_CF):
			L = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 ITA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			 ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (ITA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_fin = (ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_bss_E)
		print('Counterfactual trade in bss is ' +str(nx_cf) + ' while current net trade is ' +str(nx_bss_E))
		print('Counterfactual trade in bss is ' +str(nx_cf/nx_bss_E) + ' fold')
		print('Counterfactual productivity in bss is ' + str(A_bss) + ' fold')
		print('Current productivity in bss is ' + str(ITA.A_bss[-1]) + ' fold')
		ITA.nx_cf_bss = nx_cf

	if sec == 'fin':
		def find_nx(NX_CF):
			L = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 ITA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*NX_CF +
			(1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)
		
			share_agr = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (ITA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
			share_nps = ((1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_fin_E)
		print('Counterfactual trade in fin is ' +str(nx_cf) + ' while current net trade is ' +str(nx_fin_E))
		print('Counterfactual trade in fin is ' +str(nx_cf/nx_fin_E) + ' fold')
		print('Counterfactual productivity in fin is ' + str(A_fin) + ' fold')
		print('Current productivity in fin is ' + str(ITA.A_fin[-1]) + ' fold')
		ITA.nx_cf_fin = nx_cf

	if sec == 'nps':
		def find_nx(NX_CF):
			L = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
			 ITA.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
			 ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
			 ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
			 ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
			(1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*NX_CF)
		
			share_agr = (ITA.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
			share_man = (ITA.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
			share_trd = (ITA.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
			share_bss = (ITA.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
			share_fin = (ITA.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
			share_nps = ((1-ITA.om_agr_nps-ITA.om_man_nps-ITA.om_trd_nps-ITA.om_bss_nps-ITA.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*NX_CF)/L
		
			return E_USA[-1] - (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_bss + share_fin*A_fin + share_nps*A_nps)
		nx_cf = fsolve(find_nx, nx_nps_E)
		print('Counterfactual trade in nps is ' +str(nx_cf) + ' while current net trade is ' +str(nx_nps_E))
		print('Counterfactual trade in nps is ' +str(nx_cf/nx_nps_E) + ' fold')
		print('Counterfactual productivity in nps is ' + str(A_nps) + ' fold')
		print('Current productivity in nps is ' + str(ITA.A_nps[-1]) + ' fold')
		ITA.nx_cf_nps = nx_cf



EUR4_nx_agr_E = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_agr_E)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_agr_E)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_agr_E)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_agr_E)[-1])/EUR4_h_tot[-1]
EUR4_nx_man_E = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_man_E)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_man_E)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_man_E)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_man_E)[-1])/EUR4_h_tot[-1]
EUR4_nx_trd_E = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_trd_E)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_trd_E)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_trd_E)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_trd_E)[-1])/EUR4_h_tot[-1]
EUR4_nx_bss_E = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_bss_E)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_bss_E)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_bss_E)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_bss_E)[-1])/EUR4_h_tot[-1]
EUR4_nx_fin_E = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_fin_E)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_fin_E)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_fin_E)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_fin_E)[-1])/EUR4_h_tot[-1]
EUR4_nx_nps_E = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_nps_E)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_nps_E)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_nps_E)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_nps_E)[-1])/EUR4_h_tot[-1]

EUR4_nx_cf_agr = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_cf_agr)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_cf_agr)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_cf_agr)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_cf_agr)[-1])/EUR4_h_tot[-1]
EUR4_nx_cf_man = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_cf_man)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_cf_man)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_cf_man)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_cf_man)[-1])/EUR4_h_tot[-1]
EUR4_nx_cf_trd = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_cf_trd)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_cf_trd)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_cf_trd)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_cf_trd)[-1])/EUR4_h_tot[-1]
EUR4_nx_cf_bss = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_cf_bss)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_cf_bss)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_cf_bss)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_cf_bss)[-1])/EUR4_h_tot[-1]
EUR4_nx_cf_fin = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_cf_fin)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_cf_fin)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_cf_fin)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_cf_fin)[-1])/EUR4_h_tot[-1]
EUR4_nx_cf_nps = (np.array(DEU.h_tot)[-1]*np.array(DEU.nx_cf_nps)[-1] + np.array(GBR.h_tot)[-1]*np.array(GBR.nx_cf_nps)[-1] + np.array(FRA.h_tot)[-1]*np.array(FRA.nx_cf_nps)[-1] + np.array(ITA.h_tot)[-1]*np.array(ITA.nx_cf_nps)[-1])/EUR4_h_tot[-1]

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

ss=np.array([(np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]])
ss_base=np.array([(np.array(DEU.h_tot)[-1]*(E_USA[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E_USA[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E_USA[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E_USA[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]])
cf=np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]])
cf_base=np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]])

"""

L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_E +
 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)


share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_E)/L
share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L


L = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + (E**(1-sigma))*nx_agr_cf +
 DEU.om_man_nps*C*(A_man**(sigma - 1)) + (E**(1-sigma))*nx_man_E +
 DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + (E**(1-sigma))*nx_trd_E +
 DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + (E**(1-sigma))*nx_bss_E +
 DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + (E**(1-sigma))*nx_fin_E +
(1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + (E**(1-sigma))*nx_nps_E)


share_agr = (DEU.om_agr_nps*(C**eps_agr)*(A_agr**(sigma - 1)) + E**(1-sigma)*nx_agr_cf)/L
share_man = (DEU.om_man_nps*C*(A_man**(sigma - 1)) + E**(1-sigma)*nx_man_E)/L
share_trd = (DEU.om_trd_nps*(C**eps_trd)*(A_trd**(sigma - 1)) + E**(1-sigma)*nx_trd_E)/L
share_bss = (DEU.om_bss_nps*(C**eps_bss)*(A_bss**(sigma - 1)) + E**(1-sigma)*nx_bss_E)/L
share_fin = (DEU.om_fin_nps*(C**eps_fin)*(A_fin**(sigma - 1)) + E**(1-sigma)*nx_fin_E)/L
share_nps = ((1-DEU.om_agr_nps-DEU.om_man_nps-DEU.om_trd_nps-DEU.om_bss_nps-DEU.om_fin_nps)*(C**eps_nps)*(A_nps**(sigma - 1)) + E**(1-sigma)*nx_nps_E)/L

x = (share_agr*A_agr + share_man*A_man + share_trd*A_trd + share_bss*A_fin + share_fin*A_bss + share_nps*A_nps)
"""