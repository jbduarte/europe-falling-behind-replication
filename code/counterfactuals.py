"""
=======================================================================================
Project: Structural Transformation and Productivity in Europe (with Duarte and Saenz)
Filename: counterfactual.py
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

# The following string runs the calibration of the model on the US, and imports the parameter values that are used in this program. Also, it imports US data and model predictions, needed in the European sectoral productivity measurement.
from model_calibration_USA import sigma, eps_agr, eps_ser, eps_trd, eps_bss, eps_fin, eps_nps, GDP, E, A_agr, A_man, A_trd, A_bss, A_fin, A_nps, A_ser, A_tot, A_tot_ams, A_tot_nps, A_tot_ams_weighted, A_tot_nps_weighted

#The following string runs the measurement of productivity in Europe recovering the initial levels with our model.
from model_test_europe import model_country, EUR4_h_tot, EURCORE_h_tot, EURPERI_h_tot,  EUR15_h_tot, EUR4_A_tot, EURCORE_A_tot, EURPERI_A_tot, EUR15_A_tot, EUR4_rel_A_tot, EUR15_rel_A_tot, EUR4_E, EUR15_E, EUR4_rel_E, EUR15_rel_E, \
	EUR4_share_agr, EUR15_share_agr, EUR4_share_man, EUR15_share_man, EUR4_share_ser, EUR15_share_ser, EUR4_share_trd, EUR15_share_trd, EUR4_share_bss, EUR15_share_bss, EUR4_share_fin, EUR15_share_fin, EUR4_share_nps, EUR15_share_nps, \
	EUR4_share_agr_ams_m, EUR15_share_agr_ams_m, EUR4_share_agr_nps_m, EUR15_share_agr_nps_m, EUR4_share_man_ams_m, EUR15_share_man_ams_m, EUR4_share_man_nps_m, EUR15_share_man_nps_m, EUR4_share_ser_ams_m, EUR15_share_ser_ams_m, EUR4_share_trd_nps_m, EUR15_share_trd_nps_m, EUR4_share_bss_nps_m, EUR15_share_bss_nps_m, EUR4_share_fin_nps_m, EUR15_share_fin_nps_m, EUR4_share_nps_nps_m, EUR15_share_nps_nps_m, \
	EUR4_A_tot_ams, EUR15_A_tot_ams, EUR4_A_tot_nps, EURCORE_A_tot_nps, EURPERI_A_tot_nps, EUR15_A_tot_nps

data_export = {}  # Object to export results
data_export["EUR4_h_tot"] = EUR4_h_tot
data_export["EUR4_A_tot"] = EUR4_A_tot
data_export["EUR4_A_tot_m"] = EUR4_A_tot_nps

'Model for counterfactuals'

EU15 = model_country('EU15')
EU15.productivity_series()
EU15.predictions_ams()
EU15.predictions_nps()

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

# UNITED STATES
USA = model_country('USA')
USA.productivity_series()
USA.predictions_ams()
USA.predictions_nps()

# save results
h_data_export = pd.DataFrame()
country_labels = ['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'IRL', 'ITA', 'GRC', 'FIN', 'FRA', 'LUX', 'GBR', 'NLD', 'PRT', 'SWE']
country_models = [AUT, BEL, DEU, DNK, ESP, IRL, ITA, GRC, FIN, FRA, LUX, GBR, NLD, PRT, SWE]

j = 0
for model in country_models:
	h_temp = pd.DataFrame({"sector": ["agr"]*50 + ["man"] *50 + ["bss"]*50 + ["fin"]*50  + ["trd"]*50+ ["nps"]*50, "year": list(np.arange(1970,2020))*6})
	h_temp["country"] = country_labels[j]
	h_temp["h_tot"] = np.tile(model.h_tot.values,6)
	h_temp.loc[h_temp.sector == "agr", "LS_m"] = model.share_agr_nps_m
	h_temp.loc[h_temp.sector == "agr", "LS"] = model.h_agr.values / model.h_tot.values
	h_temp.loc[h_temp.sector == "man", "LS_m"] = model.share_man_nps_m
	h_temp.loc[h_temp.sector == "man", "LS"] = model.h_man.values / model.h_tot.values
	h_temp.loc[h_temp.sector == "bss", "LS_m"] = model.share_bss_nps_m
	h_temp.loc[h_temp.sector == "bss", "LS"] = model.h_bss.values / model.h_tot.values
	h_temp.loc[h_temp.sector == "fin", "LS_m"] = model.share_fin_nps_m
	h_temp.loc[h_temp.sector == "fin", "LS"] = model.h_fin.values / model.h_tot.values
	h_temp.loc[h_temp.sector == "trd", "LS_m"] = model.share_trd_nps_m
	h_temp.loc[h_temp.sector == "trd", "LS"] = model.h_trd.values / model.h_tot.values
	h_temp.loc[h_temp.sector == "nps", "LS_m"] = model.share_nps_nps_m
	h_temp.loc[h_temp.sector == "nps", "LS"] = model.h_nps.values / model.h_tot.values
	h_data_export = pd.concat((h_data_export, h_temp), axis=0)
	j += 1

h_temp = pd.DataFrame({"sector": ["agr"]*50 + ["man"] *50 + ["bss"]*50 + ["fin"]*50  + ["trd"]*50+ ["nps"]*50, "year": list(np.arange(1970,2020))*6})
h_temp["country"] = "EU4"
h_temp["LS"] =  (h_data_export.loc[h_data_export.country=="ITA", "LS"] * h_data_export.loc[h_data_export.country=="ITA", "h_tot"] + h_data_export.loc[h_data_export.country=="DEU", "LS"] * h_data_export.loc[h_data_export.country=="DEU", "h_tot"] + h_data_export.loc[h_data_export.country=="GBR", "LS"] * h_data_export.loc[h_data_export.country=="GBR", "h_tot"] + h_data_export.loc[h_data_export.country=="FRA", "LS"] * h_data_export.loc[h_data_export.country=="FRA", "h_tot"]) / np.tile(EUR4_h_tot, 6)
h_temp["LS_m"] =  (h_data_export.loc[h_data_export.country=="ITA", "LS_m"] * h_data_export.loc[h_data_export.country=="ITA", "h_tot"] + h_data_export.loc[h_data_export.country=="DEU", "LS_m"] * h_data_export.loc[h_data_export.country=="DEU", "h_tot"] + h_data_export.loc[h_data_export.country=="GBR", "LS_m"] * h_data_export.loc[h_data_export.country=="GBR", "h_tot"] + h_data_export.loc[h_data_export.country=="FRA", "LS_m"] * h_data_export.loc[h_data_export.country=="FRA", "h_tot"]) / np.tile(EUR4_h_tot, 6)
h_temp["h_tot"] = np.tile(EUR4_h_tot, 6)
h_data_export = pd.concat((h_data_export, h_temp), axis=0)

h_temp = pd.DataFrame({"sector": ["agr"]*50 + ["man"] *50 + ["bss"]*50 + ["fin"]*50  + ["trd"]*50+ ["nps"]*50, "year": list(np.arange(1970,2020))*6})
h_temp["country"] = "EU15"
h_temp["LS"] = (h_data_export.loc[h_data_export.country == "AUT", "LS"] * h_data_export.loc[h_data_export.country == "AUT", "h_tot"] + h_data_export.loc[h_data_export.country == "BEL", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "FIN", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "FIN", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "LUX", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "LUX", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "SWE", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "SWE", "h_tot"]) / np.tile(
		EUR15_h_tot, 6)
h_temp["LS_m"] =  (h_data_export.loc[
																				  h_data_export.country == "AUT", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "AUT", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "FIN", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "FIN", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "LUX", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "LUX", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "SWE", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "SWE", "h_tot"]) / np.tile(
		EUR15_h_tot, 6)
h_temp["h_tot"] = np.tile(EUR15_h_tot, 6)
h_data_export = pd.concat((h_data_export, h_temp), axis=0)

h_temp = pd.DataFrame({"sector": ["agr"]*50 + ["man"] *50 + ["bss"]*50 + ["fin"]*50  + ["trd"]*50+ ["nps"]*50, "year": list(np.arange(1970,2020))*6})
h_temp["country"] = "EUCORE"
h_temp["LS"] = (
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "h_tot"]) / np.tile(
		EURCORE_h_tot, 6)
h_temp["LS_m"] =  (
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "h_tot"]) / np.tile(
		EURCORE_h_tot, 6)
h_temp["h_tot"] = np.tile(EURCORE_h_tot, 6)
h_data_export = pd.concat((h_data_export, h_temp), axis=0)

h_temp = pd.DataFrame({"sector": ["agr"]*50 + ["man"] *50 + ["bss"]*50 + ["fin"]*50  + ["trd"]*50+ ["nps"]*50, "year": list(np.arange(1970,2020))*6})
h_temp["country"] = "EUPERI"
h_temp["LS"] = (
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "LS" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "h_tot"]) / np.tile(
		EURPERI_h_tot, 6)
h_temp["LS_m"] =  (
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "LS_m" ] *
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "h_tot"]) / np.tile(
		EURPERI_h_tot, 6)
h_temp["h_tot"] = np.tile(EURPERI_h_tot, 6)
h_data_export = pd.concat((h_data_export, h_temp), axis=0)

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
		self.ss_A_base_ams_init_2  = np.array(self.cou.share_agr)[25]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[25]*self.cou.A_man[-1] + np.array(self.cou.share_ser)[25]*self.cou.A_ser[-1]
		self.ss_A_base_ams  = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_ser*self.cou.A_ser

		self.ss_A_base_nps_init = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_trd)[0]*self.cou.A_trd[-1] + np.array(self.cou.share_bss)[0]*self.cou.A_bss[-1] + np.array(self.cou.share_fin)[0]*self.cou.A_fin[-1] + np.array(self.cou.share_nps)[0]*self.cou.A_nps[-1]
		self.ss_A_base_nps_init_2 = np.array(self.cou.share_agr)[25]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[25]*self.cou.A_man[-1] + np.array(self.cou.share_trd)[25]*self.cou.A_trd[-1] + np.array(self.cou.share_bss)[25]*self.cou.A_bss[-1] + np.array(self.cou.share_fin)[25]*self.cou.A_fin[-1] + np.array(self.cou.share_nps)[25]*self.cou.A_nps[-1]
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
		self.C_E_ams_baseline = [np.array(self.cou.GDP)[0]/np.array(GDP)[0]]
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
		self.C_E_nps_baseline = [np.array(self.cou.GDP)[0]/np.array(GDP)[0]]
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
		self.cou.predictions_ams()
		
		'nps'
		self.cou.p_trd_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_trd)
		self.cou.p_bss_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_bss)
		self.cou.p_fin_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_fin)
		self.cou.p_nps_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_nps)
		self.cou.C_nps = self.cou.C_ams_ser*(np.array(self.cou.C_E_nps)/np.array(self.C_E_nps_baseline))
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
			A_agr_catch = E[-1]/np.array(self.cou.share_agr)[-1] - (np.array(self.cou.h_man)[-1] * np.array(self.cou.A_man)[-1] + np.array(self.cou.h_trd)[-1] * np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1] * np.array(self.cou.A_bss)[-1] + np.array(self.cou.h_fin)[-1] * np.array(self.cou.A_fin)[-1] + np.array(self.cou.h_nps)[-1] * np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_agr)[-1]
			g_y_l_agr_catch = (A_agr_catch/np.array(self.cou.A_agr)[0])**(1/self.cou.ts_length)-1
			g_catch = np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_agr_catch)
			self.cou.g_y_l_agr[init_year:] = g_catch[init_year:]
			self.A_agr_catch = A_agr_catch
			self.g_y_l_agr_catch = g_y_l_agr_catch

		if sec == 'man':
			A_man_catch=E[-1]/np.array(self.cou.share_man)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_man)[-1]
			g_y_l_man_catch=(A_man_catch/np.array(self.cou.A_man)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_man_catch)
			self.cou.g_y_l_man[init_year:] = g_catch[init_year:]
			self.A_man_catch=A_man_catch
			self.g_y_l_man_catch=g_y_l_man_catch

		if sec == 'ser':
			A_ser_catch=E[-1]/np.array(self.cou.share_ser)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_ser)[-1]
			g_y_l_ser_catch=(A_ser_catch/np.array(self.cou.A_ser)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_ser_catch)
			self.cou.g_y_l_ser[init_year:] = g_catch[init_year:]
			self.A_ser_catch=A_ser_catch
			self.g_y_l_ser_catch=g_y_l_ser_catch

		if sec == 'trd':
			A_trd_catch=E[-1]/np.array(self.cou.share_trd)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_trd)[-1]
			g_y_l_trd_catch=(A_trd_catch/np.array(self.cou.A_trd)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_trd_catch)
			self.cou.g_y_l_trd[init_year:] = g_catch[init_year:]
			self.A_trd_catch=A_trd_catch
			self.g_y_l_trd_catch=g_y_l_trd_catch			

		if sec == 'bss':
			A_bss_catch=E[-1]/np.array(self.cou.share_bss)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_bss)[-1]
			g_y_l_bss_catch=(A_bss_catch/np.array(self.cou.A_bss)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_bss_catch)
			self.cou.g_y_l_bss[init_year:] = g_catch[init_year:]
			self.A_bss_catch=A_bss_catch
			self.g_y_l_bss_catch=g_y_l_bss_catch			

		if sec == 'fin':
			A_fin_catch=E[-1]/np.array(self.cou.share_fin)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_nps)[-1]*np.array(self.cou.A_nps)[-1])/np.array(self.cou.h_fin)[-1]
			g_y_l_fin_catch=(A_fin_catch/np.array(self.cou.A_fin)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_fin_catch)
			self.cou.g_y_l_fin[init_year:] = g_catch[init_year:]
			self.A_fin_catch=A_fin_catch
			self.g_y_l_fin_catch=g_y_l_fin_catch			

		if sec == 'nps':
			A_nps_catch=E[-1]/np.array(self.cou.share_nps)[-1] - (np.array(self.cou.h_agr)[-1]*np.array(self.cou.A_agr)[-1]+np.array(self.cou.h_man)[-1]*np.array(self.cou.A_man)[-1]+np.array(self.cou.h_trd)[-1]*np.array(self.cou.A_trd)[-1]+np.array(self.cou.h_bss)[-1]*np.array(self.cou.A_bss)[-1]+np.array(self.cou.h_fin)[-1]*np.array(self.cou.A_fin)[-1])/np.array(self.cou.h_nps)[-1]
			g_y_l_nps_catch=(A_nps_catch/np.array(self.cou.A_nps)[0])**(1/self.cou.ts_length)-1
			g_catch= np.empty(int(self.cou.ts_length)+1)
			g_catch.fill(g_y_l_nps_catch)
			self.cou.g_y_l_nps[init_year:] = g_catch[init_year:]
			self.A_nps_catch=A_nps_catch
			self.g_y_l_nps_catch=g_y_l_nps_catch			


		'Generate counterfactual series'
		self.cou.productivity_series()

		'nps'
		self.cou.p_trd_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_trd)
		self.cou.p_bss_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_bss)
		self.cou.p_fin_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_fin)
		self.cou.p_nps_p_man = np.array(self.cou.A_man)/np.array(self.cou.A_nps)
		self.cou.C_nps = self.cou.C_ams_ser*(np.array(self.cou.C_E_nps)/np.array(self.C_E_nps_baseline))
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

		self.A_tot_nps = self.cou.A_tot_nps

		'ams'
		self.cou.p_ser_p_man = np.array(self.cou.A_man) / np.array(self.cou.A_ser)
		self.cou.C_ams = self.cou.C_ams_ser * (np.array(self.cou.C_E_ams) / np.array(self.C_E_ams_baseline))
		self.cou.predictions_ams()

		self.A_agr = self.cou.A_agr
		self.A_man = self.cou.A_man
		self.A_ser = self.cou.A_ser

		self.share_agr_ams_m = self.cou.share_agr_ams_m
		self.share_man_ams_m = self.cou.share_man_ams_m
		self.share_ser_ams_m = self.cou.share_ser_ams_m
	
		self.A_tot_ams = self.cou.A_tot_ams

	def shift_share(self, sec):
		'ams'
		self.ss_A_ams_init  = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_ser)[0]*self.cou.A_ser[-1]
		self.ss_A_ams_init_2  = np.array(self.cou.share_agr)[21]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[21]*self.cou.A_man[-1] + np.array(self.cou.share_ser)[21]*self.cou.A_ser[-1]
		self.ss_A_ams  = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_ser*self.cou.A_ser

		'nps'
		self.ss_A_nps_init = np.array(self.cou.share_agr)[0]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[0]*self.cou.A_man[-1] + np.array(self.cou.share_trd)[0]*self.cou.A_trd[-1] + np.array(self.cou.share_bss)[0]*self.cou.A_bss[-1] + np.array(self.cou.share_fin)[0]*self.cou.A_fin[-1] + np.array(self.cou.share_nps)[0]*self.cou.A_nps[-1]
		self.ss_A_nps_init_2 = np.array(self.cou.share_agr)[21]*self.cou.A_agr[-1] + np.array(self.cou.share_man)[21]*self.cou.A_man[-1] + np.array(self.cou.share_trd)[21]*self.cou.A_trd[-1] + np.array(self.cou.share_bss)[21]*self.cou.A_bss[-1] + np.array(self.cou.share_fin)[21]*self.cou.A_fin[-1] + np.array(self.cou.share_nps)[21]*self.cou.A_nps[-1]
		self.ss_A_nps = self.cou.share_agr*self.cou.A_agr + self.cou.share_man*self.cou.A_man + self.cou.share_trd*self.cou.A_trd + self.cou.share_bss*self.cou.A_bss + self.cou.share_fin*self.cou.A_fin + self.cou.share_nps*self.cou.A_nps


'''
----------------------------------------------------------------------------------------
	Counterfactual 1: Each sector keeping the pace with the US for the entire period
----------------------------------------------------------------------------------------
'''

'ams'
cf_1_ams_ss_init = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EU15']]
cf_1_ams_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EU15']]
cf_1_ams = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EU15']]

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
	IRL_cf=counterfactual('IRL')
	ITA_cf=counterfactual('ITA')
	LUX_cf=counterfactual('LUX')
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
	IRL_cf.baseline()
	ITA_cf.baseline()
	LUX_cf.baseline()
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
	IRL_cf.feed_US_productivity_growth(0, sec)
	ITA_cf.feed_US_productivity_growth(0, sec)
	LUX_cf.feed_US_productivity_growth(0, sec)
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
	IRL_cf.shift_share(sec)
	ITA_cf.shift_share(sec)
	LUX_cf.shift_share(sec)
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
				(IRL_cf.ss_A_ams_init/IRL_cf.ss_A_base_ams_init-1)*100,
				(ITA_cf.ss_A_ams_init/ITA_cf.ss_A_base_ams_init-1)*100,
				(LUX_cf.ss_A_ams_init/LUX_cf.ss_A_base_ams_init-1)*100,
				(NLD_cf.ss_A_ams_init/NLD_cf.ss_A_base_ams_init-1)*100,
				(PRT_cf.ss_A_ams_init/PRT_cf.ss_A_base_ams_init-1)*100,
				(SWE_cf.ss_A_ams_init/SWE_cf.ss_A_base_ams_init-1)*100,
				np.array([(np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_ams_init/DEU_cf.ss_A_base_ams_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_ams_init/FRA_cf.ss_A_base_ams_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_ams_init/ITA_cf.ss_A_base_ams_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_ams_init/GBR_cf.ss_A_base_ams_init-1)*100)/EUR4_h_tot[0]]),
				np.array([(np.array(AUT.h_tot)[0]*(AUT_cf.ss_A_ams_init/AUT_cf.ss_A_base_ams_init-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf.ss_A_ams_init/BEL_cf.ss_A_base_ams_init-1)*100 + np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_ams_init/DEU_cf.ss_A_base_ams_init-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf.ss_A_ams_init/DNK_cf.ss_A_base_ams_init-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf.ss_A_ams_init/ESP_cf.ss_A_base_ams_init-1)*100 + np.array(FIN.h_tot)[0]*(FIN_cf.ss_A_ams_init/FIN_cf.ss_A_base_ams_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_ams_init/FRA_cf.ss_A_base_ams_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_ams_init/GBR_cf.ss_A_base_ams_init-1)*100 + np.array(GRC.h_tot)[0]*(GRC_cf.ss_A_ams_init/GRC_cf.ss_A_base_ams_init-1)*100 + np.array(IRL.h_tot)[0]*(IRL_cf.ss_A_ams_init/IRL_cf.ss_A_base_ams_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_ams_init/ITA_cf.ss_A_base_ams_init-1)*100 + np.array(LUX.h_tot)[0]*(LUX_cf.ss_A_ams_init/LUX_cf.ss_A_base_ams_init-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf.ss_A_ams_init/NLD_cf.ss_A_base_ams_init-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf.ss_A_ams_init/PRT_cf.ss_A_base_ams_init-1)*100 + np.array(SWE.h_tot)[0]*(SWE_cf.ss_A_ams_init/SWE_cf.ss_A_base_ams_init-1)*100)/EUR15_h_tot[0]])]

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
				(np.array(IRL_cf.ss_A_ams)[-1]/np.array(IRL_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(ITA_cf.ss_A_ams)[-1]/np.array(ITA_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(LUX_cf.ss_A_ams)[-1]/np.array(LUX_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(NLD_cf.ss_A_ams)[-1]/np.array(NLD_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(PRT_cf.ss_A_ams)[-1]/np.array(PRT_cf.ss_A_base_ams)[-1]-1)*100,
				(np.array(SWE_cf.ss_A_ams)[-1]/np.array(SWE_cf.ss_A_base_ams)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_ams)[-1]/np.array(DEU_cf.ss_A_base_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_ams)[-1]/np.array(FRA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_ams)[-1]/np.array(ITA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_ams)[-1]/np.array(GBR_cf.ss_A_base_ams)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.ss_A_ams)[-1]/np.array(AUT_cf.ss_A_base_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.ss_A_ams)[-1]/np.array(BEL_cf.ss_A_base_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_ams)[-1]/np.array(DEU_cf.ss_A_base_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.ss_A_ams)[-1]/np.array(DNK_cf.ss_A_base_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.ss_A_ams)[-1]/np.array(ESP_cf.ss_A_base_ams)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.ss_A_ams)[-1]/np.array(FIN_cf.ss_A_base_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_ams)[-1]/np.array(FRA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_ams)[-1]/np.array(GBR_cf.ss_A_base_ams)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.ss_A_ams)[-1]/np.array(GRC_cf.ss_A_base_ams)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf.ss_A_ams)[-1]/np.array(IRL_cf.ss_A_base_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_ams)[-1]/np.array(ITA_cf.ss_A_base_ams)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf.ss_A_ams)[-1]/np.array(LUX_cf.ss_A_base_ams)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.ss_A_ams)[-1]/np.array(NLD_cf.ss_A_base_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.ss_A_ams)[-1]/np.array(PRT_cf.ss_A_base_ams)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.ss_A_ams)[-1]/np.array(SWE_cf.ss_A_base_ams)[-1]-1)*100)/EUR15_h_tot[-1]])]
	
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
				(np.array(IRL_cf.A_tot_ams)[-1]/np.array(IRL.A_tot_ams)[-1]-1)*100,
				(np.array(ITA_cf.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100,
				(np.array(LUX_cf.A_tot_ams)[-1]/np.array(LUX.A_tot_ams)[-1]-1)*100,
				(np.array(NLD_cf.A_tot_ams)[-1]/np.array(NLD.A_tot_ams)[-1]-1)*100,
				(np.array(PRT_cf.A_tot_ams)[-1]/np.array(PRT.A_tot_ams)[-1]-1)*100,
				(np.array(SWE_cf.A_tot_ams)[-1]/np.array(SWE.A_tot_ams)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.A_tot_ams)[-1]/np.array(AUT.A_tot_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.A_tot_ams)[-1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.A_tot_ams)[-1]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.A_tot_ams)[-1]/np.array(ESP.A_tot_ams)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.A_tot_ams)[-1]/np.array(FIN.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GBR_cf.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf.A_tot_ams)[-1]/np.array(IRL.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf.A_tot_ams)[-1]/np.array(LUX.A_tot_ams)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.A_tot_ams)[-1]/np.array(NLD.A_tot_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.A_tot_ams)[-1]/np.array(PRT.A_tot_ams)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.A_tot_ams)[-1]/np.array(SWE.A_tot_ams)[-1]-1)*100)/EUR15_h_tot[-1]])]
	cf_1_sec_ams[1:] = [ '%.1f' % elem for elem in cf_1_sec_ams[1:] ]
	cf_1_ams.append(cf_1_sec_ams)

pd.DataFrame(cf_1_ams_ss_init).to_excel('../output/figures/Counterfactual_1_ams_ss_init.xlsx', index = False, header = False)
pd.DataFrame(cf_1_ams_ss).to_excel('../output/figures/Counterfactual_1_ams_ss.xlsx', index = False, header = False)
pd.DataFrame(cf_1_ams).to_excel('../output/figures/Counterfactual_1_ams.xlsx', index = False, header = False)


'nps'
cf_1_nps_ss_init = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI','EU15', 'EU15_data']]
cf_1_nps_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI', 'EU15', 'EU15_data']]
cf_1_nps = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI', 'EU15', 'EU15_data']]

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
	IRL_cf=counterfactual('IRL')
	ITA_cf=counterfactual('ITA')
	LUX_cf=counterfactual('LUX')
	NLD_cf=counterfactual('NLD')
	PRT_cf=counterfactual('PRT')
	SWE_cf=counterfactual('SWE')
	EU15_cf = counterfactual('EU15')

	AUT_cf.baseline()
	BEL_cf.baseline()
	DEU_cf.baseline()
	DNK_cf.baseline()
	ESP_cf.baseline()
	FIN_cf.baseline()
	FRA_cf.baseline()
	GBR_cf.baseline()
	GRC_cf.baseline()
	IRL_cf.baseline()
	ITA_cf.baseline()
	LUX_cf.baseline()
	NLD_cf.baseline()
	PRT_cf.baseline()
	SWE_cf.baseline()
	AUT_cf.baseline()
	EU15_cf.baseline()

	AUT_cf.feed_US_productivity_growth(0, sec)
	BEL_cf.feed_US_productivity_growth(0, sec)
	DEU_cf.feed_US_productivity_growth(0, sec)
	DNK_cf.feed_US_productivity_growth(0, sec)
	ESP_cf.feed_US_productivity_growth(0, sec)
	FIN_cf.feed_US_productivity_growth(0, sec)
	FRA_cf.feed_US_productivity_growth(0, sec)
	GBR_cf.feed_US_productivity_growth(0, sec)
	GRC_cf.feed_US_productivity_growth(0, sec)
	IRL_cf.feed_US_productivity_growth(0, sec)
	ITA_cf.feed_US_productivity_growth(0, sec)
	LUX_cf.feed_US_productivity_growth(0, sec)
	NLD_cf.feed_US_productivity_growth(0, sec)
	PRT_cf.feed_US_productivity_growth(0, sec)
	SWE_cf.feed_US_productivity_growth(0, sec)
	EU15_cf.feed_US_productivity_growth(0, sec)

	AUT_cf.shift_share(sec)
	BEL_cf.shift_share(sec)
	DEU_cf.shift_share(sec)
	DNK_cf.shift_share(sec)
	ESP_cf.shift_share(sec)
	FIN_cf.shift_share(sec)
	FRA_cf.shift_share(sec)
	GBR_cf.shift_share(sec)
	GRC_cf.shift_share(sec)
	IRL_cf.shift_share(sec)
	ITA_cf.shift_share(sec)
	LUX_cf.shift_share(sec)
	NLD_cf.shift_share(sec)
	PRT_cf.shift_share(sec)
	SWE_cf.shift_share(sec)
	EU15_cf.shift_share(sec)

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
				(IRL_cf.ss_A_nps_init/IRL_cf.ss_A_base_nps_init-1)*100,
				(ITA_cf.ss_A_nps_init/ITA_cf.ss_A_base_nps_init-1)*100,
				(LUX_cf.ss_A_nps_init/LUX_cf.ss_A_base_nps_init-1)*100,
				(NLD_cf.ss_A_nps_init/NLD_cf.ss_A_base_nps_init-1)*100,
				(PRT_cf.ss_A_nps_init/PRT_cf.ss_A_base_nps_init-1)*100,
				(SWE_cf.ss_A_nps_init/SWE_cf.ss_A_base_nps_init-1)*100,
				np.array([(np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_nps_init/DEU_cf.ss_A_base_nps_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_nps_init/FRA_cf.ss_A_base_nps_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_nps_init/ITA_cf.ss_A_base_nps_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_nps_init/GBR_cf.ss_A_base_nps_init-1)*100)/EUR4_h_tot[0]]),
			    np.array([(np.array(DEU.h_tot)[0] * (DEU_cf.ss_A_nps_init / DEU_cf.ss_A_base_nps_init - 1) * 100 + np.array(FRA.h_tot)[0] * (FRA_cf.ss_A_nps_init / FRA_cf.ss_A_base_nps_init - 1) * 100 + np.array(BEL.h_tot)[0] * (BEL_cf.ss_A_nps_init / BEL_cf.ss_A_base_nps_init - 1) * 100 + np.array(NLD.h_tot)[0] * (NLD_cf.ss_A_nps_init / NLD_cf.ss_A_base_nps_init - 1) * 100 + np.array(DNK.h_tot)[0] * (DNK_cf.ss_A_nps_init / DNK_cf.ss_A_base_nps_init - 1) * 100) / EURCORE_h_tot[0]]),
			    np.array([(np.array(GRC.h_tot)[0] * (GRC_cf.ss_A_nps_init / GRC_cf.ss_A_base_nps_init - 1) * 100 + np.array(IRL.h_tot)[0] * (IRL_cf.ss_A_nps_init / IRL_cf.ss_A_base_nps_init - 1) * 100 + np.array(PRT.h_tot)[0] * (PRT_cf.ss_A_nps_init / PRT_cf.ss_A_base_nps_init - 1) * 100 + np.array(ESP.h_tot)[0] * (ESP_cf.ss_A_nps_init / ESP_cf.ss_A_base_nps_init - 1) * 100 + np.array(ITA.h_tot)[0] * (ITA_cf.ss_A_nps_init / ITA_cf.ss_A_base_nps_init - 1) * 100 + np.array(GBR.h_tot)[0] * (GBR_cf.ss_A_nps_init / GBR_cf.ss_A_base_nps_init - 1) * 100) / EURPERI_h_tot[0]]),
				np.array([(np.array(AUT.h_tot)[0]*(AUT_cf.ss_A_nps_init/AUT_cf.ss_A_base_nps_init-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf.ss_A_nps_init/BEL_cf.ss_A_base_nps_init-1)*100 + np.array(DEU.h_tot)[0]*(DEU_cf.ss_A_nps_init/DEU_cf.ss_A_base_nps_init-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf.ss_A_nps_init/DNK_cf.ss_A_base_nps_init-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf.ss_A_nps_init/ESP_cf.ss_A_base_nps_init-1)*100 + np.array(FIN.h_tot)[0]*(FIN_cf.ss_A_nps_init/FIN_cf.ss_A_base_nps_init-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf.ss_A_nps_init/FRA_cf.ss_A_base_nps_init-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf.ss_A_nps_init/GBR_cf.ss_A_base_nps_init-1)*100 + np.array(GRC.h_tot)[0]*(GRC_cf.ss_A_nps_init/GRC_cf.ss_A_base_nps_init-1)*100 + np.array(IRL.h_tot)[0]*(IRL_cf.ss_A_nps_init/IRL_cf.ss_A_base_nps_init-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf.ss_A_nps_init/ITA_cf.ss_A_base_nps_init-1)*100 + np.array(LUX.h_tot)[0]*(LUX_cf.ss_A_nps_init/LUX_cf.ss_A_base_nps_init-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf.ss_A_nps_init/NLD_cf.ss_A_base_nps_init-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf.ss_A_nps_init/PRT_cf.ss_A_base_nps_init-1)*100 + np.array(SWE.h_tot)[0]*(SWE_cf.ss_A_nps_init/SWE_cf.ss_A_base_nps_init-1)*100)/EUR15_h_tot[0]]),
			    (EU15_cf.ss_A_nps_init/EU15_cf.ss_A_base_nps_init-1)*100]

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
				(np.array(IRL_cf.ss_A_nps)[-1]/np.array(IRL_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(LUX_cf.ss_A_nps)[-1]/np.array(LUX_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(NLD_cf.ss_A_nps)[-1]/np.array(NLD_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(PRT_cf.ss_A_nps)[-1]/np.array(PRT_cf.ss_A_base_nps)[-1]-1)*100,
				(np.array(SWE_cf.ss_A_nps)[-1]/np.array(SWE_cf.ss_A_base_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.ss_A_nps)[-1]/np.array(BEL_cf.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.ss_A_nps)[-1]/np.array(NLD_cf.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.ss_A_nps)[-1]/np.array(DNK_cf.ss_A_base_nps)[-1]-1)*100)/EURCORE_h_tot[-1]]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.ss_A_nps)[-1]/np.array(GRC_cf.ss_A_base_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf.ss_A_nps)[-1]/np.array(IRL_cf.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.ss_A_nps)[-1]/np.array(PRT_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.ss_A_nps)[-1]/np.array(ESP_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.ss_A_nps)[-1]/np.array(AUT_cf.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.ss_A_nps)[-1]/np.array(BEL_cf.ss_A_base_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.ss_A_nps)[-1]/np.array(DEU_cf.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.ss_A_nps)[-1]/np.array(DNK_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.ss_A_nps)[-1]/np.array(ESP_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.ss_A_nps)[-1]/np.array(FIN_cf.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.ss_A_nps)[-1]/np.array(FRA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.ss_A_nps)[-1]/np.array(GBR_cf.ss_A_base_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.ss_A_nps)[-1]/np.array(GRC_cf.ss_A_base_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf.ss_A_nps)[-1]/np.array(IRL_cf.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.ss_A_nps)[-1]/np.array(ITA_cf.ss_A_base_nps)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf.ss_A_nps)[-1]/np.array(LUX_cf.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.ss_A_nps)[-1]/np.array(NLD_cf.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.ss_A_nps)[-1]/np.array(PRT_cf.ss_A_base_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.ss_A_nps)[-1]/np.array(SWE_cf.ss_A_base_nps)[-1]-1)*100)/EUR15_h_tot[-1]]),
				(np.array(EU15_cf.ss_A_nps)[-1] / np.array(EU15_cf.ss_A_base_nps)[-1] - 1) * 100]

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
				(np.array(IRL_cf.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100,
				(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100,
				(np.array(LUX_cf.A_tot_nps)[-1]/np.array(LUX.A_tot_nps)[-1]-1)*100,
				(np.array(NLD_cf.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100,
				(np.array(PRT_cf.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100,
				(np.array(SWE_cf.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100)/EURCORE_h_tot[-1]]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GBR_cf.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf.A_tot_nps)[-1]/np.array(LUX.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100)/EUR15_h_tot[-1]]),
				(np.array(EU15_cf.A_tot_nps)[-1] / np.array(EU15.A_tot_nps)[-1] - 1) * 100]
	cf_1_sec_nps[1:] = [ '%.1f' % elem for elem in cf_1_sec_nps[1:] ]
	cf_1_nps.append(cf_1_sec_nps)

	# save results
	h_data_export.loc[(h_data_export.sector == "agr") & (h_data_export.country == "FRA"), "LS_m_cf1_" + sec] = FRA_cf.share_agr_nps_m
	h_data_export.loc[(h_data_export.sector == "man") & (h_data_export.country == "FRA"), "LS_m_cf1_" + sec] = FRA_cf.share_man_nps_m
	h_data_export.loc[(h_data_export.sector == "bss") & (h_data_export.country == "FRA"), "LS_m_cf1_" + sec] = FRA_cf.share_bss_nps_m
	h_data_export.loc[(h_data_export.sector == "fin") & (h_data_export.country == "FRA"), "LS_m_cf1_" + sec] = FRA_cf.share_fin_nps_m
	h_data_export.loc[(h_data_export.sector == "trd") & (h_data_export.country == "FRA"), "LS_m_cf1_" + sec] = FRA_cf.share_trd_nps_m
	h_data_export.loc[(h_data_export.sector == "nps") & (h_data_export.country == "FRA"), "LS_m_cf1_" + sec] = FRA_cf.share_nps_nps_m

	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "DEU"), "LS_m_cf1_" + sec] = DEU_cf.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "DEU"), "LS_m_cf1_" + sec] = DEU_cf.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "DEU"), "LS_m_cf1_" + sec] = DEU_cf.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "DEU"), "LS_m_cf1_" + sec] = DEU_cf.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "DEU"), "LS_m_cf1_" + sec] = DEU_cf.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "DEU"), "LS_m_cf1_" + sec] = DEU_cf.share_nps_nps_m

	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "GBR"), "LS_m_cf1_" + sec] = GBR_cf.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "GBR"), "LS_m_cf1_" + sec] = GBR_cf.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "GBR"), "LS_m_cf1_" + sec] = GBR_cf.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "GBR"), "LS_m_cf1_" + sec] = GBR_cf.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "GBR"), "LS_m_cf1_" + sec] = GBR_cf.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "GBR"), "LS_m_cf1_" + sec] = GBR_cf.share_nps_nps_m

	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "ITA"), "LS_m_cf1_" + sec] = ITA_cf.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "ITA"), "LS_m_cf1_" + sec] = ITA_cf.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "ITA"), "LS_m_cf1_" + sec] = ITA_cf.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "ITA"), "LS_m_cf1_" + sec] = ITA_cf.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "ITA"), "LS_m_cf1_" + sec] = ITA_cf.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "ITA"), "LS_m_cf1_" + sec] = ITA_cf.share_nps_nps_m

	h_data_export.loc[(h_data_export.country == "EU4"), "LS_m_cf1_" + sec] = (h_data_export.loc[h_data_export.country == "ITA", "LS_m_cf1_" + sec] * h_data_export.loc[
		h_data_export.country == "ITA", "h_tot"] + h_data_export.loc[h_data_export.country == "DEU", "LS_m_cf1_" + sec] *
					h_data_export.loc[h_data_export.country == "DEU", "h_tot"] + h_data_export.loc[
						h_data_export.country == "GBR", "LS_m_cf1_" + sec] * h_data_export.loc[
						h_data_export.country == "GBR", "h_tot"] + h_data_export.loc[
						h_data_export.country == "FRA", "LS_m_cf1_" + sec] * h_data_export.loc[
						h_data_export.country == "FRA", "h_tot"]) / np.tile(EUR4_h_tot, 6)

	h_data_export.loc[(h_data_export.country == "EU15"), "LS_m_cf1_" + sec] = (h_data_export.loc[
																				  h_data_export.country == "AUT", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "AUT", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "BEL", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "DNK", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "ESP", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "FIN", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "FIN", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "GRC", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "IRL", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "LUX", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "LUX", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "NLD", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "PRT", "h_tot"]+
																			  h_data_export.loc[
																				  h_data_export.country == "SWE", "LS_m_cf1_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "SWE", "h_tot"]) / np.tile(
		EUR15_h_tot, 6)

pd.DataFrame(cf_1_nps_ss_init).to_excel('../output/figures/Counterfactual_1_nps_ss_init.xlsx', index = False, header = False)
pd.DataFrame(cf_1_nps_ss).to_excel('../output/figures/Counterfactual_1_nps_ss.xlsx', index = False, header = False)
pd.DataFrame(cf_1_nps).to_excel('../output/figures/Counterfactual_1_nps.xlsx', index = False, header = False)


'''
---------------------------------------------------------------------------------
	Counterfactual 2: Each sector keeping the pace with the US since 1990
---------------------------------------------------------------------------------
'''
'ams'
cf_2_ams_ss_init = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EU15']]
cf_2_ams_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EU15']]
cf_2_ams = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EU15']]

for sec in ['agr', 'man', 'ser']:
	AUT_cf2=counterfactual('AUT')
	BEL_cf2=counterfactual('BEL')
	DEU_cf2=counterfactual('DEU')
	DNK_cf2=counterfactual('DNK')
	ESP_cf2=counterfactual('ESP')
	FIN_cf2=counterfactual('FIN')
	FRA_cf2=counterfactual('FRA')
	GBR_cf2=counterfactual('GBR')
	GRC_cf2=counterfactual('GRC')
	IRL_cf2=counterfactual('IRL')
	ITA_cf2=counterfactual('ITA')
	LUX_cf2=counterfactual('LUX')
	NLD_cf2=counterfactual('NLD')
	PRT_cf2=counterfactual('PRT')
	SWE_cf2=counterfactual('SWE')

	AUT_cf2.baseline()
	BEL_cf2.baseline()
	DEU_cf2.baseline()
	DNK_cf2.baseline()
	ESP_cf2.baseline()
	FIN_cf2.baseline()
	FRA_cf2.baseline()
	GBR_cf2.baseline()
	GRC_cf2.baseline()
	IRL_cf2.baseline()
	ITA_cf2.baseline()
	LUX_cf2.baseline()
	NLD_cf2.baseline()
	PRT_cf2.baseline()
	SWE_cf2.baseline()
	AUT_cf2.baseline()

	AUT_cf2.feed_US_productivity_growth(21, sec)
	BEL_cf2.feed_US_productivity_growth(21, sec)
	DEU_cf2.feed_US_productivity_growth(21, sec)
	DNK_cf2.feed_US_productivity_growth(21, sec)
	ESP_cf2.feed_US_productivity_growth(21, sec)
	FIN_cf2.feed_US_productivity_growth(21, sec)
	FRA_cf2.feed_US_productivity_growth(21, sec)
	GBR_cf2.feed_US_productivity_growth(21, sec)
	GRC_cf2.feed_US_productivity_growth(21, sec)
	IRL_cf2.feed_US_productivity_growth(21, sec)
	ITA_cf2.feed_US_productivity_growth(21, sec)
	LUX_cf2.feed_US_productivity_growth(21, sec)
	NLD_cf2.feed_US_productivity_growth(21, sec)
	PRT_cf2.feed_US_productivity_growth(21, sec)
	SWE_cf2.feed_US_productivity_growth(21, sec)

	AUT_cf2.shift_share(sec)
	BEL_cf2.shift_share(sec)
	DEU_cf2.shift_share(sec)
	DNK_cf2.shift_share(sec)
	ESP_cf2.shift_share(sec)
	FIN_cf2.shift_share(sec)
	FRA_cf2.shift_share(sec)
	GBR_cf2.shift_share(sec)
	GRC_cf2.shift_share(sec)
	IRL_cf2.shift_share(sec)
	ITA_cf2.shift_share(sec)
	LUX_cf2.shift_share(sec)
	NLD_cf2.shift_share(sec)
	PRT_cf2.shift_share(sec)
	SWE_cf2.shift_share(sec)

	cf_2_sec_ams_init_ss = 	[sec,
				(AUT_cf2.ss_A_ams_init_2/AUT_cf2.ss_A_base_ams_init_2-1)*100,
				(BEL_cf2.ss_A_ams_init_2/BEL_cf2.ss_A_base_ams_init_2-1)*100,
				(DEU_cf2.ss_A_ams_init_2/DEU_cf2.ss_A_base_ams_init_2-1)*100,
				(DNK_cf2.ss_A_ams_init_2/DNK_cf2.ss_A_base_ams_init_2-1)*100,
				(ESP_cf2.ss_A_ams_init_2/ESP_cf2.ss_A_base_ams_init_2-1)*100,
				(FIN_cf2.ss_A_ams_init_2/FIN_cf2.ss_A_base_ams_init_2-1)*100,
				(FRA_cf2.ss_A_ams_init_2/FRA_cf2.ss_A_base_ams_init_2-1)*100,
				(GBR_cf2.ss_A_ams_init_2/GBR_cf2.ss_A_base_ams_init_2-1)*100,
				(GRC_cf2.ss_A_ams_init_2/GRC_cf2.ss_A_base_ams_init_2-1)*100,
				(IRL_cf2.ss_A_ams_init_2/IRL_cf2.ss_A_base_ams_init_2-1)*100,
				(ITA_cf2.ss_A_ams_init_2/ITA_cf2.ss_A_base_ams_init_2-1)*100,
				(LUX_cf2.ss_A_ams_init_2/LUX_cf2.ss_A_base_ams_init_2-1)*100,
				(NLD_cf2.ss_A_ams_init_2/NLD_cf2.ss_A_base_ams_init_2-1)*100,
				(PRT_cf2.ss_A_ams_init_2/PRT_cf2.ss_A_base_ams_init_2-1)*100,
				(SWE_cf2.ss_A_ams_init_2/SWE_cf2.ss_A_base_ams_init_2-1)*100,
				np.array([(np.array(DEU.h_tot)[0]*(DEU_cf2.ss_A_ams_init_2/DEU_cf2.ss_A_base_ams_init_2-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf2.ss_A_ams_init_2/FRA_cf2.ss_A_base_ams_init_2-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf2.ss_A_ams_init_2/ITA_cf2.ss_A_base_ams_init_2-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf2.ss_A_ams_init_2/GBR_cf2.ss_A_base_ams_init_2-1)*100)/EUR4_h_tot[0]]),
				np.array([(np.array(AUT.h_tot)[0]*(AUT_cf2.ss_A_ams_init_2/AUT_cf2.ss_A_base_ams_init_2-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf2.ss_A_ams_init_2/BEL_cf2.ss_A_base_ams_init_2-1)*100 + np.array(DEU.h_tot)[0]*(DEU_cf2.ss_A_ams_init_2/DEU_cf2.ss_A_base_ams_init_2-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf2.ss_A_ams_init_2/DNK_cf2.ss_A_base_ams_init_2-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf2.ss_A_ams_init_2/ESP_cf2.ss_A_base_ams_init_2-1)*100 + np.array(FIN.h_tot)[0]*(FIN_cf2.ss_A_ams_init_2/FIN_cf2.ss_A_base_ams_init_2-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf2.ss_A_ams_init_2/FRA_cf2.ss_A_base_ams_init_2-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf2.ss_A_ams_init_2/GBR_cf2.ss_A_base_ams_init_2-1)*100 + np.array(GRC.h_tot)[0]*(GRC_cf2.ss_A_ams_init_2/GRC_cf2.ss_A_base_ams_init_2-1)*100 + np.array(IRL.h_tot)[0]*(IRL_cf2.ss_A_ams_init_2/IRL_cf2.ss_A_base_ams_init_2-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf2.ss_A_ams_init_2/ITA_cf2.ss_A_base_ams_init_2-1)*100 + np.array(LUX.h_tot)[0]*(LUX_cf2.ss_A_ams_init_2/LUX_cf2.ss_A_base_ams_init_2-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf2.ss_A_ams_init_2/NLD_cf2.ss_A_base_ams_init_2-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf2.ss_A_ams_init_2/PRT_cf2.ss_A_base_ams_init_2-1)*100 + np.array(SWE.h_tot)[0]*(SWE_cf2.ss_A_ams_init_2/SWE_cf2.ss_A_base_ams_init_2-1)*100)/EUR15_h_tot[0]])]

	cf_2_sec_ams_init_ss[1:] = [ '%.1f' % elem for elem in cf_2_sec_ams_init_ss[1:] ]
	cf_2_ams_ss_init.append(cf_2_sec_ams_init_ss)

	cf_2_sec_ams_ss = 	[sec,
				(np.array(AUT_cf2.ss_A_ams)[-1]/np.array(AUT_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(BEL_cf2.ss_A_ams)[-1]/np.array(BEL_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(DEU_cf2.ss_A_ams)[-1]/np.array(DEU_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(DNK_cf2.ss_A_ams)[-1]/np.array(DNK_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(ESP_cf2.ss_A_ams)[-1]/np.array(ESP_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(FIN_cf2.ss_A_ams)[-1]/np.array(FIN_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(FRA_cf2.ss_A_ams)[-1]/np.array(FRA_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(GBR_cf2.ss_A_ams)[-1]/np.array(GBR_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(GRC_cf2.ss_A_ams)[-1]/np.array(GRC_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(IRL_cf2.ss_A_ams)[-1]/np.array(IRL_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(ITA_cf2.ss_A_ams)[-1]/np.array(ITA_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(LUX_cf2.ss_A_ams)[-1]/np.array(LUX_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(NLD_cf2.ss_A_ams)[-1]/np.array(NLD_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(PRT_cf2.ss_A_ams)[-1]/np.array(PRT_cf2.ss_A_base_ams)[-1]-1)*100,
				(np.array(SWE_cf2.ss_A_ams)[-1]/np.array(SWE_cf2.ss_A_base_ams)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.ss_A_ams)[-1]/np.array(DEU_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.ss_A_ams)[-1]/np.array(FRA_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.ss_A_ams)[-1]/np.array(ITA_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.ss_A_ams)[-1]/np.array(GBR_cf2.ss_A_base_ams)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2.ss_A_ams)[-1]/np.array(AUT_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf2.ss_A_ams)[-1]/np.array(BEL_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.ss_A_ams)[-1]/np.array(DEU_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2.ss_A_ams)[-1]/np.array(DNK_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2.ss_A_ams)[-1]/np.array(ESP_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2.ss_A_ams)[-1]/np.array(FIN_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.ss_A_ams)[-1]/np.array(FRA_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.ss_A_ams)[-1]/np.array(GBR_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2.ss_A_ams)[-1]/np.array(GRC_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2.ss_A_ams)[-1]/np.array(IRL_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.ss_A_ams)[-1]/np.array(ITA_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf2.ss_A_ams)[-1]/np.array(LUX_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2.ss_A_ams)[-1]/np.array(NLD_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2.ss_A_ams)[-1]/np.array(PRT_cf2.ss_A_base_ams)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2.ss_A_ams)[-1]/np.array(SWE_cf2.ss_A_base_ams)[-1]-1)*100)/EUR15_h_tot[-1]])]

	cf_2_sec_ams_ss[1:] = [ '%.1f' % elem for elem in cf_2_sec_ams_ss[1:] ]
	cf_2_ams_ss.append(cf_2_sec_ams_ss)

	cf_2_sec_ams = 	[sec,
				(np.array(AUT_cf2.A_tot_ams)[-1]/np.array(AUT.A_tot_ams)[-1]-1)*100,
				(np.array(BEL_cf2.A_tot_ams)[-1]/np.array(BEL.A_tot_ams)[-1]-1)*100,
				(np.array(DEU_cf2.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100,
				(np.array(DNK_cf2.A_tot_ams)[-1]/np.array(DNK.A_tot_ams)[-1]-1)*100,
				(np.array(ESP_cf2.A_tot_ams)[-1]/np.array(ESP.A_tot_ams)[-1]-1)*100,
				(np.array(FIN_cf2.A_tot_ams)[-1]/np.array(FIN.A_tot_ams)[-1]-1)*100,
				(np.array(FRA_cf2.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100,
				(np.array(GBR_cf2.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100,
				(np.array(GRC_cf2.A_tot_ams)[-1]/np.array(GRC.A_tot_ams)[-1]-1)*100,
				(np.array(IRL_cf2.A_tot_ams)[-1]/np.array(IRL.A_tot_ams)[-1]-1)*100,
				(np.array(ITA_cf2.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100,
				(np.array(LUX_cf2.A_tot_ams)[-1]/np.array(LUX.A_tot_ams)[-1]-1)*100,
				(np.array(NLD_cf2.A_tot_ams)[-1]/np.array(NLD.A_tot_ams)[-1]-1)*100,
				(np.array(PRT_cf2.A_tot_ams)[-1]/np.array(PRT.A_tot_ams)[-1]-1)*100,
				(np.array(SWE_cf2.A_tot_ams)[-1]/np.array(SWE.A_tot_ams)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2.A_tot_ams)[-1]/np.array(AUT.A_tot_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf2.A_tot_ams)[-1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.A_tot_ams)[-1]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2.A_tot_ams)[-1]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2.A_tot_ams)[-1]/np.array(ESP.A_tot_ams)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2.A_tot_ams)[-1]/np.array(FIN.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.A_tot_ams)[-1]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GBR_cf2.A_tot_ams)[-1]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2.A_tot_ams)[-1]/np.array(IRL.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.A_tot_ams)[-1]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf2.A_tot_ams)[-1]/np.array(LUX.A_tot_ams)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2.A_tot_ams)[-1]/np.array(NLD.A_tot_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2.A_tot_ams)[-1]/np.array(PRT.A_tot_ams)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2.A_tot_ams)[-1]/np.array(SWE.A_tot_ams)[-1]-1)*100)/EUR15_h_tot[-1]])]
	cf_2_sec_ams[1:] = [ '%.1f' % elem for elem in cf_2_sec_ams[1:] ]
	cf_2_ams.append(cf_2_sec_ams)

pd.DataFrame(cf_2_ams_ss_init).to_excel('../output/figures/Counterfactual_2_ams_ss_init.xlsx', index = False, header = False)
pd.DataFrame(cf_2_ams_ss).to_excel('../output/figures/Counterfactual_2_ams_ss.xlsx', index = False, header = False)
pd.DataFrame(cf_2_ams).to_excel('../output/figures/Counterfactual_2_ams.xlsx', index = False, header = False)


'nps'
cf_2_nps_ss_init = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI', 'EU15', 'EU15_data']]
cf_2_nps_ss = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE','EUPERI', 'EU15', 'EU15_data']]
cf_2_nps = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', 'EUCORE', 'EUPERI','EU15', 'EU15_data']]


AUT_cf2=counterfactual('AUT')
BEL_cf2=counterfactual('BEL')
DEU_cf2=counterfactual('DEU')
DNK_cf2=counterfactual('DNK')
ESP_cf2=counterfactual('ESP')
FIN_cf2=counterfactual('FIN')
FRA_cf2=counterfactual('FRA')
GBR_cf2=counterfactual('GBR')
GRC_cf2=counterfactual('GRC')
IRL_cf2=counterfactual('IRL')
ITA_cf2=counterfactual('ITA')
LUX_cf2=counterfactual('LUX')
NLD_cf2=counterfactual('NLD')
PRT_cf2=counterfactual('PRT')
SWE_cf2=counterfactual('SWE')
EU15_cf2 = counterfactual('EU15')

AUT_cf2.baseline()
BEL_cf2.baseline()
DEU_cf2.baseline()
DNK_cf2.baseline()
ESP_cf2.baseline()
FIN_cf2.baseline()
FRA_cf2.baseline()
GBR_cf2.baseline()
GRC_cf2.baseline()
IRL_cf2.baseline()
ITA_cf2.baseline()
LUX_cf2.baseline()
NLD_cf2.baseline()
PRT_cf2.baseline()
SWE_cf2.baseline()
AUT_cf2.baseline()
EU15_cf2.baseline()

for sec in ['all']:
	AUT_cf2.feed_US_productivity_growth(21, sec)
	BEL_cf2.feed_US_productivity_growth(21, sec)
	DEU_cf2.feed_US_productivity_growth(21, sec)
	DNK_cf2.feed_US_productivity_growth(21, sec)
	ESP_cf2.feed_US_productivity_growth(21, sec)
	FIN_cf2.feed_US_productivity_growth(21, sec)
	FRA_cf2.feed_US_productivity_growth(21, sec)
	GBR_cf2.feed_US_productivity_growth(21, sec)
	GRC_cf2.feed_US_productivity_growth(21, sec)
	IRL_cf2.feed_US_productivity_growth(21, sec)
	ITA_cf2.feed_US_productivity_growth(21, sec)
	LUX_cf2.feed_US_productivity_growth(21, sec)
	NLD_cf2.feed_US_productivity_growth(21, sec)
	PRT_cf2.feed_US_productivity_growth(21, sec)
	SWE_cf2.feed_US_productivity_growth(21, sec)
	EU15_cf2.feed_US_productivity_growth(21, sec)

	AUT_cf2.shift_share(sec)
	BEL_cf2.shift_share(sec)
	DEU_cf2.shift_share(sec)
	DNK_cf2.shift_share(sec)
	ESP_cf2.shift_share(sec)
	FIN_cf2.shift_share(sec)
	FRA_cf2.shift_share(sec)
	GBR_cf2.shift_share(sec)
	GRC_cf2.shift_share(sec)
	IRL_cf2.shift_share(sec)
	ITA_cf2.shift_share(sec)
	LUX_cf2.shift_share(sec)
	NLD_cf2.shift_share(sec)
	PRT_cf2.shift_share(sec)
	SWE_cf2.shift_share(sec)
	EU15_cf2.shift_share(sec)

cf_2_sec_nps_init_ss = 	[
			(AUT_cf2.ss_A_nps_init_2/AUT_cf2.ss_A_base_nps_init_2-1)*100,
			(BEL_cf2.ss_A_nps_init_2/BEL_cf2.ss_A_base_nps_init_2-1)*100,
			(DEU_cf2.ss_A_nps_init_2/DEU_cf2.ss_A_base_nps_init_2-1)*100,
			(DNK_cf2.ss_A_nps_init_2/DNK_cf2.ss_A_base_nps_init_2-1)*100,
			(ESP_cf2.ss_A_nps_init_2/ESP_cf2.ss_A_base_nps_init_2-1)*100,
			(FIN_cf2.ss_A_nps_init_2/FIN_cf2.ss_A_base_nps_init_2-1)*100,
			(FRA_cf2.ss_A_nps_init_2/FRA_cf2.ss_A_base_nps_init_2-1)*100,
			(GBR_cf2.ss_A_nps_init_2/GBR_cf2.ss_A_base_nps_init_2-1)*100,
			(GRC_cf2.ss_A_nps_init_2/GRC_cf2.ss_A_base_nps_init_2-1)*100,
			(IRL_cf2.ss_A_nps_init_2/IRL_cf2.ss_A_base_nps_init_2-1)*100,
			(ITA_cf2.ss_A_nps_init_2/ITA_cf2.ss_A_base_nps_init_2-1)*100,
			(LUX_cf2.ss_A_nps_init_2/LUX_cf2.ss_A_base_nps_init_2-1)*100,
			(NLD_cf2.ss_A_nps_init_2/NLD_cf2.ss_A_base_nps_init_2-1)*100,
			(PRT_cf2.ss_A_nps_init_2/PRT_cf2.ss_A_base_nps_init_2-1)*100,
			(SWE_cf2.ss_A_nps_init_2/SWE_cf2.ss_A_base_nps_init_2-1)*100,
			np.array([(np.array(DEU.h_tot)[0]*(DEU_cf2.ss_A_nps_init_2/DEU_cf2.ss_A_base_nps_init_2-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf2.ss_A_nps_init_2/FRA_cf2.ss_A_base_nps_init_2-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf2.ss_A_nps_init_2/ITA_cf2.ss_A_base_nps_init_2-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf2.ss_A_nps_init_2/GBR_cf2.ss_A_base_nps_init_2-1)*100)/EUR4_h_tot[0]]),
			np.array([(np.array(DEU.h_tot)[0]*(DEU_cf2.ss_A_nps_init_2/DEU_cf2.ss_A_base_nps_init_2-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf2.ss_A_nps_init_2/FRA_cf2.ss_A_base_nps_init_2-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf2.ss_A_nps_init_2/BEL_cf2.ss_A_base_nps_init_2-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf2.ss_A_nps_init_2/NLD_cf2.ss_A_base_nps_init_2-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf2.ss_A_nps_init_2/DNK_cf2.ss_A_base_nps_init_2-1)*100)/EURCORE_h_tot[0]]),
			np.array([(np.array(GRC.h_tot)[0]*(GRC_cf2.ss_A_nps_init_2/GRC_cf2.ss_A_base_nps_init_2-1)*100 + np.array(IRL.h_tot)[0]*(IRL_cf2.ss_A_nps_init_2/IRL_cf2.ss_A_base_nps_init_2-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf2.ss_A_nps_init_2/PRT_cf2.ss_A_base_nps_init_2-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf2.ss_A_nps_init_2/ESP_cf2.ss_A_base_nps_init_2-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf2.ss_A_nps_init_2/ITA_cf2.ss_A_base_nps_init_2-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf2.ss_A_nps_init_2/GBR_cf2.ss_A_base_nps_init_2-1)*100)/EURPERI_h_tot[0]]),
			np.array([(np.array(AUT.h_tot)[0]*(AUT_cf2.ss_A_nps_init_2/AUT_cf2.ss_A_base_nps_init_2-1)*100 + np.array(BEL.h_tot)[0]*(BEL_cf2.ss_A_nps_init_2/BEL_cf2.ss_A_base_nps_init_2-1)*100 + np.array(DEU.h_tot)[0]*(DEU_cf2.ss_A_nps_init_2/DEU_cf2.ss_A_base_nps_init_2-1)*100 + np.array(DNK.h_tot)[0]*(DNK_cf2.ss_A_nps_init_2/DNK_cf2.ss_A_base_nps_init_2-1)*100 + np.array(ESP.h_tot)[0]*(ESP_cf2.ss_A_nps_init_2/ESP_cf2.ss_A_base_nps_init_2-1)*100 + np.array(FIN.h_tot)[0]*(FIN_cf2.ss_A_nps_init_2/FIN_cf2.ss_A_base_nps_init_2-1)*100 + np.array(FRA.h_tot)[0]*(FRA_cf2.ss_A_nps_init_2/FRA_cf2.ss_A_base_nps_init_2-1)*100 + np.array(GBR.h_tot)[0]*(GBR_cf2.ss_A_nps_init_2/GBR_cf2.ss_A_base_nps_init_2-1)*100 + np.array(GRC.h_tot)[0]*(GRC_cf2.ss_A_nps_init_2/GRC_cf2.ss_A_base_nps_init_2-1)*100 + np.array(IRL.h_tot)[0]*(IRL_cf2.ss_A_nps_init_2/IRL_cf2.ss_A_base_nps_init_2-1)*100 + np.array(ITA.h_tot)[0]*(ITA_cf2.ss_A_nps_init_2/ITA_cf2.ss_A_base_nps_init_2-1)*100 + np.array(LUX.h_tot)[0]*(LUX_cf2.ss_A_nps_init_2/LUX_cf2.ss_A_base_nps_init_2-1)*100 + np.array(NLD.h_tot)[0]*(NLD_cf2.ss_A_nps_init_2/NLD_cf2.ss_A_base_nps_init_2-1)*100 + np.array(PRT.h_tot)[0]*(PRT_cf2.ss_A_nps_init_2/PRT_cf2.ss_A_base_nps_init_2-1)*100 + np.array(SWE.h_tot)[0]*(SWE_cf2.ss_A_nps_init_2/SWE_cf2.ss_A_base_nps_init_2-1)*100)/EUR15_h_tot[0]]),
			(EU15_cf2.ss_A_nps_init_2 / EU15_cf2.ss_A_base_nps_init_2 - 1) * 100]

cf_2_sec_nps_init_ss[1:] = [ '%.1f' % elem for elem in cf_2_sec_nps_init_ss[1:] ]
cf_2_nps_ss_init.append(cf_2_sec_nps_init_ss)

cf_2_sec_nps_ss = 	[
			(np.array(AUT_cf2.ss_A_nps)[-1]/np.array(AUT_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(BEL_cf2.ss_A_nps)[-1]/np.array(BEL_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(DEU_cf2.ss_A_nps)[-1]/np.array(DEU_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(DNK_cf2.ss_A_nps)[-1]/np.array(DNK_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(ESP_cf2.ss_A_nps)[-1]/np.array(ESP_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(FIN_cf2.ss_A_nps)[-1]/np.array(FIN_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(FRA_cf2.ss_A_nps)[-1]/np.array(FRA_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(GBR_cf2.ss_A_nps)[-1]/np.array(GBR_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(GRC_cf2.ss_A_nps)[-1]/np.array(GRC_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(IRL_cf2.ss_A_nps)[-1]/np.array(IRL_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(ITA_cf2.ss_A_nps)[-1]/np.array(ITA_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(LUX_cf2.ss_A_nps)[-1]/np.array(LUX_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(NLD_cf2.ss_A_nps)[-1]/np.array(NLD_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(PRT_cf2.ss_A_nps)[-1]/np.array(PRT_cf2.ss_A_base_nps)[-1]-1)*100,
			(np.array(SWE_cf2.ss_A_nps)[-1]/np.array(SWE_cf2.ss_A_base_nps)[-1]-1)*100,
			np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.ss_A_nps)[-1]/np.array(DEU_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.ss_A_nps)[-1]/np.array(FRA_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.ss_A_nps)[-1]/np.array(ITA_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.ss_A_nps)[-1]/np.array(GBR_cf2.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
			np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.ss_A_nps)[-1]/np.array(DEU_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.ss_A_nps)[-1]/np.array(FRA_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf2.ss_A_nps)[-1]/np.array(BEL_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2.ss_A_nps)[-1]/np.array(NLD_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2.ss_A_nps)[-1]/np.array(DNK_cf2.ss_A_base_nps)[-1]-1)*100)/EURCORE_h_tot[-1]]),
			np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2.ss_A_nps)[-1]/np.array(GRC_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2.ss_A_nps)[-1]/np.array(IRL_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2.ss_A_nps)[-1]/np.array(PRT_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2.ss_A_nps)[-1]/np.array(ESP_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.ss_A_nps)[-1]/np.array(ITA_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.ss_A_nps)[-1]/np.array(GBR_cf2.ss_A_base_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
			np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2.ss_A_nps)[-1]/np.array(AUT_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf2.ss_A_nps)[-1]/np.array(BEL_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.ss_A_nps)[-1]/np.array(DEU_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2.ss_A_nps)[-1]/np.array(DNK_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2.ss_A_nps)[-1]/np.array(ESP_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2.ss_A_nps)[-1]/np.array(FIN_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.ss_A_nps)[-1]/np.array(FRA_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.ss_A_nps)[-1]/np.array(GBR_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2.ss_A_nps)[-1]/np.array(GRC_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2.ss_A_nps)[-1]/np.array(IRL_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.ss_A_nps)[-1]/np.array(ITA_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf2.ss_A_nps)[-1]/np.array(LUX_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2.ss_A_nps)[-1]/np.array(NLD_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2.ss_A_nps)[-1]/np.array(PRT_cf2.ss_A_base_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2.ss_A_nps)[-1]/np.array(SWE_cf2.ss_A_base_nps)[-1]-1)*100)/EUR15_h_tot[-1]]),
			(np.array(EU15_cf2.ss_A_nps)[-1] / np.array(EU15_cf2.ss_A_base_nps)[-1] - 1) * 100]

cf_2_sec_nps_ss[1:] = [ '%.1f' % elem for elem in cf_2_sec_nps_ss[1:] ]
cf_2_nps_ss.append(cf_2_sec_nps_ss)

cf_2_sec_nps = 	[
			(np.array(AUT_cf2.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100,
			(np.array(BEL_cf2.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100,
			(np.array(DEU_cf2.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100,
			(np.array(DNK_cf2.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100,
			(np.array(ESP_cf2.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100,
			(np.array(FIN_cf2.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100,
			(np.array(FRA_cf2.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100,
			(np.array(GBR_cf2.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100,
			(np.array(GRC_cf2.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100,
			(np.array(IRL_cf2.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100,
			(np.array(ITA_cf2.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100,
			(np.array(LUX_cf2.A_tot_nps)[-1]/np.array(LUX.A_tot_nps)[-1]-1)*100,
			(np.array(NLD_cf2.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100,
			(np.array(PRT_cf2.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100,
			(np.array(SWE_cf2.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100,
			np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
			np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf2.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100)/EURCORE_h_tot[-1]]),
			np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
			np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(np.array(BEL_cf2.A_tot_nps)[-1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GBR_cf2.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(np.array(LUX_cf2.A_tot_nps)[-1]/np.array(LUX.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100)/EUR15_h_tot[-1]]),
			(np.array(EU15_cf2.A_tot_nps)[-1] / np.array(EU15.A_tot_nps)[-1] - 1) * 100]

cf_2_sec_nps[1:] = [ '%.1f' % elem for elem in cf_2_sec_nps[1:] ]
cf_2_nps.append(cf_2_sec_nps)

pd.DataFrame(cf_2_nps_ss_init).to_excel('../output/figures/Counterfactual_2_nps_ss_init.xlsx', index = False, header = False)
pd.DataFrame(cf_2_nps_ss).to_excel('../output/figures/Counterfactual_2_nps_ss.xlsx', index = False, header = False)
pd.DataFrame(cf_2_nps).to_excel('../output/figures/Counterfactual_2_nps.xlsx', index = False, header = False)

EUR4_A_tot_nps_cf2 = ( np.array(DEU.h_tot)*np.array(DEU_cf2.A_tot_nps).astype(float)+ np.array(FRA.h_tot)* np.array(FRA_cf2.A_tot_nps).astype(float) + np.array(ITA.h_tot)*np.array(ITA_cf2.A_tot_nps).astype(float) + np.array(GBR.h_tot)*np.array(GBR_cf2.A_tot_nps).astype(float) ) / EUR4_h_tot
#EUR4_A_tot_nps_cf2_ss = ( np.array(DEU.h_tot)*DEU_cf2.ss_A_nps.astype(float)+ np.array(FRA.h_tot)* np.array(FRA_cf2.ss_A_nps).astype(float) + np.array(ITA.h_tot)*np.array(ITA_cf2.ss_A_nps).astype(float) + np.array(GBR.h_tot)*np.array(GBR_cf2.ss_A_nps).astype(float) ) / EUR4_h_tot

rel_EU4_cf2 = EUR4_A_tot_nps_cf2/np.array(A_tot_nps).flatten()
rel_EU4_obs = EUR4_A_tot_nps/np.array(A_tot_nps).flatten()
#rel_EU4_cf2_ss = EUR4_A_tot_nps_cf2_ss/np.array(A_tot_nps).flatten()
#rel_EU4_cf2_ss = rel_EU4_cf2_ss.values



'''
------------------------------------------------------------
	Counterfactual 3: Catch up Productivity with the US
------------------------------------------------------------
'''

'ams'
cf_2_catch_ams_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15', 'EU15_data']]
cf_2_catch_ams = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15', 'EU15_data']]
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
	IRL_cf2_catch=counterfactual('IRL')
	ITA_cf2_catch=counterfactual('ITA')
	LUX_cf2_catch=counterfactual('LUX')
	NLD_cf2_catch=counterfactual('NLD')
	PRT_cf2_catch=counterfactual('PRT')
	SWE_cf2_catch=counterfactual('SWE')
	EU15_cf2_catch = counterfactual('EU15')

	AUT_cf2_catch.baseline()
	BEL_cf2_catch.baseline()
	DEU_cf2_catch.baseline()
	DNK_cf2_catch.baseline()
	ESP_cf2_catch.baseline()
	FIN_cf2_catch.baseline()
	FRA_cf2_catch.baseline()
	GBR_cf2_catch.baseline()
	GRC_cf2_catch.baseline()
	IRL_cf2_catch.baseline()
	ITA_cf2_catch.baseline()
	LUX_cf2_catch.baseline()
	NLD_cf2_catch.baseline()
	PRT_cf2_catch.baseline()
	SWE_cf2_catch.baseline()
	AUT_cf2_catch.baseline()
	EU15_cf2_catch.baseline()

	AUT_cf2_catch.feed_catch_up_growth(0, sec)
	BEL_cf2_catch.feed_catch_up_growth(0, sec)
	DEU_cf2_catch.feed_catch_up_growth(0, sec)
	DNK_cf2_catch.feed_catch_up_growth(0, sec)
	ESP_cf2_catch.feed_catch_up_growth(0, sec)
	FIN_cf2_catch.feed_catch_up_growth(0, sec)
	FRA_cf2_catch.feed_catch_up_growth(0, sec)
	GBR_cf2_catch.feed_catch_up_growth(0, sec)
	GRC_cf2_catch.feed_catch_up_growth(0, sec)
	IRL_cf2_catch.feed_catch_up_growth(0, sec)
	ITA_cf2_catch.feed_catch_up_growth(0, sec)
	LUX_cf2_catch.feed_catch_up_growth(0, sec)
	NLD_cf2_catch.feed_catch_up_growth(0, sec)
	PRT_cf2_catch.feed_catch_up_growth(0, sec)
	SWE_cf2_catch.feed_catch_up_growth(0, sec)
	EU15_cf2_catch.feed_catch_up_growth(0, sec)

	cf_2_catch_sec_ams_ss = [sec,
							 (E[-1] / np.array(AUT_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(BEL_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(DEU_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(DNK_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(ESP_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(FIN_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(FRA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(GBR_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(GRC_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(IRL_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(ITA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(LUX_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(NLD_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(PRT_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 (E[-1] / np.array(SWE_cf2_catch.ss_A_base_ams)[-1] - 1) * 100,
							 np.array([(np.array(DEU.h_tot)[-1] * (
										 E[-1] / np.array(DEU_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(FRA.h_tot)[-1] * (
													E[-1] / np.array(FRA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(GBR.h_tot)[-1] * (
													E[-1] / np.array(GBR_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(ITA.h_tot)[-1] * (
													E[-1] / np.array(ITA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100) /
									   EUR4_h_tot[-1]]),
							 np.array([(np.array(DEU.h_tot)[-1] * (
										 E[-1] / np.array(DEU_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(FRA.h_tot)[-1] * (
													E[-1] / np.array(FRA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(NLD.h_tot)[-1] * (
													E[-1] / np.array(NLD_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(DNK.h_tot)[-1] * (
													E[-1] / np.array(DNK_cf2_catch.ss_A_base_ams)[-1] - 1) * 100) / (
												   EURCORE_h_tot[-1] - np.array(BEL.h_tot)[-1])]),
							 np.array([(np.array(GRC.h_tot)[-1] * (
										 E[-1] / np.array(GRC_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(IRL.h_tot)[-1] * (
													E[-1] / np.array(IRL_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(PRT.h_tot)[-1] * (
													E[-1] / np.array(PRT_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(ESP.h_tot)[-1] * (
													E[-1] / np.array(ESP_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(ITA.h_tot)[-1] * (
													E[-1] / np.array(ITA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(GBR.h_tot)[-1] * (
													E[-1] / np.array(GBR_cf2_catch.ss_A_base_ams)[-1] - 1) * 100) /
									   EURPERI_h_tot[-1]]),
							 np.array([(np.array(AUT.h_tot)[-1] * (
										 E[-1] / np.array(AUT_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(BEL.h_tot)[-1] * (
													E[-1] / np.array(BEL_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(DEU.h_tot)[-1] * (
													E[-1] / np.array(DEU_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(DNK.h_tot)[-1] * (
													E[-1] / np.array(DNK_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(ESP.h_tot)[-1] * (
													E[-1] / np.array(ESP_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(FIN.h_tot)[-1] * (
													E[-1] / np.array(FIN_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(FRA.h_tot)[-1] * (
													E[-1] / np.array(FRA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(GBR.h_tot)[-1] * (
													E[-1] / np.array(GBR_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(GRC.h_tot)[-1] * (
													E[-1] / np.array(GRC_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(IRL.h_tot)[-1] * (
													E[-1] / np.array(IRL_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(ITA.h_tot)[-1] * (
													E[-1] / np.array(ITA_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(LUX.h_tot)[-1] * (
													E[-1] / np.array(LUX_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(NLD.h_tot)[-1] * (
													E[-1] / np.array(NLD_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(PRT.h_tot)[-1] * (
													E[-1] / np.array(PRT_cf2_catch.ss_A_base_ams)[-1] - 1) * 100 +
										np.array(SWE.h_tot)[-1] * (
													E[-1] / np.array(SWE_cf2_catch.ss_A_base_ams)[-1] - 1) * 100) /
									   EUR15_h_tot[-1]]),
							 (E[-1] / np.array(EU15_cf2_catch.ss_A_base_ams)[-1] - 1) * 100]

	cf_2_catch_sec_ams_ss[1:] = ['%.1f' % elem for elem in cf_2_catch_sec_ams_ss[1:]]
	cf_2_catch_ams_ss.append(cf_2_catch_sec_ams_ss)

	cf_2_catch_sec_ams = [sec,
						  (np.array(AUT_cf2_catch.A_tot_ams)[-1] / np.array(AUT.A_tot_ams)[-1] - 1) * 100,
						  (np.array(BEL_cf2_catch.A_tot_ams)[-1] / np.array(BEL.A_tot_ams)[-1] - 1) * 100,
						  (np.array(DEU_cf2_catch.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[-1] - 1) * 100,
						  (np.array(DNK_cf2_catch.A_tot_ams)[-1] / np.array(DNK.A_tot_ams)[-1] - 1) * 100,
						  (np.array(ESP_cf2_catch.A_tot_ams)[-1] / np.array(ESP.A_tot_ams)[-1] - 1) * 100,
						  (np.array(FIN_cf2_catch.A_tot_ams)[-1] / np.array(FIN.A_tot_ams)[-1] - 1) * 100,
						  (np.array(FRA_cf2_catch.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[-1] - 1) * 100,
						  (np.array(GBR_cf2_catch.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[-1] - 1) * 100,
						  (np.array(GRC_cf2_catch.A_tot_ams)[-1] / np.array(GRC.A_tot_ams)[-1] - 1) * 100,
						  (np.array(IRL_cf2_catch.A_tot_ams)[-1] / np.array(IRL.A_tot_ams)[-1] - 1) * 100,
						  (np.array(ITA_cf2_catch.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[-1] - 1) * 100,
						  (np.array(LUX_cf2_catch.A_tot_ams)[-1] / np.array(LUX.A_tot_ams)[-1] - 1) * 100,
						  (np.array(NLD_cf2_catch.A_tot_ams)[-1] / np.array(NLD.A_tot_ams)[-1] - 1) * 100,
						  (np.array(PRT_cf2_catch.A_tot_ams)[-1] / np.array(PRT.A_tot_ams)[-1] - 1) * 100,
						  (np.array(SWE_cf2_catch.A_tot_ams)[-1] / np.array(SWE.A_tot_ams)[-1] - 1) * 100,
						  np.array([(np.array(DEU.h_tot)[-1] * (
									  np.array(DEU_cf2_catch.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[-1] - 1) * 100 +
									 np.array(FRA.h_tot)[-1] * (
												 np.array(FRA_cf2_catch.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[
											 -1] - 1) * 100 + np.array(GBR.h_tot)[-1] * (
												 np.array(GBR_cf2_catch.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[
											 -1] - 1) * 100 + np.array(ITA.h_tot)[-1] * (
												 np.array(ITA_cf2_catch.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[
											 -1] - 1) * 100) / EUR4_h_tot[-1]]),
						  np.array([(np.array(DEU.h_tot)[-1] * (
									  np.array(DEU_cf2_catch.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[-1] - 1) * 100 +
									 np.array(FRA.h_tot)[-1] * (
												 np.array(FRA_cf2_catch.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[
											 -1] - 1) * 100 + np.array(NLD.h_tot)[-1] * (
												 np.array(NLD_cf2_catch.A_tot_ams)[-1] / np.array(NLD.A_tot_ams)[
											 -1] - 1) * 100 + np.array(DNK.h_tot)[-1] * (
												 np.array(DNK_cf2_catch.A_tot_ams)[-1] / np.array(DNK.A_tot_ams)[
											 -1] - 1) * 100) / (EURCORE_h_tot[-1] - np.array(BEL.h_tot)[-1])]),
						  np.array([(np.array(GRC.h_tot)[-1] * (
									  np.array(GRC_cf2_catch.A_tot_ams)[-1] / np.array(GRC.A_tot_ams)[-1] - 1) * 100 +
									 np.array(IRL.h_tot)[-1] * (
												 np.array(IRL_cf2_catch.A_tot_ams)[-1] / np.array(IRL.A_tot_ams)[
											 -1] - 1) * 100 + np.array(PRT.h_tot)[-1] * (
												 np.array(PRT_cf2_catch.A_tot_ams)[-1] / np.array(PRT.A_tot_ams)[
											 -1] - 1) * 100 + np.array(ESP.h_tot)[-1] * (
												 np.array(ESP_cf2_catch.A_tot_ams)[-1] / np.array(ESP.A_tot_ams)[
											 -1] - 1) * 100 + np.array(ITA.h_tot)[-1] * (
												 np.array(ITA_cf2_catch.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[
											 -1] - 1) * 100 + np.array(GBR.h_tot)[-1] * (
												 np.array(GBR_cf2_catch.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[
											 -1] - 1) * 100) / EURPERI_h_tot[-1]]),
						  np.array([(np.array(AUT.h_tot)[-1] * (
									  np.array(AUT_cf2_catch.A_tot_ams)[-1] / np.array(AUT.A_tot_ams)[-1] - 1) * 100 +
									 np.array(DEU.h_tot)[-1] * (
												 np.array(DEU_cf2_catch.A_tot_ams)[-1] / np.array(DEU.A_tot_ams)[
											 -1] - 1) * 100 + np.array(DNK.h_tot)[-1] * (
												 np.array(DNK_cf2_catch.A_tot_ams)[-1] / np.array(DNK.A_tot_ams)[
											 -1] - 1) * 100 + np.array(ESP.h_tot)[-1] * (
												 np.array(ESP_cf2_catch.A_tot_ams)[-1] / np.array(ESP.A_tot_ams)[
											 -1] - 1) * 100 + np.array(FIN.h_tot)[-1] * (
												 np.array(FIN_cf2_catch.A_tot_ams)[-1] / np.array(FIN.A_tot_ams)[
											 -1] - 1) * 100 + np.array(FRA.h_tot)[-1] * (
												 np.array(FRA_cf2_catch.A_tot_ams)[-1] / np.array(FRA.A_tot_ams)[
											 -1] - 1) * 100 + np.array(GBR.h_tot)[-1] * (
												 np.array(GBR_cf2_catch.A_tot_ams)[-1] / np.array(GBR.A_tot_ams)[
											 -1] - 1) * 100 + np.array(GRC.h_tot)[-1] * (
												 np.array(GRC_cf2_catch.A_tot_ams)[-1] / np.array(GRC.A_tot_ams)[
											 -1] - 1) * 100 + np.array(IRL.h_tot)[-1] * (
												 np.array(IRL_cf2_catch.A_tot_ams)[-1] / np.array(IRL.A_tot_ams)[
											 -1] - 1) * 100 + np.array(ITA.h_tot)[-1] * (
												 np.array(ITA_cf2_catch.A_tot_ams)[-1] / np.array(ITA.A_tot_ams)[
											 -1] - 1) * 100 + np.array(NLD.h_tot)[-1] * (
												 np.array(NLD_cf2_catch.A_tot_ams)[-1] / np.array(NLD.A_tot_ams)[
											 -1] - 1) * 100 + np.array(PRT.h_tot)[-1] * (
												 np.array(PRT_cf2_catch.A_tot_ams)[-1] / np.array(PRT.A_tot_ams)[
											 -1] - 1) * 100 + np.array(SWE.h_tot)[-1] * (
												 np.array(SWE_cf2_catch.A_tot_ams)[-1] / np.array(SWE.A_tot_ams)[
											 -1] - 1) * 100) / (
												EUR15_h_tot[-1] - np.array(BEL.h_tot)[-1] - np.array(LUX.h_tot)[-1])]),
						  (np.array(EU15_cf2_catch.A_tot_ams)[-1] / np.array(EU15.A_tot_ams)[-1] - 1) * 100]

	cf_2_catch_sec_ams[1:] = ['%.1f' % elem for elem in cf_2_catch_sec_ams[1:]]
	cf_2_catch_ams.append(cf_2_catch_sec_ams)

pd.DataFrame(cf_2_catch_ams_ss).to_excel('../output/figures/Counterfactual_2_catch_ams_ss.xlsx', index=False, header=False)
pd.DataFrame(cf_2_catch_ams).to_excel('../output/figures/Counterfactual_2_catch_ams.xlsx', index=False, header=False)


'nps'
cf_2_catch_nps_ss = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15', 'EU15_data']]
cf_2_catch_nps = [['','AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15', 'EU15_data']]
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
	IRL_cf2_catch=counterfactual('IRL')
	ITA_cf2_catch=counterfactual('ITA')
	LUX_cf2_catch=counterfactual('LUX')
	NLD_cf2_catch=counterfactual('NLD')
	PRT_cf2_catch=counterfactual('PRT')
	SWE_cf2_catch=counterfactual('SWE')
	EU15_cf2_catch = counterfactual('EU15')

	AUT_cf2_catch.baseline()
	BEL_cf2_catch.baseline()
	DEU_cf2_catch.baseline()
	DNK_cf2_catch.baseline()
	ESP_cf2_catch.baseline()
	FIN_cf2_catch.baseline()
	FRA_cf2_catch.baseline()
	GBR_cf2_catch.baseline()
	GRC_cf2_catch.baseline()
	IRL_cf2_catch.baseline()
	ITA_cf2_catch.baseline()
	LUX_cf2_catch.baseline()
	NLD_cf2_catch.baseline()
	PRT_cf2_catch.baseline()
	SWE_cf2_catch.baseline()
	AUT_cf2_catch.baseline()
	EU15_cf2_catch.baseline()

	AUT_cf2_catch.feed_catch_up_growth(0, sec)
	BEL_cf2_catch.feed_catch_up_growth(0, sec)
	DEU_cf2_catch.feed_catch_up_growth(0, sec)
	DNK_cf2_catch.feed_catch_up_growth(0, sec)
	ESP_cf2_catch.feed_catch_up_growth(0, sec)
	FIN_cf2_catch.feed_catch_up_growth(0, sec)
	FRA_cf2_catch.feed_catch_up_growth(0, sec)
	GBR_cf2_catch.feed_catch_up_growth(0, sec)
	GRC_cf2_catch.feed_catch_up_growth(0, sec)
	IRL_cf2_catch.feed_catch_up_growth(0, sec)
	ITA_cf2_catch.feed_catch_up_growth(0, sec)
	LUX_cf2_catch.feed_catch_up_growth(0, sec)
	NLD_cf2_catch.feed_catch_up_growth(0, sec)
	PRT_cf2_catch.feed_catch_up_growth(0, sec)
	SWE_cf2_catch.feed_catch_up_growth(0, sec)
	EU15_cf2_catch.feed_catch_up_growth(0, sec)


	cf_2_catch_sec_nps_ss = 	[sec,
				(E[-1]/np.array(AUT_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(BEL_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(FIN_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(IRL_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(LUX_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				(E[-1]/np.array(SWE_cf2_catch.ss_A_base_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(E[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(E[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(E[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(E[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100)/(EURCORE_h_tot[-1] -np.array(BEL.h_tot)[-1])]),
				np.array([(np.array(GRC.h_tot)[-1]*(E[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(E[-1]/np.array(IRL_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(E[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(E[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(E[-1]/np.array(AUT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(E[-1]/np.array(BEL_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(E[-1]/np.array(DEU_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(E[-1]/np.array(DNK_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(E[-1]/np.array(ESP_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(E[-1]/np.array(FIN_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(E[-1]/np.array(FRA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(E[-1]/np.array(GBR_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(E[-1]/np.array(GRC_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(E[-1]/np.array(IRL_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(E[-1]/np.array(ITA_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(LUX.h_tot)[-1]*(E[-1]/np.array(LUX_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(E[-1]/np.array(NLD_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(E[-1]/np.array(PRT_cf2_catch.ss_A_base_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(E[-1]/np.array(SWE_cf2_catch.ss_A_base_nps)[-1]-1)*100)/EUR15_h_tot[-1]]),
				(E[-1] / np.array(EU15_cf2_catch.ss_A_base_nps)[-1] - 1) * 100]

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
				(np.array(IRL_cf2_catch.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100,
				(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100,
				(np.array(LUX_cf2_catch.A_tot_nps)[-1]/np.array(LUX.A_tot_nps)[-1]-1)*100,
				(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100,
				(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100,
				(np.array(SWE_cf2_catch.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100,
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1]]),
				np.array([(np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100  + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100)/(EURCORE_h_tot[-1] -np.array(BEL.h_tot)[-1])]),
				np.array([(np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2_catch.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1]]),
				np.array([(np.array(AUT.h_tot)[-1]*(np.array(AUT_cf2_catch.A_tot_nps)[-1]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(np.array(DEU_cf2_catch.A_tot_nps)[-1]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(np.array(DNK_cf2_catch.A_tot_nps)[-1]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(np.array(ESP_cf2_catch.A_tot_nps)[-1]/np.array(ESP.A_tot_nps)[-1]-1)*100 + np.array(FIN.h_tot)[-1]*(np.array(FIN_cf2_catch.A_tot_nps)[-1]/np.array(FIN.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(np.array(FRA_cf2_catch.A_tot_nps)[-1]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(np.array(GBR_cf2_catch.A_tot_nps)[-1]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(GRC.h_tot)[-1]*(np.array(GRC_cf2_catch.A_tot_nps)[-1]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(IRL.h_tot)[-1]*(np.array(IRL_cf2_catch.A_tot_nps)[-1]/np.array(IRL.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(np.array(ITA_cf2_catch.A_tot_nps)[-1]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(NLD.h_tot)[-1]*(np.array(NLD_cf2_catch.A_tot_nps)[-1]/np.array(NLD.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(np.array(PRT_cf2_catch.A_tot_nps)[-1]/np.array(PRT.A_tot_nps)[-1]-1)*100 + np.array(SWE.h_tot)[-1]*(np.array(SWE_cf2_catch.A_tot_nps)[-1]/np.array(SWE.A_tot_nps)[-1]-1)*100)/(EUR15_h_tot[-1]-np.array(BEL.h_tot)[-1]-np.array(LUX.h_tot)[-1])]),
				(np.array(EU15_cf2_catch.A_tot_nps)[-1] / np.array(EU15.A_tot_nps)[-1] - 1) * 100]

	cf_2_catch_sec_nps[1:] = [ '%.1f' % elem for elem in cf_2_catch_sec_nps[1:] ]
	cf_2_catch_nps.append(cf_2_catch_sec_nps)

	# save results
	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "FRA"), "LS_m_cf2_" + sec] = FRA_cf2_catch.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "FRA"), "LS_m_cf2_" + sec] = FRA_cf2_catch.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "FRA"), "LS_m_cf2_" + sec] = FRA_cf2_catch.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "FRA"), "LS_m_cf2_" + sec] = FRA_cf2_catch.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "FRA"), "LS_m_cf2_" + sec] = FRA_cf2_catch.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "FRA"), "LS_m_cf2_" + sec] = FRA_cf2_catch.share_nps_nps_m

	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "DEU"), "LS_m_cf2_" + sec] = DEU_cf2_catch.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "DEU"), "LS_m_cf2_" + sec] = DEU_cf2_catch.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "DEU"), "LS_m_cf2_" + sec] = DEU_cf2_catch.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "DEU"), "LS_m_cf2_" + sec] = DEU_cf2_catch.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "DEU"), "LS_m_cf2_" + sec] = DEU_cf2_catch.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "DEU"), "LS_m_cf2_" + sec] = DEU_cf2_catch.share_nps_nps_m

	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "GBR"), "LS_m_cf2_" + sec] = GBR_cf2_catch.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "GBR"), "LS_m_cf2_" + sec] = GBR_cf2_catch.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "GBR"), "LS_m_cf2_" + sec] = GBR_cf2_catch.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "GBR"), "LS_m_cf2_" + sec] = GBR_cf2_catch.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "GBR"), "LS_m_cf2_" + sec] = GBR_cf2_catch.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "GBR"), "LS_m_cf2_" + sec] = GBR_cf2_catch.share_nps_nps_m

	h_data_export.loc[
		(h_data_export.sector == "agr") & (h_data_export.country == "ITA"), "LS_m_cf2_" + sec] = ITA_cf2_catch.share_agr_nps_m
	h_data_export.loc[
		(h_data_export.sector == "man") & (h_data_export.country == "ITA"), "LS_m_cf2_" + sec] = ITA_cf2_catch.share_man_nps_m
	h_data_export.loc[
		(h_data_export.sector == "bss") & (h_data_export.country == "ITA"), "LS_m_cf2_" + sec] = ITA_cf2_catch.share_bss_nps_m
	h_data_export.loc[
		(h_data_export.sector == "fin") & (h_data_export.country == "ITA"), "LS_m_cf2_" + sec] = ITA_cf2_catch.share_fin_nps_m
	h_data_export.loc[
		(h_data_export.sector == "trd") & (h_data_export.country == "ITA"), "LS_m_cf2_" + sec] = ITA_cf2_catch.share_trd_nps_m
	h_data_export.loc[
		(h_data_export.sector == "nps") & (h_data_export.country == "ITA"), "LS_m_cf2_" + sec] = ITA_cf2_catch.share_nps_nps_m

	h_data_export.loc[(h_data_export.country == "EU4"), "LS_m_cf2_" + sec] = (h_data_export.loc[
																				  h_data_export.country == "ITA", "LS_m_cf2_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "ITA", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "LS_m_cf2_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "DEU", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "LS_m_cf2_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "GBR", "h_tot"] +
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "LS_m_cf2_" + sec] *
																			  h_data_export.loc[
																				  h_data_export.country == "FRA", "h_tot"]) / np.tile(
		EUR4_h_tot, 6)

pd.DataFrame(cf_2_catch_nps_ss).to_excel('../output/figures/Counterfactual_2_catch_nps_ss.xlsx', index=False, header=False)
pd.DataFrame(cf_2_catch_nps).to_excel('../output/figures/Counterfactual_2_catch_nps.xlsx', index=False, header=False)

data_export["data_predictions"] = h_data_export

filehandler = open('../output/data/predictions_vs_data.obj', 'wb')
pickle.dump(data_export, filehandler)
filehandler.close()


"""
C1 referee suggestion: reallocation gap with US is zero after 1990
"""

'ams'
id_1990 = 21

cf_3 = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15']]
A_cfs = []
cf_results = []
country_models = [AUT, BEL, DEU, DNK, ESP, IRL, ITA, GRC, FIN, FRA, LUX, GBR, NLD, PRT, SWE]

for country in country_models:
	cf4 = country
	agr_cf = np.array(cf4.share_agr_ams_m).copy()
	for i in range(id_1990, 49):
		agr_cf[i + 1] = USA.share_agr.diff().values[i + 1] + agr_cf[i]
	man_cf = np.array(cf4.share_man_ams_m).copy()
	for i in range(id_1990, 49):
		man_cf[i + 1] = USA.share_man.diff().values[i + 1] + man_cf[i]
	ser_cf = np.array(cf4.share_ser_ams_m).copy()
	for i in range(id_1990, 49):
		ser_cf[i + 1] = USA.share_ser.diff().values[i + 1] + ser_cf[i]

	A_cfs.append((cf4.A_agr*agr_cf + cf4.A_man*man_cf + cf4.A_ser*ser_cf)[-1])
	cf_results.append((A_cfs[-1] / np.array(cf4.A_tot_ams)[-1] - 1) * 100)

# EU4
cf_results.append((np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_ams)[-1]-1)*100)/EUR4_h_tot[-1])
# EU15
cf_results.append((np.array(AUT.h_tot)[-1]*(A_cfs[0]/np.array(AUT.A_tot_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_ams)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_ams)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_ams)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_ams)[-1]-1)*100+ np.array(FIN.h_tot)[-1]*(A_cfs[8]/np.array(FIN.A_tot_ams)[-1]-1)*100+ np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_ams)[-1]-1)*100+ np.array(LUX.h_tot)[-1]*(A_cfs[10]/np.array(LUX.A_tot_ams)[-1]-1)*100+ np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_ams)[-1]-1)*100+ np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_ams)[-1]-1)*100+ np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_ams)[-1]-1)*100+ np.array(SWE.h_tot)[-1]*(A_cfs[14]/np.array(SWE.A_tot_ams)[-1]-1)*100)/EUR15_h_tot[-1])
# EUCORE
cf_results.append((np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_ams)[-1]-1)*100+np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_ams)[-1]-1)*100)/EURCORE_h_tot[-1])
# EUPERI
cf_results.append((np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_ams)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_ams)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_ams)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_ams)[-1]-1)*100)/EURPERI_h_tot[-1])

cf_3.append(cf_results)

pd.DataFrame(cf_3).to_excel('../output/figures/Counterfactual_3_ams.xlsx', index = False, header = False)

'nps'
id_1990 = 21

cf_3 = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15']]
A_cfs_ts = []
A_cfs = []
cf_results = []
country_models = [AUT, BEL, DEU, DNK, ESP, IRL, ITA, GRC, FIN, FRA, LUX, GBR, NLD, PRT, SWE]

for country in country_models:
	cf4 = country
	agr_cf = np.array(cf4.share_agr_nps_m).copy()
	for i in range(id_1990, 49):
		agr_cf[i + 1] = USA.share_agr.diff().values[i + 1] + agr_cf[i]
	man_cf = np.array(cf4.share_man_nps_m).copy()
	for i in range(id_1990, 49):
		man_cf[i + 1] = USA.share_man.diff().values[i + 1] + man_cf[i]
	trd_cf = np.array(cf4.share_trd_nps_m).copy()
	for i in range(id_1990, 49):
		trd_cf[i + 1] = USA.share_trd.diff().values[i + 1] + trd_cf[i]
	fin_cf = np.array(cf4.share_fin_nps_m).copy()
	for i in range(id_1990, 49):
		fin_cf[i + 1] = USA.share_fin.diff().values[i + 1] + fin_cf[i]
	bss_cf = np.array(cf4.share_bss_nps_m).copy()
	for i in range(id_1990, 49):
		bss_cf[i + 1] = USA.share_bss.diff().values[i + 1] + bss_cf[i]
	nps_cf = np.array(cf4.share_nps_nps_m).copy()
	for i in range(id_1990, 49):
		nps_cf[i + 1] = USA.share_nps.diff().values[i + 1] + nps_cf[i]
	A_cfs_ts.append((cf4.A_agr*agr_cf + cf4.A_man*man_cf + cf4.A_trd*trd_cf + cf4.A_fin*fin_cf + cf4.A_bss*bss_cf + cf4.A_nps*nps_cf)[:])
	A_cfs.append((cf4.A_agr*agr_cf + cf4.A_man*man_cf + cf4.A_trd*trd_cf + cf4.A_fin*fin_cf + cf4.A_bss*bss_cf + cf4.A_nps*nps_cf)[-1])

	cf_results.append((A_cfs[-1] / np.array(cf4.A_tot_nps)[-1] - 1) * 100)

A_cfs2 = []
A_cfs2_ts = []
for country in country_models:
	cf4 = country
	agr_cf = np.array(cf4.share_agr_nps_m).copy()
	for i in range(id_1990, 49):
		agr_cf[i + 1] = agr_cf[i]
	man_cf = np.array(cf4.share_man_nps_m).copy()
	for i in range(id_1990, 49):
		man_cf[i + 1] = man_cf[i]
	trd_cf = np.array(cf4.share_trd_nps_m).copy()
	for i in range(id_1990, 49):
		trd_cf[i + 1] = trd_cf[i]
	fin_cf = np.array(cf4.share_fin_nps_m).copy()
	for i in range(id_1990, 49):
		fin_cf[i + 1] =  fin_cf[i]
	bss_cf = np.array(cf4.share_bss_nps_m).copy()
	for i in range(id_1990, 49):
		bss_cf[i + 1] =  bss_cf[i]
	nps_cf = np.array(cf4.share_nps_nps_m).copy()
	for i in range(id_1990, 49):
		nps_cf[i + 1] =  nps_cf[i]
	A_cfs2_ts.append((cf4.A_agr*agr_cf + cf4.A_man*man_cf + cf4.A_trd*trd_cf + cf4.A_fin*fin_cf + cf4.A_bss*bss_cf + cf4.A_nps*nps_cf)[:])
	A_cfs2.append((cf4.A_agr*agr_cf + cf4.A_man*man_cf + cf4.A_trd*trd_cf + cf4.A_fin*fin_cf + cf4.A_bss*bss_cf + cf4.A_nps*nps_cf)[-1])

# EU4
EU4_cf3 = (np.array(DEU.h_tot)[:]*A_cfs_ts[2][:] + np.array(FRA.h_tot)[:]*A_cfs_ts[9][:] + np.array(ITA.h_tot)[:]*A_cfs_ts[6][:] + np.array(GBR.h_tot)[:]*A_cfs_ts[11][:])/EUR4_h_tot[:]
EU4_cf3_init = (np.array(DEU.h_tot)[:]*A_cfs2_ts[2][:] + np.array(FRA.h_tot)[:]*A_cfs2_ts[9][:] + np.array(ITA.h_tot)[:]*A_cfs2_ts[6][:] + np.array(GBR.h_tot)[:]*A_cfs2_ts[11][:])/EUR4_h_tot[:]
rel_EU4_cf3 = EU4_cf3 / np.array(A_tot_nps).flatten()
rel_EU4_cf3_init = EU4_cf3_init / np.array(A_tot_nps).flatten()

pd.DataFrame({'cf2':rel_EU4_cf2, 'obs': rel_EU4_obs, 'cf3':rel_EU4_cf3, 'cf3_init':rel_EU4_cf3_init}).to_excel('../output/figures/Counterfactual_ts.xlsx', index = False, header = True)

# EU4
cf_results.append((np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1])
# EU15
cf_results.append((np.array(AUT.h_tot)[-1]*(A_cfs[0]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_nps)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_nps)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_nps)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_nps)[-1]-1)*100+ np.array(FIN.h_tot)[-1]*(A_cfs[8]/np.array(FIN.A_tot_nps)[-1]-1)*100+ np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_nps)[-1]-1)*100+ np.array(LUX.h_tot)[-1]*(A_cfs[10]/np.array(LUX.A_tot_nps)[-1]-1)*100+ np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_nps)[-1]-1)*100+ np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_nps)[-1]-1)*100+ np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_nps)[-1]-1)*100+ np.array(SWE.h_tot)[-1]*(A_cfs[14]/np.array(SWE.A_tot_nps)[-1]-1)*100)/EUR15_h_tot[-1])
# EUCORE
cf_results.append((np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_nps)[-1]-1)*100+np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_nps)[-1]-1)*100)/EURCORE_h_tot[-1])
# EUPERI
cf_results.append((np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_nps)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_nps)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_nps)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1])

cf_3.append(cf_results)

pd.DataFrame(cf_3).to_excel('../output/figures/Counterfactual_3.xlsx', index = False, header = False)

"""
C2 referee suggestion: productivity gap with US is zero after 1990
"""

'ams'
cf_4 = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15']]
A_cfs = []
cf_results = []

for country in country_models:
	cf4 = country
	agr_cf = np.array(cf4.A_agr).copy()
	# for i in range(id_1990, 49):
	# 	agr_cf[i + 1] = np.array(USA.A_agr)[i + 1]/np.array(USA.A_agr)[i] * agr_cf[i]
	man_cf = np.array(cf4.A_man).copy()
	# for i in range(id_1990, 49):
	# 	man_cf[i + 1] = np.array(USA.A_man)[i + 1]/np.array(USA.A_man)[i] * man_cf[i]
	ser_cf = np.array(cf4.A_ser).copy()
	for i in range(id_1990, 49):
		ser_cf[i + 1] = np.array(USA.A_ser)[i + 1]/np.array(USA.A_ser)[i] * ser_cf[i]
	# for i in range(id_1990, 49):
	# 	ams_cf[i + 1] = np.array(USA.A_ams)[i + 1]/np.array(USA.A_ams)[i] * ams_cf[i]
	A_cfs.append((cf4.share_agr_ams_m*agr_cf + cf4.share_man_ams_m*man_cf + cf4.share_ser_ams_m*ser_cf )[-1])
	cf_results.append((A_cfs[-1] / np.array(cf4.A_tot_ams)[-1] - 1) * 100)

# EU4
cf_results.append((np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_ams)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_ams)[-1]-1)*100)/EUR4_h_tot[-1])
# EU15
cf_results.append((np.array(AUT.h_tot)[-1]*(A_cfs[0]/np.array(AUT.A_tot_ams)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_ams)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_ams)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_ams)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_ams)[-1]-1)*100+ np.array(FIN.h_tot)[-1]*(A_cfs[8]/np.array(FIN.A_tot_ams)[-1]-1)*100+ np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_ams)[-1]-1)*100+ np.array(LUX.h_tot)[-1]*(A_cfs[10]/np.array(LUX.A_tot_ams)[-1]-1)*100+ np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_ams)[-1]-1)*100+ np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_ams)[-1]-1)*100+ np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_ams)[-1]-1)*100+ np.array(SWE.h_tot)[-1]*(A_cfs[14]/np.array(SWE.A_tot_ams)[-1]-1)*100)/EUR15_h_tot[-1])
# EUCORE
cf_results.append((np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_ams)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_ams)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_ams)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_ams)[-1]-1)*100+np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_ams)[-1]-1)*100)/EURCORE_h_tot[-1])
# EUPERI
cf_results.append((np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_ams)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_ams)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_ams)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_ams)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_ams)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_ams)[-1]-1)*100)/EURPERI_h_tot[-1])

cf_4.append(cf_results)

pd.DataFrame(cf_4).to_excel('../output/figures/Counterfactual_4_ams.xlsx', index = False, header = False)

'nps'
cf_4 = [['AUT', 'BEL', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SWE', 'EU4', "EUCORE", "EUPERI", 'EU15']]
A_cfs = []
cf_results = []

for country in country_models:
	cf4 = country
	agr_cf = np.array(cf4.A_agr).copy()
	# for i in range(id_1990, 49):
	# 	agr_cf[i + 1] = np.array(USA.A_agr)[i + 1]/np.array(USA.A_agr)[i] * agr_cf[i]
	man_cf = np.array(cf4.A_man).copy()
	# for i in range(id_1990, 49):
	# 	man_cf[i + 1] = np.array(USA.A_man)[i + 1]/np.array(USA.A_man)[i] * man_cf[i]
	trd_cf = np.array(cf4.A_trd).copy()
	for i in range(id_1990, 49):
		trd_cf[i + 1] = np.array(USA.A_trd)[i + 1]/np.array(USA.A_trd)[i] * trd_cf[i]
	fin_cf = np.array(cf4.A_fin).copy()
	for i in range(id_1990, 49):
		fin_cf[i + 1] = np.array(USA.A_fin)[i + 1]/np.array(USA.A_fin)[i] * fin_cf[i]
	bss_cf = np.array(cf4.A_bss).copy()
	for i in range(id_1990, 49):
		bss_cf[i + 1] = np.array(USA.A_bss)[i + 1]/np.array(USA.A_bss)[i] * bss_cf[i]
	nps_cf = np.array(cf4.A_nps).copy()
	# for i in range(id_1990, 49):
	# 	nps_cf[i + 1] = np.array(USA.A_nps)[i + 1]/np.array(USA.A_nps)[i] * nps_cf[i]
	A_cfs.append((cf4.share_agr_nps_m*agr_cf + cf4.share_man_nps_m*man_cf + cf4.share_trd_nps_m*trd_cf + cf4.share_fin_nps_m*fin_cf + cf4.share_bss_nps_m*bss_cf + cf4.share_nps_nps_m*nps_cf)[-1])
	cf_results.append((A_cfs[-1] / np.array(cf4.A_tot_nps)[-1] - 1) * 100)

# EU4
cf_results.append((np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_nps)[-1]-1)*100 + np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_nps)[-1]-1)*100)/EUR4_h_tot[-1])
# EU15
cf_results.append((np.array(AUT.h_tot)[-1]*(A_cfs[0]/np.array(AUT.A_tot_nps)[-1]-1)*100 + np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_nps)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_nps)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_nps)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_nps)[-1]-1)*100+ np.array(FIN.h_tot)[-1]*(A_cfs[8]/np.array(FIN.A_tot_nps)[-1]-1)*100+ np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_nps)[-1]-1)*100+ np.array(LUX.h_tot)[-1]*(A_cfs[10]/np.array(LUX.A_tot_nps)[-1]-1)*100+ np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_nps)[-1]-1)*100+ np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_nps)[-1]-1)*100+ np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_nps)[-1]-1)*100+ np.array(SWE.h_tot)[-1]*(A_cfs[14]/np.array(SWE.A_tot_nps)[-1]-1)*100)/EUR15_h_tot[-1])
# EUCORE
cf_results.append((np.array(BEL.h_tot)[-1]*(A_cfs[1]/np.array(BEL.A_tot_nps)[-1]-1)*100 + np.array(DEU.h_tot)[-1]*(A_cfs[2]/np.array(DEU.A_tot_nps)[-1]-1)*100 + np.array(DNK.h_tot)[-1]*(A_cfs[3]/np.array(DNK.A_tot_nps)[-1]-1)*100 + np.array(FRA.h_tot)[-1]*(A_cfs[9]/np.array(FRA.A_tot_nps)[-1]-1)*100+np.array(NLD.h_tot)[-1]*(A_cfs[12]/np.array(NLD.A_tot_nps)[-1]-1)*100)/EURCORE_h_tot[-1])
# EUPERI
cf_results.append((np.array(ESP.h_tot)[-1]*(A_cfs[4]/np.array(ESP.A_tot_nps)[-1]-1)*100+ np.array(IRL.h_tot)[-1]*(A_cfs[5]/np.array(IRL.A_tot_nps)[-1]-1)*100+ np.array(ITA.h_tot)[-1]*(A_cfs[6]/np.array(ITA.A_tot_nps)[-1]-1)*100+ np.array(GRC.h_tot)[-1]*(A_cfs[7]/np.array(GRC.A_tot_nps)[-1]-1)*100 + np.array(GBR.h_tot)[-1]*(A_cfs[11]/np.array(GBR.A_tot_nps)[-1]-1)*100 + np.array(PRT.h_tot)[-1]*(A_cfs[13]/np.array(PRT.A_tot_nps)[-1]-1)*100)/EURPERI_h_tot[-1])

cf_4.append(cf_results)

pd.DataFrame(cf_4).to_excel('../output/figures/Counterfactual_4.xlsx', index = False, header = False)