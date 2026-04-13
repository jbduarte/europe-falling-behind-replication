"""
Replication code for:
    Buiatti, C., Duarte, J. B., & Sáenz, L. F. (2026).
    "Europe Falling Behind: Structural Transformation and Labor Productivity
    Growth Differences Between Europe and the U.S."
    Journal of International Economics.

File:        table_ss_core_vs_periphery.py
Purpose:     Shift-share decomposition of aggregate labour productivity growth for
             EU Core (DE, FR, BE, NL, DK) and EU Periphery (EL, IE, PT, ES, IT, GB)
             alongside the U.S., for 1970-2019 and 1995-2019.
Pipeline:    Step 17/19 — Generates Tables A.9 and A.10 (core vs. periphery
             shift-share decomposition).
Inputs:      ../data/euklems_2023.csv (sectoral VA, VA_Q, H by country, year, sector).
Outputs:     ../output/tables/table_c2_core_vs_periphery_new.tex (Table A.9, 1970-2019),
             ../output/tables/table_c3_core_vs_periphery_new.tex (Table A.10, 1995-2019).
Dependencies: None — self-contained robustness companion to utils.table_1_ss_eu4.
"""

import pandas as pd
from matplotlib import rc
import numpy as np

rc('text', usetex=True)
rc('font', family='serif')

data = pd.read_csv('../data/euklems_2023.csv', index_col=[0, 1])
data.rename(index={'US': 'USA'}, inplace=True)


# Rebuild 'tot' as the sum of the six base sectors (agr, man, bss, fin, trd, nps).
# 'ser' and 'prs' are alternate composite labels in EUKLEMS that overlap the
# base sectors; dropping them keeps the aggregation a clean partition.
data = data.reset_index()
data = data[data.sector != "tot"]
sector_filter = (data.sector != 'ser') & (data.sector != 'prs')
data_total = data.loc[sector_filter, ['country', 'year', 'VA', 'H', 'VA_Q']].groupby(
    ["country", "year"]).aggregate(sum)
data_total = data_total.reset_index()
data_total.columns = ['country', 'year', 'VA', 'H', 'VA_Q']
data_total['sector'] = "tot"
data_total = data_total[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
final_data = data[['country', 'year', 'sector', 'VA', 'H', 'VA_Q']]
final_data = pd.concat((final_data, data_total), axis=0)
final_data = final_data.sort_values(['country', 'sector', 'year'])
data = final_data.copy()

'Labor Productivity'
data['y_l'] = (data['VA_Q'] / data['H']) * 100

'US data'
data_us = data.loc[data.country == 'USA']

# EU Core = Germany, France, Benelux, Denmark. Definition follows the
# Bayoumi-Eichengreen (1993) optimal-currency-area split repeated in the
# euro-crisis literature, with Italy and the southern periphery moved out.
# Aggregated on hours and real VA, then y_l rederived (same logic as table_1).
tot_sector_filter = data.sector == "tot"
EUCORE_countries = ['DE', "FR", "BE", "NL", "DK"]
data_EUCORE = pd.DataFrame()
data_EUCORE['H'] = data.loc[data.country.isin(EUCORE_countries), ["sector", "year", "H"]].groupby(["sector", "year"]).agg(sum)
data_EUCORE['VA_Q'] = data.loc[data.country.isin(EUCORE_countries), ["sector", "year", "VA_Q"]].groupby(["sector", "year"]).agg(sum)
data_EUCORE['y_l'] = data_EUCORE['VA_Q'] / data_EUCORE['H'] * 100
data_eu_core = data_EUCORE.copy()
data_eu_core = data_eu_core.reset_index()

# EU Periphery = Greece, Ireland, Iberia, Italy, UK. Includes the GIIPS
# economies most affected by the post-1999 productivity slowdown plus the UK
# (kept for cross-reference with the EU4 main-text aggregate).
tot_sector_filter = data.sector == "tot"
EUPERI_countries = ['EL', "IE", "PT", "ES", "IT", "GB"]
data_EUPERI = pd.DataFrame()
data_EUPERI['H'] = data.loc[data.country.isin(EUPERI_countries), ["sector", "year", "H"]].groupby(["sector", "year"]).agg(sum)
data_EUPERI['VA_Q'] = data.loc[data.country.isin(EUPERI_countries), ["sector", "year", "VA_Q"]].groupby(["sector", "year"]).agg(sum)
data_EUPERI['y_l'] = data_EUPERI['VA_Q'] / data_EUPERI['H'] * 100
data_eu_peri = data_EUPERI.copy()
data_eu_peri = data_eu_peri.reset_index()

# Compute emp. shares

"US"
grouped = data_us.groupby(['sector'])
data_us['LS'] = data_us.groupby(['sector'])['H'].transform(lambda x: x / grouped['H'].get_group('tot').values)

data_us_nps = data_us[(data_us.sector != "tot") & (data_us.sector != "ser") & (data_us.sector != "prs")]

lp_1970_us_nps = data_us_nps.loc[data_us_nps.year == 1970, ["sector", "y_l"]]
lp_1995_us_nps = data_us_nps.loc[data_us_nps.year == 1995, ["sector", "y_l"]]
lp_2019_us_nps = data_us_nps.loc[data_us_nps.year == 2019, ["sector", "y_l"]]

l_1970_us_nps = data_us_nps.loc[data_us_nps.year == 1970, ["sector", "LS"]]
l_1995_us_nps = data_us_nps.loc[data_us_nps.year == 1995, ["sector", "LS"]]
l_2019_us_nps = data_us_nps.loc[data_us_nps.year == 2019, ["sector", "LS"]]

"EU CORE"
grouped = data_eu_core.groupby(['sector'])
data_eu_core['LS'] = data_eu_core.groupby(['sector'])['H'].transform(lambda x: x / grouped['H'].get_group('tot').values)

data_eu_core_nps = data_eu_core[(data_eu_core.sector != "tot") & (data_eu_core.sector != "ser") & (data_eu_core.sector != "prs")]

lp_1970_eu_core_nps = data_eu_core_nps.loc[data_eu_core_nps.year == 1970, ["sector", "y_l"]]
lp_1995_eu_core_nps = data_eu_core_nps.loc[data_eu_core_nps.year == 1995, ["sector", "y_l"]]
lp_2019_eu_core_nps = data_eu_core_nps.loc[data_eu_core_nps.year == 2019, ["sector", "y_l"]]

l_1970_eu_core_nps = data_eu_core_nps.loc[data_eu_core_nps.year == 1970, ["sector", "LS"]]
l_1995_eu_core_nps = data_eu_core_nps.loc[data_eu_core_nps.year == 1995, ["sector", "LS"]]
l_2019_eu_core_nps = data_eu_core_nps.loc[data_eu_core_nps.year == 2019, ["sector", "LS"]]

"EU PERI"
grouped = data_eu_peri.groupby(['sector'])
data_eu_peri['LS'] = data_eu_peri.groupby(['sector'])['H'].transform(lambda x: x / grouped['H'].get_group('tot').values)

data_eu_peri_nps = data_eu_peri[(data_eu_peri.sector != "tot") & (data_eu_peri.sector != "ser") & (data_eu_peri.sector != "prs")]

lp_1970_eu_peri_nps = data_eu_peri_nps.loc[data_eu_peri_nps.year == 1970, ["sector", "y_l"]]
lp_1995_eu_peri_nps = data_eu_peri_nps.loc[data_eu_peri_nps.year == 1995, ["sector", "y_l"]]
lp_2019_eu_peri_nps = data_eu_peri_nps.loc[data_eu_peri_nps.year == 2019, ["sector", "y_l"]]

l_1970_eu_peri_nps = data_eu_peri_nps.loc[data_eu_peri_nps.year == 1970, ["sector", "LS"]]
l_1995_eu_peri_nps = data_eu_peri_nps.loc[data_eu_peri_nps.year == 1995, ["sector", "LS"]]
l_2019_eu_peri_nps = data_eu_peri_nps.loc[data_eu_peri_nps.year == 2019, ["sector", "LS"]]


def shift_share(lp_0, lp_T, l_0, l_T):
    """
    Shift-share decomposition of aggregate labour productivity growth.
    See utils.table_1_ss_eu4.shift_share for the full formula. Inputs are 1-D
    arrays of length = number of sectors; returns dict with LP_growth,
    within_effect and shift_effect as sector-level vectors.
    """
    # Aggregate LP in base and final years (sector-level products)
    LP_0 = (lp_0 * l_0)
    LP_T = (lp_T * l_T)

    LP_growth = (LP_T - LP_0)

    # Within-sector productivity growth effect (shares held fixed at base year)
    within_growth = ((lp_T - lp_0) * l_0)

    # Reallocation (shift) effect, including the interaction term
    shift_growth = (((l_T - l_0) * lp_0) + ((l_T - l_0) * (lp_T - lp_0)))

    return {"LP_growth": LP_growth, "within_effect": within_growth, "shift_effect": shift_growth}

## With normalized LP

"1970-2019 period for main text"

ss_us = shift_share(np.ones(6),
            1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values,
            l_1970_us_nps['LS'].values,
            l_2019_us_nps['LS'].values)

# Counterfactual: replace EU Core manufacturing growth (sector index 1 in
# the alphabetised order agr/man/bss/fin/trd/nps) with the U.S. figure to
# isolate the contribution of slower European manufacturing productivity.
cf_eu_core = 1 + (lp_2019_eu_core_nps['y_l'].values - lp_1970_eu_core_nps['y_l'].values)/lp_1970_eu_core_nps['y_l'].values
cf_eu_core[1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[1]
#cf_eu_core[-1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[-1]

ss_eu_core = shift_share(np.ones(6),
            1 + (lp_2019_eu_core_nps['y_l'].values - lp_1970_eu_core_nps['y_l'].values)/lp_1970_eu_core_nps['y_l'].values,
            l_1970_eu_core_nps['LS'].values,
            l_2019_eu_core_nps['LS'].values)


ss_eu_core_cf = shift_share(np.ones(6),
            cf_eu_core,
            l_1970_eu_core_nps['LS'].values,
            l_2019_eu_core_nps['LS'].values)

# Same counterfactual swap for the periphery aggregate.
cf_eu_peri = 1 + (lp_2019_eu_peri_nps['y_l'].values - lp_1970_eu_peri_nps['y_l'].values)/lp_1970_eu_peri_nps['y_l'].values
cf_eu_peri[1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[1]
#cf_eu_peri[-1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[-1]

ss_eu_peri = shift_share(np.ones(6),
            1 + (lp_2019_eu_peri_nps['y_l'].values - lp_1970_eu_peri_nps['y_l'].values)/lp_1970_eu_peri_nps['y_l'].values,
            l_1970_eu_peri_nps['LS'].values,
            l_2019_eu_peri_nps['LS'].values)


ss_eu_peri_cf = shift_share(np.ones(6),
            cf_eu_peri,
            l_1970_eu_peri_nps['LS'].values,
            l_2019_eu_peri_nps['LS'].values)

# 49-year annualisation horizon (1970-2019) for the Table A.9 panel.
def annualized(x):
    return ((x) ** (1 / 49) - 1) * 100

LP_us = annualized(1 + ss_us["LP_growth"].sum())
LP_eu_core = annualized(1 + ss_eu_core["LP_growth"].sum())
LP_eu_peri = annualized(1 + ss_eu_peri["LP_growth"].sum())
LP_eu_core_cf = annualized(1 + ss_eu_core_cf["LP_growth"].sum())
LP_eu_peri_cf = annualized(1 + ss_eu_peri_cf["LP_growth"].sum())

print("\n\n 1970-2019 period")

print("LP")
print("US LP", round(LP_us, 2))
print("EU Core LP", round(LP_eu_core, 2))
print("EU Periphery LP", round(LP_eu_peri, 2))

# Sectoral contribution
sectors = lp_2019_eu_core_nps["sector"].values
print("\nSectoral contribution")
for i in range(6):
    print("US", sectors[i], round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU Core", sectors[i], round(ss_eu_core["LP_growth"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))

print("")
for i in range(6):
    print("EU Periphery", sectors[i], round(ss_eu_peri["LP_growth"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))


# Shit-share decomposition
print("\n Shift share for aggregate LP")
print("US within effect", round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))
print("US shift effect", round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))

print("EU Core within effect", round(ss_eu_core["within_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))
print("EU Core shift effect", round(ss_eu_core["shift_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))

print("EU Periphery within effect", round(ss_eu_peri["within_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))
print("EU Periphery shift effect", round(ss_eu_peri["shift_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))

# Shit-share decomposition by sector
print("\n Shift share for each sector")
for i in range(6):
    print("US within effect", sectors[i], round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
    print("US shift effect", sectors[i], round(ss_us["shift_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU Core within effect", sectors[i], round(ss_eu_core["within_effect"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))
    print("EU Core shift effect", sectors[i], round(ss_eu_core["shift_effect"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))

print("")
for i in range(6):
    print("EU Periphery within effect", sectors[i], round(ss_eu_peri["within_effect"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))
    print("EU Periphery shift effect", sectors[i], round(ss_eu_peri["shift_effect"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))

table_position_dic = {"agr": 1, "man": 2, "ser": 3, "bss": 4, "fin": 5, "trd": 6, "nps": 7}

table_results = np.zeros((8, 9))
table_results[0, 0] = round(LP_us, 2)
table_results[0, 1] = round(LP_eu_core, 2)
table_results[0, 2] = round(LP_eu_peri, 2)

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 0] = sectors_dic_temp[sector]
table_results[3, 0] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_eu_core["LP_growth"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 1] = sectors_dic_temp[sector]
table_results[3, 1] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_eu_peri["LP_growth"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 2] = sectors_dic_temp[sector]
table_results[3, 2] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

table_results[0, 3] = round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)
table_results[0, 4] = round(ss_eu_core["within_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2)
table_results[0, 5] = round(ss_eu_peri["within_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2)

table_results[0, 6] = round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)
table_results[0, 7] = round(ss_eu_core["shift_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2)
table_results[0, 8] = round(ss_eu_peri["shift_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2)

# Within effect
sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 3] = sectors_dic_temp[sector]
table_results[3, 3] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_eu_core["within_effect"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 4] = sectors_dic_temp[sector]
table_results[3, 4] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_eu_peri["within_effect"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 5] = sectors_dic_temp[sector]
table_results[3, 5] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

# Shift effect: computed as residual (LP - Growth) to enforce row identity
for row_idx in [1, 2, 4, 5, 6, 7]:  # agr, man, bss, fin, trd, nps
    table_results[row_idx, 6] = round(table_results[row_idx, 0] - table_results[row_idx, 3], 2)  # US
    table_results[row_idx, 7] = round(table_results[row_idx, 1] - table_results[row_idx, 4], 2)  # EU Core
    table_results[row_idx, 8] = round(table_results[row_idx, 2] - table_results[row_idx, 5], 2)  # EU Peri
# Services row: sum of subsectors
table_results[3, 6] = table_results[4, 6] + table_results[5, 6] + table_results[6, 6] + table_results[7, 6]
table_results[3, 7] = table_results[4, 7] + table_results[5, 7] + table_results[6, 7] + table_results[7, 7]
table_results[3, 8] = table_results[4, 8] + table_results[5, 8] + table_results[6, 8] + table_results[7, 8]


index_table = ["tot"] + list(table_position_dic.keys())
arrays = [
    ["LP growth"] * 3 + ["Growth effect"] * 3 + ["Shift effect"] * 3,
    ["U.S.", "EU Core", 'EU Periphery'] * 3,
]
tuples = list(zip(*arrays))
table = pd.DataFrame(table_results, index=index_table, columns=tuples)
table.style.format("{:.2f}").to_latex("../output/tables/table_c2_core_vs_periphery_new.tex")

"1995-2019 period"

A_1995_us = 1 + (lp_1995_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values
ss_us = shift_share(A_1995_us,
            A_1995_us*(1 + (lp_2019_us_nps['y_l'].values - lp_1995_us_nps['y_l'].values)/lp_1995_us_nps['y_l'].values),
            l_1995_us_nps['LS'].values,
            l_2019_us_nps['LS'].values)

A_1995_eu_core = 1 + (lp_1995_eu_core_nps['y_l'].values - lp_1970_eu_core_nps['y_l'].values)/lp_1970_eu_core_nps['y_l'].values
ss_eu_core = shift_share(A_1995_eu_core,
            A_1995_eu_core*(1 + (lp_2019_eu_core_nps['y_l'].values - lp_1995_eu_core_nps['y_l'].values)/lp_1995_eu_core_nps['y_l'].values),
            l_1995_eu_core_nps['LS'].values,
            l_2019_eu_core_nps['LS'].values)

A_1995_eu_peri = 1 + (lp_1995_eu_peri_nps['y_l'].values - lp_1970_eu_peri_nps['y_l'].values)/lp_1970_eu_peri_nps['y_l'].values
ss_eu_peri = shift_share(A_1995_eu_peri,
            A_1995_eu_peri*(1 + (lp_2019_eu_peri_nps['y_l'].values - lp_1995_eu_peri_nps['y_l'].values)/lp_1995_eu_peri_nps['y_l'].values),
            l_1995_eu_peri_nps['LS'].values,
            l_2019_eu_peri_nps['LS'].values)

# 24-year annualisation horizon (1995-2019) for the Table A.10 panel.
def annualized(x):
    return ((x) ** (1 / 24) - 1) * 100

LP_us = annualized(1 + ss_us["LP_growth"].sum())
LP_eu_core = annualized(1 + ss_eu_core["LP_growth"].sum())
LP_eu_peri = annualized(1 + ss_eu_peri["LP_growth"].sum())

print("\n\n 1995-2019 period")

print("LP")
print("US LP", round(LP_us, 2))
print("EU Core LP", round(LP_eu_core, 2))
print("EU Periphery LP", round(LP_eu_peri, 2))

# Sectoral contribution
sectors = lp_2019_eu_core_nps["sector"].values
print("\nSectoral contribution")
for i in range(6):
    print("US", sectors[i], round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU Core", sectors[i], round(ss_eu_core["LP_growth"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))

print("")
for i in range(6):
    print("EU Periphery", sectors[i], round(ss_eu_peri["LP_growth"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))

# Shit-share decomposition
print("\n Shift share for aggregate LP")
print("US within effect", round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))
print("US shift effect", round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))

print("EU Core within effect", round(ss_eu_core["within_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))
print("EU Core shift effect", round(ss_eu_core["shift_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))

print("EU Periphery within effect", round(ss_eu_peri["within_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))
print("EU Periphery shift effect", round(ss_eu_peri["shift_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))

# Shit-share decomposition by sector
print("\n Shift share for each sector")
for i in range(6):
    print("US within effect", sectors[i], round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
    print("US shift effect", sectors[i], round(ss_us["shift_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU Core within effect", sectors[i], round(ss_eu_core["within_effect"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))
    print("EU Core shift effect", sectors[i], round(ss_eu_core["shift_effect"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))

print("")
for i in range(6):
    print("EU Periphery within effect", sectors[i], round(ss_eu_peri["within_effect"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))
    print("EU Periphery shift effect", sectors[i], round(ss_eu_peri["shift_effect"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))

table_position_dic = {"agr": 1, "man": 2, "ser": 3, "bss": 4, "fin": 5, "trd": 6, "nps": 7}


table_results = np.zeros((8, 9))
table_results[0, 0] = round(LP_us, 2)
table_results[0, 1] = round(LP_eu_core, 2)
table_results[0, 2] = round(LP_eu_peri, 2)

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 0] = sectors_dic_temp[sector]
table_results[3, 0] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_eu_core["LP_growth"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 1] = sectors_dic_temp[sector]
table_results[3, 1] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_eu_peri["LP_growth"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 2] = sectors_dic_temp[sector]
table_results[3, 2] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

table_results[0, 3] = round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)
table_results[0, 4] = round(ss_eu_core["within_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2)
table_results[0, 5] = round(ss_eu_peri["within_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2)

table_results[0, 6] = round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)
table_results[0, 7] = round(ss_eu_core["shift_effect"].sum() / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2)
table_results[0, 8] = round(ss_eu_peri["shift_effect"].sum() / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2)

# Within effect
sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 3] = sectors_dic_temp[sector]
table_results[3, 3] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_eu_core["within_effect"][i] / ss_eu_core["LP_growth"].sum() * LP_eu_core, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 4] = sectors_dic_temp[sector]
table_results[3, 4] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_eu_peri["within_effect"][i] / ss_eu_peri["LP_growth"].sum() * LP_eu_peri, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 5] = sectors_dic_temp[sector]
table_results[3, 5] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

# Shift effect: computed as residual (LP - Growth) to enforce row identity
for row_idx in [1, 2, 4, 5, 6, 7]:  # agr, man, bss, fin, trd, nps
    table_results[row_idx, 6] = round(table_results[row_idx, 0] - table_results[row_idx, 3], 2)  # US
    table_results[row_idx, 7] = round(table_results[row_idx, 1] - table_results[row_idx, 4], 2)  # EU Core
    table_results[row_idx, 8] = round(table_results[row_idx, 2] - table_results[row_idx, 5], 2)  # EU Peri
# Services row: sum of subsectors
table_results[3, 6] = table_results[4, 6] + table_results[5, 6] + table_results[6, 6] + table_results[7, 6]
table_results[3, 7] = table_results[4, 7] + table_results[5, 7] + table_results[6, 7] + table_results[7, 7]
table_results[3, 8] = table_results[4, 8] + table_results[5, 8] + table_results[6, 8] + table_results[7, 8]


index_table = ["tot"] + list(table_position_dic.keys())
arrays = [
    ["LP growth"] * 3 + ["Growth effect"] * 3 + ["Shift effect"] * 3,
    ["U.S.", "EU Core", 'EU Periphery'] * 3,
]
tuples = list(zip(*arrays))
table = pd.DataFrame(table_results, index=index_table, columns=tuples)
table.style.format("{:.2f}").to_latex("../output/tables/table_c3_core_vs_periphery_new.tex")