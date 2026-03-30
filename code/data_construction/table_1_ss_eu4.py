import pandas as pd
import statsmodels.api as sm
from matplotlib import rc
import numpy as np

rc('text', usetex=True)
rc('font', family='serif')

'''
-----------
Data
-----------
'''

'KLEMS'
data = pd.read_csv('../Data/Final Data/euklems_2023.csv', index_col=[0, 1])
data.rename(index={'US': 'USA'}, inplace=True)


# compute Total
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

'EU4 data'
tot_sector_filter = data.sector == "tot"
EU4_countries = ['DE', "FR", "IT", "GB"]
data_EU4 = pd.DataFrame()
data_EU4['H'] = data.loc[data.country.isin(EU4_countries), ["sector", "year", "H"]].groupby(["sector", "year"]).agg(sum)
data_EU4['VA_Q'] = data.loc[data.country.isin(EU4_countries), ["sector", "year", "VA_Q"]].groupby(["sector", "year"]).agg(sum)
data_EU4['y_l'] = data_EU4['VA_Q'] / data_EU4['H'] * 100
data_eu = data_EU4.copy()
data_eu = data_eu.reset_index()

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

"EU4"
grouped = data_eu.groupby(['sector'])
data_eu['LS'] = data_eu.groupby(['sector'])['H'].transform(lambda x: x / grouped['H'].get_group('tot').values)

data_eu_nps = data_eu[(data_eu.sector != "tot") & (data_eu.sector != "ser") & (data_eu.sector != "prs")]

lp_1970_eu_nps = data_eu_nps.loc[data_eu_nps.year == 1970, ["sector", "y_l"]]
lp_1995_eu_nps = data_eu_nps.loc[data_eu_nps.year == 1995, ["sector", "y_l"]]
lp_2019_eu_nps = data_eu_nps.loc[data_eu_nps.year == 2019, ["sector", "y_l"]]

l_1970_eu_nps = data_eu_nps.loc[data_eu_nps.year == 1970, ["sector", "LS"]]
l_1995_eu_nps = data_eu_nps.loc[data_eu_nps.year == 1995, ["sector", "LS"]]
l_2019_eu_nps = data_eu_nps.loc[data_eu_nps.year == 2019, ["sector", "LS"]]


def shift_share(lp_0, lp_T, l_0, l_T):
    """
    all vectors with values for all sectors
    """
    # Compute aggregate LP growth
    LP_0 = (lp_0 * l_0)
    LP_T = (lp_T * l_T)

    LP_growth = (LP_T - LP_0)

    # Within-sector productivity growth effect
    within_growth = ((lp_T - lp_0) * l_0)

    # Shift effect
    shift_growth = (((l_T - l_0) * lp_0) + ((l_T - l_0) * (lp_T - lp_0)))

    return {"LP_growth": LP_growth, "within_effect": within_growth, "shift_effect": shift_growth}

## With normalized LP

"1970-2019 period for main text"

ss_us = shift_share(np.ones(6),
            1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values,
            l_1970_us_nps['LS'].values,
            l_2019_us_nps['LS'].values)

cf_eu = 1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
cf_eu[1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[1]
#cf_eu[-1] = (1 + (lp_2019_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values)[-1]

ss_eu = shift_share(np.ones(6),
            1 + (lp_2019_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values,
            l_1970_eu_nps['LS'].values,
            l_2019_eu_nps['LS'].values)


ss_eu_cf = shift_share(np.ones(6),
            cf_eu,
            l_1970_eu_nps['LS'].values,
            l_2019_eu_nps['LS'].values)

def annualized(x):
    return ((x) ** (1 / 49) - 1) * 100

LP_us = annualized(1 + ss_us["LP_growth"].sum())
LP_eu = annualized(1 + ss_eu["LP_growth"].sum())
LP_eu_cf = annualized(1 + ss_eu_cf["LP_growth"].sum())

print("\n\n 1970-2019 period")

print("LP")
print("US LP", round(LP_us, 2))
print("EU LP", round(LP_eu, 2))

# Sectoral contribution
sectors = lp_2019_eu_nps["sector"].values
print("\nSectoral contribution")
for i in range(6):
    print("US", sectors[i], round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU", sectors[i], round(ss_eu["LP_growth"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))

# Shit-share decomposition
print("\n Shift share for aggregate LP")
print("US within effect", round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))
print("US shift effect", round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))

print("EU within effect", round(ss_eu["within_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2))
print("EU shift effect", round(ss_eu["shift_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2))

# Shit-share decomposition by sector
print("\n Shift share for each sector")
for i in range(6):
    print("US within effect", sectors[i], round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
    print("US shift effect", sectors[i], round(ss_us["shift_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU within effect", sectors[i], round(ss_eu["within_effect"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))
    print("EU shift effect", sectors[i], round(ss_eu["shift_effect"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))

table_position_dic = {"agr": 1, "man": 2, "ser": 3, "bss": 4, "fin": 5, "trd": 6, "nps": 7}

table_results = np.zeros((8, 6))
table_results[0, 0] = round(LP_us, 2)
table_results[0, 1] = round(LP_eu, 2)

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 0] = sectors_dic_temp[sector]
table_results[3, 0] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_eu["LP_growth"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))
for sector in sectors:
     table_results[table_position_dic[sector], 1] = sectors_dic_temp[sector]
table_results[3, 1] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

table_results[0, 2] = round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)
table_results[0, 4] = round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)

table_results[0, 3] = round(ss_eu["within_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2)
table_results[0, 5] = round(ss_eu["shift_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2)

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 2] = sectors_dic_temp[sector]
table_results[3, 2] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_eu["within_effect"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2)
for sector in sectors:
     table_results[table_position_dic[sector], 3] = sectors_dic_temp[sector]
table_results[3, 3] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

# Shift effect: computed as residual (LP - Growth) to enforce row identity
for row_idx in [1, 2, 4, 5, 6, 7]:  # agr, man, bss, fin, trd, nps
    table_results[row_idx, 4] = round(table_results[row_idx, 0] - table_results[row_idx, 2], 2)
    table_results[row_idx, 5] = round(table_results[row_idx, 1] - table_results[row_idx, 3], 2)
table_results[3, 4] = table_results[4, 4] + table_results[5, 4] + table_results[6, 4] + table_results[7, 4]
table_results[3, 5] = table_results[4, 5] + table_results[5, 5] + table_results[6, 5] + table_results[7, 5]


index_table = ["tot"] + list(table_position_dic.keys())
arrays = [
    ["LP growth"] * 2 + ["Growth effect"] * 2 + ["Shift effect"] * 2,
    ["U.S.", "Europe"] * 3,
]
tuples = list(zip(*arrays))
table = pd.DataFrame(table_results, index=index_table, columns=tuples)
table.to_excel("../Tables/table1_ss.xlsx")

"1995-2019 period for main text"

A_1995_us = 1 + (lp_1995_us_nps['y_l'].values - lp_1970_us_nps['y_l'].values)/lp_1970_us_nps['y_l'].values
ss_us = shift_share(A_1995_us,
            A_1995_us*(1 + (lp_2019_us_nps['y_l'].values - lp_1995_us_nps['y_l'].values)/lp_1995_us_nps['y_l'].values),
            l_1995_us_nps['LS'].values,
            l_2019_us_nps['LS'].values)

A_1995_eu = 1 + (lp_1995_eu_nps['y_l'].values - lp_1970_eu_nps['y_l'].values)/lp_1970_eu_nps['y_l'].values
ss_eu = shift_share(A_1995_eu,
            A_1995_eu*(1 + (lp_2019_eu_nps['y_l'].values - lp_1995_eu_nps['y_l'].values)/lp_1995_eu_nps['y_l'].values),
            l_1995_eu_nps['LS'].values,
            l_2019_eu_nps['LS'].values)

def annualized(x):
    return ((x) ** (1 / 24) - 1) * 100

LP_us = annualized(1 + ss_us["LP_growth"].sum())
LP_eu = annualized(1 + ss_eu["LP_growth"].sum())

print("\n\n 1995-2019 period")

print("LP")
print("US LP", round(LP_us, 2))
print("EU LP", round(LP_eu, 2))

# Sectoral contribution
sectors = lp_2019_eu_nps["sector"].values
print("\nSectoral contribution")
for i in range(6):
    print("US", sectors[i], round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU", sectors[i], round(ss_eu["LP_growth"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))

# Shit-share decomposition
print("\n Shift share for aggregate LP")
print("US within effect", round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))
print("US shift effect", round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2))

print("EU within effect", round(ss_eu["within_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2))
print("EU shift effect", round(ss_eu["shift_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2))

# Shit-share decomposition by sector
print("\n Shift share for each sector")
for i in range(6):
    print("US within effect", sectors[i], round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
    print("US shift effect", sectors[i], round(ss_us["shift_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2))

print("")
for i in range(6):
    print("EU within effect", sectors[i], round(ss_eu["within_effect"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))
    print("EU shift effect", sectors[i], round(ss_eu["shift_effect"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))

table_position_dic = {"agr": 1, "man": 2, "ser": 3, "bss": 4, "fin": 5, "trd": 6, "nps": 7}

table_results_2 = np.zeros((8, 6))
table_results_2[0, 0] = round(LP_us, 2)
table_results_2[0, 1] = round(LP_eu, 2)

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_us["LP_growth"][i] / ss_us["LP_growth"].sum() * LP_us, 2))
for sector in sectors:
     table_results_2[table_position_dic[sector], 0] = sectors_dic_temp[sector]
table_results_2[3, 0] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = (round(ss_eu["LP_growth"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2))
for sector in sectors:
     table_results_2[table_position_dic[sector], 1] = sectors_dic_temp[sector]
table_results_2[3, 1] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

table_results_2[0, 2] = round(ss_us["within_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)
table_results_2[0, 4] = round(ss_us["shift_effect"].sum() / ss_us["LP_growth"].sum() * LP_us, 2)

table_results_2[0, 3] = round(ss_eu["within_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2)
table_results_2[0, 5] = round(ss_eu["shift_effect"].sum() / ss_eu["LP_growth"].sum() * LP_eu, 2)

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_us["within_effect"][i] / ss_us["LP_growth"].sum() * LP_us, 2)
for sector in sectors:
     table_results_2[table_position_dic[sector], 2] = sectors_dic_temp[sector]
table_results_2[3, 2] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

sectors_dic_temp = {}
for i, sector in enumerate(sectors):
     sectors_dic_temp[sector] = round(ss_eu["within_effect"][i] / ss_eu["LP_growth"].sum() * LP_eu, 2)
for sector in sectors:
     table_results_2[table_position_dic[sector], 3] = sectors_dic_temp[sector]
table_results_2[3, 3] = sectors_dic_temp["bss"] + sectors_dic_temp["fin"] + sectors_dic_temp["trd"] + sectors_dic_temp["nps"]

# Shift effect: computed as residual (LP - Growth) to enforce row identity
for row_idx in [1, 2, 4, 5, 6, 7]:  # agr, man, bss, fin, trd, nps
    table_results_2[row_idx, 4] = round(table_results_2[row_idx, 0] - table_results_2[row_idx, 2], 2)
    table_results_2[row_idx, 5] = round(table_results_2[row_idx, 1] - table_results_2[row_idx, 3], 2)
table_results_2[3, 4] = table_results_2[4, 4] + table_results_2[5, 4] + table_results_2[6, 4] + table_results_2[7, 4]
table_results_2[3, 5] = table_results_2[4, 5] + table_results_2[5, 5] + table_results_2[6, 5] + table_results_2[7, 5]
