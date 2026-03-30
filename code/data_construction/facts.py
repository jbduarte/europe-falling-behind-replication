"""
=======================================================================================
Project: Structural Transformation and Productivity in Europe (with Duarte and Saenz)
Filename: facts.py
Description: This program constructs tables and plots for the facts section of the paper

Author: Joao B. Duarte
Last Modified: Apr 2023
=======================================================================================
"""

import matplotlib.pyplot as plt
from construct_dataset_facts import *
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


data = pd.read_csv('../Data/Final Data/euklems_2021.csv')
countries = ["AUT", "BEL", "DK", "FI", "FRA", "DEU", "ITA", "LU", "NL", "PT", "ES", "SE", "GB"]
data_lp = pd.read_excel('../Data/Raw Data/OECD_GDP_ph.xlsx')

'''
Construct EU BIG 4 Labor Productivity 
'''

big4 = ['DEU', 'ITA', 'GBR', 'FRA']

rel_lp = np.empty((len(np.arange(1970, 2019)), 4))

from statsmodels.tsa.filters.hp_filter import hpfilter

for i, country in enumerate(big4):
    filter = (data_lp["LOCATION"] == country) & (data_lp["MEASURE"] == "USD") & (data_lp["TIME"] <= 2018)
    filter_usa = (data_lp["LOCATION"] == "USA") & (data_lp["MEASURE"] == "USD") & (data_lp["TIME"] <= 2018)
    _, trend = hpfilter(data_lp.loc[filter, "Value"].values / data_lp.loc[filter_usa, "Value"].values, lamb=6.25)
    rel_lp[:,i] = trend

'''
------------
Plots
------------
'''
'Figure 1a: EU15 and EU Big Four'
fig, ax = plt.subplots()
ax.plot(np.arange(1970, 2019), np.ones(len(np.arange(1970, 2019))), 'grey', linestyle='dashed')
for i in range(4):
    ax.plot(np.arange(1970, 2019), rel_lp[:,i],
        lw=2,
        label=big4[i])
ax.legend(loc=9, bbox_to_anchor=(0.5, -0.13), ncol=3)
ax.set_ylabel('GDP per hour relative to U.S.', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
# ax.set_ylim(0.6, 1.1)
plt.xticks(np.arange(1975, 2020 + 5, 5))
ax.legend()
ax.grid(alpha=0.2)
fig.savefig('../Figures/fig_1a.pdf', bbox_inches="tight")
plt.show()

'Figure 1b: LP growth and the Rise of the Services sector'
gdp = pd.read_excel('../Data/Final Data/penn_gdp.xlsx')
gdp['gdp'] = gdp['gdp'] * 1_000_000

countries = ["US", "EU15", "AT", "BE", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LU", "NL", "PT", "ES", "SE", "GB"]
markers = ['v', 'o']
fig, ax = plt.subplots()
for i in range(len(countries)):
    if i == 0:
        scatter_us = ax.scatter(
            np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
                i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
            countries_fulldata[i].loc[(countries_fulldata[i].sector == 'trd'), 'LS'],
            color='red',
            alpha=1,
            marker=markers[0],
            s=15)
    # if i == 11 or i == 9:
    #     continue
    scatter = ax.scatter(np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
        i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
                         countries_fulldata[i].loc[(countries_fulldata[i].sector == 'trd'), 'LS'],
                         color='red',
                         alpha=0.1,
                         marker=markers[1])
scatter_us.set_label('Wholesale and retail trade: US')
scatter.set_label('Wholesale and retail trade: EU-15')
for i in range(len(countries)):
    if i == 0:
        scatter_us = ax.scatter(
            np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
                i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
            countries_fulldata[i].loc[(countries_fulldata[i].sector == 'bss'), 'LS'],
            color='blue',
            alpha=1,
            marker=markers[0],
            s=15)
    # if i == 11 or i == 9:
    #     continue
    scatter = ax.scatter(np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
        i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
                         countries_fulldata[i].loc[(countries_fulldata[i].sector == 'bss'), 'LS'],
                         color='blue',
                         alpha=0.1)
scatter_us.set_label('Business services: US')
scatter.set_label('Business services: EU-15')
for i in range(len(countries)):
    if i == 0:
        scatter_us = ax.scatter(
            np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
                i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
            countries_fulldata[i].loc[(countries_fulldata[i].sector == 'fin'), 'LS'],
            color='purple',
            alpha=1,
            marker=markers[0],
            s=15)
    # if i == 11 or i == 9:
    #     continue
    scatter = ax.scatter(np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
        i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
                         countries_fulldata[i].loc[(countries_fulldata[i].sector == 'fin'), 'LS'],
                         color='purple',
                         alpha=0.1)
scatter_us.set_label('Financial services: US')
scatter.set_label('Financial services: EU-15')
# for i in range(len(countries)):
#     if i == 0:
#         scatter_us = ax.scatter(
#             np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
#                 i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'edu'), 'LS'].values +
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'gov'), 'LS'].values +
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'hlt'), 'LS'].values +
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'per'), 'LS'].values +
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'res'), 'LS'].values +
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'rst'), 'LS'].values +
#             countries_fulldata[i].loc[(countries_fulldata[i].sector == 'trs'), 'LS'].values,
#             color='purple',
#             alpha=1,
#             marker=markers[0],
#             s=15)
#     # if i == 11 or i == 9:
#     #     continue
#     scatter = ax.scatter(np.log(gdp.loc[(gdp.country == countries[i]) & (gdp.year >= 1977) & (gdp.year <= 2018), 'gdp'].values / countries_fulldata[
#         i].loc[(countries_fulldata[i].sector == 'tot'), 'H'].values),
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'edu'), 'LS'].values +
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'gov'), 'LS'].values +
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'hlt'), 'LS'].values +
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'per'), 'LS'].values +
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'res'), 'LS'].values +
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'rst'), 'LS'].values +
#                          countries_fulldata[i].loc[(countries_fulldata[i].sector == 'trs'), 'LS'].values,
#                          color='purple',
#                          alpha=0.1)
# scatter_us.set_label('Other services: US')
# scatter.set_label('Other services: EU-15')
ax.set_ylabel('Share in total employment', fontsize=12)
ax.set_xlabel('Log of GDP per hour', fontsize=12)
ax.grid(alpha=0.2)
ax.legend(fontsize=10)
plt.savefig('../Figures/fig_1b.pdf', bbox_inches="tight")
fig.show()

'''
Table 1
'''


def annualized(x1, x2, t):
    return ((x2 / x1) ** (1 / t) - 1) * 100


'''
Compute total LP growth
'''
# get LS in 1995 and 2018
sectors = countries_fulldata[0].sector.unique().tolist()
LS_95 = np.empty((2, len(sectors)))
LS_18 = np.empty((2, len(sectors)))
for i in range(2):
    for k, sector in enumerate(sectors):
        LS_95[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 1995), 'LS']
        LS_18[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 2018), 'LS']
LS_data_95 = pd.DataFrame(LS_95[:, :-1], index=['US', 'EU-15'], columns=sectors[:-1])
LS_data_18 = pd.DataFrame(LS_18[:, :-1], index=['US', 'EU-15'], columns=sectors[:-1])

# get A_i in 1995 and 2018
ai_95 = np.empty((2, len(sectors)))
ai_18 = np.empty((2, len(sectors)))
for i in range(2):
    for k, sector in enumerate(sectors):
        ai_95[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 1995), 'L_PROD_normalized']
        ai_18[i, k] = countries_fulldata[i].loc[(countries_fulldata[i].sector == sector)
                                                & (countries_fulldata[i].year == 2018), 'L_PROD_normalized']
ai_data_95 = pd.DataFrame(ai_95, index=['US', 'EU-15'], columns=sectors)
ai_data_18 = pd.DataFrame(ai_18, index=['US', 'EU-15'], columns=sectors)

# compute aggregate labor productivity growth
LP_growth = annualized(np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1), np.sum(ai_data_18.iloc[:, :-1] * LS_data_18, axis=1), 2018 - 1995)
LP_growth = pd.DataFrame(LP_growth.values.reshape(1, 2), columns=['US', 'EU-15'])

# Shift-share Decomposition of LP growth
# compute growth effect
growth_effect = annualized(np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1), np.sum(ai_data_18.iloc[:, :-1] * LS_data_95, axis=1), 2018 - 1995)
growth_effect = pd.DataFrame(growth_effect.values.reshape(1, 2), columns=['US', 'EU-15'])

# compute reallocation effect
reallocation_effect = LP_growth - growth_effect

# Decomposition of LP growth across sectors
# lets try agr first
LP_growth_i = (ai_data_18.loc[:, 'agr'] * LS_data_18.loc[:, 'agr'] - ai_data_95.loc[:, 'agr'] * LS_data_95.loc[:, 'agr']) / \
              (np.sum(ai_data_18.iloc[:, :-1] * LS_data_18, axis=1) - np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1))
LP_growth_i = pd.DataFrame(LP_growth_i.values.reshape(1, 2), columns=['US', 'EU-15']) * LP_growth

growth_effect_i = (ai_data_18.loc[:, 'agr'] * LS_data_95.loc[:, 'agr'] - ai_data_95.loc[:, 'agr'] * LS_data_95.loc[:, 'agr']) / \
                  (np.sum(ai_data_18.iloc[:, :-1] * LS_data_95, axis=1) - np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1))
growth_effect_i = pd.DataFrame(growth_effect_i.values.reshape(1, 2), columns=['US', 'EU-15']) * growth_effect

reallocation_effect_i = LP_growth_i - growth_effect_i

# loop over remaining sectors
for sector in sectors[1:-1]:
    LP_growth_temp = (ai_data_18.loc[:, sector] * LS_data_18.loc[:, sector] - ai_data_95.loc[:, sector] * LS_data_95.loc[:, sector]) / \
                     (np.sum(ai_data_18.iloc[:, :-1] * LS_data_18, axis=1) - np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1))
    LP_growth_temp = pd.DataFrame(LP_growth_temp.values.reshape(1, 2), columns=['US', 'EU-15']) * LP_growth

    growth_effect_temp = (ai_data_18.loc[:, sector] * LS_data_95.loc[:, sector] - ai_data_95.loc[:, sector] * LS_data_95.loc[:, sector]) / \
                         (np.sum(ai_data_18.iloc[:, :-1] * LS_data_95, axis=1) - np.sum(ai_data_95.iloc[:, :-1] * LS_data_95, axis=1))
    growth_effect_temp = pd.DataFrame(growth_effect_temp.values.reshape(1, 2), columns=['US', 'EU-15']) * growth_effect

    reallocation_effect_temp = LP_growth_temp - growth_effect_temp

    # update dataframes
    LP_growth_i = LP_growth_i.append(LP_growth_temp)
    growth_effect_i = growth_effect_i.append(growth_effect_temp)
    reallocation_effect_i = reallocation_effect_i.append(reallocation_effect_temp)

LP_growth_i.index = sectors[:-1]
growth_effect_i.index = sectors[:-1]
reallocation_effect_i.index = sectors[:-1]

# Data for table 1
# Total LP growth: inside LP_growth dataframe
# Shift-share decomposition of LP growth: inside growth_effect and reallocation_effect dataframes
# Sectoral decomposition: inside LP_growth_i, growth_effect_i and reallocation_effect_i dataframes

LP_growth_i = round(LP_growth_i, 2)
growth_effect_i = round(growth_effect_i, 2)
reallocation_effect_i = round(reallocation_effect_i, 2)
